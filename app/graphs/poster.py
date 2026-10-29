"""Poster graph: generate a ready-to-publish post about kitchen restoration."""
import logging
import uuid
from pathlib import Path
from typing import TypedDict, Optional, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from app.llm import get_poster_llm
from app.db import save_draft, get_recent_posts
from app.config import settings

logger = logging.getLogger(__name__)

class PosterState(TypedDict):
    topic: str
    text: Optional[str]
    image_prompt: Optional[str]
    image_path: Optional[str]
    image_paths: Optional[list]
    draft_id: Optional[int]
    error: Optional[str]
    short_topic: Optional[str]

class PostOutput(BaseModel):
    """Structured output for a channel post."""
    post_text: str = Field(description="Готовый текст поста на русском, 400–650 символов. Живой, понятный, процессный и результатный. Без продажи и воды.")
    short_topic: str = Field(description="Короткая суть в 2–4 словах")
    image_prompt: str = Field(
        description=(
            "Короткий промпт на английском для реалистичной фотографии кухни. "
            "Простой вид обновлённой кухни, фасадов или столешницы. "
            "Без сюрреализма, без лишних объектов, без предметов внутри шкафов."
        )
    )

SYSTEM_POSTER = """Ты — мастер по реставрации кухонь, который ведёт Telegram-канал.

Пиши живые, понятные посты про реальные подходы к реставрации: что оставляем, что обновляем, как готовим и красим, какой получается результат.

Один пост — одна конкретная идея или типичный сценарий работы.

Стиль:
- Начинай с сути: зачем это делают и в чём смысл работы (чтобы кухня снова выглядела цельно и ею было удобно пользоваться).
- Коротко и просто описывай процесс: каркас часто оставляем если он крепкий, обновляем покрытие фасадов, фурнитуру, при необходимости столешницу. Поверхность готовим, грунтуем и красим плотным ровным слоем. Цвет и ручки подбираем так, чтобы всё смотрелось одним комплектом.
- Показывай результат: свежий вид, ровные фасады, спокойное открывание, кухня выглядит собранно и ею проще пользоваться.
- Короткие абзацы. Простой уверенный язык без пафоса и литературщины.
- Длина 400–650 символов.
- Эмодзи почти не используй (0–1).
- Без жёсткой продажи, без призывов «звоните», без точных цен и сроков. Можно мягко намекнуть, что делаете такие работы, если основа ещё живая.
- Не выдумывай редкие поломки и странные ситуации.
- Не повторяй последние темы.

Для картинки:
- Только простой реалистичный фото готовой или обновлённой кухни / фасадов / столешницы.
- Без сюрреализма, без лишних предметов, без объектов внутри шкафов.
"""

def generate_post_node(state: PosterState) -> PosterState:
    llm = get_poster_llm().with_structured_output(PostOutput)
    recent = get_recent_posts(12)
    recent_text = ""
    if recent:
        recent_text = "\n\nПоследние темы (не повторяй):\n" + "\n".join(f"- {t}" for t in recent[:12])

    custom_topic = state.get("topic")
    image_paths = state.get("image_paths") or []

    if custom_topic and custom_topic.lower() not in ("реставрация кухонь", "кухни", ""):
        text_prompt = (
            f"Напиши пост строго на тему: «{custom_topic}».\n"
            "Сохрани стиль мастера.\n"
            "Если приложены фото — выделяй главное по теме, не описывай всё подряд."
            f"{recent_text}"
        )
    else:
        text_prompt = f"Сгенерируй один новый готовый пост про типичную реставрацию кухни. Придумай свежую конкретную тему внутри процесса или результата.{recent_text}"

    content_parts = [{"type": "text", "text": text_prompt}]

    for p in image_paths[:8]:
        if Path(p).exists():
            import base64
            b64 = base64.b64encode(Path(p).read_bytes()).decode()
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })

    try:
        if len(content_parts) > 1:
            result: PostOutput = llm.invoke([
                SystemMessage(content=SYSTEM_POSTER),
                HumanMessage(content=content_parts)
            ])
        else:
            result: PostOutput = llm.invoke([
                SystemMessage(content=SYSTEM_POSTER),
                HumanMessage(content=text_prompt)
            ])
        state["text"] = result.post_text
        state["short_topic"] = result.short_topic
        state["image_prompt"] = result.image_prompt
        logger.info(f"Poster: post generated, length={len(result.post_text)}, topic={result.short_topic}")
    except Exception as e:
        logger.error(f"Poster structured generation failed: {e}")
        state["text"] = (
            "Реставрация кухни часто обходится в 3–5 раз дешевле новой. "
            "Покраска фасадов, замена фурнитуры и столешницы дают эффект «как новая» "
            "без полного демонтажа. Главное — правильная подготовка поверхности."
        )
        state["short_topic"] = "реставрация выгоднее"
        state["image_prompt"] = "realistic photo of kitchen cabinet restoration before and after, professional lighting, modern interior"
        state["error"] = str(e)

    return state

def _safe_image_prompt(raw: str) -> str:
    """Make image prompt short, realistic and safe for Flux."""
    if not raw:
        return "realistic photo of modern kitchen cabinets, clean lighting, no clutter"
    # Keep it short and add strong constraints
    base = raw.strip()[:180]
    return (
        f"{base}, realistic photography, modern kitchen, clean, professional lighting, "
        "no surreal elements, no objects inside cabinets unless essential, no extra items"
    )

def generate_image_node(state: PosterState) -> PosterState:
    """Generate image via OpenRouter /api/v1/images endpoint."""
    import base64
    import httpx

    image_path = settings.temp_images_dir / f"{uuid.uuid4().hex}.png"
    raw_prompt = state.get("image_prompt") or "realistic photo of modern kitchen cabinets"
    prompt = _safe_image_prompt(raw_prompt)
    logger.info(f"Poster image prompt: {prompt}")

    try:
        resp = httpx.post(
            "https://openrouter.ai/api/v1/images",
            headers={
                "Authorization": f"Bearer {settings.openrouter_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.model_image,
                "prompt": prompt,
                "aspect_ratio": "1:1",
            },
            timeout=120.0,
        )
        resp.raise_for_status()
        data = resp.json()
        b64 = data["data"][0]["b64_json"]
        image_bytes = base64.b64decode(b64)
        image_path.write_bytes(image_bytes)
        state["image_path"] = str(image_path)
        logger.info(f"Poster: real image saved to {image_path}")
    except Exception as e:
        logger.error(f"OpenRouter image generation failed: {e}")
        try:
            from PIL import Image, ImageDraw
            img = Image.new("RGB", (1024, 1024), color=(245, 245, 245))
            d = ImageDraw.Draw(img)
            d.text((80, 480), "Kitchen Restoration", fill=(60, 60, 60))
            img.save(image_path)
            state["image_path"] = str(image_path)
            logger.info("Poster: fell back to placeholder image")
        except Exception as e2:
            logger.error(f"Placeholder also failed: {e2}")
            state["image_path"] = None
        state["error"] = str(e)
    return state

def save_draft_node(state: PosterState) -> PosterState:
    if not state.get("text"):
        state["error"] = "No text generated"
        return state
    draft_id = save_draft(state["text"], state.get("image_path"))
    state["draft_id"] = draft_id
    logger.info(f"Poster: draft saved id={draft_id}, topic={state.get('short_topic')}")
    return state

def build_poster_graph():
    workflow = StateGraph(PosterState)
    workflow.add_node("generate_post", generate_post_node)
    workflow.add_node("generate_image", generate_image_node)
    workflow.add_node("save_draft", save_draft_node)

    workflow.set_entry_point("generate_post")
    workflow.add_edge("generate_post", "generate_image")
    workflow.add_edge("generate_image", "save_draft")
    workflow.add_edge("save_draft", END)

    return workflow.compile()

poster_graph = build_poster_graph()

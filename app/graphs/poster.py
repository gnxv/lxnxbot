"""Poster graph: generate a ready-to-publish post about kitchen restoration."""
import logging
import uuid
import re
import random
from pathlib import Path
from typing import TypedDict, Optional, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from app.llm import get_poster_llm
from app.db import save_draft, get_recent_posts
from app.config import settings

logger = logging.getLogger(__name__)

PASTEL_TONES = [
    "warm beige",
    "light gray",
    "soft sage",
    "powder pink",
    "warm white",
    "muted blue-gray",
    "soft cream",
    "pale olive",
    "dusty rose",
    "light taupe",
]

def _get_color_hint() -> str:
    """Pick 1 or 2 random pastel tones for variety."""
    k = random.choice([1, 2])
    chosen = random.sample(PASTEL_TONES, k=k)
    return ", ".join(chosen)

# Soft image style hints per angle (not forced, just a gentle bias)
ANGLE_IMAGE_HINTS = {
    "process": "clean modern restored kitchen, soft pastel tones",
    "problem_solution": "realistic kitchen showing neat updated cabinets and hardware, natural lighting",
    "cozy_character": "warm cozy kitchen with wooden or soft natural tones, inviting and lived-in but clean, characterful",
    "single_element": "close-up or focused view of updated kitchen cabinets, handles or countertop, soft lighting",
    "custom": "realistic photo of modern restored kitchen",
}

# Four post angles for variety.
POST_ANGLES = [
    {
        "name": "process",
        "instruction": (
            "Напиши пост про типичный процесс реставрации.\n"
            "Каркас часто оставляем, если он крепкий. Обновляем покрытие фасадов, фурнитуру, при необходимости столешницу.\n"
            "Поверхность готовим, грунтуем и красим плотным ровным слоем. Цвет и ручки подбираем так, чтобы всё смотрелось одним комплектом.\n"
            "Покажи результат: свежий вид, ровные фасады, спокойное открывание, кухня выглядит собранно и ею проще пользоваться."
        ),
    },
    {
        "name": "problem_solution",
        "instruction": (
            "Напиши пост от конкретной проблемы.\n"
            "Начни с узнаваемой ситуации (тугие или дребезжащие ящики, потёртый торец, глянец подчёркивает царапины, разрозненный вид и т.п.).\n"
            "Кратко объясни, почему так происходит, и покажи, как это решаем: что оставляем, что обновляем, как готовим поверхность.\n"
            "Закончи тем, как кухня стала удобнее и выглядит цельно.\n"
            "Не уходи в редкие поломки."
        ),
    },
    {
        "name": "cozy_character",
        "instruction": (
            "Напиши пост про уютную кухню с характером.\n"
            "Фокус на тёплом дереве, мягких натуральных тонах, живом ощущении от кухни.\n"
            "Можно коротко упомянуть, что сделали (обновили фасады, оставили дерево, подобрали спокойный цвет), но главное — как кухня стала уютнее и живее, с характером, а не просто «как у всех».\n"
            "Без сказочности и без прямой фэнтези — просто уютная реальная кухня, которой приятно пользоваться."
        ),
    },
    {
        "name": "single_element",
        "instruction": (
            "Напиши пост про один элемент: направляющие, боковину, столешницу, ручки или кромку.\n"
            "Объясни, зачем его обновляют и как это влияет на общий вид и удобство кухни.\n"
            "Кратко покажи процесс (что оставляем, как готовим/меняем) и результат.\n"
            "Не размазывай на всю кухню — держи фокус на выбранном элементе."
        ),
    },
]

def _pick_angle() -> dict:
    return random.choice(POST_ANGLES)

class PosterState(TypedDict):
    topic: str
    text: Optional[str]
    image_prompt: Optional[str]
    image_path: Optional[str]
    image_paths: Optional[list]
    draft_id: Optional[int]
    error: Optional[str]
    short_topic: Optional[str]
    angle_name: Optional[str]

class PostOutput(BaseModel):
    """Structured output for a channel post."""
    post_text: str = Field(description="Готовый текст поста на русском, 450–650 символов. Живой, понятный, процессный и результатный. Без продажи и воды.")
    short_topic: str = Field(description="Короткая суть в 2–4 словах")
    image_prompt: str = Field(
        description=(
            "Короткий промпт на английском для одной реалистичной фотографии кухни. "
            "Только один кадр: обновлённая кухня, фасады или столешница. "
            "Без before/after, без split, без comparison, без сюрреализма, без лишних объектов, без текста на картинке."
        )
    )

SYSTEM_POSTER = """Ты — мастер по реставрации кухонь, который ведёт Telegram-канал.

Пиши живые, понятные посты про реальные подходы к реставрации: что оставляем, что обновляем, как готовим и красим, какой получается результат.

Один пост — одна конкретная идея или типичный сценарий работы.

Стиль:
- Начинай с сути: зачем это делают и в чём смысл работы (чтобы кухня снова выглядела цельно и ею было удобно пользоваться).
- Короткие абзацы. Простой уверенный язык без пафоса и литературщины.
- Длина 450–650 символов.
- Эмодзи используй только если усиливают смысл, а не как украшательство (0–2).
- Без жёсткой продажи, без призывов «звоните», без точных цен и сроков. Можно мягко намекнуть, что делаете такие работы, если основа ещё живая.
- Не выдумывай редкие поломки и странные ситуации.
- Не повторяй последние темы.

Для картинки:
- Только одна простая реалистичная фотография готовой или обновлённой кухни / фасадов / столешницы.
- Запрещено: before/after, split image, comparison, two kitchens, сюрреализм, лишние предметы, текст на картинке.
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
        angle_name = "custom"
    else:
        angle = _pick_angle()
        angle_name = angle["name"]
        text_prompt = (
            f"{angle['instruction']}\n\n"
            f"Сгенерируй один новый готовый пост в этом ключе. Не повторяй недавние темы.{recent_text}"
        )

    state["angle_name"] = angle_name

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
        logger.info(f"Poster: post generated, length={len(result.post_text)}, topic={result.short_topic}, angle={angle_name}")
    except Exception as e:
        logger.error(f"Poster structured generation failed: {e}")
        state["text"] = (
            "Реставрация кухни часто обходится в 3–5 раз дешевле новой. "
            "Покраска фасадов, замена фурнитуры и столешницы дают эффект «как новая» "
            "без полного демонтажа. Главное — правильная подготовка поверхности."
        )
        state["short_topic"] = "реставрация выгоднее"
        state["image_prompt"] = "realistic photo of modern restored kitchen cabinets"
        state["error"] = str(e)

    return state

def _safe_image_prompt(raw: str, angle_name: str = "process") -> str:
    """Make image prompt short, realistic and safe for Flux.
    Soft style hint depends on the chosen angle.
    """
    hint = ANGLE_IMAGE_HINTS.get(angle_name, ANGLE_IMAGE_HINTS["process"])

    if not raw or not raw.strip():
        return f"realistic photo of {hint}, clean lighting, no clutter, no text"

    text = raw.strip()
    bad_patterns = [
        r"\bbefore\s*(and|&)?\s*after\b",
        r"\bbefore\b",
        r"\bafter\b",
        r"\bcomparison\b",
        r"\bsplit\s*(image|view|screen)?\b",
        r"\btwo\s*(kitchens|images|photos)\b",
        r"\bold\s*(kitchen|cabinets)?\b",
        r"\bnew\s*(kitchen|cabinets)?\b",
        r"\bsurreal\b",
        r"\bfloating\b",
    ]
    for pat in bad_patterns:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)

    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*,\s*,", ",", text)
    text = text.strip(" ,.")

    if len(text) > 160:
        text = text[:160].rsplit(" ", 1)[0]

    suffix = f", realistic photography, single photo, {hint}, clean professional lighting, no clutter, no text, no surreal elements"
    return (text + suffix).strip()

def generate_image_node(state: PosterState) -> PosterState:
    """Generate image via OpenRouter /api/v1/images endpoint."""
    import base64
    import httpx

    image_path = settings.temp_images_dir / f"{uuid.uuid4().hex}.png"
    raw_prompt = state.get("image_prompt") or "realistic photo of modern kitchen cabinets"
    angle_name = state.get("angle_name") or "process"
    prompt = _safe_image_prompt(raw_prompt, angle_name)
    logger.info(f"Poster image prompt (angle={angle_name}): {prompt}")

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

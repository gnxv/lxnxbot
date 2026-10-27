"""Poster graph: generate a ready-to-publish post about kitchen restoration."""
import logging
import uuid
from pathlib import Path
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from app.llm import get_poster_llm
from app.db import save_draft
from app.config import settings

logger = logging.getLogger(__name__)

class PosterState(TypedDict):
    topic: str
    text: Optional[str]
    image_prompt: Optional[str]
    image_path: Optional[str]
    draft_id: Optional[int]
    error: Optional[str]

SYSTEM_POSTER = """Ты — контент-менеджер Telegram-канала про реставрацию кухонь.
Пиши только готовые посты. Никаких рассуждений, планов, «сейчас сверюсь», «затем соберу» и т.п.

Тема строго: реставрация кухонь (покраска фасадов, замена столешниц, фурнитура, перетяжка, ремонт, до/после).

Правила поста:
- Готовый текст, который можно сразу публиковать.
- Полезный, конкретный, живой язык.
- Примеры подачи: сравнение «новая кухня 300к vs реставрация 50-80к», конкретные советы, типичные ошибки, кейсы.
- Длина 350-800 символов.
- Умеренно эмодзи.
- В конце короткий призыв (вопрос к читателям или «пишите в комментарии»).
- Не выдумывай точные цены и факты. Если цифры нужны — пиши диапазоны или «ориентировочно».

Формат ответа строго:
POST:
<готовый текст поста>

IMAGE_PROMPT:
<короткий промпт на английском для реалистичной картинки кухни до/после или процесса реставрации>
"""

def generate_post_node(state: PosterState) -> PosterState:
    llm = get_poster_llm()
    prompt = f"Сгенерируй один готовый пост на тему: {state['topic']}. Сразу в нужном формате."
    resp = llm.invoke([SystemMessage(content=SYSTEM_POSTER), HumanMessage(content=prompt)])
    content = resp.content.strip()

    text = None
    image_prompt = "realistic photo of kitchen cabinet restoration before and after, professional lighting, modern interior"

    if "POST:" in content and "IMAGE_PROMPT:" in content:
        try:
            post_part = content.split("POST:")[1].split("IMAGE_PROMPT:")[0].strip()
            img_part = content.split("IMAGE_PROMPT:")[1].strip()
            if post_part:
                text = post_part
            if img_part:
                image_prompt = img_part
        except Exception:
            pass

    if not text:
        # fallback: take everything as text if parsing failed
        text = content.replace("POST:", "").replace("IMAGE_PROMPT:", "").strip()
        if len(text) < 50:
            text = (
                "Реставрация кухни часто обходится в 3–5 раз дешевле новой. "
                "Покраска фасадов, замена фурнитуры и столешницы дают эффект «как новая» "
                "без полного демонтажа. Главное — правильная подготовка поверхности.\n\n"
                "А вы уже делали реставрацию или только планируете?"
            )

    state["text"] = text
    state["image_prompt"] = image_prompt
    logger.info(f"Poster: post generated, length={len(text)}")
    return state

def generate_image_node(state: PosterState) -> PosterState:
    """Generate image via OpenRouter /api/v1/images endpoint."""
    import base64
    import httpx
    from app.config import settings

    image_path = settings.temp_images_dir / f"{uuid.uuid4().hex}.png"
    prompt = state.get("image_prompt") or "realistic photo of kitchen cabinet restoration before and after, professional lighting"

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
        # fallback to placeholder
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
    logger.info(f"Poster: draft saved id={draft_id}")
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

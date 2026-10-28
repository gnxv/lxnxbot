"""Custom Poster: for posts with user photos and/or explicit work description.
No memory of previous posts. Strictly follows the request and looks at images.
"""
import logging
import uuid
import base64
from pathlib import Path
from typing import TypedDict, Optional, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from app.llm import get_poster_llm
from app.db import save_draft
from app.config import settings

logger = logging.getLogger(__name__)

class CustomPosterState(TypedDict):
    topic: str                  # user's description / request
    text: Optional[str]
    image_prompt: Optional[str]
    image_path: Optional[str]   # generated image (if needed)
    image_paths: Optional[List[str]]  # user photos
    draft_id: Optional[int]
    error: Optional[str]

SYSTEM_CUSTOM = """Ты — мастер по реставрации кухонь, который ведёт Telegram-канал.

Твоя задача — написать живой, понятный и полезный пост про конкретную работу (или по заданной теме).

Правила для постов с фото:
- Не описывай фото как рассказчик. Не перечисляй всё, что видишь (чайник, кофеварка, жалюзи, пол, потолок и т.д.).
- Выделяй главное: что сделали, какой результат получился, в чём смысл работы.
- Пиши так, чтобы человеку было интересно и понятно, зачем это делали и что получилось.
- Можно упомянуть материалы, цвет, практичность, внешний вид, если это важно для результата.
- Не выдумывай цены, сроки и проблемы, которых нет в комментарии и на фото.
- Стиль: простой, уверенный, без пафоса и без литературщины. Короткие абзацы.
- Длина 350–650 символов.
- Эмодзи почти не используй (0–1).
- Не заканчивай формальным вопросом. Можно мягко намекнуть, что делаете такие работы.

Если фото нет — напиши пост на заданную тему и дай хороший IMAGE_PROMPT на английском для реалистичной картинки.

Формат ответа строго:
POST:
<готовый текст поста>

IMAGE_PROMPT:
<промпт на английском или пусто, если есть фото пользователя>
"""

def generate_custom_node(state: CustomPosterState) -> CustomPosterState:
    llm = get_poster_llm()
    topic = state.get("topic") or "наша новая работа"
    image_paths = state.get("image_paths") or []

    if image_paths:
        user_text = (
            f"Комментарий заказчика: {topic}\n\n"
            "Напиши пост про эту работу. Фокусируйся на том, что сделали и какой результат получился. "
            "Не описывай всё, что видно на фото. IMAGE_PROMPT оставь пустым."
        )
    else:
        user_text = (
            f"Тема поста: {topic}\n\n"
            "Фотографий нет. Напиши пост на эту тему и дай IMAGE_PROMPT для генерации подходящей картинки."
        )

    content_parts = [{"type": "text", "text": user_text}]

    for p in image_paths[:8]:
        if Path(p).exists():
            b64 = base64.b64encode(Path(p).read_bytes()).decode()
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })

    resp = llm.invoke([
        SystemMessage(content=SYSTEM_CUSTOM),
        HumanMessage(content=content_parts if len(content_parts) > 1 else user_text)
    ])
    content = resp.content.strip()

    text = None
    image_prompt = None
    if "POST:" in content:
        try:
            post_part = content.split("POST:")[1].split("IMAGE_PROMPT:")[0].strip()
            if post_part:
                text = post_part
            if "IMAGE_PROMPT:" in content:
                img = content.split("IMAGE_PROMPT:")[1].strip()
                if img and not image_paths:
                    image_prompt = img
        except Exception:
            pass

    if not text:
        text = content.replace("POST:", "").replace("IMAGE_PROMPT:", "").strip()

    state["text"] = text
    state["image_prompt"] = image_prompt
    logger.info(f"CustomPoster: generated, length={len(text or '')}, photos={len(image_paths)}")
    return state

def maybe_generate_image_node(state: CustomPosterState) -> CustomPosterState:
    """Generate image only if no user photos and we have an image_prompt."""
    if state.get("image_paths"):
        # User provided photos — don't generate
        state["image_path"] = None
        return state
    if not state.get("image_prompt"):
        return state

    import httpx
    image_path = settings.temp_images_dir / f"custom_{uuid.uuid4().hex}.png"
    try:
        resp = httpx.post(
            "https://openrouter.ai/api/v1/images",
            headers={"Authorization": f"Bearer {settings.openrouter_api_key}", "Content-Type": "application/json"},
            json={"model": settings.model_image, "prompt": state["image_prompt"], "aspect_ratio": "1:1"},
            timeout=120.0,
        )
        resp.raise_for_status()
        b64 = resp.json()["data"][0]["b64_json"]
        image_path.write_bytes(base64.b64decode(b64))
        state["image_path"] = str(image_path)
        logger.info(f"CustomPoster: generated image {image_path}")
    except Exception as e:
        logger.error(f"CustomPoster image gen failed: {e}")
        state["image_path"] = None
    return state

def save_node(state: CustomPosterState) -> CustomPosterState:
    if state.get("text"):
        # For custom we don't always need a draft, but save for consistency
        did = save_draft(state["text"], state.get("image_path"))
        state["draft_id"] = did
    return state

def build_custom_poster():
    g = StateGraph(CustomPosterState)
    g.add_node("generate", generate_custom_node)
    g.add_node("image", maybe_generate_image_node)
    g.add_node("save", save_node)
    g.set_entry_point("generate")
    g.add_edge("generate", "image")
    g.add_edge("image", "save")
    g.add_edge("save", END)
    return g.compile()

custom_poster_graph = build_custom_poster()

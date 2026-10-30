"""Custom Poster: for posts with user photos and/or explicit work description.
No memory of previous posts. Strictly follows the request and looks at images.
Uses structured output.
"""
import logging
import uuid
import base64
import re
from pathlib import Path
from typing import TypedDict, Optional, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from app.llm import get_poster_llm
from app.db import save_draft
from app.config import settings

logger = logging.getLogger(__name__)

def _safe_image_prompt(raw: str) -> str:
    """Clean image prompt: remove bad patterns, keep realistic single photo.
    Do not force color palette — topic can be anything (wood, high-tech, etc.).
    """
    if not raw or not raw.strip():
        return "realistic photo of modern kitchen, clean lighting, no clutter, no text"

    text = raw.strip()
    bad_patterns = [
        r"\bbefore\s*(and|&)?\s*after\b",
        r"\bbefore\b",
        r"\bafter\b",
        r"\bcomparison\b",
        r"\bsplit\s*(image|view|screen)?\b",
        r"\btwo\s*(kitchens|images|photos)\b",
        r"\bsurreal\b",
    ]
    for pat in bad_patterns:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)

    text = re.sub(r"\s+", " ", text).strip(" ,.")
    if len(text) > 180:
        text = text[:180].rsplit(" ", 1)[0]

    suffix = ", realistic photography, single photo, clean professional lighting, no clutter, no text, no surreal elements"
    return (text + suffix).strip()

class CustomPosterState(TypedDict):
    topic: str
    text: Optional[str]
    image_prompt: Optional[str]
    image_path: Optional[str]
    image_paths: Optional[List[str]]
    draft_id: Optional[int]
    error: Optional[str]

class CustomPostOutput(BaseModel):
    post_text: str = Field(description="Готовый текст поста на русском, 450-650 символов, фокус на результате работы. С 2–5 эмодзи.")
    image_prompt: Optional[str] = Field(default=None, description="Промпт на английском для картинки, только если нет фото пользователя. Стиль и цвет под тему.")

SYSTEM_CUSTOM = """Ты — мастер по реставрации кухонь, который ведёт Telegram-канал организации «РЕСТАВРАТОР» (Воронеж и область).

Твоя задача — написать живой, понятный и полезный пост про конкретную работу (или по заданной теме).

Правила для постов с фото:
- Не описывай фото как рассказчик. Не перечисляй всё, что видишь (чайник, кофеварка, жалюзи, пол, потолок и т.д.).
- Выделяй главное: что сделали, какой результат получился, в чём смысл работы.
- Пиши так, чтобы человеку было интересно и понятно, зачем это делали и что получилось.
- Можно упомянуть материалы, цвет, практичность, внешний вид, если это важно для результата.
- Не выдумывай цены, сроки и проблемы, которых нет в комментарии и на фото.
- Стиль: простой, уверенный, без пафоса и без литературщины. Короткие абзацы.
- Длина 450–650 символов.
- Эмодзи используй обязательно, естественно: минимум 2, максимум 5.
- В конце мягко и лаконично упомяни, что реставрируем кухни в Воронеже и области. Если подходит — обыграй с темой. Можно коротко упомянуть «РЕСТАВРАТОР».
- Не заканчивай формальным вопросом. Без жёсткой продажи.

Если фото нет — напиши пост на заданную тему и дай хороший IMAGE_PROMPT на английском для реалистичной картинки, которая соответствует теме (стиль, материал, цвет — как подходит).
"""

def generate_custom_node(state: CustomPosterState) -> CustomPosterState:
    llm = get_poster_llm().with_structured_output(CustomPostOutput)
    topic = state.get("topic") or "наша новая работа"
    image_paths = state.get("image_paths") or []

    if image_paths:
        user_text = (
            f"Комментарий заказчика: {topic}\n\n"
            "Напиши пост про эту работу. Фокусируйся на том, что сделали и какой результат получился. "
            "Не описывай всё, что видно на фото. image_prompt оставь пустым."
        )
    else:
        user_text = (
            f"Тема поста: {topic}\n\n"
            "Фотографий нет. Напиши пост на эту тему и дай image_prompt для генерации подходящей картинки."
        )

    content_parts = [{"type": "text", "text": user_text}]

    for p in image_paths[:8]:
        if Path(p).exists():
            b64 = base64.b64encode(Path(p).read_bytes()).decode()
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })

    try:
        result: CustomPostOutput = llm.invoke([
            SystemMessage(content=SYSTEM_CUSTOM),
            HumanMessage(content=content_parts if len(content_parts) > 1 else user_text)
        ])
        state["text"] = result.post_text
        state["image_prompt"] = result.image_prompt if not image_paths else None
        logger.info(f"CustomPoster: generated, length={len(result.post_text or '')}, photos={len(image_paths)}")
    except Exception as e:
        logger.error(f"CustomPoster structured generation failed: {e}")
        state["text"] = f"Выполнили работу: {topic}. Результат получился аккуратным и практичным."
        state["image_prompt"] = None
        state["error"] = str(e)

    return state

def maybe_generate_image_node(state: CustomPosterState) -> CustomPosterState:
    if state.get("image_paths"):
        state["image_path"] = None
        return state
    if not state.get("image_prompt"):
        return state

    import httpx
    image_path = settings.temp_images_dir / f"custom_{uuid.uuid4().hex}.png"
    prompt = _safe_image_prompt(state["image_prompt"])
    logger.info(f"CustomPoster image prompt: {prompt}")
    try:
        resp = httpx.post(
            "https://openrouter.ai/api/v1/images",
            headers={"Authorization": f"Bearer {settings.openrouter_api_key}", "Content-Type": "application/json"},
            json={"model": settings.model_image, "prompt": prompt, "aspect_ratio": "1:1"},
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

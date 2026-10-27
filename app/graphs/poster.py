"""Poster graph: search → generate text → generate image → save draft → send to manager."""
import logging
import uuid
from pathlib import Path
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from app.llm import get_poster_llm
from app.db import save_draft
from app.config import settings
import httpx

logger = logging.getLogger(__name__)

class PosterState(TypedDict):
    topic: str
    search_results: Optional[str]
    text: Optional[str]
    image_prompt: Optional[str]
    image_path: Optional[str]
    draft_id: Optional[int]
    error: Optional[str]

SYSTEM_POSTER = """Ты — контент-менеджер канала про реставрацию кухонь.
Тема строго: реставрация кухонь (фасады, столешницы, фурнитура, покраска, перетяжка, замена, ремонт).

Правила:
- Не выдумывай факты. Если нужны цифры или примеры — опирайся на поиск.
- Полезный, конкретный пост. Примеры подачи: «Новая кухня 300к, реставрация — 50к», сравнение, кейсы, советы.
- Длина: 400-900 символов, с эмодзи умеренно.
- В конце можно добавить призыв к комментариям.
- Язык: русский, живой.

Сначала подумай, нужен ли поиск. Если да — сформулируй запрос.
Затем напиши текст поста и отдельно промпт для картинки (реалистичное фото/рендер кухни до/после или процесс)."""

def search_node(state: PosterState) -> PosterState:
    """Gather context. For production, bind OpenRouter web_search tool if the model supports it."""
    llm = get_poster_llm()
    prompt = f"Тема поста: {state['topic']}. Нужны ли актуальные факты/цены/идеи по реставрации кухонь? Если да, кратко перечисли ключевые моменты (цены, популярные решения 2024-2025). Не выдумывай точные цифры без основания."
    resp = llm.invoke([SystemMessage(content=SYSTEM_POSTER), HumanMessage(content=prompt)])
    state["search_results"] = resp.content
    logger.info("Poster: search/context gathered")
    return state

def generate_text_node(state: PosterState) -> PosterState:
    llm = get_poster_llm()
    prompt = f"""Тема: {state['topic']}
Контекст из поиска/размышлений:
{state.get('search_results', 'нет')}

Напиши готовый текст поста и отдельно строку IMAGE_PROMPT: <промпт для картинки на английском, реалистичный>.
Формат ответа строго:
TEXT:
<текст поста>

IMAGE_PROMPT:
<промпт>
"""
    resp = llm.invoke([SystemMessage(content=SYSTEM_POSTER), HumanMessage(content=prompt)])
    content = resp.content
    # Parse
    if "TEXT:" in content and "IMAGE_PROMPT:" in content:
        parts = content.split("IMAGE_PROMPT:")
        text_part = parts[0].replace("TEXT:", "").strip()
        image_prompt = parts[1].strip()
        state["text"] = text_part
        state["image_prompt"] = image_prompt
    else:
        state["text"] = content
        state["image_prompt"] = "realistic photo of kitchen cabinet restoration before and after, professional lighting"
    logger.info("Poster: text generated")
    return state

def generate_image_node(state: PosterState) -> PosterState:
    """Generate image via OpenRouter image models (if supported) or placeholder."""
    # Note: OpenRouter image generation varies by model. This is a best-effort implementation.
    # For flux models, check current OpenRouter docs for exact endpoint.
    image_path = settings.temp_images_dir / f"{uuid.uuid4().hex}.png"
    try:
        # Attempt via OpenRouter chat completions with image model if available
        # Fallback: create a simple placeholder or skip
        # Real implementation should use the specific OpenRouter image API when stable.
        logger.warning("Image generation: using placeholder. Implement proper OpenRouter image call for MODEL_IMAGE.")
        # Create a tiny placeholder PNG
        from PIL import Image, ImageDraw, ImageFont
        img = Image.new("RGB", (1024, 1024), color=(240, 240, 240))
        d = ImageDraw.Draw(img)
        d.text((50, 450), "Kitchen Restoration\n(placeholder)", fill=(80, 80, 80))
        img.save(image_path)
        state["image_path"] = str(image_path)
    except Exception as e:
        logger.error(f"Image gen failed: {e}")
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
    workflow.add_node("search", search_node)
    workflow.add_node("generate_text", generate_text_node)
    workflow.add_node("generate_image", generate_image_node)
    workflow.add_node("save_draft", save_draft_node)

    workflow.set_entry_point("search")
    workflow.add_edge("search", "generate_text")
    workflow.add_edge("generate_text", "generate_image")
    workflow.add_edge("generate_image", "save_draft")
    workflow.add_edge("save_draft", END)

    return workflow.compile()

poster_graph = build_poster_graph()

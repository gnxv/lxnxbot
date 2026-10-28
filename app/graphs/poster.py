"""Poster graph: generate a ready-to-publish post about kitchen restoration."""
import logging
import uuid
from pathlib import Path
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from app.llm import get_poster_llm
from app.db import save_draft, get_recent_posts
from app.config import settings

logger = logging.getLogger(__name__)

class PosterState(TypedDict):
    topic: str
    text: Optional[str]
    image_prompt: Optional[str]
    image_path: Optional[str]
    image_paths: Optional[list]  # user-provided photos for vision
    draft_id: Optional[int]
    error: Optional[str]

SYSTEM_POSTER = """Ты — мастер по реставрации кухонь, который ведёт свой Telegram-канал.
Пиши живым, точным и понятным языком: короткие фразы, конкретные детали, лёгкая ирония где уместно, без пафоса и канцелярита. Как будто опытный человек рассказывает из практики.

Не просто про покраску фасадов, а про кухни, ремонт, интерьер и бытовые ситуации. Людям должно хотеться читать и понимать пользу.

Каждый пост — про что-то конкретное. Сам придумывай новую тему внутри одной из рубрик (не повторяйся):

1. Кухонный детектив — почему дверца перекосилась, откуда вздулись фасады, почему скрипят ящики.
2. Миф или правда? — правда ли, что любую кухню можно перекрасить? Плёнка или эмаль?
3. Кухонные преступления — смешные и нелепые ошибки при ремонте.
4. РЕСТАВРАТОР объясняет — простыми словами про материалы и уход.
5. Кухонные мемы — жизненный юмор про ремонт, клиентов, семейные споры.
6. А вы бы что выбрали? — два варианта. Можно мягко предложить выбрать.
7. Не спешите выбрасывать — что реально можно восстановить.
8. Кухня будущего — новые материалы и решения.
9. Маленькие хитрости — практические советы.
10. Кухонные истории — необычные наблюдения.

Правила:
- Один пост = одна узкая тема.
- Пиши от лица мастера. Можно «я обычно делаю так», «частая ошибка».
- Стиль: короткие предложения, точные детали, без воды. Пост должен быть полезным или интересным, а не просто описательным.
- Не выдумывай точные цены.
- Длина 400–700 символов.
- Эмодзи редко (0–2).
- В конце не задавай формальных вопросов. Заканчивай мысль.
- Картинка должна точно соответствовать теме поста.
- Не повторяй темы из последних постов.
- Если есть фото — не описывай всё подряд. Выделяй главное по теме.

Формат ответа строго:
POST:
<готовый текст поста>

IMAGE_PROMPT:
<детальный промпт на английском для реалистичной фотографии именно по теме поста>
"""

def generate_post_node(state: PosterState) -> PosterState:
    llm = get_poster_llm()
    recent = get_recent_posts(10)
    recent_text = ""
    if recent:
        recent_text = "\n\nПоследние посты (не повторяй эти темы и сюжеты):\n" + "\n---\n".join(recent[:10])

    custom_topic = state.get("topic")
    image_paths = state.get("image_paths") or []

    # Build multimodal content if photos are provided
    content_parts = []
    if custom_topic and custom_topic.lower() not in ("реставрация кухонь", "кухни", ""):
        text_prompt = (
            f"Напиши пост строго на тему: «{custom_topic}».\n"
            "Сохрани стиль мастера.\n"
            "Если приложены фото — выделяй главное по теме, не описывай всё подряд."
            f"{recent_text}"
        )
    else:
        text_prompt = f"Сгенерируй один новый готовый пост. Придумай свежую тему внутри одной из рубрик.{recent_text}"

    content_parts.append({"type": "text", "text": text_prompt})

    # Attach images if any (vision)
    for p in image_paths[:8]:  # limit to 8
        if Path(p).exists():
            import base64
            b64 = base64.b64encode(Path(p).read_bytes()).decode()
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })

    if len(content_parts) > 1:
        # Multimodal
        resp = llm.invoke([SystemMessage(content=SYSTEM_POSTER), HumanMessage(content=content_parts)])
    else:
        resp = llm.invoke([SystemMessage(content=SYSTEM_POSTER), HumanMessage(content=text_prompt)])

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

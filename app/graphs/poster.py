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
    draft_id: Optional[int]
    error: Optional[str]

SYSTEM_POSTER = """Ты — «РЕСТАВРАТОР», опытный мастер по реставрации кухонь, который ведёт свой Telegram-канал как живой блог.
Не просто про покраску фасадов, а про кухни, ремонт, интерьер и бытовые ситуации. Людям должно хотеться читать, даже если они пока ничего не заказывают.

Тон: живой, с лёгкой иронией и самоиронией. Как будто рассказывает знакомый мастер за чаем — просто, по делу, иногда с юмором. Без канцелярита, без заумных терминов и без навязчивых продаж.

Каждый пост — про что-то конкретное и разное. Сам придумывай новую тему внутри одной из рубрик (не повторяйся):

1. Кухонный детектив — почему дверца перекосилась, откуда вздулись фасады, почему скрипят ящики. Разбираем причины.
2. Миф или правда? — правда ли, что любую кухню можно перекрасить? Что лучше: плёнка или эмаль?
3. Кухонные преступления — смешные и нелепые ошибки при ремонте и эксплуатации кухни.
4. РЕСТАВРАТОР объясняет — простыми словами про МДФ, ЛДСП, фурнитуру, покрытия и уход.
5. Кухонные мемы — жизненный юмор про ремонт, выбор цвета, семейные споры, клиентов.
6. А вы бы что выбрали? — два варианта (оттенка, ручек, подхода). Можно мягко предложить выбрать.
7. Не спешите выбрасывать — что реально можно восстановить, а что лучше заменить.
8. Кухня будущего — новые материалы, технологии, мебельные решения.
9. Маленькие хитрости — как ухаживать за столешницей, регулировать дверцы, сохранять покрытие.
10. Кухонные истории — необычные факты о кухнях, дизайне и привычках.

Правила:
- Один пост = одна узкая тема + конкретная польза, объяснение или лёгкий юмор.
- Пиши от лица РЕСТАВРАТОРА. Можно «я обычно делаю так», «частая ошибка», «вот простой способ».
- Живой язык. Короткие предложения. Можно лёгкий сарказм.
- Не выдумывай точные цены и цифры. Если упоминаешь стоимость — только ориентировочно.
- Длина 400–750 символов.
- Умеренно эмодзи (1–3 на пост).
- В конце НЕ задавай тупых или формальных вопросов («а у вас как?», «напишите в комментарии»). Если очень хочется вовлечь — сделай это естественно и по делу, или просто закончи мысль.
- Картинка должна точно соответствовать теме именно этого поста.
- Не повторяй темы и сюжеты из последних постов (они будут переданы ниже).

Формат ответа строго:
POST:
<готовый текст поста>

IMAGE_PROMPT:
<детальный промпт на английском для реалистичной фотографии или иллюстрации именно по теме поста. Укажи, что именно должно быть на картинке.>
"""

def generate_post_node(state: PosterState) -> PosterState:
    llm = get_poster_llm()
    recent = get_recent_posts(10)
    recent_text = ""
    if recent:
        recent_text = "\n\nПоследние посты (не повторяй эти темы и сюжеты):\n" + "\n---\n".join(recent[:10])

    custom_topic = state.get("topic")
    if custom_topic and custom_topic.lower() not in ("реставрация кухонь", "кухни", ""):
        # Force the custom topic while keeping the style
        prompt = (
            f"Напиши пост строго на тему: «{custom_topic}».\n"
            "Сохрани стиль и тон РЕСТАВРАТОРА (ирония, простые объяснения, польза или лёгкий юмор).\n"
            "Если тема мемная или юмористическая — сделай пост в духе рубрики «Кухонные мемы».\n"
            "Не уходи в другую тему."
            f"{recent_text}"
        )
    else:
        prompt = f"Сгенерируй один новый готовый пост. Придумай свежую тему внутри одной из рубрик.{recent_text}"

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

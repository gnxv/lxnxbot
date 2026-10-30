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
from app.db import save_draft, get_recent_posts, get_recent_post_texts
from app.config import settings

logger = logging.getLogger(__name__)

# Soft image style hints per angle
# Soft bias toward typical apartment kitchens (compact / L-shaped, limited space)
ANGLE_IMAGE_HINTS = {
    "example_style": "realistic photo of modern restored apartment kitchen, compact or L-shaped cabinets along the wall, limited floor space, soft pastel beige milk olive or deep royal blue tones, clean and neat",
    "problem_focus": "realistic close-up or unusual angle of a kitchen detail in a compact apartment kitchen (drawers, worn edge, glossy doors, texture), natural lighting, expressive",
    "cozy_old": "warm cozy older apartment kitchen with soft natural light, compact layout, lived-in but clean, wooden or muted warm tones, tea cups, everyday life details, occasionally a cat resting on a chair or windowsill",
    "fairytale": "warm storybook-style kitchen illustration feel, soft light, natural wood, gentle old-fashioned cozy atmosphere, calm colors, not hyper-fantasy",
    "custom": "realistic photo of a kitchen",
}

# Four diverse post angles
POST_ANGLES = [
    {
        "name": "example_style",
        "instruction": (
            "Напиши пост в живом разговорном стиле от лица мастера.\n"
            "Можно начать с вопросов заказчиков, сомнений или короткого профессионального совета.\n"
            "Объясни простым языком, что оставляем каркас и обновляем фасады, фурнитуру, при необходимости столешницу.\n"
            "Тон дружелюбный и уверенный. Эмодзи используй естественно."
        ),
    },
    {
        "name": "problem_focus",
        "instruction": (
            "Напиши пост про одну конкретную проблему или интересную деталь кухни.\n"
            "Можешь разрушать заблуждение, удивлять или коротко объяснять, почему так бывает.\n"
            "Покажи, как это решаем в рамках обновления, не уходя в редкие поломки.\n"
            "Сделай даже обычную ситуацию интересной."
        ),
    },
    {
        "name": "cozy_old",
        "instruction": (
            "Напиши пост про уютную старую кухню с историей.\n"
            "Тёплые наблюдения про привычки, семейные моменты или вещи, с которыми не хочется расставаться.\n"
            "Допускаются ностальгия и мягкий юмор. Не обязательно прямо говорить про реставрацию в каждом посте.\n"
            "Не начинай каждый раз со слов «старая кухня» — меняй заход.\n"
            "Иногда можно упомянуть кота, но не в каждом посте.\n"
            "Тон спокойный и душевный."
        ),
    },
    {
        "name": "fairytale",
        "instruction": (
            "Напиши пост с лёгким сказочным или уютно-историческим вайбом.\n"
            "Можно оживлять предметы через повествование или придумать короткую добрую историю.\n"
            "Без гипер-фэнтези, без магии и без копирования известных персонажей.\n"
            "Главное — уют, дерево и спокойная атмосфера."
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
    post_text: str = Field(description="Готовый текст поста на русском, 450–650 символов. Живой, понятный, разговорный. С 3–5 эмодзи. Без жёсткой продажи.")
    short_topic: str = Field(description="Короткая суть в 2–4 словах")
    image_prompt: str = Field(
        description=(
            "Короткий промпт на английском для одной реалистичной фотографии. "
            "Только один кадр. Стиль под тему поста. "
            "Без before/after, без split, без comparison, без любого текста, надписей, логотипов."
        )
    )

class RefineOutput(BaseModel):
    """Improved version of the post after self-check."""
    post_text: str = Field(description="Улучшенный текст поста, 450–650 символов. Живой разговорный стиль, 3–5 эмодзи.")
    short_topic: str = Field(description="Короткая суть в 2–4 словах")
    image_prompt: str = Field(
        description=(
            "Короткий промпт на английском для одной реалистичной фотографии под обновлённый текст. "
            "Без before/after, без split, без comparison, без любого текста и надписей."
        )
    )

SYSTEM_POSTER = """Ты — мастер по реставрации кухонь, который пишет живые посты для Telegram-канала.

Пиши как будто объясняешь человеку, который думает обновить кухню, но боится большого ремонта.

Важно:
- Не пиши так, будто ты лично сделал именно ту кухню, которая на картинке. Картинка — просто иллюстрация. Пиши про подход и типичные ситуации.
- Один пост — одна идея, наблюдение или маленькая история.
- Короткие абзацы. Простой живой язык без канцеляризмов.
- Длина 450–650 символов.
- Эмодзи используй обязательно, естественно, как в живом разговоре: минимум 3, максимум 5. Только если усиливают смысл.
- Не повторяй одни и те же идеи и формулировки.
- Без точных цен. Можно мягко предложить прислать фото, но без навязчивых продаж и одинаковых концовок.
- Не выдумывай редкие поломки и неподтверждённые обещания.
- Не упоминай название организации и город.

Для картинки:
- Одна реалистичная фотография, которая подходит под тему поста.
- Запрещено: before/after, split, comparison, любой текст, надписи на одежде, логотипы.
"""

SYSTEM_REFINE = """Ты редактор постов для Telegram-канала про реставрацию кухонь.

Тебе дают уже написанный пост и последние опубликованные посты. Твоя задача — улучшить текст, а не переписывать его с нуля.

Что сделать:
- Сделай язык живее и естественнее, убери воду и канцеляризмы.
- Сохрани основную идею и тон угла.
- Длина 450–650 символов, 3–5 эмодзи естественно.
- Если тема или формулировки сильно пересекаются с последними постами — мягко смести акцент или переформулируй так, чтобы не было ощущения повтора. Не меняй тему полностью, если можно обойтись уточнением.
- Не добавляй название организации и город.
- Обнови image_prompt, если текст изменился.

Верни улучшенную версию.
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
            "Сохрани разговорный стиль мастера.\n"
            "Если приложены фото — выделяй главное по теме, не описывай всё подряд."
            f"{recent_text}"
        )
        angle_name = "custom"
    else:
        angle = _pick_angle()
        angle_name = angle["name"]
        text_prompt = (
            f"{angle['instruction']}\n\n"
            f"Сгенерируй один новый готовый пост в этом ключе. Не повторяй недавние темы и одинаковые начала.{recent_text}"
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
            "Обновление кухни часто обходится в 3–5 раз дешевле новой. "
            "Покраска фасадов, замена фурнитуры и столешницы дают эффект «как новая» "
            "без полного демонтажа. Главное — правильная подготовка поверхности."
        )
        state["short_topic"] = "обновление выгоднее"
        state["image_prompt"] = "realistic photo of modern restored kitchen"
        state["error"] = str(e)

    return state

def refine_post_node(state: PosterState) -> PosterState:
    """One refinement pass: improve the text and gently avoid recent repeats."""
    if not state.get("text"):
        return state

    recent_texts = get_recent_post_texts(4)
    recent_block = ""
    if recent_texts:
        recent_block = "\n\nПоследние опубликованные посты (не повторяй близкие идеи и формулировки):\n" + "\n---\n".join(recent_texts)

    refine_prompt = (
        f"Текущий пост:\n{state['text']}\n\n"
        f"Короткая тема: {state.get('short_topic', '')}\n"
        f"Угол: {state.get('angle_name', '')}"
        f"{recent_block}\n\n"
        "Улучши текст. Если тема слишком близка к недавним постам — мягко смести акцент или переформулируй. "
        "Сохрани живой разговорный стиль."
    )

    try:
        llm = get_poster_llm().with_structured_output(RefineOutput)
        result: RefineOutput = llm.invoke([
            SystemMessage(content=SYSTEM_REFINE),
            HumanMessage(content=refine_prompt)
        ])
        state["text"] = result.post_text
        state["short_topic"] = result.short_topic
        state["image_prompt"] = result.image_prompt
        logger.info(f"Poster: post refined, length={len(result.post_text)}, topic={result.short_topic}")
    except Exception as e:
        logger.warning(f"Poster refinement failed, keeping original: {e}")

    return state

def _safe_image_prompt(raw: str, angle_name: str = "example_style") -> str:
    """Make image prompt short, realistic and safe.
    Soft style hint depends on the chosen angle.
    """
    hint = ANGLE_IMAGE_HINTS.get(angle_name, ANGLE_IMAGE_HINTS["example_style"])

    if not raw or not raw.strip():
        return f"realistic photo of {hint}, clean lighting, no text, no logos, no writing"

    text = raw.strip()
    bad_patterns = [
        r"\bbefore\s*(and|&)?\s*after\b",
        r"\bbefore\b",
        r"\bafter\b",
        r"\bcomparison\b",
        r"\bsplit\s*(image|view|screen)?\b",
        r"\btwo\s*(kitchens|images|photos)\b",
        r"\bsurreal\b",
        r"\bfloating\b",
        r"\bmagic\b",
    ]
    for pat in bad_patterns:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)

    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*,\s*,", ",", text)
    text = text.strip(" ,.")

    if len(text) > 180:
        text = text[:180].rsplit(" ", 1)[0]

    suffix = (
        f", realistic photography, single photo, {hint}, clean professional lighting, "
        "no clutter, no text, no letters, no logos, no writing on clothes, no inscriptions"
    )
    return (text + suffix).strip()

def generate_image_node(state: PosterState) -> PosterState:
    """Generate image via OpenRouter /api/v1/images endpoint."""
    import base64
    import httpx

    image_path = settings.temp_images_dir / f"{uuid.uuid4().hex}.png"
    raw_prompt = state.get("image_prompt") or "realistic photo of a kitchen"
    angle_name = state.get("angle_name") or "example_style"
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
            d.text((80, 480), "Kitchen", fill=(60, 60, 60))
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
    workflow.add_node("refine_post", refine_post_node)
    workflow.add_node("generate_image", generate_image_node)
    workflow.add_node("save_draft", save_draft_node)

    workflow.set_entry_point("generate_post")
    workflow.add_edge("generate_post", "refine_post")
    workflow.add_edge("refine_post", "generate_image")
    workflow.add_edge("generate_image", "save_draft")
    workflow.add_edge("save_draft", END)

    return workflow.compile()

poster_graph = build_poster_graph()

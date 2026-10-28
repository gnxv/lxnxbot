"""Reliable scheduler: polls DB every 30s and auto-publishes when a slot is due (MSK)."""
import logging
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo
from pathlib import Path
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from app.db import get_schedule, update_draft_status, save_post, get_pending_extras, mark_extra_published
from app.graphs.poster import poster_graph
from app.graphs.custom_poster import custom_poster_graph
from app.config import settings
from telegram.ext import Application
from telegram import InputFile, InputMediaPhoto

logger = logging.getLogger(__name__)

MSK = ZoneInfo("Europe/Moscow")
scheduler = AsyncIOScheduler(timezone=MSK)
# Track which slots we already published today: "YYYY-MM-DD_HH:MM"
_published_today: set[str] = set()

def _already_published(slot: dtime) -> bool:
    key = f"{datetime.now(MSK).strftime('%Y-%m-%d')}_{slot.hour:02d}:{slot.minute:02d}"
    return key in _published_today

def _mark_published(slot: dtime):
    key = f"{datetime.now(MSK).strftime('%Y-%m-%d')}_{slot.hour:02d}:{slot.minute:02d}"
    _published_today.add(key)

async def check_and_publish(app: Application):
    """Check regular schedule and extra posts."""
    now = datetime.now(MSK)
    current = now.time().replace(second=0, microsecond=0)
    try:
        # Regular slots
        slots = get_schedule()
        for slot in slots:
            if slot.hour == current.hour and slot.minute == current.minute:
                if _already_published(slot):
                    continue
                logger.info(f"Scheduler: slot {slot} (MSK) is due, publishing")
                await _publish_scheduled(app, slot)
                _mark_published(slot)

        # Extra posts
        extras = get_pending_extras()
        for extra in extras:
            logger.info(f"Scheduler: extra post {extra['id']} is due")
            await _publish_extra(app, extra)
            mark_extra_published(extra["id"])
    except Exception as e:
        logger.error(f"Scheduler check failed: {e}")

async def _publish_extra(app: Application, extra: dict):
    """Publish a one-time extra post (text and/or photos)."""
    try:
        chat_id = settings.telegram_channel_id
        text = extra.get("text") or ""
        topic = extra.get("topic")
        paths = extra.get("image_paths") or []
        logger.info(f"Publishing extra {extra['id']}: topic={topic}, text_len={len(text)}, photos={len(paths)}")

        # Generate text if needed
        generated_image = None
        if not text:
            use_custom = bool(paths) or (topic and "реставрация кухонь" not in (topic or "").lower())
            graph = custom_poster_graph if use_custom else poster_graph
            logger.info(f"Extra {extra['id']}: using {'custom_poster' if use_custom else 'regular poster'}, photos={len(paths)}")
            result = graph.invoke({
                "topic": topic or "наша новая работа",
                "text": None,
                "image_prompt": None,
                "image_path": None,
                "image_paths": paths,
                "draft_id": None,
                "error": None,
            })
            text = result.get("text") or ""
            generated_image = result.get("image_path")
            logger.info(f"Generated for extra {extra['id']}: len={len(text)}, gen_image={generated_image}")

        # Prefer user photos; fall back to generated image
        if not paths and generated_image and Path(generated_image).exists():
            paths = [generated_image]
            logger.info(f"Using generated image for extra {extra['id']}")

        if not text and not paths:
            raise ValueError("Nothing to publish (no text and no photos)")

        if paths:
            media = []
            open_files = []
            for i, p in enumerate(paths):
                if Path(p).exists():
                    f = open(p, "rb")
                    open_files.append(f)
                    media.append(InputMediaPhoto(media=f, caption=text[:1024] if i == 0 and text else None))
            if media:
                await app.bot.send_media_group(chat_id=chat_id, media=media)
                for f in open_files:
                    f.close()
            else:
                # photos listed but files missing
                if text:
                    await app.bot.send_message(chat_id=chat_id, text=text)
        elif text:
            await app.bot.send_message(chat_id=chat_id, text=text)

        if settings.manager_chat_id:
            await app.bot.send_message(
                chat_id=settings.manager_chat_id,
                text=f"Внеплановый пост опубликован (id {extra['id']})."
            )
        logger.info(f"Extra {extra['id']} published successfully")
    except Exception as e:
        logger.error(f"Extra publish failed: {e}", exc_info=True)
        if settings.manager_chat_id:
            try:
                await app.bot.send_message(
                    chat_id=settings.manager_chat_id,
                    text=f"Ошибка публикации внепланового (id {extra.get('id')}): {e}"
                )
            except Exception:
                pass

async def _publish_scheduled(app: Application, slot: dtime):
    try:
        result = poster_graph.invoke({
            "topic": "реставрация кухонь",
            "text": None,
            "image_prompt": None,
            "image_path": None,
            "draft_id": None,
            "error": None,
        })
        draft_id = result.get("draft_id")
        text = result.get("text", "")
        image_path = result.get("image_path")

        if not draft_id or not text:
            logger.error("Scheduler: no draft generated")
            return

        chat_id = settings.telegram_channel_id
        msg = None
        caption = text
        rest = None
        if len(text) > 1024:
            # Safety: split so photo has short caption, rest goes as separate message
            caption = text[:1000].rsplit(" ", 1)[0] + "…"
            rest = text[len(caption)-1:].strip()
            logger.info(f"Scheduler: text too long ({len(text)}), splitting")

        if image_path and Path(image_path).exists():
            with open(image_path, "rb") as f:
                msg = await app.bot.send_photo(
                    chat_id=chat_id,
                    photo=InputFile(f),
                    caption=caption
                )
        else:
            msg = await app.bot.send_message(chat_id=chat_id, text=caption)

        if rest and msg:
            await app.bot.send_message(chat_id=chat_id, text=rest, reply_to_message_id=msg.message_id)

        update_draft_status(draft_id, "published")
        if msg:
            save_post(draft_id, msg.message_id, text, topic=short_topic)

        if image_path:
            try:
                Path(image_path).unlink(missing_ok=True)
            except Exception:
                pass

        logger.info(f"Scheduler: published draft {draft_id} for slot {slot}")
        if settings.manager_chat_id:
            await app.bot.send_message(
                chat_id=settings.manager_chat_id,
                text=f"По расписанию ({slot.hour:02d}:{slot.minute:02d}) опубликован пост #{draft_id}."
            )
    except Exception as e:
        logger.error(f"Scheduled publish failed: {e}")
        if settings.manager_chat_id:
            try:
                await app.bot.send_message(
                    chat_id=settings.manager_chat_id,
                    text=f"Ошибка публикации по расписанию: {e}"
                )
            except Exception:
                pass

def setup_scheduler(app: Application):
    # Poll every 30 seconds — reliable and picks up schedule changes immediately
    scheduler.add_job(
        check_and_publish,
        IntervalTrigger(seconds=30),
        args=[app],
        id="schedule_poller",
        replace_existing=True,
    )
    scheduler.start()
    logger.info("Scheduler started (polling every 30s)")

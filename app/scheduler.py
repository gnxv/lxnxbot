"""Reliable scheduler: polls DB every 30s and auto-publishes when a slot is due (MSK)."""
import logging
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo
from pathlib import Path
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from app.db import get_schedule, update_draft_status, save_post
from app.graphs.poster import poster_graph
from app.config import settings
from telegram.ext import Application
from telegram import InputFile

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
    """Check schedule every interval and publish if a slot is due (MSK)."""
    now = datetime.now(MSK)
    current = now.time().replace(second=0, microsecond=0)
    try:
        slots = get_schedule()
        for slot in slots:
            if slot.hour == current.hour and slot.minute == current.minute:
                if _already_published(slot):
                    continue
                logger.info(f"Scheduler: slot {slot} (MSK) is due, publishing")
                await _publish_scheduled(app, slot)
                _mark_published(slot)
    except Exception as e:
        logger.error(f"Scheduler check failed: {e}")

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
        if image_path and Path(image_path).exists():
            with open(image_path, "rb") as f:
                msg = await app.bot.send_photo(
                    chat_id=chat_id,
                    photo=InputFile(f),
                    caption=text[:1024]
                )
        else:
            msg = await app.bot.send_message(chat_id=chat_id, text=text)

        update_draft_status(draft_id, "published")
        if msg:
            save_post(draft_id, msg.message_id, text)

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

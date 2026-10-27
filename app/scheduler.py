"""APScheduler that checks schedule table and triggers Poster (auto-publish)."""
import logging
from datetime import datetime
from pathlib import Path
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from app.db import get_schedule, update_draft_status, save_post
from app.graphs.poster import poster_graph
from app.config import settings
from telegram.ext import Application
from telegram import InputFile

logger = logging.getLogger(__name__)

scheduler = AsyncIOScheduler()

async def trigger_poster(app: Application):
    """Generate and immediately publish a scheduled post."""
    logger.info("Scheduler: triggering Poster (auto-publish)")
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

        # Publish immediately to channel
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

        # Cleanup image
        if image_path:
            try:
                Path(image_path).unlink(missing_ok=True)
            except Exception:
                pass

        logger.info(f"Scheduler: published draft {draft_id}")

        # Notify manager
        if settings.manager_chat_id:
            await app.bot.send_message(
                chat_id=settings.manager_chat_id,
                text=f"По расписанию опубликован пост #{draft_id}."
            )
    except Exception as e:
        logger.error(f"Scheduled poster failed: {e}")
        if settings.manager_chat_id:
            try:
                await app.bot.send_message(
                    chat_id=settings.manager_chat_id,
                    text=f"Ошибка публикации по расписанию: {e}"
                )
            except Exception:
                pass

def setup_scheduler(app: Application):
    times = get_schedule()
    for t in times:
        hour = t.hour
        minute = t.minute
        scheduler.add_job(
            trigger_poster,
            CronTrigger(hour=hour, minute=minute),
            args=[app],
            id=f"post_{hour}_{minute}",
            replace_existing=True,
        )
        logger.info(f"Scheduled post at {hour:02d}:{minute:02d}")
    scheduler.start()
    logger.info("Scheduler started")

"""APScheduler that checks schedule table and triggers Poster."""
import logging
from datetime import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from app.db import get_schedule
from app.graphs.poster import poster_graph
from app.config import settings
from telegram.ext import Application

logger = logging.getLogger(__name__)

scheduler = AsyncIOScheduler()

async def trigger_poster(app: Application):
    logger.info("Scheduler: triggering Poster")
    try:
        result = poster_graph.invoke({
            "topic": "реставрация кухонь",
            "search_results": None,
            "text": None,
            "image_prompt": None,
            "image_path": None,
            "draft_id": None,
            "error": None,
        })
        draft_id = result.get("draft_id")
        if draft_id and settings.manager_chat_id:
            text = result.get("text", "")
            # Send to manager for approval
            await app.bot.send_message(
                chat_id=settings.manager_chat_id,
                text=f"Черновик по расписанию #{draft_id}:\n\n{text[:1000]}\n\nОдобрить? Напиши «да»."
            )
    except Exception as e:
        logger.error(f"Scheduled poster failed: {e}")

def setup_scheduler(app: Application):
    times = get_schedule()
    for t in times:
        # t is time object
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

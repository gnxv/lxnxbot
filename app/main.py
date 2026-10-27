"""Entry point."""
import logging
import asyncio
from app.config import settings
from app.db import init_db
from app.telegram.handlers import build_application
from app.scheduler import setup_scheduler

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

async def main():
    logger.info("Starting Kitchen Autoposter")
    init_db()
    application = build_application()
    setup_scheduler(application)
    await application.initialize()
    await application.start()
    await application.updater.start_polling()
    logger.info("Bot is polling")
    # Keep running
    try:
        await asyncio.Event().wait()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutting down")
        await application.updater.stop()
        await application.stop()
        await application.shutdown()

if __name__ == "__main__":
    asyncio.run(main())

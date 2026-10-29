"""Utility to clean old temporary images."""
import logging
from datetime import datetime, timedelta
from pathlib import Path
from app.config import settings

logger = logging.getLogger(__name__)

def cleanup_temp_images(older_than_hours: int = 24):
    """Remove image files older than N hours from temp_images_dir."""
    cutoff = datetime.now() - timedelta(hours=older_than_hours)
    removed = 0
    for p in settings.temp_images_dir.glob("*"):
        if p.is_file() and p.stat().st_mtime < cutoff.timestamp():
            try:
                p.unlink()
                removed += 1
            except Exception as e:
                logger.warning(f"Failed to remove {p}: {e}")
    if removed:
        logger.info(f"Cleaned up {removed} old temp images")
    return removed

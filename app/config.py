import os
from pathlib import Path
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    
    model_poster: str = os.getenv("MODEL_POSTER", "x-ai/grok-4")
    model_image: str = os.getenv("MODEL_IMAGE", "black-forest-labs/flux-1.1-pro")
    model_moderator: str = os.getenv("MODEL_MODERATOR", "openai/gpt-4o-mini")
    model_responder: str = os.getenv("MODEL_RESPONDER", "openai/gpt-4o-mini")
    model_manager: str = os.getenv("MODEL_MANAGER", "openai/gpt-4o-mini")
    
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_channel_id: str = os.getenv("TELEGRAM_CHANNEL_ID", "")
    telegram_discussion_group_id: str = os.getenv("TELEGRAM_DISCUSSION_GROUP_ID", "")
    manager_chat_id: str = os.getenv("MANAGER_CHAT_ID", "")
    manager_user_id: str = os.getenv("MANAGER_USER_ID", "")
    
    database_url: str = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/kitchenbot")
    
    temp_images_dir: Path = Path(os.getenv("TEMP_IMAGES_DIR", "./temp_images"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
settings.temp_images_dir.mkdir(parents=True, exist_ok=True)

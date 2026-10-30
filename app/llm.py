import logging
from typing import Optional
from langchain_openai import ChatOpenAI
from app.config import settings

logger = logging.getLogger(__name__)

def get_llm(model: str, temperature: float = 0.7) -> ChatOpenAI:
    """Create ChatOpenAI client pointed at OpenRouter."""
    return ChatOpenAI(
        model=model,
        api_key=settings.openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=temperature,
        default_headers={
            "HTTP-Referer": "https://github.com/gnxv/lxnxbot",
            "X-Title": "Kitchen Autoposter",
        },
    )

def get_poster_llm():
    return get_llm(settings.model_poster, temperature=0.75)

def get_manager_llm():
    return get_llm(settings.model_manager, temperature=0.3)

def get_moderator_llm():
    return get_llm(settings.model_moderator, temperature=0.1)

def get_responder_llm():
    return get_llm(settings.model_responder, temperature=0.5)

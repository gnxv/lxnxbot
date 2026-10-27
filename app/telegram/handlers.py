"""Telegram bot handlers. Routes to graphs."""
import logging
import asyncio
from telegram import Update, InputFile
from telegram.ext import Application, MessageHandler, filters, ContextTypes
from app.config import settings
from app.graphs.manager import manager_graph
from app.graphs.moderator import moderator_graph
from app.graphs.responder import responder_graph
from app.graphs.poster import poster_graph
from app.db import get_pending_draft, update_draft_status, save_post
from pathlib import Path

logger = logging.getLogger(__name__)

async def is_manager(update: Update) -> bool:
    if not update.effective_chat or not update.effective_user:
        return False
    return (
        str(update.effective_chat.id) == settings.manager_chat_id
        and str(update.effective_user.id) == settings.manager_user_id
    )

async def handle_manager_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_manager(update):
        return  # complete silence
    text = update.message.text or ""
    state = {
        "messages": [{"role": "user", "content": text}],
        "action": None,
        "pending_confirm": context.user_data.get("pending_confirm"),
        "response": None,
    }
    result = manager_graph.invoke(state)
    context.user_data["pending_confirm"] = result.get("pending_confirm")
    if result.get("response"):
        await update.message.reply_text(result["response"])

    # Handle side effects
    pending = result.get("pending_confirm") or context.user_data.get("pending_confirm")
    if pending and pending.get("type") == "do_publish":
        await publish_draft(pending["draft_id"], context)
        context.user_data["pending_confirm"] = None
    elif pending and pending.get("type") == "extra_post":
        # Trigger poster
        asyncio.create_task(run_poster_and_send(update, context))

async def publish_draft(draft_id: int, context: ContextTypes.DEFAULT_TYPE):
    from app.db import get_conn
    with get_conn() as conn:
        cur = conn.execute("SELECT * FROM drafts WHERE id = %s", (draft_id,))
        draft = cur.fetchone()
    if not draft:
        return
    chat_id = settings.telegram_channel_id
    try:
        if draft["image_path"] and Path(draft["image_path"]).exists():
            with open(draft["image_path"], "rb") as f:
                msg = await context.bot.send_photo(
                    chat_id=chat_id,
                    photo=InputFile(f),
                    caption=draft["text"][:1024]
                )
        else:
            msg = await context.bot.send_message(chat_id=chat_id, text=draft["text"])
        update_draft_status(draft_id, "published")
        save_post(draft_id, msg.message_id, draft["text"])
        # cleanup image
        if draft["image_path"]:
            Path(draft["image_path"]).unlink(missing_ok=True)
        logger.info(f"Published draft {draft_id}")
    except Exception as e:
        logger.error(f"Publish failed: {e}")

async def run_poster_and_send(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        result = poster_graph.invoke({"topic": "реставрация кухонь", "search_results": None, "text": None, "image_prompt": None, "image_path": None, "draft_id": None, "error": None})
        draft_id = result.get("draft_id")
        if draft_id and update.effective_chat:
            text = result.get("text", "")
            image_path = result.get("image_path")
            if image_path and Path(image_path).exists():
                with open(image_path, "rb") as f:
                    await context.bot.send_photo(
                        chat_id=update.effective_chat.id,
                        photo=InputFile(f),
                        caption=f"Черновик #{draft_id}:\n\n{text[:900]}"
                    )
            else:
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text=f"Черновик #{draft_id}:\n\n{text}"
                )
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="Одобрить публикацию в канал? Напиши «да»."
            )
    except Exception as e:
        logger.error(f"Poster failed: {e}")
        if update.effective_chat:
            await context.bot.send_message(update.effective_chat.id, f"Ошибка генерации: {e}")

async def handle_comment(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Incoming comment from discussion group."""
    if not update.message or not update.message.text:
        return
    if str(update.effective_chat.id) != settings.telegram_discussion_group_id:
        return
    # Ignore own messages and manager
    if update.effective_user and str(update.effective_user.id) == settings.manager_user_id:
        return

    text = update.message.text
    # 1. Moderator
    mod_result = moderator_graph.invoke({"comment_text": text, "decision": "pass", "reason": ""})
    if mod_result["decision"] == "delete":
        try:
            await update.message.delete()
            logger.info(f"Deleted comment: {mod_result.get('reason')}")
        except Exception as e:
            logger.error(f"Failed to delete: {e}")
        return

    # 2. Responder (only if passed)
    resp_result = responder_graph.invoke({"comment_text": text, "reply": ""})
    reply = resp_result.get("reply", "")
    if reply:
        try:
            await update.message.reply_text(reply)
        except Exception as e:
            logger.error(f"Reply failed: {e}")

def build_application() -> Application:
    app = Application.builder().token(settings.telegram_bot_token).build()
    # Manager chat only
    app.add_handler(MessageHandler(
        filters.Chat(chat_id=int(settings.manager_chat_id)) & filters.TEXT & ~filters.COMMAND,
        handle_manager_message
    ))
    # Discussion group comments
    app.add_handler(MessageHandler(
        filters.Chat(chat_id=int(settings.telegram_discussion_group_id)) & filters.TEXT & ~filters.COMMAND,
        handle_comment
    ))
    return app

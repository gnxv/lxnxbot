"""Telegram bot handlers. Routes to graphs."""
import logging
import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo
from telegram import Update, InputFile
from telegram.ext import Application, MessageHandler, filters, ContextTypes
from app.config import settings
from app.graphs.manager import manager_graph
from app.graphs.moderator import moderator_graph
from app.graphs.responder import responder_graph
from app.graphs.custom_poster import custom_poster_graph
from app.graphs.poster import poster_graph
from app.db import get_pending_draft, update_draft_status, save_post
from pathlib import Path

logger = logging.getLogger(__name__)

async def is_manager(update: Update) -> bool:
    if not update.effective_chat or not update.effective_user:
        return False
    return (
        str(update.effective_chat.id) == settings.manager_chat_id
        and str(update.effective_user.id) in settings.manager_user_ids
    )

# Temporary storage for media groups: media_group_id -> {"photos": [], "caption": "", "task": None}
_media_groups: dict = {}

async def _process_media_group(media_group_id: str, update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Process accumulated photos from one media group after a short delay."""
    await asyncio.sleep(1.5)  # wait for remaining photos in the album
    group = _media_groups.pop(media_group_id, None)
    if not group:
        return
    photos = group["photos"]
    caption = group.get("caption") or ""
    if not photos:
        return

    context.user_data.setdefault("pending_photos", []).extend(photos)
    logger.info(f"Media group {media_group_id}: collected {len(photos)} photos, caption={bool(caption)}")

    if not caption:
        await update.message.reply_text(
            f"Получил {len(photos)} фото. На какую тему написать пост или пришли готовый текст?"
        )
        return

    # Process as one request with all photos
    text = caption
    state = {
        "messages": [{"role": "user", "content": text}],
        "action": None,
        "pending_confirm": context.user_data.get("pending_confirm"),
        "response": None,
    }
    result = manager_graph.invoke(state)
    new_pending = result.get("pending_confirm")
    if new_pending and new_pending.get("type") in ("extra_confirm", "extra_waiting_details", "extra_generate_preview", "extra_preview_confirm"):
        new_pending["image_paths"] = list(context.user_data.get("pending_photos") or [])
    context.user_data["pending_confirm"] = new_pending
    if result.get("response"):
        await update.message.reply_text(result["response"])

    if new_pending and new_pending.get("type") == "extra_generate_preview":
        asyncio.create_task(generate_and_show_preview(update, context, new_pending))

async def handle_manager_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_manager(update):
        return  # complete silence

    # Handle photos (including media groups / albums)
    if update.message.photo:
        photo = update.message.photo[-1]
        file = await photo.get_file()
        path = settings.temp_images_dir / f"user_{update.message.message_id}.jpg"
        await file.download_to_drive(path)

        media_group_id = update.message.media_group_id
        if media_group_id:
            # Album: accumulate and process after short delay
            if media_group_id not in _media_groups:
                _media_groups[media_group_id] = {"photos": [], "caption": "", "task": None}
            _media_groups[media_group_id]["photos"].append(str(path))
            if update.message.caption:
                _media_groups[media_group_id]["caption"] = update.message.caption
            # Restart delay timer so we wait for the last photo
            if _media_groups[media_group_id]["task"]:
                _media_groups[media_group_id]["task"].cancel()
            _media_groups[media_group_id]["task"] = asyncio.create_task(
                _process_media_group(media_group_id, update, context)
            )
            return
        else:
            # Single photo
            context.user_data.setdefault("pending_photos", []).append(str(path))
            caption = update.message.caption or ""
            if caption:
                text = caption
            else:
                await update.message.reply_text("Фото получил. На какую тему написать пост или пришли готовый текст?")
                return
    else:
        text = update.message.text or ""

    state = {
        "messages": [{"role": "user", "content": text}],
        "action": None,
        "pending_confirm": context.user_data.get("pending_confirm"),
        "response": None,
    }
    result = manager_graph.invoke(state)
    new_pending = result.get("pending_confirm")
    # Attach photos if we have them and this is an extra-related pending
    photos = context.user_data.get("pending_photos") or []
    if new_pending and photos and new_pending.get("type") in ("extra_confirm", "extra_waiting_details", "extra_generate_preview", "extra_preview_confirm"):
        new_pending["image_paths"] = list(photos)
        logger.info(f"Attached {len(photos)} photos to pending {new_pending.get('type')}")
    context.user_data["pending_confirm"] = new_pending
    if result.get("response"):
        await update.message.reply_text(result["response"])

    # Clear photos only after they have been saved into an extra post
    if new_pending is None and photos:
        context.user_data["pending_photos"] = []

    # Handle side effects
    if new_pending and new_pending.get("type") == "do_publish":
        await publish_draft(new_pending["draft_id"], context)
        context.user_data["pending_confirm"] = None
    elif new_pending and new_pending.get("type") == "do_publish_extra":
        await publish_extra_now(new_pending, context)
        context.user_data["pending_confirm"] = None
        context.user_data["pending_photos"] = []
    elif new_pending and new_pending.get("type") == "extra_generate_preview":
        asyncio.create_task(generate_and_show_preview(update, context, new_pending))

async def publish_draft(draft_id: int, context: ContextTypes.DEFAULT_TYPE):
    from app.db import get_conn
    with get_conn() as conn:
        cur = conn.execute("SELECT * FROM drafts WHERE id = %s", (draft_id,))
        draft = cur.fetchone()
    if not draft:
        logger.warning(f"Draft {draft_id} not found for publish")
        return
    chat_id = settings.telegram_channel_id
    try:
        caption = draft["text"]
        rest = None
        if len(caption) > 1024:
            caption = draft["text"][:1000].rsplit(" ", 1)[0] + "…"
            rest = draft["text"][len(caption)-1:].strip()
            logger.info(f"Publish: text too long ({len(draft['text'])}), splitting")

        if draft["image_path"] and Path(draft["image_path"]).exists():
            with open(draft["image_path"], "rb") as f:
                msg = await context.bot.send_photo(
                    chat_id=chat_id,
                    photo=InputFile(f),
                    caption=caption
                )
        else:
            msg = await context.bot.send_message(chat_id=chat_id, text=caption)

        if rest and msg:
            await context.bot.send_message(chat_id=chat_id, text=rest, reply_to_message_id=msg.message_id)
        update_draft_status(draft_id, "published")
        save_post(draft_id, msg.message_id, draft["text"])
        # cleanup image
        if draft["image_path"]:
            try:
                Path(draft["image_path"]).unlink(missing_ok=True)
                logger.info(f"Deleted temp image {draft['image_path']}")
            except Exception as e:
                logger.warning(f"Failed to delete image: {e}")
        logger.info(f"Published draft {draft_id} to channel")
        # Notify manager
        if settings.manager_chat_id:
            await context.bot.send_message(
                chat_id=settings.manager_chat_id,
                text=f"Пост опубликован в канал (черновик #{draft_id})."
            )
    except Exception as e:
        logger.error(f"Publish failed: {e}")
        if settings.manager_chat_id:
            await context.bot.send_message(
                chat_id=settings.manager_chat_id,
                text=f"Не удалось опубликовать: {e}\nПроверь, что бот — админ в канале и ID канала верный."
            )

async def generate_and_show_preview(update: Update, context: ContextTypes.DEFAULT_TYPE, pending: dict):
    """Generate text (if needed) and show preview to admin before publishing."""
    try:
        topic = pending.get("topic") or "наша новая работа"
        ready_text = pending.get("text")
        image_paths = pending.get("image_paths") or context.user_data.get("pending_photos") or []
        run_at = pending.get("run_at")

        text = ready_text
        generated_image = None
        if not text:
            use_custom = bool(image_paths) or (topic and "реставрация кухонь" not in topic.lower())
            graph = custom_poster_graph if use_custom else poster_graph
            result = graph.invoke({
                "topic": topic,
                "text": None,
                "image_prompt": None,
                "image_path": None,
                "image_paths": image_paths,
                "draft_id": None,
                "error": None,
            })
            text = result.get("text") or "Не удалось сгенерировать текст."
            generated_image = result.get("image_path")
            if generated_image and not image_paths:
                image_paths = [generated_image]

        # Prefer user photos, fall back to generated
        preview_paths = image_paths if image_paths else ([generated_image] if generated_image else [])

        chat_id = update.effective_chat.id
        caption = f"Черновик на {run_at.strftime('%d.%m %H:%M') if run_at else 'сейчас'}:\n\n{text[:900]}"

        valid_paths = [p for p in preview_paths if Path(p).exists()]
        if len(valid_paths) > 1:
            from telegram import InputMediaPhoto
            media = []
            open_files = []
            for i, p in enumerate(valid_paths[:10]):  # Telegram limit
                f = open(p, "rb")
                open_files.append(f)
                media.append(InputMediaPhoto(media=f, caption=caption if i == 0 else None))
            await context.bot.send_media_group(chat_id=chat_id, media=media)
            for f in open_files:
                f.close()
        elif valid_paths:
            with open(valid_paths[0], "rb") as f:
                await context.bot.send_photo(
                    chat_id=chat_id,
                    photo=InputFile(f),
                    caption=caption
                )
        else:
            await context.bot.send_message(chat_id=chat_id, text=caption)

        await context.bot.send_message(
            chat_id=chat_id,
            text=(
                "Опубликовать этот пост сейчас? Напиши «да» или «нет» / «отмена»."
                if (not run_at or abs((run_at - datetime.now(ZoneInfo("Europe/Moscow"))).total_seconds()) < 120)
                else f"Запланировать этот пост на {run_at.strftime('%d.%m %H:%M')}? Напиши «да» или «нет» / «отмена»."
            )
        )

        # Set confirmation state
        # "сейчас" = within 2 minutes of the requested time (or no time)
        now = datetime.now(ZoneInfo("Europe/Moscow"))
        is_now = (not run_at) or abs((run_at - now).total_seconds()) < 120
        context.user_data["pending_confirm"] = {
            "type": "extra_preview_confirm" if is_now else "extra_confirm",
            "run_at": run_at,
            "text": text,
            "topic": topic,
            "image_paths": image_paths,
        }
    except Exception as e:
        logger.error(f"Preview generation failed: {e}")
        if update.effective_chat:
            await context.bot.send_message(update.effective_chat.id, f"Ошибка генерации черновика: {e}")

async def publish_extra_now(pending: dict, context: ContextTypes.DEFAULT_TYPE):
    """Publish already-generated extra post immediately."""
    from telegram import InputMediaPhoto
    text = pending.get("text") or ""
    paths = pending.get("image_paths") or []
    chat_id = settings.telegram_channel_id
    try:
        if paths:
            media = []
            open_files = []
            for i, p in enumerate(paths):
                if Path(p).exists():
                    f = open(p, "rb")
                    open_files.append(f)
                    media.append(InputMediaPhoto(media=f, caption=text[:1024] if i == 0 and text else None))
            if media:
                await context.bot.send_media_group(chat_id=chat_id, media=media)
                for f in open_files:
                    f.close()
            elif text:
                await context.bot.send_message(chat_id=chat_id, text=text)
        elif text:
            await context.bot.send_message(chat_id=chat_id, text=text)
        if settings.manager_chat_id:
            await context.bot.send_message(settings.manager_chat_id, "Пост опубликован в канал.")
    except Exception as e:
        logger.error(f"Immediate extra publish failed: {e}")
        if settings.manager_chat_id:
            await context.bot.send_message(settings.manager_chat_id, f"Ошибка публикации: {e}")

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
    # Manager chat (text + photos)
    app.add_handler(MessageHandler(
        filters.Chat(chat_id=int(settings.manager_chat_id)) & (filters.TEXT | filters.PHOTO) & ~filters.COMMAND,
        handle_manager_message
    ))
    # Discussion group comments
    app.add_handler(MessageHandler(
        filters.Chat(chat_id=int(settings.telegram_discussion_group_id)) & filters.TEXT & ~filters.COMMAND,
        handle_comment
    ))
    return app

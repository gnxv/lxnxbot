"""Manager graph: only responds in manager chat from manager user. Handles schedule, approval, extra posts."""
import logging
from typing import TypedDict, Optional, Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from app.llm import get_manager_llm
from app.db import get_schedule, set_schedule, get_pending_draft, update_draft_status
from app.config import settings

logger = logging.getLogger(__name__)

class ManagerState(TypedDict):
    messages: list
    action: Optional[str]
    pending_confirm: Optional[dict]
    response: Optional[str]

SYSTEM_MANAGER = """Ты — помощник админа канала про реставрацию кухонь.
Отвечай коротко, по-русски, без технического жаргона.

Распознавай намерения админа (даже если пишет неформально):
- show_schedule — показать расписание, слоты, что запланировано
- change_schedule — изменить / поставить / добавить / убрать время постов
- extra_post — сделать пост сейчас или на время (с фото, темой, готовым текстом или просто «выложи это»)
- approve_draft — одобрить черновик / «да» после показа текста
- confirm — подтверждение «да», «ок», «публикуй»
- cancel — отмена, сброс, «отмени», «нет», «передумал»
- other

Для extra_post старайся понять тему из свободного текста.
Верни только одно слово действия из списка выше."""

def classify_node(state: ManagerState) -> ManagerState:
    last = state["messages"][-1] if state["messages"] else ""
    if isinstance(last, dict):
        last = last.get("content", "")
    llm = get_manager_llm()
    prompt = f"""Сообщение админа: {last}
Текущий pending_confirm: {state.get('pending_confirm')}

Определи действие: show_schedule | change_schedule | extra_post | approve_draft | confirm | other
Верни только одно слово из списка."""
    resp = llm.invoke([SystemMessage(content=SYSTEM_MANAGER), HumanMessage(content=prompt)])
    action = resp.content.strip().lower()
    state["action"] = action
    return state

import re
from datetime import time

def extract_topic(text: str) -> str:
    """Best-effort topic extraction from free-form admin message."""
    lower = text.lower().strip()
    # Remove common request prefixes
    for prefix in [
        "напиши пост", "сделай пост", "выложи", "опубликуй", "пост про",
        "напиши про", "сделай про", "это наша", "наша новая работа",
        "новая работа", "работа", "пост", "тема"
    ]:
        if lower.startswith(prefix):
            lower = lower[len(prefix):].strip(" :,.-")
            break
    # Cut at time mentions
    lower = re.sub(r"\s*(сейчас|в\s+\d{1,2}:\d{2}|на\s+\d{1,2}:\d{2}).*", "", lower).strip()
    if not lower or len(lower) < 3:
        return "наша новая работа"
    return lower[0].upper() + lower[1:] if lower else "наша новая работа"

def parse_times(text: str) -> list:
    """Extract HH:MM times from text."""
    matches = re.findall(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", text)
    times = []
    for h, m in matches:
        times.append(time(int(h), int(m)))
    return times

from app.db import (
    get_schedule, set_schedule, get_pending_draft, update_draft_status,
    get_upcoming_extras, save_extra_post, cancel_extra, get_pending_extras
)
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

MSK = ZoneInfo("Europe/Moscow")

def handle_node(state: ManagerState) -> ManagerState:
    action = state.get("action", "other")
    last_raw = state["messages"][-1] if state["messages"] else ""
    last_msg = last_raw.get("content", "") if isinstance(last_raw, dict) else str(last_raw)
    last_msg_lower = last_msg.strip().lower()
    response = ""

    # Handle time choice for extra post
    pending = state.get("pending_confirm") or {}
    if pending.get("type") == "extra_waiting_details":
        choice = last_msg_lower.strip()
        run_at = None
        if choice in ("1", "сейчас", "сейчас."):
            run_at = datetime.now(MSK)
        elif choice in ("2", "17:30", "вечером"):
            run_at = pending.get("evening")
        elif choice in ("3", "вместо"):
            # Replace evening — for simplicity schedule at 18:00 and note it
            run_at = datetime.now(MSK).replace(hour=18, minute=0, second=0, microsecond=0)
            if run_at < datetime.now(MSK):
                run_at += timedelta(days=1)
        else:
            times = parse_times(last_msg)
            if times:
                t = times[0]
                run_at = datetime.now(MSK).replace(hour=t.hour, minute=t.minute, second=0, microsecond=0)
                if run_at < datetime.now(MSK):
                    run_at += timedelta(days=1)

        if run_at:
            # Time chosen — request preview generation
            response = (
                "Генерирую черновик. Сейчас покажу текст, потом подтвердишь публикацию."
            )
            state["pending_confirm"] = {
                "type": "extra_generate_preview",
                "run_at": run_at,
                "text": pending.get("text"),
                "topic": pending.get("topic"),
                "image_paths": pending.get("image_paths"),
            }
            state["response"] = response
            return state

    if action == "confirm" or last_msg_lower == "да":
        pending = state.get("pending_confirm")
        if pending:
            if pending["type"] == "schedule":
                set_schedule(pending["times"])
                response = f"Расписание обновлено: {', '.join(f'{t.hour:02d}:{t.minute:02d}' for t in pending['times'])}"
                state["pending_confirm"] = None
            elif pending["type"] == "publish":
                draft = get_pending_draft()
                if draft:
                    update_draft_status(draft["id"], "approved")
                    response = "Черновик одобрен. Публикую в канал..."
                    state["pending_confirm"] = {"type": "do_publish", "draft_id": draft["id"]}
                else:
                    response = "Нет черновика на одобрение."
                    state["pending_confirm"] = None
            elif pending["type"] == "extra_confirm":
                # This type is now set after preview is shown and admin confirms
                eid = save_extra_post(
                    run_at=pending["run_at"],
                    text=pending.get("text"),
                    topic=pending.get("topic"),
                    image_paths=pending.get("image_paths") or [],
                )
                response = f"Пост запланирован на {pending['run_at'].strftime('%d.%m %H:%M')} (id {eid})."
                state["pending_confirm"] = None
            elif pending["type"] == "extra_preview_confirm":
                # Immediate publish after preview
                state["pending_confirm"] = {
                    "type": "do_publish_extra",
                    "text": pending.get("text"),
                    "image_paths": pending.get("image_paths") or [],
                }
                response = "Публикую в канал..."
            else:
                state["pending_confirm"] = None
        else:
            response = "Нечего подтверждать."

    elif action == "show_schedule":
        times = get_schedule()
        extras = get_upcoming_extras()
        times_str = ", ".join(f"{t.hour:02d}:{t.minute:02d}" for t in times) if times else "не задано"
        lines = [f"Обычное расписание: {times_str}"]
        if extras:
            lines.append("Внеплановые:")
            for e in extras:
                t = e["run_at"].astimezone(MSK).strftime("%d.%m %H:%M")
                desc = (e["text"][:50] + "…") if e["text"] else (e["topic"] or "без темы")
                lines.append(f"— {t} — {desc}")
        response = "\n".join(lines)

    elif action == "change_schedule":
        times = parse_times(last_msg)
        if len(times) >= 1:
            times = times[:10]
            times_str = [f"{t.hour:02d}:{t.minute:02d}" for t in times]
            response = f"Меняю расписание на {', '.join(times_str)}. Подтвердите: да"
            state["pending_confirm"] = {"type": "schedule", "times": times}
        else:
            response = "Напиши времена в формате 10:00, 14:30 и т.д., затем подтверди «да»."

    elif any(w in last_msg_lower for w in ["отмени", "отмена", "сброс", "передумал", "нет"]):
        state["pending_confirm"] = None
        response = "Ок, отменил. Что дальше?"

    elif "отмени" in last_msg_lower and ("внеплан" in last_msg_lower or "пост" in last_msg_lower):
        extras = get_upcoming_extras()
        if not extras:
            response = "Нет запланированных внеплановых постов."
        elif len(extras) == 1:
            cancel_extra(extras[0]["id"])
            response = "Внеплановый пост отменён."
        else:
            response = "Какой отменить? Напиши id или время.\n" + "\n".join(
                f"id {e['id']} — {e['run_at'].astimezone(MSK).strftime('%H:%M')}" for e in extras
            )

    elif action == "extra_post" or any(w in last_msg_lower for w in ["опубликуй", "выложи", "внеплан", "пост про", "напиши текст", "сгенерируй"]):
        now = datetime.now(MSK)
        evening = now.replace(hour=17, minute=30, second=0, microsecond=0)
        if evening < now:
            evening += timedelta(days=1)

        # Extract explicit time if present
        explicit_times = parse_times(last_msg)
        run_at = None
        if explicit_times:
            t = explicit_times[0]
            run_at = now.replace(hour=t.hour, minute=t.minute, second=0, microsecond=0)
            if run_at < now:
                run_at += timedelta(days=1)

        # Decide if text is ready or needs generation
        ready_text = None
        topic = extract_topic(last_msg)
        generate = any(w in last_msg_lower for w in ["сгенерируй", "напиши текст", "напиши пост", "придумай", "сделай пост"])

        # If message looks like ready text (has colon and is long enough, no generation words)
        if not generate and ":" in last_msg and len(last_msg.split(":", 1)[1].strip()) > 40:
            ready_text = last_msg.split(":", 1)[1].strip()
            topic = None

        if run_at:
            response = (
                "Понял.\n"
                + (f"Тема: {topic}\n" if topic and not ready_text else "")
                + ("Текст сгенерирую и покажу.\n" if not ready_text else "Готовый текст принят.\n")
                + f"Время: {run_at.strftime('%d.%m %H:%M')}. Сейчас покажу черновик."
            )
            state["pending_confirm"] = {
                "type": "extra_generate_preview",
                "run_at": run_at,
                "text": ready_text,
                "topic": topic if not ready_text else None,
                "image_paths": None,
            }
        else:
            response = (
                "Понял.\n"
                + (f"Тема: {topic}\n" if topic and not ready_text else "")
                + ("Текст сгенерирую и покажу.\n" if not ready_text else "Готовый текст принят.\n")
                + "Когда опубликовать?\n"
                "— сейчас\n"
                "— или напиши время (например 19:15)"
            )
            state["pending_confirm"] = {
                "type": "extra_waiting_details",
                "topic": topic if not ready_text else None,
                "text": ready_text,
                "evening": evening,
            }

    else:
        response = "Понял. Могу: показать расписание, изменить его, сделать пост (с фото или без), отменить текущее. Напиши простыми словами."

    state["response"] = response
    if "messages" in state:
        state["messages"].append(AIMessage(content=response))
    return state

def build_manager_graph():
    workflow = StateGraph(ManagerState)
    workflow.add_node("classify", classify_node)
    workflow.add_node("handle", handle_node)
    workflow.set_entry_point("classify")
    workflow.add_edge("classify", "handle")
    workflow.add_edge("handle", END)
    return workflow.compile()

manager_graph = build_manager_graph()

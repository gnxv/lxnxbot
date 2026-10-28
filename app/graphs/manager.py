"""Manager graph: only responds in manager chat from manager user. Handles schedule, approval, extra posts."""
import logging
import re
from datetime import datetime, time, timedelta
from typing import TypedDict, Optional, Literal
from zoneinfo import ZoneInfo
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from app.llm import get_manager_llm
from app.db import (
    get_schedule, set_schedule, get_pending_draft, update_draft_status,
    get_upcoming_extras, save_extra_post, cancel_extra, get_pending_extras
)
from app.config import settings

logger = logging.getLogger(__name__)

MSK = ZoneInfo("Europe/Moscow")


class ManagerState(TypedDict):
    messages: list
    action: Optional[str]
    pending_confirm: Optional[dict]
    response: Optional[str]

SYSTEM_MANAGER = """Ты — помощник админа канала про реставрацию кухонь.
Отвечай коротко, по-русски, без технического жаргона.

Распознавай намерения админа (даже если пишет неформально):
- show_schedule — показать расписание, слоты, что запланировано
- change_schedule — изменить обычное расписание постов (несколько времён в день)
- extra_post — сделать один пост сейчас или на конкретное время (с фото, темой или просто «опубликуй пост»). Даже если в сообщении есть время — это extra_post, если речь про один пост.
- approve_draft — одобрить черновик / «да» после показа текста
- confirm — подтверждение «да», «ок»
- cancel — отмена, сброс, «отмени», «нет», «передумал»
- other

Важно: «опубликуй пост в 20:40» — это extra_post, а не change_schedule.
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


def extract_topic(text: str) -> str:
    """Best-effort topic extraction from free-form admin message."""
    lower = text.lower().strip()

    # Prefer explicit topic markers
    for marker in ["на тему ", "про ", "тема ", "пост про ", "текст про "]:
        if marker in lower:
            topic = lower.split(marker, 1)[1].strip()
            # Clean trailing time/date mentions
            topic = re.sub(r"\s*(сейчас|завтра|послезавтра|в\s+\d{1,2}:\d{2}|на\s+\d{1,2}:\d{2}|\d{1,2}\.\d{1,2}.*)", "", topic).strip(" .,:-")
            if topic and len(topic) >= 3:
                return topic[0].upper() + topic[1:]

    # Fallback: strip request prefixes and time expressions
    for prefix in [
        "напиши пост", "сделай пост", "выложи", "опубликуй пост", "опубликуй",
        "пост про", "напиши про", "сделай про", "это наша", "наша новая работа",
        "новая работа", "работа", "пост", "тема"
    ]:
        if lower.startswith(prefix):
            lower = lower[len(prefix):].strip(" :,.-")
            break

    # Remove time/date phrases anywhere
    lower = re.sub(r"\b(сейчас|завтра|послезавтра|вечером)\b", " ", lower)
    lower = re.sub(r"\b\d{1,2}:\d{2}\b", " ", lower)
    lower = re.sub(r"\b\d{1,2}\.\d{1,2}(?:\.\d{2,4})?\b", " ", lower)
    lower = re.sub(r"\s+", " ", lower).strip(" .,:-")

    if not lower or len(lower) < 3:
        return None
    return lower[0].upper() + lower[1:]

def parse_times(text: str) -> list:
    """Extract HH:MM times from text."""
    matches = re.findall(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", text)
    times = []
    for h, m in matches:
        times.append(time(int(h), int(m)))
    return times

def parse_datetime(text: str, now: datetime = None) -> Optional[datetime]:
    """Parse flexible date/time expressions into a datetime (MSK)."""
    if now is None:
        now = datetime.now(MSK)
    lower = text.lower().strip()

    # Relative days
    target_date = now.date()
    if "послезавтра" in lower:
        target_date = now.date() + timedelta(days=2)
    elif "завтра" in lower:
        target_date = now.date() + timedelta(days=1)

    # Explicit date DD.MM or DD.MM.YYYY
    date_match = re.search(r"\b(\d{1,2})\.(\d{1,2})(?:\.(\d{2,4}))?\b", lower)
    if date_match:
        d, m = int(date_match.group(1)), int(date_match.group(2))
        y = int(date_match.group(3)) if date_match.group(3) else now.year
        if y < 100:
            y += 2000
        try:
            target_date = datetime(y, m, d).date()
        except ValueError:
            pass

    # Time HH:MM
    times = parse_times(text)
    if times:
        t = times[0]
        run_at = datetime.combine(target_date, t).replace(tzinfo=MSK)
        # If only time given and it's already passed today, move to tomorrow
        if "завтра" not in lower and "послезавтра" not in lower and not date_match and run_at <= now:
            run_at += timedelta(days=1)
        return run_at

    # No explicit time — only accept explicit "сейчас"
    if lower in ("сейчас", "сейчас.", "1"):
        return now

    return None



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
        if choice in ("сейчас", "сейчас.", "1"):
            run_at = datetime.now(MSK)
        else:
            # Only accept clear time or date+time
            run_at = parse_datetime(last_msg)

        if run_at:
            response = "Генерирую черновик. Сейчас покажу текст, потом подтвердишь публикацию."
            state["pending_confirm"] = {
                "type": "extra_generate_preview",
                "run_at": run_at,
                "text": pending.get("text"),
                "topic": pending.get("topic"),
                "image_paths": pending.get("image_paths"),
            }
            state["response"] = response
            return state
        else:
            response = (
                "Не понял время. Напиши:\n"
                "— сейчас\n"
                "— или время (19:15)\n"
                "— или дату и время (29.10 10:00)"
            )
            state["response"] = response
            return state

    # Strong override: clear post request beats classifier mistakes
    if any(w in last_msg_lower for w in ["опубликуй пост", "выложи пост", "сделай пост", "напиши пост", "внеплановый пост"]):
        action = "extra_post"

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

        # Decide if text is ready or needs generation
        ready_text = None
        topic = extract_topic(last_msg)
        generate = any(w in last_msg_lower for w in ["сгенерируй", "напиши текст", "напиши пост", "придумай", "сделай пост"])

        # If message looks like ready text (has colon and is long enough, no generation words)
        if not generate and ":" in last_msg and len(last_msg.split(":", 1)[1].strip()) > 40:
            ready_text = last_msg.split(":", 1)[1].strip()
            topic = None

        # If explicit time/date is already in the message — don't ask again
        run_at = parse_datetime(last_msg)
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
            }
        else:
            # No time given — ask
            response = (
                "Понял.\n"
                + (f"Тема: {topic}\n" if topic and not ready_text else "")
                + ("Текст сгенерирую и покажу.\n" if not ready_text else "Готовый текст принят.\n")
                + "Когда опубликовать?\n"
                "— сейчас\n"
                "— или напиши время (19:15) или дату и время (29.10 10:00)"
            )
            state["pending_confirm"] = {
                "type": "extra_waiting_details",
                "topic": topic if not ready_text else None,
                "text": ready_text,
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

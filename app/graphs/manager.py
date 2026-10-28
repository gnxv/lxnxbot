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

Доступные команды (распознавай намерение):
- Показать расписание
- Изменить расписание (на два времени в формате HH:MM)
- Сделать внеочередной пост
- Одобрить черновик (да / нет)

ВАЖНО:
- Любое изменение (расписание, публикация) требует явного подтверждения словом «да».
- Сначала предложи действие и спроси «Подтвердите: да».
- Только после «да» выполняй.
- Если сообщение не от админа — игнорируй (но сюда доходят только от него)."""

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

    if action == "confirm" or last_msg_lower == "да":
        pending = state.get("pending_confirm")
        if pending:
            if pending["type"] == "schedule":
                set_schedule(pending["times"])
                response = f"Расписание обновлено: {', '.join(str(t) for t in pending['times'])}"
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
                # Save the extra post
                eid = save_extra_post(
                    run_at=pending["run_at"],
                    text=pending.get("text"),
                    topic=pending.get("topic"),
                    image_paths=pending.get("image_paths"),
                )
                response = f"Запланировала внеплановый пост на {pending['run_at'].strftime('%H:%M')} (id {eid})."
                state["pending_confirm"] = None
            else:
                state["pending_confirm"] = None
        else:
            response = "Нечего подтверждать."

    elif action == "show_schedule":
        times = get_schedule()
        extras = get_upcoming_extras()
        lines = ["Обычное расписание: " + (", ".join(str(t) for t in times) if times else "не задано")]
        if extras:
            lines.append("Внеплановые:")
            for e in extras:
                t = e["run_at"].astimezone(MSK).strftime("%d.%m %H:%M")
                desc = e["text"][:60] + "…" if e["text"] else (e["topic"] or "без темы")
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

    elif action == "extra_post" or "опубликуй" in last_msg_lower or "внеплан" in last_msg_lower:
        # Simple clarification flow
        now = datetime.now(MSK)
        # Default evening suggestion
        evening = now.replace(hour=17, minute=30, second=0, microsecond=0)
        if evening < now:
            evening += timedelta(days=1)
        response = (
            "Поняла, внеплановый пост. Как сделать?\n"
            "1. Сейчас\n"
            "2. Сегодня в 17:30 (перед обычным вечерним)\n"
            "3. Вместо обычного вечернего\n"
            "4. Напиши точное время (например 19:15)\n\n"
            "Если есть готовый текст — пришли его. Если только фото — напиши тему."
        )
        state["pending_confirm"] = {"type": "extra_waiting_details", "evening": evening}

    else:
        response = "Понял. Могу показать расписание, изменить его, сделать внеплановый пост или одобрить черновик."

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

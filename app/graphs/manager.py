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

def handle_node(state: ManagerState) -> ManagerState:
    action = state.get("action", "other")
    last_raw = state["messages"][-1] if state["messages"] else ""
    last_msg = last_raw.get("content", "") if isinstance(last_raw, dict) else str(last_raw)
    last_msg = last_msg.strip().lower()
    response = ""

    if action == "confirm" or last_msg == "да":
        pending = state.get("pending_confirm")
        if pending:
            if pending["type"] == "schedule":
                set_schedule(pending["times"])
                response = f"Расписание обновлено: {', '.join(pending['times'])}"
                state["pending_confirm"] = None
            elif pending["type"] == "publish":
                draft = get_pending_draft()
                if draft:
                    update_draft_status(draft["id"], "approved")
                    response = "Черновик одобрен. Публикую в канал..."
                    # Keep do_publish so handler can publish
                    state["pending_confirm"] = {"type": "do_publish", "draft_id": draft["id"]}
                else:
                    response = "Нет черновика на одобрение."
                    state["pending_confirm"] = None
            else:
                state["pending_confirm"] = None
        else:
            response = "Нечего подтверждать."
    elif action == "show_schedule":
        times = get_schedule()
        response = "Текущее расписание: " + (", ".join(str(t) for t in times) if times else "не задано")
    elif action == "change_schedule":
        # Extract times roughly via LLM or simple
        response = "Напиши два времени в формате HH:MM и HH:MM, затем подтверди «да»."
        # In real use the next message will be parsed
    elif action == "extra_post":
        response = "Запускаю генерацию внеочередного поста. Черновик придёт сюда на одобрение."
        state["pending_confirm"] = {"type": "extra_post"}
    elif action == "approve_draft":
        draft = get_pending_draft()
        if draft:
            response = f"Черновик:\n\n{draft['text'][:500]}...\n\nОдобрить публикацию? Напиши «да»."
            state["pending_confirm"] = {"type": "publish", "draft_id": draft["id"]}
        else:
            response = "Нет черновиков на одобрение."
    else:
        response = "Понял. Могу показать расписание, изменить его, сделать внеочередной пост или одобрить черновик."

    state["response"] = response
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

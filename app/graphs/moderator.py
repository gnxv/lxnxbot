"""Moderator graph: decide keep or delete comment."""
import logging
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from app.llm import get_moderator_llm

logger = logging.getLogger(__name__)

class ModeratorState(TypedDict):
    comment_text: str
    decision: Literal["delete", "pass"]
    reason: str

SYSTEM_MODERATOR = """Ты — модератор комментариев в канале про реставрацию кухонь.
Твоя задача: решить, удалить комментарий или пропустить дальше.

Удалять если:
- мат, оскорбления, токсичность
- спам, реклама, ссылки на услуги/товары
- попытки увести в личку с предложениями услуг
- оффтоп полностью не по теме

Пропускать если:
- нормальный вопрос/комментарий по теме реставрации кухонь
- нейтральный отзыв, уточнение, благодарность

Ответь строго в формате:
DECISION: delete|pass
REASON: коротко почему
"""

def moderate_node(state: ModeratorState) -> ModeratorState:
    llm = get_moderator_llm()
    resp = llm.invoke([
        SystemMessage(content=SYSTEM_MODERATOR),
        HumanMessage(content=f"Комментарий: {state['comment_text']}")
    ])
    content = resp.content
    decision = "pass"
    reason = ""
    if "DECISION: delete" in content or "delete" in content.lower():
        decision = "delete"
    if "REASON:" in content:
        reason = content.split("REASON:")[-1].strip()
    state["decision"] = decision
    state["reason"] = reason
    logger.info(f"Moderator: {decision} — {reason}")
    return state

def build_moderator_graph():
    workflow = StateGraph(ModeratorState)
    workflow.add_node("moderate", moderate_node)
    workflow.set_entry_point("moderate")
    workflow.add_edge("moderate", END)
    return workflow.compile()

moderator_graph = build_moderator_graph()

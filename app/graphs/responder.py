"""Responder graph: short safe reply to approved comments."""
import logging
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from app.llm import get_responder_llm

logger = logging.getLogger(__name__)

class ResponderState(TypedDict):
    comment_text: str
    reply: str

SYSTEM_RESPONDER = """Ты отвечаешь на комментарии в канале про реставрацию кухонь.
Отвечай коротко (1-2 предложения), вежливо, по делу.
Не знаешь деталей постов и расписания — не выдумывай.
Если вопрос про конкретный заказ/цену/замер — направляй к менеджеру.
Примеры:
- «Напишите менеджеру по этому вопросу, там лучше подскажут.»
- «Да, фасады можно перекрасить, главное подготовить поверхность.»
- «Спасибо за комментарий!»

Никакой рекламы и контактов от себя."""

def respond_node(state: ResponderState) -> ResponderState:
    llm = get_responder_llm()
    resp = llm.invoke([
        SystemMessage(content=SYSTEM_RESPONDER),
        HumanMessage(content=f"Комментарий: {state['comment_text']}\nНапиши короткий ответ.")
    ])
    state["reply"] = resp.content.strip()
    logger.info(f"Responder: {state['reply'][:100]}")
    return state

def build_responder_graph():
    workflow = StateGraph(ResponderState)
    workflow.add_node("respond", respond_node)
    workflow.set_entry_point("respond")
    workflow.add_edge("respond", END)
    return workflow.compile()

responder_graph = build_responder_graph()

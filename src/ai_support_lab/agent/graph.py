from typing import Literal

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ai_support_lab.agent.nodes import AnalysisNodes
from ai_support_lab.agent.state import AnalysisState


def route_ticket(state: AnalysisState) -> Literal["context", "simple"]:
    complex_ticket = (
        state["priority"] in ("high", "critical") or state["classification"]["score"] < 0.6
    )
    return "context" if state["include_context"] and complex_ticket else "simple"


def build_graph(
    nodes: AnalysisNodes, checkpointer: InMemorySaver | None = None
) -> CompiledStateGraph:
    graph = StateGraph(AnalysisState)
    graph.add_node("load_ticket", nodes.load_ticket)
    graph.add_node("classify_ticket", nodes.classify_ticket)
    graph.add_node("assess_priority", nodes.assess_priority)
    graph.add_node("gather_context", nodes.gather_context)
    graph.add_node("generate_response", nodes.generate_response)
    graph.add_node("validate_response", nodes.validate_response)
    graph.add_node("persist_agent_run", nodes.persist_agent_run)
    graph.add_edge(START, "load_ticket")
    graph.add_edge("load_ticket", "classify_ticket")
    graph.add_edge("classify_ticket", "assess_priority")
    graph.add_conditional_edges(
        "assess_priority",
        route_ticket,
        {"context": "gather_context", "simple": "generate_response"},
    )
    graph.add_edge("gather_context", "generate_response")
    graph.add_edge("generate_response", "validate_response")
    graph.add_edge("validate_response", "persist_agent_run")
    graph.add_edge("persist_agent_run", END)
    # Checkpointer передаётся явно. Default не накапливает все сообщения в памяти
    # процесса; InMemorySaver включается в тесте/учебном исследовании history.
    return graph.compile(checkpointer=checkpointer)

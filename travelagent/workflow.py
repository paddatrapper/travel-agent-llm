from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langchain_core.messages import HumanMessage
from .agent import PlannerState, Agent

def run_travel_planner(workflow: CompiledStateGraph, user_request: str):
    print(f"Initial Request: {user_request}")
    state = {
        "messages": [HumanMessage(content=user_request)],
        "city": "",
        "interests": [],
        "itinerary": "",
    }

    for _ in workflow.stream(state):
        pass

def compile_workflow(agent: Agent) -> CompiledStateGraph:
    workflow = StateGraph(PlannerState)
    workflow.add_node("input_city", agent.input_city)
    workflow.add_node("input_interests", agent.input_interests)
    workflow.add_node("create_itinerary", agent.create_itinerary)
    workflow.set_entry_point("input_city")

    workflow.add_edge("input_city", "input_interests")
    workflow.add_edge("input_interests", "create_itinerary")
    workflow.add_edge("create_itinerary", END)

    return workflow.compile()



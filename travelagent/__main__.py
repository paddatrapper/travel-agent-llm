import os
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

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

if __name__ == "__main__":
    load_dotenv()
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite")
    itinerary_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful travel assistant. Create a day trip itinerary for {city} based on the user's interests: {interests}. Provide a brief, bulleted itinerary."),
        ("human", "Create an itinerary for my day trip."),
    ])
    
    agent = Agent(llm, itinerary_prompt)
    workflow = compile_workflow(agent)
    user_request = "I want to plan a day trip."
    run_travel_planner(workflow, user_request)

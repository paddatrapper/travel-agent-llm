import os
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

from .agent import PlannerState

def input_city(state: PlannerState) -> PlannerState:
    print("Please enter the city you want to visit for your day trip:")
    user_message = input("Your destination: ")
    return {
        **state,
        "city": user_message,
        "messages": state["messages"] + [HumanMessage(content=user_message)],
    }

def input_interests(state: PlannerState) -> PlannerState:
    print("Please enter your interests:")
    user_message = input("Your interests: ")
    return {
        **state,
        "interests": [interest.strip() for interest in user_message.split(",")],
        "messages": state["messages"] + [HumanMessage(content=user_message)],
    }

def create_itinerary(llm: ChatGoogleGenerativeAI, prompt: ChatPromptTemplate, state: PlannerState) -> PlannerState:
    print(f"Creating an itinerary for {state['city']} based on your interests: {', '.join(state['interests'])}...")
    response = llm.invoke(prompt.format_messsages(city=state['city'], interests=", ".join(state['interests'])))
    print(response.content)
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=response.content)],
        "itinerary": response.content,
    }

if __name__ == "__main__":
    load_dotenv()
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite")
    itinerary_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful travel assistant. Create a day trip itinerary for {city} based on the user's interests: {interests}. Provide a brief, bulleted itinerary."),
        ("human", "Create an itinerary for my day trip."),
    ])

    workflow = StateGraph(PlannerState)
    workflow.add_node("input_city", input_city)
    workflow.add_node("input_interests", input_interests)
    workflow.add_node("create_itinerary", create_itinerary)
    workflow.set_entry_point("input_city")

    workflow.add_edge("input_city", "input_interests")
    workflow.add_edge("input_interests", "create_itinerary")
    workflow.add_edge("create_itinerary", END)

    app = workflow.compile()

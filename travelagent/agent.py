from typing import TypedDict, Annotated, List
from langgraph.graph.state import CompiledStateGraph
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

class PlannerState(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], "The messages in the conversation"]
    city: str
    interests: List[str]
    itenerary: str

class Agent:
    def __init__(self, llm: ChatGoogleGenerativeAI, prompt: ChatPromptTemplate):
        self.llm = llm
        self.prompt = prompt

    def input_city(self, state: PlannerState) -> PlannerState:
        print("Please enter the city you want to visit for your day trip:")
        user_message = input("Your destination: ")
        return {
            **state,
            "city": user_message,
            "messages": state["messages"] + [HumanMessage(content=user_message)],
        }

    def input_interests(self, state: PlannerState) -> PlannerState:
        print("Please enter your interests:")
        user_message = input("Your interests: ")
        return {
            **state,
            "interests": [interest.strip() for interest in user_message.split(",")],
            "messages": state["messages"] + [HumanMessage(content=user_message)],
        }

    def create_itinerary(self, state: PlannerState) -> PlannerState:
        print(f"Creating an itinerary for {state['city']} based on your interests: {', '.join(state['interests'])}...")
        response = self.llm.invoke(self.prompt.format_messages(city=state['city'], interests=", ".join(state['interests'])))
        print(response.content)
        return {
            **state,
            "messages": state["messages"] + [AIMessage(content=response.content)],
            "itinerary": response.content,
        }

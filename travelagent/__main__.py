from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from .agent import Agent
from .workflow import compile_workflow, run_travel_planner

if __name__ == "__main__":
    load_dotenv()

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite")
    itinerary_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful travel assistant. Create a day trip itinerary for {city} based on the user's interests: {interests}. Provide a brief, bulleted itinerary.",
            ),
            ("human", "Create an itinerary for my day trip."),
        ]
    )

    agent = Agent(llm, itinerary_prompt)
    workflow = compile_workflow(agent)
    USER_REQUEST = "I want to plan a day trip."
    run_travel_planner(workflow, USER_REQUEST)

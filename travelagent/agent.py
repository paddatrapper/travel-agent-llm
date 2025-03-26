from typing import TypedDict, Annotated, List
from langchain_core.messages import HumanMessage, AIMessage

class PlannerState(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], "The messages in the conversation"]
    city: str
    interests: List[str]
    itenerary: str

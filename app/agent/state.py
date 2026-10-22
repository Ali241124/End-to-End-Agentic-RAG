
from typing import TypedDict


class AgentState(TypedDict):
    question: str
    search_query: str
    documents: list[dict]
    attempts: int
    relevant: bool
    answer: str

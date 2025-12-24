from typing import TypeDict , List , Literal
from langchain_core.messages import BaseMessage


class AgentState(TypeDict, total=False):
    messages: List[BaseMessage]
    route: Literal["rag", "web" , "answer" , "end"]
    rag: str
    web:str
    web_search_enabled: bool



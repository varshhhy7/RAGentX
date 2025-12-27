from typing import TypeDict , List , Literal
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
import os
from config import GROQ_API_KEY, TRAVILY_API_KEY , PINECONE_API_KEY



class RouteDecision(BaseModel):
    route: Literal["rag", "web" , "answer" , "end"] 
    reply: str | None = Field(None, description="Filled only when route == 'end' ")

class RagJudge(BaseModel):
    sufficinet: bool = Field(..., description="True if the retrieved documents are sufficient to answer the question otherwise False")

os.environ["GROQ_API_KEY"]=GROQ_API_KEY

router_llm=ChatGroq(model="llama3-70b-8192", temperature=0).with_structured_output(RouteDecision)
judege_llm=ChatGroq(model="llama3-70b-8192", temperature=0).with_structured_output(RagJudge)
answer_llm=ChatGroq(model="llama3-70b-8192", temperature=0.7)


class AgentState(TypeDict, total=False):
    messages: List[BaseMessage]
    route: Literal["rag", "web" , "answer" , "end"]
    rag: str
    web:str
    web_search_enabled: bool






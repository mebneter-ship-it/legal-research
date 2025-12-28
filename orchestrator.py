
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from agents import run_primary_agent, run_case_agent, run_analysis_agent

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.1)

def supervisor_executor(payload: dict):
    query = payload["input"]

    primary = run_primary_agent(query)
    case = run_case_agent(query)

    if "USER DOCUMENT" in query:
        analysis = run_analysis_agent(f"{query}\nPRIMARY:\n{primary}\nCASE:\n{case}")
        output = f"PRIMARY LAW:\n{primary}\n\nCASE LAW:\n{case}\n\nDOCUMENT ANALYSIS:\n{analysis}"
    else:
        output = f"PRIMARY LAW:\n{primary}\n\nCASE LAW:\n{case}"

    return {"output": output}

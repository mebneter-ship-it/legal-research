"""
Swiss Legal Research - Multi-Agent System using LangGraph

This module implements a supervisor-based multi-agent architecture:
- Supervisor: Routes queries to appropriate specialist agents
- Primary Law Agent: Searches and analyzes Swiss federal law
- Case Law Agent: Searches and analyzes BGE decisions  
- Analysis Agent: Synthesizes research into comprehensive analysis
"""

import os
from typing import TypedDict, Annotated, Literal, Sequence
from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from tools import search_swiss_primary_law, search_swiss_case_law, search_general_legal


def get_llm():
    """Get configured LLM based on environment settings"""
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0.1)
    else:
        from langchain_openai import ChatOpenAI
        # Fix OpenAI key if it has double prefix
        api_key = os.getenv("OPENAI_API_KEY", "")
        if api_key.startswith("sk-sk-"):
            api_key = api_key.replace("sk-sk-", "sk-", 1)
            os.environ["OPENAI_API_KEY"] = api_key
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.1)


# ============================================================
# STATE DEFINITION
# ============================================================

class ResearchState(TypedDict):
    """State passed between agents in the research pipeline"""
    # The original legal question
    question: str
    # Optional user document to analyze
    document: str
    # Results from primary law search
    primary_law_results: str
    # Results from case law search  
    case_law_results: str
    # Final synthesized analysis
    final_analysis: str
    # Current stage in the pipeline
    current_stage: str
    # Any errors encountered
    errors: list[str]


# ============================================================
# AGENT DEFINITIONS
# ============================================================

PRIMARY_LAW_PROMPT = """You are a Swiss primary law specialist. Your role is to:
1. Analyze search results from Swiss federal law sources (Fedlex, admin.ch)
2. Identify relevant legal provisions (Articles, Paragraphs, SR numbers)
3. Cite precisely: Art. X Abs. Y [Law Name] (SR [number])
4. NEVER invent or hallucinate legal provisions

Based on the search results below, provide a structured summary of relevant primary law.
If the search results are insufficient, clearly state what's missing.

SEARCH RESULTS:
{search_results}

USER QUESTION:
{question}

Respond in the same language as the question (German/French/Italian/English)."""


CASE_LAW_PROMPT = """You are a Swiss case law specialist. Your role is to:
1. Analyze search results from Swiss Federal Court (Bundesgericht) decisions
2. Identify relevant BGE decisions and their key holdings
3. Cite precisely: BGE [volume] [section] [page] (year)
4. NEVER invent or hallucinate case references

Based on the search results below, provide a structured summary of relevant case law.
If the search results are insufficient, clearly state what's missing.

SEARCH RESULTS:
{search_results}

USER QUESTION:
{question}

Respond in the same language as the question."""


ANALYSIS_PROMPT = """You are a Swiss legal analyst. Your role is to:
1. Synthesize primary law and case law research into a comprehensive analysis
2. Apply the law to any user document or specific situation
3. Identify legal risks, uncertainties, and open questions
4. Provide practical recommendations where appropriate

IMPORTANT GUIDELINES:
- Cite all sources precisely (Art., BGE references)
- Clearly distinguish between established law and interpretation
- Note any gaps in the research or areas needing further investigation
- If analyzing a document, identify specific clauses and their legal implications

PRIMARY LAW FINDINGS:
{primary_law}

CASE LAW FINDINGS:
{case_law}

USER QUESTION:
{question}

{document_section}

Provide a structured legal analysis in the same language as the question."""


def run_primary_law_agent(state: ResearchState) -> ResearchState:
    """Agent that searches and analyzes Swiss primary law"""
    llm = get_llm()
    
    try:
        # Perform search
        search_results = search_swiss_primary_law(state["question"])
        
        # Analyze results
        prompt = ChatPromptTemplate.from_template(PRIMARY_LAW_PROMPT)
        response = llm.invoke(prompt.format(
            search_results=search_results,
            question=state["question"]
        ))
        
        return {
            **state,
            "primary_law_results": response.content,
            "current_stage": "primary_law_complete"
        }
    except Exception as e:
        return {
            **state,
            "primary_law_results": f"Error in primary law search: {str(e)}",
            "errors": state.get("errors", []) + [str(e)]
        }


def run_case_law_agent(state: ResearchState) -> ResearchState:
    """Agent that searches and analyzes Swiss case law"""
    llm = get_llm()
    
    try:
        # Perform search
        search_results = search_swiss_case_law(state["question"])
        
        # Analyze results
        prompt = ChatPromptTemplate.from_template(CASE_LAW_PROMPT)
        response = llm.invoke(prompt.format(
            search_results=search_results,
            question=state["question"]
        ))
        
        return {
            **state,
            "case_law_results": response.content,
            "current_stage": "case_law_complete"
        }
    except Exception as e:
        return {
            **state,
            "case_law_results": f"Error in case law search: {str(e)}",
            "errors": state.get("errors", []) + [str(e)]
        }


def run_analysis_agent(state: ResearchState) -> ResearchState:
    """Agent that synthesizes research into final analysis"""
    llm = get_llm()
    
    try:
        # Prepare document section if present
        document_section = ""
        if state.get("document"):
            document_section = f"USER DOCUMENT TO ANALYZE:\n{state['document']}"
        
        # Generate analysis
        prompt = ChatPromptTemplate.from_template(ANALYSIS_PROMPT)
        response = llm.invoke(prompt.format(
            primary_law=state.get("primary_law_results", "No primary law research available"),
            case_law=state.get("case_law_results", "No case law research available"),
            question=state["question"],
            document_section=document_section
        ))
        
        return {
            **state,
            "final_analysis": response.content,
            "current_stage": "analysis_complete"
        }
    except Exception as e:
        return {
            **state,
            "final_analysis": f"Error in analysis: {str(e)}",
            "errors": state.get("errors", []) + [str(e)]
        }


# ============================================================
# GRAPH CONSTRUCTION
# ============================================================

def create_research_graph():
    """Create the LangGraph workflow for legal research"""
    
    # Initialize graph with state schema
    workflow = StateGraph(ResearchState)
    
    # Add nodes (agents)
    workflow.add_node("primary_law", run_primary_law_agent)
    workflow.add_node("case_law", run_case_law_agent)
    workflow.add_node("analysis", run_analysis_agent)
    
    # Define edges (workflow)
    # Start with primary law, then case law, then analysis
    workflow.set_entry_point("primary_law")
    workflow.add_edge("primary_law", "case_law")
    workflow.add_edge("case_law", "analysis")
    workflow.add_edge("analysis", END)
    
    # Compile the graph
    return workflow.compile()


# Global graph instance
research_graph = None


def get_research_graph():
    """Get or create the research graph"""
    global research_graph
    if research_graph is None:
        research_graph = create_research_graph()
    return research_graph


def run_legal_research(question: str, document: str = "") -> dict:
    """
    Execute the full legal research pipeline
    
    Args:
        question: The legal question to research
        document: Optional document to analyze
        
    Returns:
        Dictionary with research results and final analysis
    """
    graph = get_research_graph()
    
    # Initialize state
    initial_state: ResearchState = {
        "question": question,
        "document": document,
        "primary_law_results": "",
        "case_law_results": "",
        "final_analysis": "",
        "current_stage": "started",
        "errors": []
    }
    
    # Run the graph
    final_state = graph.invoke(initial_state)
    
    return {
        "question": question,
        "has_document": bool(document),
        "primary_law": final_state.get("primary_law_results", ""),
        "case_law": final_state.get("case_law_results", ""),
        "analysis": final_state.get("final_analysis", ""),
        "errors": final_state.get("errors", [])
    }


if __name__ == "__main__":
    # Test the graph
    print("Testing LangGraph Research Pipeline...")
    result = run_legal_research("Was sind die Kündigungsfristen im Schweizer Arbeitsrecht?")
    print("\n" + "=" * 60)
    print("FINAL ANALYSIS:")
    print(result["analysis"])

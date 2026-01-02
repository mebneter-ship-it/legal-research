"""
Swiss Legal Research - Multi-Agent System using LangGraph

This module implements a smart orchestrator-based multi-agent architecture:
- Orchestrator: Analyzes query, enriches context, dispatches to agents in parallel
- Primary Law Agent: Searches Swiss federal law (Fedlex)
- Cantonal Law Agent: Searches cantonal law (if canton detected)
- Case Law Agent: Searches BGE decisions
- Analysis Agent: Synthesizes all research into comprehensive analysis
"""

import os
import concurrent.futures
from typing import TypedDict, Annotated, Literal, Sequence
from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from tools import (
    search_swiss_primary_law, 
    search_swiss_case_law, 
    search_cantonal_law,
    search_cantonal_case_law,
    search_communal_law,
    detect_canton,
    detect_commune,
    CANTON_NAMES
)


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
    # Original question
    question: str
    # Optional user document
    document: str
    
    # Orchestrator context (enriched by analyze step)
    detected_canton: str          # e.g., "AI", "ZH"
    detected_canton_name: str     # e.g., "Appenzell Innerrhoden"
    detected_commune: str         # e.g., "Appenzell"
    legal_domain: str             # e.g., "Baurecht", "Mietrecht"
    enriched_queries: dict        # Queries tailored for each agent
    
    # Agent results
    primary_law_results: str
    cantonal_law_results: str
    case_law_results: str
    
    # Final output
    final_analysis: str
    current_stage: str
    errors: list[str]


# ============================================================
# ORCHESTRATOR: ANALYZE & ENRICH
# ============================================================

def run_orchestrator_analyze(state: ResearchState) -> ResearchState:
    """
    Orchestrator Step 1: Analyze query and enrich context for all agents.
    
    This is the BRAIN of the system:
    - Detects canton/commune from query
    - Identifies legal domain
    - Creates tailored queries for each agent
    """
    question = state["question"]
    
    # 1. Detect geographic context
    canton = detect_canton(question)
    canton_name = ""
    if canton:
        canton_info = CANTON_NAMES.get(canton, {})
        canton_name = canton_info.get("de", canton)
    
    commune = detect_commune(question)
    
    # 2. Detect legal domain (simple keyword matching)
    question_lower = question.lower()
    legal_domain = "Allgemein"
    
    domain_keywords = {
        "Baurecht": ["zaun", "einfriedung", "bauen", "grenzabstand", "baugesetz", "höhe", "mauer", "hecke"],
        "Mietrecht": ["miete", "kündigung", "mietvertrag", "vermieter", "mieter", "nebenkosten", "kaution"],
        "Arbeitsrecht": ["arbeit", "lohn", "ferien", "kündigung", "arbeitsvertrag", "überstunden"],
        "Nachbarrecht": ["nachbar", "immissionen", "lärm", "grenze", "schatten"],
        "Familienrecht": ["scheidung", "unterhalt", "sorgerecht", "ehe", "kind"],
        "Vertragsrecht": ["vertrag", "schadenersatz", "haftung", "schuld"],
    }
    
    for domain, keywords in domain_keywords.items():
        if any(kw in question_lower for kw in keywords):
            legal_domain = domain
            break
    
    # 3. Create enriched queries for each agent
    enriched_queries = {}
    
    # Primary Law Query - focus on federal law
    if canton:
        # If cantonal question, primary law might still be relevant (e.g., ZGB Nachbarrecht)
        enriched_queries["primary_law"] = f"{question} Bundesrecht ZGB OR"
    else:
        enriched_queries["primary_law"] = question
    
    # Cantonal Law Query - only if canton detected
    if canton:
        # Add specific legal terms based on domain
        domain_terms = {
            "Baurecht": "Baugesetz Bauverordnung Einfriedung Grenzabstand",
            "Mietrecht": "Mietrecht VMWG",
            "Nachbarrecht": "Nachbarrecht Einfriedung",
        }
        extra_terms = domain_terms.get(legal_domain, "")
        enriched_queries["cantonal_law"] = f"{question} {extra_terms}"
        
        # Generate specific search queries for cantonal search
        enriched_queries["cantonal_search_queries"] = [
            f'"{canton_name}" {legal_domain} Gesetz',
            f'site:lexfind.ch "{canton_name}" {question}',
            f'Merkblatt "{canton_name}" {extra_terms}',
        ]
    
    # Case Law Query - add BGE and relevant terms
    case_law_terms = {
        "Baurecht": "Grenzabstand Baute BGE",
        "Nachbarrecht": "Immissionen Einfriedung BGE",
        "Mietrecht": "Kündigung Mietzins BGE",
    }
    extra_case_terms = case_law_terms.get(legal_domain, "BGE")
    enriched_queries["case_law"] = f"{question} {extra_case_terms}"
    
    # Log what we detected
    print(f"\n🎯 ORCHESTRATOR ANALYSIS:")
    print(f"   Question: {question[:80]}...")
    print(f"   Canton: {canton} ({canton_name})" if canton else "   Canton: None (Bundesrecht)")
    print(f"   Commune: {commune}" if commune else "   Commune: None")
    print(f"   Legal Domain: {legal_domain}")
    print(f"   Enriched Queries: {list(enriched_queries.keys())}")
    
    return {
        **state,
        "detected_canton": canton or "",
        "detected_canton_name": canton_name,
        "detected_commune": commune or "",
        "legal_domain": legal_domain,
        "enriched_queries": enriched_queries,
        "current_stage": "analyzed"
    }


# ============================================================
# PARALLEL SEARCH AGENTS
# ============================================================

def run_parallel_search(state: ResearchState) -> ResearchState:
    """
    Execute all relevant searches in PARALLEL.
    
    This is much faster than sequential execution!
    """
    enriched = state.get("enriched_queries", {})
    canton = state.get("detected_canton", "")
    canton_name = state.get("detected_canton_name", "")
    commune = state.get("detected_commune", "")
    
    results = {
        "primary_law": "",
        "cantonal_law": "",
        "case_law": ""
    }
    errors = state.get("errors", [])
    
    def search_primary():
        """Search federal law"""
        try:
            query = enriched.get("primary_law", state["question"])
            return search_swiss_primary_law(query, max_results=5)
        except Exception as e:
            return f"Error: {str(e)}"
    
    def search_cantonal():
        """Search cantonal law (only if canton detected)"""
        if not canton:
            return "Kein Kanton erkannt - kantonale Suche übersprungen."
        try:
            query = enriched.get("cantonal_law", state["question"])
            orchestrator_queries = enriched.get("cantonal_search_queries", [])
            return search_cantonal_law(
                query=query,
                canton=canton,
                canton_name=canton_name,
                orchestrator_queries=orchestrator_queries,
                max_results=5,
                commune=commune
            )
        except Exception as e:
            return f"Error: {str(e)}"
    
    def search_case_law():
        """Search BGE decisions"""
        try:
            query = enriched.get("case_law", state["question"])
            return search_swiss_case_law(query, max_results=5)
        except Exception as e:
            return f"Error: {str(e)}"
    
    # Execute ALL searches in parallel!
    print(f"\n🔄 PARALLEL SEARCH: Starting 3 agents...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_primary = executor.submit(search_primary)
        future_cantonal = executor.submit(search_cantonal)
        future_case = executor.submit(search_case_law)
        
        # Collect results
        results["primary_law"] = future_primary.result()
        results["cantonal_law"] = future_cantonal.result()
        results["case_law"] = future_case.result()
    
    print(f"✅ PARALLEL SEARCH: All agents complete!")
    print(f"   Primary Law: {len(results['primary_law'])} chars")
    print(f"   Cantonal Law: {len(results['cantonal_law'])} chars")
    print(f"   Case Law: {len(results['case_law'])} chars")
    
    return {
        **state,
        "primary_law_results": results["primary_law"],
        "cantonal_law_results": results["cantonal_law"],
        "case_law_results": results["case_law"],
        "current_stage": "search_complete",
        "errors": errors
    }


# ============================================================
# ANALYSIS AGENT
# ============================================================

ANALYSIS_PROMPT = """Du bist ein Schweizer Rechtsexperte. Deine Aufgabe ist es, die Rechercheergebnisse zu einer präzisen rechtlichen Analyse zusammenzufassen.

WICHTIGE REGELN:
1. Zitiere ALLE Quellen präzise (Art. X Abs. Y Gesetz, BGE X II Y)
2. Unterscheide klar zwischen Bundesrecht und kantonalem Recht
3. Wenn ein Kanton erkannt wurde, priorisiere kantonales Recht für lokale Fragen
4. Gib konkrete Antworten mit Zahlen/Massen wenn vorhanden
5. Weise auf Unsicherheiten oder fehlende Informationen hin

KONTEXT:
- Erkannter Kanton: {canton} {canton_name}
- Erkannte Gemeinde: {commune}
- Rechtsgebiet: {legal_domain}

BUNDESRECHT (Fedlex):
{primary_law}

KANTONALES RECHT:
{cantonal_law}

RECHTSPRECHUNG (BGE):
{case_law}

FRAGE:
{question}

{document_section}

Antworte strukturiert in der Sprache der Frage:
1. KURZE ANTWORT (1-2 Sätze mit konkreter Antwort)
2. RECHTSGRUNDLAGE (relevante Artikel mit Zitaten)
3. DETAILS (weitere relevante Informationen)
4. QUELLEN (alle verwendeten Quellen auflisten)"""


def run_analysis_agent(state: ResearchState) -> ResearchState:
    """Agent that synthesizes all research into final analysis"""
    llm = get_llm()
    
    try:
        # Prepare document section if present
        document_section = ""
        if state.get("document"):
            document_section = f"DOKUMENT ZUR ANALYSE:\n{state['document']}"
        
        # Generate analysis
        prompt = ChatPromptTemplate.from_template(ANALYSIS_PROMPT)
        response = llm.invoke(prompt.format(
            canton=state.get("detected_canton", "Keiner"),
            canton_name=f"({state.get('detected_canton_name', '')})" if state.get("detected_canton_name") else "",
            commune=state.get("detected_commune", "Keine"),
            legal_domain=state.get("legal_domain", "Allgemein"),
            primary_law=state.get("primary_law_results", "Keine Ergebnisse"),
            cantonal_law=state.get("cantonal_law_results", "Keine Ergebnisse"),
            case_law=state.get("case_law_results", "Keine Ergebnisse"),
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
    """
    Create the LangGraph workflow for legal research.
    
    Flow:
    1. ORCHESTRATOR_ANALYZE: Detect context, enrich queries
    2. PARALLEL_SEARCH: Run all search agents simultaneously
    3. ANALYSIS: Synthesize results into final answer
    """
    
    workflow = StateGraph(ResearchState)
    
    # Add nodes
    workflow.add_node("orchestrator_analyze", run_orchestrator_analyze)
    workflow.add_node("parallel_search", run_parallel_search)
    workflow.add_node("analysis", run_analysis_agent)
    
    # Define flow: analyze → search (parallel) → analyze
    workflow.set_entry_point("orchestrator_analyze")
    workflow.add_edge("orchestrator_analyze", "parallel_search")
    workflow.add_edge("parallel_search", "analysis")
    workflow.add_edge("analysis", END)
    
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
        "detected_canton": "",
        "detected_canton_name": "",
        "detected_commune": "",
        "legal_domain": "",
        "enriched_queries": {},
        "primary_law_results": "",
        "cantonal_law_results": "",
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
        "detected_canton": final_state.get("detected_canton", ""),
        "detected_canton_name": final_state.get("detected_canton_name", ""),
        "legal_domain": final_state.get("legal_domain", ""),
        "primary_law": final_state.get("primary_law_results", ""),
        "cantonal_law": final_state.get("cantonal_law_results", ""),
        "case_law": final_state.get("case_law_results", ""),
        "analysis": final_state.get("final_analysis", ""),
        "errors": final_state.get("errors", [])
    }


if __name__ == "__main__":
    # Test the graph
    print("=" * 70)
    print("Testing Smart Orchestrator Pipeline...")
    print("=" * 70)
    
    # Test 1: Cantonal question
    result = run_legal_research("Wie hoch darf ein Zaun in Appenzell sein?")
    print("\n" + "=" * 70)
    print("FINAL ANALYSIS:")
    print("=" * 70)
    print(result["analysis"])

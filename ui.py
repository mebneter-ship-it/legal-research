"""
Swiss Legal Research Assistant - Developer UI v2

Enhanced interface with:
- Document upload (PDF, DOCX, TXT)
- Per-agent activity panels
- Orchestrator visibility
- Real-time state tracking

Run with: streamlit run ui.py
"""

import os
import sys
import time
import json
import tempfile
from datetime import datetime
from typing import Generator, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
import re

# Load environment variables
load_dotenv("keys.env")

def make_links_open_in_new_tab(text: str) -> str:
    """Convert markdown links to HTML links that open in new tab"""
    # Pattern for markdown links: [text](url)
    markdown_link_pattern = r'\[([^\]]+)\]\((https?://[^\)]+)\)'
    
    def replace_link(match):
        link_text = match.group(1)
        url = match.group(2)
        return f'<a href="{url}" target="_blank" rel="noopener noreferrer">{link_text}</a>'
    
    # Replace markdown links with HTML
    result = re.sub(markdown_link_pattern, replace_link, text)
    
    # Also handle bare URLs that aren't already in markdown link format
    # Match URLs not preceded by ]( or href="
    bare_url_pattern = r'(?<!\]\()(?<!href=")(https?://[^\s\)<>\]]+)'
    
    def replace_bare_url(match):
        url = match.group(1)
        # Don't replace if it's already part of an HTML tag
        return f'<a href="{url}" target="_blank" rel="noopener noreferrer">{url}</a>'
    
    result = re.sub(bare_url_pattern, replace_bare_url, result)
    
    return result

# Fix API key prefixes
def fix_api_keys():
    tavily_key = os.getenv("TAVILY_API_KEY", "")
    if tavily_key.startswith("tvly-tvly-"):
        os.environ["TAVILY_API_KEY"] = tavily_key.replace("tvly-tvly-", "tvly-", 1)
    
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if openai_key.startswith("sk-sk-"):
        os.environ["OPENAI_API_KEY"] = openai_key.replace("sk-sk-", "sk-", 1)

fix_api_keys()

from tools import (
    search_swiss_primary_law, 
    search_swiss_case_law, 
    search_general_legal,
    search_cantonal_law,
    search_cantonal_case_law,
    search_communal_law
)
from smart_search import SmartLegalSearch, search_cantonal, search_federal, search_case_law
from prompts import (
    ANALYSIS_SYSTEM_PROMPT,
    ANALYSIS_USER_PROMPT,
    get_analysis_prompt  # Only need this one now - agents pass raw results to Claude
)


# ============================================================
# DOCUMENT PROCESSING
# ============================================================

def extract_text_with_llm(image_bytes: bytes, file_type: str = "image") -> str:
    """Use LLM to extract text from an image or scanned PDF"""
    try:
        import base64
        
        # Encode image to base64
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        # Determine media type
        media_type = "image/png"
        if file_type.lower() in ["jpg", "jpeg"]:
            media_type = "image/jpeg"
        elif file_type.lower() == "pdf":
            media_type = "application/pdf"
        
        llm, model_name = get_llm()
        
        from langchain_core.messages import HumanMessage
        
        # Create message with image
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Extract ALL text from this document image. Return ONLY the extracted text, preserving the structure and formatting as much as possible. Do not add any commentary or explanation."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{base64_image}"
                    }
                }
            ]
        )
        
        response = llm.invoke([message])
        return response.content
        
    except Exception as e:
        return f"[LLM text extraction failed: {str(e)}]"


def convert_pdf_to_images(pdf_bytes: bytes) -> list:
    """Convert PDF pages to images for LLM processing"""
    try:
        from pdf2image import convert_from_bytes
        images = convert_from_bytes(pdf_bytes, dpi=150)
        return images
    except ImportError:
        return None
    except Exception as e:
        return None


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF bytes using multiple methods"""
    text = ""
    
    # Method 1: Try pypdf first (fast, works for simple PDFs)
    try:
        import pypdf
        import io
        
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        text = "\n\n".join(text_parts)
        
        if text.strip() and len(text.strip()) > 50:  # Meaningful text found
            return text
    except ImportError:
        pass
    except Exception as e:
        pass
    
    # Method 2: Try pdfplumber (better for complex layouts)
    try:
        import pdfplumber
        import io
        
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            text_parts = []
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            text = "\n\n".join(text_parts)
            
            if text.strip() and len(text.strip()) > 50:
                return text
    except ImportError:
        pass
    except Exception as e:
        pass
    
    # Method 3: Use LLM for scanned PDFs (OCR via vision)
    # Convert PDF to image and send to LLM
    try:
        images = convert_pdf_to_images(file_bytes)
        if images:
            import io
            text_parts = []
            for i, img in enumerate(images[:5]):  # Max 5 pages
                img_buffer = io.BytesIO()
                img.save(img_buffer, format='PNG')
                img_bytes = img_buffer.getvalue()
                page_text = extract_text_with_llm(img_bytes, "png")
                if page_text and not page_text.startswith("["):
                    text_parts.append(f"--- Page {i+1} ---\n{page_text}")
            if text_parts:
                return "\n\n".join(text_parts)
    except Exception as e:
        pass
    
    # If all methods fail
    return "[PDF_NEEDS_LLM_EXTRACTION]"


def extract_text_from_image(file_bytes: bytes, file_extension: str) -> str:
    """Extract text from image using LLM vision"""
    return extract_text_with_llm(file_bytes, file_extension)


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract text from DOCX bytes"""
    try:
        import docx
        import io
        
        doc = docx.Document(io.BytesIO(file_bytes))
        text_parts = []
        for para in doc.paragraphs:
            text_parts.append(para.text)
        return "\n".join(text_parts)
    except ImportError:
        return "[DOCX extraction requires python-docx: pip install python-docx]"
    except Exception as e:
        return f"[DOCX extraction error: {str(e)}]"


def extract_text_from_file(uploaded_file) -> tuple[str, str]:
    """
    Extract text from uploaded file.
    Returns (text, file_type)
    """
    if uploaded_file is None:
        return "", ""
    
    file_bytes = uploaded_file.read()
    file_name = uploaded_file.name.lower()
    
    if file_name.endswith('.pdf'):
        return extract_text_from_pdf(file_bytes), "PDF"
    elif file_name.endswith('.docx'):
        return extract_text_from_docx(file_bytes), "DOCX"
    elif file_name.endswith('.doc'):
        return "[.doc format not supported - please convert to .docx]", "DOC"
    elif file_name.endswith('.txt') or file_name.endswith('.md'):
        return file_bytes.decode('utf-8', errors='replace'), "TXT"
    elif file_name.endswith(('.jpg', '.jpeg', '.png', '.webp', '.gif')):
        # Use LLM vision to extract text from image
        ext = file_name.split('.')[-1]
        return extract_text_from_image(file_bytes, ext), "IMAGE"
    else:
        # Try to decode as text
        try:
            return file_bytes.decode('utf-8'), "TXT"
        except:
            return "[Unsupported file format]", "UNKNOWN"


# ============================================================
# AGENT STATE & LOGGING
# ============================================================

class AgentStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class AgentState:
    """State for a single agent"""
    name: str
    status: AgentStatus = AgentStatus.IDLE
    current_action: str = ""
    logs: list = field(default_factory=list)
    search_query: str = ""
    search_results: str = ""
    planned_queries: list = field(default_factory=list)  # NEW: What the agent decided to search
    planning_duration_ms: float = 0  # NEW: How long planning took
    # Separate system and user prompts for visibility
    system_prompt: str = ""
    user_prompt: str = ""
    llm_prompt: str = ""  # Combined for backward compatibility
    llm_response: str = ""
    # Data flow tracking
    data_received: dict = field(default_factory=dict)  # What this agent received
    data_sent: dict = field(default_factory=dict)  # What this agent sends out
    # Citations extracted
    citations: list = field(default_factory=list)
    duration_ms: float = 0
    error: str = ""
    
    def add_log(self, message: str, event_type: str = "info", data: dict = None):
        self.logs.append({
            "timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3],
            "message": message,
            "event_type": event_type,
            "data": data or {}
        })


@dataclass
class OrchestratorState:
    """State for the orchestrator"""
    status: AgentStatus = AgentStatus.IDLE
    current_step: str = ""
    pipeline: list = field(default_factory=list)
    logs: list = field(default_factory=list)
    total_duration_ms: float = 0
    
    # Track what orchestrator passes to each agent
    agent_inputs: dict = field(default_factory=dict)
    
    def add_log(self, message: str, event_type: str = "info"):
        self.logs.append({
            "timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3],
            "message": message,
            "event_type": event_type
        })
    
    def set_agent_input(self, agent_name: str, inputs: dict):
        """Track what is passed to each agent"""
        self.agent_inputs[agent_name] = inputs


@dataclass
class ResearchSession:
    """Complete session state"""
    question: str = ""
    document_text: str = ""
    document_type: str = ""
    document_name: str = ""
    
    # Canton/Commune detection (by LLM orchestrator)
    canton: str = ""
    canton_name: dict = field(default_factory=dict)
    commune: str = ""
    has_cantonal_scope: bool = False
    response_language: str = "German"  # Determined by LLM orchestrator
    
    # Legal context from orchestrator
    legal_domain: str = ""  # Specific domain: Mietrecht, Arbeitsrecht, etc.
    related_domains: list = field(default_factory=list)  # Related domains to also search
    legal_context: str = ""  # Brief explanation of the issue
    relevant_articles: list = field(default_factory=list)  # Articles that ARE relevant
    irrelevant_articles: list = field(default_factory=list)  # Articles to AVOID (wrong domain)
    
    # Enhanced context for agents (NEW!)
    key_terms: list = field(default_factory=list)  # Key search terms
    synonyms: dict = field(default_factory=dict)  # Alternative terms
    search_hints: dict = field(default_factory=dict)  # Hints per agent type
    
    # Document analysis from orchestrator (new!)
    document_analysis: dict = field(default_factory=dict)  # Structured analysis of uploaded document
    
    search_topics: list = field(default_factory=list)  # Legacy: Topics to search for
    search_queries: dict = field(default_factory=dict)  # New: Specific queries per agent
    
    # Benchmark
    benchmark_output: str = ""  # Direct ChatGPT response for comparison
    
    orchestrator: OrchestratorState = field(default_factory=OrchestratorState)
    primary_law_agent: AgentState = field(default_factory=lambda: AgentState(name="Primary Law Agent"))
    case_law_agent: AgentState = field(default_factory=lambda: AgentState(name="Case Law Agent"))
    cantonal_law_agent: AgentState = field(default_factory=lambda: AgentState(name="Cantonal Law Agent"))
    cantonal_case_law_agent: AgentState = field(default_factory=lambda: AgentState(name="Cantonal Case Law Agent"))
    analysis_agent: AgentState = field(default_factory=lambda: AgentState(name="Analysis Agent"))
    
    final_output: str = ""
    errors: list = field(default_factory=list)


# ============================================================
# LLM CONFIGURATION
# ============================================================

def get_llm(role: str = "agent"):
    """
    Get configured LLM based on role.
    
    Roles:
    - "orchestrator": Uses GPT-4o for critical legal domain detection
    - "analysis": Uses Claude Sonnet (if available) OR GPT-4o for synthesis
    - "agent": Uses GPT-4o-mini for search agents
    
    Hybrid Mode (default when both API keys present):
    - Orchestrator: GPT-4o (fast, good at structured output)
    - Search Agents: GPT-4o-mini (cheap, has context)
    - Analysis: Claude Sonnet (best reasoning for legal synthesis)
    """
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    use_claude_for_analysis = os.getenv("USE_CLAUDE_FOR_ANALYSIS", "true").lower() == "true"
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    
    # Option A: Claude for Analysis (if key available and enabled)
    if role == "analysis" and use_claude_for_analysis and anthropic_key:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0.1), "claude-sonnet-4"
    
    # Full Anthropic mode
    if provider == "anthropic" and anthropic_key:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0.1), "claude-sonnet-4"
    
    # Default: OpenAI
    from langchain_openai import ChatOpenAI
    if role in ["orchestrator", "analysis"]:
        # Stronger model for critical decisions
        return ChatOpenAI(model="gpt-4o", temperature=0.1), "gpt-4o"
    else:
        # Faster/cheaper model for search agents
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.1), "gpt-4o-mini"


def get_llm_simple():
    """Get default LLM (for backward compatibility)"""
    return get_llm("agent")


def run_benchmark_direct(question: str, language: str = "German") -> str:
    """
    Run direct LLM query WITHOUT agents for comparison.
    Uses the SAME strong model (gpt-4o) as Analysis Agent for fair comparison.
    """
    llm, model_name = get_llm("analysis")  # Use same model as Analysis Agent for fair comparison
    
    # Language-specific headers for fair comparison
    headers = {
        "German": "## Kurze Antwort\n## Rechtliche Grundlagen\n## Relevante Rechtsprechung\n## Erläuterung\n## Empfehlung\n## Quellen",
        "French": "## Réponse courte\n## Base juridique\n## Jurisprudence pertinente\n## Explications\n## Recommandation\n## Sources",
        "Italian": "## Risposta breve\n## Base giuridica\n## Giurisprudenza rilevante\n## Spiegazioni\n## Raccomandazione\n## Fonti",
        "English": "## Short Answer\n## Legal Basis\n## Relevant Case Law\n## Explanation\n## Recommendation\n## Sources"
    }
    
    structure = headers.get(language, headers["German"])
    
    benchmark_prompt = f"""{question}

Beantworte diese Schweizer Rechtsfrage komplett auf {language}.
Verwende Gesetzesabkürzungen in der korrekten Sprache (OR/CO, ZGB/CC, BGE/ATF/DTF).

Struktur:
{structure}"""

    try:
        from langchain_core.messages import HumanMessage
        response = llm.invoke([HumanMessage(content=benchmark_prompt)])
        return response.content
    except Exception as e:
        return f"Benchmark error: {str(e)}"


# ============================================================
# PARALLEL AGENT EXECUTION HELPERS
# ============================================================

def run_primary_law_agent_work(session: 'ResearchSession') -> dict:
    """
    Run Primary Law Agent work - AGENTIC VERSION
    
    The agent DECIDES ITSELF what to search based on:
    1. The question
    2. Context from orchestrator (legal domain, relevant articles)
    
    Flow:
    1. PLAN: LLM decides what searches to do
    2. EXECUTE: Run the planned searches
    3. ANALYZE: LLM analyzes results
    """
    from prompts import PRIMARY_LAW_PLANNING_PROMPT, parse_search_queries
    
    start_time = time.time()
    result = {
        "agent": "primary_law",
        "search_results": "",
        "llm_response": "",
        "system_prompt": "",
        "user_prompt": "",
        "search_query": "",
        "planned_queries": [],  # NEW: What the agent decided to search
        "error": None,
        "search_duration_ms": 0,
        "llm_duration_ms": 0,
        "planning_duration_ms": 0,
        "total_duration_ms": 0
    }
    
    try:
        agent_llm, _ = get_llm("agent")
        
        # ========== STEP 1: PLANNING ==========
        planning_start = time.time()
        
        # Format related_domains for display
        related_domains_str = ", ".join(session.related_domains) if session.related_domains else "Keine"
        search_hint = (session.search_hints or {}).get("primary_law", "")
        
        planning_prompt = PRIMARY_LAW_PLANNING_PROMPT.format(
            question=session.question,
            legal_domain=session.legal_domain or "Nicht spezifiziert",
            related_domains=related_domains_str,
            relevant_articles=", ".join(session.relevant_articles) if session.relevant_articles else "Keine",
            key_terms=", ".join(session.key_terms) if session.key_terms else "Keine",
            search_hint=search_hint
        )
        
        from langchain_core.messages import SystemMessage, HumanMessage
        planning_response = agent_llm.invoke([
            HumanMessage(content=planning_prompt)
        ])
        
        # Parse the search queries from LLM response
        planned_queries = parse_search_queries(planning_response.content)
        result["planned_queries"] = planned_queries
        result["planning_duration_ms"] = (time.time() - planning_start) * 1000
        
        # Fallback if parsing failed
        if not planned_queries:
            planned_queries = [session.question]
        
        result["search_query"] = " | ".join(planned_queries)
        
        # ========== STEP 2: EXECUTE SEARCHES ==========
        search_start = time.time()
        all_results = []
        
        for query in planned_queries:
            try:
                search_result = search_swiss_primary_law(query, max_results=5)
                if search_result and len(search_result) > 100:
                    all_results.append(f"### Query: {query}\n{search_result}")
            except Exception as e:
                all_results.append(f"### Query: {query}\nError: {str(e)}")
        
        result["search_results"] = "\n\n---\n\n".join(all_results) if all_results else "Keine Ergebnisse gefunden"
        result["search_duration_ms"] = (time.time() - search_start) * 1000
        
        # Store planning prompt for UI visibility
        result["user_prompt"] = f"PLANNING PROMPT:\n{planning_prompt}\n\nAGENT RESPONSE:\n{planning_response.content}"
        
        # NO STEP 3! Raw results go directly to Analysis Agent (Claude)
        result["llm_response"] = result["search_results"]  # Pass raw results
        result["llm_duration_ms"] = 0
        
    except Exception as e:
        result["error"] = str(e)
    
    result["total_duration_ms"] = (time.time() - start_time) * 1000
    return result


def run_case_law_agent_work(session: 'ResearchSession') -> dict:
    """
    Run Case Law Agent work - AGENTIC VERSION
    
    The agent DECIDES ITSELF what BGE/case law to search based on:
    1. The question
    2. Context from orchestrator (legal domain, relevant articles)
    
    Flow:
    1. PLAN: LLM decides what case law searches to do
    2. EXECUTE: Run the planned searches
    3. ANALYZE: LLM analyzes results
    """
    from prompts import CASE_LAW_PLANNING_PROMPT, parse_search_queries
    
    start_time = time.time()
    result = {
        "agent": "case_law",
        "search_results": "",
        "llm_response": "",
        "system_prompt": "",
        "user_prompt": "",
        "search_query": "",
        "planned_queries": [],
        "error": None,
        "search_duration_ms": 0,
        "llm_duration_ms": 0,
        "planning_duration_ms": 0,
        "total_duration_ms": 0
    }
    
    try:
        agent_llm, _ = get_llm("agent")
        
        # ========== STEP 1: PLANNING ==========
        planning_start = time.time()
        
        # Format related_domains for display
        related_domains_str = ", ".join(session.related_domains) if session.related_domains else "Keine"
        search_hint = (session.search_hints or {}).get("case_law", "")
        
        planning_prompt = CASE_LAW_PLANNING_PROMPT.format(
            question=session.question,
            legal_domain=session.legal_domain or "Nicht spezifiziert",
            related_domains=related_domains_str,
            relevant_articles=", ".join(session.relevant_articles) if session.relevant_articles else "Keine",
            key_terms=", ".join(session.key_terms) if session.key_terms else "Keine",
            search_hint=search_hint
        )
        
        from langchain_core.messages import SystemMessage, HumanMessage
        planning_response = agent_llm.invoke([
            HumanMessage(content=planning_prompt)
        ])
        
        planned_queries = parse_search_queries(planning_response.content)
        result["planned_queries"] = planned_queries
        result["planning_duration_ms"] = (time.time() - planning_start) * 1000
        
        if not planned_queries:
            planned_queries = [f"BGE {session.question}"]
        
        result["search_query"] = " | ".join(planned_queries)
        
        # ========== STEP 2: EXECUTE SEARCHES ==========
        search_start = time.time()
        all_results = []
        
        for query in planned_queries:
            try:
                search_result = search_swiss_case_law(query, max_results=5)
                if search_result and len(search_result) > 100:
                    all_results.append(f"### Query: {query}\n{search_result}")
            except Exception as e:
                all_results.append(f"### Query: {query}\nError: {str(e)}")
        
        result["search_results"] = "\n\n---\n\n".join(all_results) if all_results else "Keine Ergebnisse gefunden"
        result["search_duration_ms"] = (time.time() - search_start) * 1000
        
        # Store planning prompt for UI visibility
        result["user_prompt"] = f"PLANNING PROMPT:\n{planning_prompt}\n\nAGENT RESPONSE:\n{planning_response.content}"
        
        # NO STEP 3! Raw results go directly to Analysis Agent (Claude)
        result["llm_response"] = result["search_results"]
        result["llm_duration_ms"] = 0
        
    except Exception as e:
        result["error"] = str(e)
    
    result["total_duration_ms"] = (time.time() - start_time) * 1000
    return result


def run_cantonal_law_agent_work(session: 'ResearchSession') -> dict:
    """
    Run Cantonal Law Agent work - AGENTIC VERSION
    
    The agent DECIDES ITSELF what cantonal laws to search based on:
    1. The question
    2. The canton/commune
    3. Context from orchestrator (legal domain)
    
    Flow:
    1. PLAN: LLM decides what cantonal law searches to do
    2. EXECUTE: Run the planned searches
    3. ANALYZE: LLM analyzes results
    """
    from prompts import CANTONAL_LAW_PLANNING_PROMPT, parse_search_queries
    
    start_time = time.time()
    result = {
        "agent": "cantonal_law",
        "search_results": "",
        "llm_response": "",
        "system_prompt": "",
        "user_prompt": "",
        "search_query": "",
        "planned_queries": [],
        "error": None,
        "search_duration_ms": 0,
        "llm_duration_ms": 0,
        "planning_duration_ms": 0,
        "total_duration_ms": 0
    }
    
    if not session.canton:
        result["error"] = "No canton specified"
        return result
    
    try:
        agent_llm, _ = get_llm("agent")
        canton_display = session.canton_name.get("de", session.canton) if session.canton_name else session.canton
        
        # Canton domain mapping
        canton_domains = {
            "AI": "ai.clex.ch", "AR": "ar.clex.ch", "ZH": "zh.clex.ch",
            "BE": "be.clex.ch", "BS": "bs.clex.ch", "BL": "bl.clex.ch",
            "LU": "lu.clex.ch", "SG": "sg.clex.ch", "AG": "ag.clex.ch",
            "TG": "tg.clex.ch", "GR": "gr.clex.ch", "VS": "vs.clex.ch",
            "TI": "ti.clex.ch", "VD": "vd.clex.ch", "GE": "ge.clex.ch",
            "NE": "ne.clex.ch", "JU": "ju.clex.ch", "FR": "fr.clex.ch",
            "SO": "so.clex.ch", "SH": "sh.clex.ch", "ZG": "zg.clex.ch",
            "SZ": "sz.clex.ch", "OW": "ow.clex.ch", "NW": "nw.clex.ch",
            "GL": "gl.clex.ch", "UR": "ur.clex.ch"
        }
        canton_domain = canton_domains.get(session.canton, f"{session.canton.lower()}.clex.ch")
        
        # ========== STEP 1: PLANNING ==========
        planning_start = time.time()
        
        search_hint = (session.search_hints or {}).get("cantonal_law", "")
        
        planning_prompt = CANTONAL_LAW_PLANNING_PROMPT.format(
            question=session.question,
            canton=session.canton,
            canton_name=canton_display,
            commune=session.commune or "Nicht spezifiziert",
            canton_domain=canton_domain,
            legal_domain=session.legal_domain or "Nicht spezifiziert",
            key_terms=", ".join(session.key_terms) if session.key_terms else "Keine",
            search_hint=search_hint
        )
        
        from langchain_core.messages import SystemMessage, HumanMessage
        planning_response = agent_llm.invoke([
            HumanMessage(content=planning_prompt)
        ])
        
        planned_queries = parse_search_queries(planning_response.content)
        result["planned_queries"] = planned_queries
        result["planning_duration_ms"] = (time.time() - planning_start) * 1000
        
        if not planned_queries:
            planned_queries = [f"site:{canton_domain} {session.question}"]
        
        result["search_query"] = " | ".join(planned_queries)
        
        # ========== STEP 2: EXECUTE SEARCHES using Smart Search from tools.py ==========
        search_start = time.time()
        
        # Use the comprehensive search_cantonal_law function from tools.py
        # It has: include_raw_content, Merkblatt detection, PDF extraction, relevance scoring
        search_results = search_cantonal_law(
            query=session.question,
            canton=session.canton,
            canton_name=canton_display,
            orchestrator_queries=planned_queries,  # Pass agent-planned queries
            max_results=5,
            commune=session.commune
        )
        
        result["search_results"] = search_results
        result["search_duration_ms"] = (time.time() - search_start) * 1000
        
        # Store planning prompt for UI visibility
        result["user_prompt"] = f"PLANNING PROMPT:\n{planning_prompt}\n\nAGENT RESPONSE:\n{planning_response.content}"
        
        # NO STEP 3! Raw results go directly to Analysis Agent (Claude)
        result["llm_response"] = result["search_results"]
        result["llm_duration_ms"] = 0
        
    except Exception as e:
        result["error"] = str(e)
    
    result["total_duration_ms"] = (time.time() - start_time) * 1000
    return result


def run_cantonal_case_law_agent_work(session: 'ResearchSession') -> dict:
    """
    Run Cantonal Case Law Agent work - AGENTIC VERSION
    
    The agent DECIDES ITSELF what cantonal case law to search based on:
    1. The question
    2. The canton
    3. Context from orchestrator (legal domain)
    
    Flow:
    1. PLAN: LLM decides what cantonal case law searches to do
    2. EXECUTE: Run the planned searches
    3. ANALYZE: LLM analyzes results
    """
    from prompts import CANTONAL_CASE_LAW_PLANNING_PROMPT, parse_search_queries
    
    start_time = time.time()
    result = {
        "agent": "cantonal_case_law",
        "search_results": "",
        "llm_response": "",
        "system_prompt": "",
        "user_prompt": "",
        "search_query": "",
        "planned_queries": [],
        "error": None,
        "search_duration_ms": 0,
        "llm_duration_ms": 0,
        "planning_duration_ms": 0,
        "total_duration_ms": 0
    }
    
    if not session.canton:
        result["error"] = "No canton specified"
        return result
    
    try:
        agent_llm, _ = get_llm("agent")
        canton_display = session.canton_name.get("de", session.canton) if session.canton_name else session.canton
        
        # ========== STEP 1: PLANNING ==========
        planning_start = time.time()
        
        search_hint = (session.search_hints or {}).get("cantonal_law", "")
        
        planning_prompt = CANTONAL_CASE_LAW_PLANNING_PROMPT.format(
            question=session.question,
            canton=session.canton,
            canton_name=canton_display,
            legal_domain=session.legal_domain or "Nicht spezifiziert",
            key_terms=", ".join(session.key_terms) if session.key_terms else "Keine",
            search_hint=search_hint
        )
        
        from langchain_core.messages import SystemMessage, HumanMessage
        planning_response = agent_llm.invoke([
            HumanMessage(content=planning_prompt)
        ])
        
        planned_queries = parse_search_queries(planning_response.content)
        result["planned_queries"] = planned_queries
        result["planning_duration_ms"] = (time.time() - planning_start) * 1000
        
        if not planned_queries:
            planned_queries = [f"site:entscheidsuche.ch {canton_display} {session.question}"]
        
        result["search_query"] = " | ".join(planned_queries)
        
        # ========== STEP 2: EXECUTE SEARCHES ==========
        search_start = time.time()
        
        # Use the comprehensive search_cantonal_case_law function from tools.py
        search_results = search_cantonal_case_law(
            query=session.question,
            canton=session.canton,
            canton_name=canton_display,
            orchestrator_queries=planned_queries,  # Pass agent-planned queries
            max_results=5
        )
        
        result["search_results"] = search_results
        result["search_duration_ms"] = (time.time() - search_start) * 1000
        
        # Store planning prompt for UI visibility
        result["user_prompt"] = f"PLANNING PROMPT:\n{planning_prompt}\n\nAGENT RESPONSE:\n{planning_response.content}"
        
        # NO STEP 3! Raw results go directly to Analysis Agent (Claude)
        result["llm_response"] = result["search_results"]
        result["llm_duration_ms"] = 0
        
    except Exception as e:
        result["error"] = str(e)
    
    result["total_duration_ms"] = (time.time() - start_time) * 1000
    return result


# ============================================================
# INSTRUMENTED RESEARCH PIPELINE
# ============================================================

def run_research_pipeline(session: ResearchSession) -> Generator[ResearchSession, None, None]:
    """
    Run the full research pipeline with instrumentation.
    Yields updated session after each significant event.
    """
    from langchain_core.prompts import ChatPromptTemplate
    import json
    
    # Get orchestrator LLM (stronger model)
    llm, model_name = get_llm("orchestrator")
    pipeline_start = time.time()
    
    # ========== ORCHESTRATOR: LLM-BASED ANALYSIS ==========
    session.orchestrator.status = AgentStatus.RUNNING
    session.orchestrator.current_step = "Analyzing question"
    session.orchestrator.add_log("Analyzing question with LLM...", "start")
    session.orchestrator.add_log(f"Using model: {model_name}", "config")
    yield session
    
    # Let the LLM analyze the question
    # Build orchestrator prompt with optional document context
    doc_context_for_orchestrator = ""
    doc_analysis_instruction = ""
    if session.document_text:
        doc_context_for_orchestrator = f"\n\nDOCUMENT TO ANALYZE:\n\"\"\"\n{session.document_text[:3000]}\n\"\"\"\n"
        doc_analysis_instruction = """
6. ANALYZE THE DOCUMENT and extract key facts:
   - Document type (contract type, letter, etc.)
   - Parties involved
   - Key dates, amounts, obligations
   - The specific problem/issue
   - Legal questions that arise
"""
    
    orchestrator_prompt = """Du bist ein erfahrener SCHWEIZER RECHTSRECHERCHE-SPEZIALIST.

Deine Aufgabe: Analysiere die Rechtsfrage und liefere optimalen Kontext für die Such-Agents. 
Die Qualität der Recherche hängt direkt von deiner Analyse ab!

FRAGE: {question}
{document_context}

ANALYSE-AUFGABEN:
1. Sprache erkennen
2. Kanton/Gemeinde identifizieren (Schweizer Geographie)
3. Rechtsgebiet präzise bestimmen
4. Relevante Gesetzesartikel auflisten
5. Schlüsselbegriffe für die Suche - denke wie ein Jurist:
   - Fachbegriffe, Synonyme, verwandte Konzepte
   - Was würde ein Anwalt suchen um diese Frage zu beantworten?
6. Spezifische Such-Hinweise für jeden Agent-Typ
{doc_analysis_task}
Respond in JSON format ONLY:
{{
    "canton": null,
    "canton_name": null,
    "commune": null, 
    "response_language": "German/French/Italian/English",
    "legal_domain": "primary domain name",
    "related_domains": ["related domain 1", "related domain 2"],
    "legal_context": "detailed explanation of the legal issue",
    "relevant_articles": ["Art. X OR", "Art. Y ZGB"],
    "irrelevant_articles": [],
    "key_terms": ["Schlüsselbegriff1", "Schlüsselbegriff2", "Schlüsselbegriff3"],
    "synonyms": {{"Begriff1": ["Synonym1", "Synonym2"], "Begriff2": ["Synonym3"]}},
    "search_hints": {{
        "primary_law": "Hinweise für Bundesrecht-Suche",
        "case_law": "Hinweise für BGE-Suche",
        "cantonal_law": "Hinweise für kantonale Suche"
    }},
    "document_analysis": {{
        "document_type": "type of document or null if no document",
        "parties": ["Party A", "Party B"],
        "key_facts": ["fact 1", "fact 2", "fact 3"],
        "amounts_dates": ["CHF X", "date Y"],
        "problem": "the specific issue",
        "legal_questions": ["question 1", "question 2"]
    }},
    "reasoning": "brief explanation"
}}

KONTEXT FÜR AGENTS - denke wie ein erfahrener Anwalt:
- related_domains: Welche verwandten Rechtsgebiete sind auch relevant? (z.B. Erbrecht → Pflichtteilsrecht, Steuerrecht)
- key_terms: Die wichtigsten Suchbegriffe (juristisch korrekt!)
- synonyms: Alternative Begriffe die dasselbe meinen
- search_hints: Spezifische Hinweise für jeden Agent-Typ

Beispiel für "Wie hoch darf ein Zaun sein?":
- key_terms: ["Zaunhöhe", "Einfriedung", "Grenzabstand", "Nachbarrecht"]
- synonyms: {{"Zaun": ["Einfriedung", "Einzäunung"], "Höhe": ["Maximalhöhe", "Höchstmass"]}}
- search_hints: {{
    "primary_law": "Art. 684-686 ZGB, Sachenrecht Nachbarrecht",
    "case_law": "BGE zu Einfriedungen, Immissionen, Grenzabstand",
    "cantonal_law": "Bauverordnung, Baugesetz, Einfriedungsvorschriften"
  }}

HINWEIS zu "irrelevant_articles":
- Setze irrelevant_articles NUR wenn echte Verwechslungsgefahr besteht (z.B. Mietrecht vs. Arbeitsrecht bei "Kündigung")
- Bei den meisten Fragen: irrelevant_articles sollte LEER sein []

KANTONS-ABKÜRZUNGEN (für canton field):
ZH, BE, LU, UR, SZ, OW, NW, GL, ZG, FR, SO, BS, BL, SH, AR, AI, SG, GR, AG, TG, TI, VD, VS, NE, GE, JU


=== EXAMPLES ===

Example 1 - MIETRECHT:
Question: "Kann mein Vermieter mir einfach kündigen?"
{{
    "legal_domain": "Mietrecht",
    "related_domains": ["Prozessrecht (Anfechtung)", "Schlichtungsverfahren"],
    "relevant_articles": ["Art. 271 OR", "Art. 271a OR", "Art. 266a OR"],
    "key_terms": ["Kündigung", "Mietvertrag", "Kündigungsschutz", "Anfechtung"],
    "search_hints": {{
        "primary_law": "Art. 266a-271a OR Mietrecht Kündigung",
        "case_law": "BGE Kündigungsschutz missbräuchliche Kündigung"
    }}
}}

Example 2 - ERBRECHT mit Liegenschaft:
Question: "Was muss ich bei einem Erbvorbezug einer Liegenschaft beachten?"
{{
    "legal_domain": "Erbrecht",
    "related_domains": ["Pflichtteilsrecht", "Herabsetzungsklage", "Grundstücksteuerrecht"],
    "relevant_articles": ["Art. 626 ZGB", "Art. 627 ZGB", "Art. 522 ZGB", "Art. 560 ZGB"],
    "key_terms": ["Erbvorbezug", "Ausgleichung", "Pflichtteil", "Herabsetzung", "Liegenschaft"],
    "search_hints": {{
        "primary_law": "Art. 626-628 ZGB Ausgleichung, Art. 522 ZGB Pflichtteil",
        "case_law": "BGE Erbvorbezug Ausgleichung Pflichtteil"
    }}
}}

Example 3 - NACHBARRECHT kantonal:
Question: "Wie hoch darf ein Zaun in Appenzell sein?"
{{
    "canton": "AI",
    "canton_name": "Appenzell Innerrhoden",
    "legal_domain": "Nachbarrecht / Baurecht",
    "related_domains": ["Kantonales Baurecht"],
    "relevant_articles": ["Art. 684 ZGB", "Art. 686 ZGB"],
    "key_terms": ["Zaunhöhe", "Einfriedung", "Grenzabstand"],
    "search_hints": {{
        "primary_law": "Art. 684-686 ZGB Nachbarrecht",
        "cantonal_law": "Bauverordnung BauV Einfriedung"
    }}
}}

Now analyze the question above and provide RICH CONTEXT for the agents."""

    try:
        from langchain_core.messages import HumanMessage
        response = llm.invoke([HumanMessage(content=orchestrator_prompt.format(
            question=session.question,
            document_context=doc_context_for_orchestrator,
            doc_analysis_task=doc_analysis_instruction
        ))])
        
        # Parse JSON response
        response_text = response.content.strip()
        session.orchestrator.add_log(f"LLM raw response length: {len(response_text)} chars", "info")
        
        # Extract JSON from markdown code blocks more robustly
        import re
        
        # Try to find JSON in code block first
        json_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(1)
        else:
            # Try to find the outermost JSON object by matching braces
            start_idx = response_text.find('{')
            if start_idx != -1:
                brace_count = 0
                end_idx = start_idx
                for i, char in enumerate(response_text[start_idx:], start_idx):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i
                            break
                response_text = response_text[start_idx:end_idx + 1]
        
        response_text = response_text.strip()
        session.orchestrator.add_log(f"Extracted JSON: {response_text[:200]}...", "debug")
        
        analysis = json.loads(response_text)
        session.orchestrator.add_log(f"Parsed keys: {list(analysis.keys())}", "debug")
        
        # Handle null values properly (JSON null becomes Python None)
        canton_val = analysis.get("canton")
        session.orchestrator.add_log(f"Raw canton value: {canton_val} (type: {type(canton_val).__name__})", "debug")
        session.canton = canton_val if canton_val and canton_val != "null" and canton_val != "None" and canton_val is not None else ""
        
        canton_name_val = analysis.get("canton_name")
        session.canton_name = {"de": canton_name_val} if canton_name_val and canton_name_val != "null" and canton_name_val is not None else {}
        
        commune_val = analysis.get("commune")
        session.commune = commune_val if commune_val and commune_val != "null" and commune_val != "None" and commune_val is not None else ""
        
        # Get response language, with fallback detection
        lang = analysis.get("response_language", "German")
        if lang == "DETECT_FROM_QUESTION" or not lang:
            # Fallback: simple detection based on common words
            q_lower = session.question.lower()
            if any(w in q_lower for w in ["est-ce", "peut", "mon", "employeur", "je", "licencier", "loyer"]):
                lang = "French"
            elif any(w in q_lower for w in ["il mio", "può", "posso", "datore", "lavoro"]):
                lang = "Italian"
            elif any(w in q_lower for w in ["can my", "employer", "fired", "employment"]):
                lang = "English"
            else:
                lang = "German"
        session.response_language = lang
        session.has_cantonal_scope = bool(session.canton)
        
        # Extract legal domain and context
        session.legal_domain = analysis.get("legal_domain") or ""
        session.related_domains = analysis.get("related_domains") or []
        session.legal_context = analysis.get("legal_context") or ""
        session.relevant_articles = analysis.get("relevant_articles") or []
        session.irrelevant_articles = analysis.get("irrelevant_articles") or []
        
        # Extract enhanced context for agents (NEW!)
        session.key_terms = analysis.get("key_terms") or []
        session.synonyms = analysis.get("synonyms") or {}
        session.search_hints = analysis.get("search_hints") or {}
        
        # Extract document analysis (new!)
        doc_analysis = analysis.get("document_analysis") or {}
        if doc_analysis and isinstance(doc_analysis, dict):
            session.document_analysis = doc_analysis
            if doc_analysis.get("document_type"):
                session.orchestrator.add_log(f"📄 Document: {doc_analysis.get('document_type')}", "info")
            if doc_analysis.get("problem"):
                session.orchestrator.add_log(f"⚠️ Issue: {doc_analysis.get('problem')}", "info")
            if doc_analysis.get("key_facts"):
                facts = doc_analysis.get("key_facts", [])[:3]
                session.orchestrator.add_log(f"📋 Key facts: {', '.join(facts)}", "info")
        
        # Get search queries (new format) or fall back to search_topics (old format)
        # NOTE: Agents now plan their own searches - this is kept for backward compatibility only
        session.search_queries = {}  # Not used anymore - agents are agentic
        
        # Log results
        session.orchestrator.add_log(f"🌐 Response language: {session.response_language}", "info")
        if session.legal_domain:
            session.orchestrator.add_log(f"⚖️ Legal domain: {session.legal_domain}", "info")
        if session.related_domains:
            session.orchestrator.add_log(f"🔗 Related: {', '.join(session.related_domains[:3])}", "info")
        if session.legal_context:
            session.orchestrator.add_log(f"📚 Context: {session.legal_context}", "info")
        if session.relevant_articles:
            session.orchestrator.add_log(f"✅ Relevant articles: {', '.join(session.relevant_articles[:4])}", "info")
        if session.irrelevant_articles:
            session.orchestrator.add_log(f"❌ AVOID: {', '.join(session.irrelevant_articles[:2])}", "warning")
        if session.key_terms:
            session.orchestrator.add_log(f"🔑 Key terms: {', '.join(session.key_terms[:5])}", "info")
        if session.search_hints:
            session.orchestrator.add_log(f"💡 Search hints provided for agents", "info")
        if session.has_cantonal_scope:
            session.orchestrator.add_log(f"🏔️ Canton: {session.canton} ({session.canton_name.get('de', '')})", "info")
            if session.commune:
                session.orchestrator.add_log(f"🏘️ Commune: {session.commune}", "info")
        else:
            session.orchestrator.add_log("📍 No specific canton detected", "info")
        if analysis.get("reasoning"):
            session.orchestrator.add_log(f"💭 {analysis.get('reasoning')}", "info")
        
    except Exception as e:
        # Fallback: no canton, German response
        import traceback
        session.orchestrator.add_log(f"⚠️ Analysis failed: {str(e)}", "error")
        session.orchestrator.add_log(f"Traceback: {traceback.format_exc()[:300]}", "error")
        session.canton = ""
        session.canton_name = {}
        session.commune = ""
        session.response_language = "German"
        session.has_cantonal_scope = False
        session.legal_domain = ""
        session.legal_context = ""
        session.relevant_articles = []
        session.irrelevant_articles = []
        session.search_queries = {"primary_law": [], "case_law": []}
    
    yield session
    
    # Build dynamic pipeline based on whether canton is detected
    # Note: Primary Law, Case Law, Cantonal Law, and Cantonal Case Law run IN PARALLEL
    if session.has_cantonal_scope:
        session.orchestrator.pipeline = [
            {"step": 1, "agent": "Primary Law Agent", "status": "pending", "parallel": True},
            {"step": 1, "agent": "Cantonal Law Agent", "status": "pending", "parallel": True},
            {"step": 1, "agent": "Cantonal Case Law Agent", "status": "pending", "parallel": True},
            {"step": 1, "agent": "Case Law Agent", "status": "pending", "parallel": True},
            {"step": 2, "agent": "Analysis Agent", "status": "pending"},
        ]
    else:
        session.orchestrator.pipeline = [
            {"step": 1, "agent": "Primary Law Agent", "status": "pending", "parallel": True},
            {"step": 1, "agent": "Case Law Agent", "status": "pending", "parallel": True},
            {"step": 2, "agent": "Analysis Agent", "status": "pending"},
        ]
    
    session.orchestrator.add_log("Pipeline initialized", "start")
    session.orchestrator.add_log(f"Question: {session.question[:100]}...", "info")
    session.orchestrator.add_log("⚡ Parallel execution enabled for search agents", "config")
    
    if session.has_cantonal_scope:
        session.orchestrator.add_log("→ Including cantonal law + cantonal case law search in pipeline", "info")
    
    if session.document_text:
        session.orchestrator.add_log(f"Document loaded: {session.document_name} ({len(session.document_text)} chars)", "info")
    
    # Show hybrid model configuration
    _, agent_model = get_llm("agent")
    session.orchestrator.add_log(f"🧠 Orchestrator/Analysis: {model_name}", "config")
    session.orchestrator.add_log(f"⚡ Search Agents: {agent_model}", "config")
    yield session
    

    # ========== PARALLEL AGENT EXECUTION ==========
    session.orchestrator.current_step = "Dispatching Search Agents (Parallel)"
    session.orchestrator.add_log("🚀 Starting parallel agent execution", "dispatch")
    
    # Set all search agents to running
    for item in session.orchestrator.pipeline:
        if item.get("parallel"):
            item["status"] = "running"
    
    # Build AGENT-SPECIFIC contexts (saves tokens - each agent only gets what it needs!)
    
    # Helper to convert lists to compact strings
    def to_str(lst):
        return ", ".join(lst) if lst else ""
    
    # Common base context
    base_context = {
        "question": session.question,
        "language": session.response_language,
        "legal_domain": session.legal_domain,
    }
    
    # PRIMARY LAW AGENT - Federal law focus
    primary_law_context = {
        **base_context,
        "related_domains": to_str(session.related_domains),
        "relevant_articles": to_str(session.relevant_articles),
        "key_terms": to_str(session.key_terms),
        "search_hint": (session.search_hints or {}).get("primary_law", ""),
    }
    
    # CASE LAW AGENT - BGE focus
    case_law_context = {
        **base_context,
        "related_domains": to_str(session.related_domains),
        "relevant_articles": to_str(session.relevant_articles),
        "key_terms": to_str(session.key_terms),
        "search_hint": (session.search_hints or {}).get("case_law", ""),
    }
    
    # CANTONAL LAW AGENT - Canton-specific
    cantonal_law_context = {
        **base_context,
        "canton": session.canton,
        "canton_name": session.canton_name.get("de", session.canton) if session.canton_name else session.canton,
        "commune": session.commune or "",
        "key_terms": to_str(session.key_terms),
        "search_hint": (session.search_hints or {}).get("cantonal_law", ""),
    }
    
    # CANTONAL CASE LAW AGENT - Canton court decisions
    cantonal_case_law_context = {
        **base_context,
        "canton": session.canton,
        "canton_name": session.canton_name.get("de", session.canton) if session.canton_name else session.canton,
        "key_terms": to_str(session.key_terms),
        "search_hint": (session.search_hints or {}).get("cantonal_law", ""),
    }
    
    # Set agent inputs (for UI display)
    session.orchestrator.set_agent_input("Primary Law Agent", primary_law_context)
    session.orchestrator.set_agent_input("Case Law Agent", case_law_context)
    if session.has_cantonal_scope:
        session.orchestrator.set_agent_input("Cantonal Law Agent", cantonal_law_context)
        session.orchestrator.set_agent_input("Cantonal Case Law Agent", cantonal_case_law_context)
    
    # Initialize agent states
    session.primary_law_agent.status = AgentStatus.RUNNING
    session.primary_law_agent.current_action = "Running in parallel"
    session.primary_law_agent.add_log("Agent activated (parallel mode)", "start")
    if session.legal_domain:
        session.primary_law_agent.add_log(f"⚖️ Rechtsgebiet: {session.legal_domain}", "info")
    
    session.case_law_agent.status = AgentStatus.RUNNING
    session.case_law_agent.current_action = "Running in parallel"
    session.case_law_agent.add_log("Agent activated (parallel mode)", "start")
    
    if session.has_cantonal_scope:
        session.cantonal_law_agent.status = AgentStatus.RUNNING
        session.cantonal_law_agent.current_action = "Running in parallel"
        session.cantonal_law_agent.add_log("Agent activated (parallel mode)", "start")
        session.cantonal_law_agent.add_log(f"🏛️ Kanton: {session.canton}", "info")
        
        session.cantonal_case_law_agent.status = AgentStatus.RUNNING
        session.cantonal_case_law_agent.current_action = "Running in parallel"
        session.cantonal_case_law_agent.add_log("Agent activated (parallel mode)", "start")
        session.cantonal_case_law_agent.add_log(f"🏛️ Kanton: {session.canton} - Rechtsprechung", "info")
    
    yield session
    
    # Submit all agents to thread pool
    parallel_start = time.time()
    futures = {}
    
    # max_workers = 4 if cantonal scope (primary, case, cantonal_law, cantonal_case)
    num_workers = 4 if session.has_cantonal_scope else 2
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit Primary Law Agent
        futures[executor.submit(run_primary_law_agent_work, session)] = "primary_law"
        session.orchestrator.add_log("→ Primary Law Agent started", "dispatch")
        
        # Submit Case Law Agent
        futures[executor.submit(run_case_law_agent_work, session)] = "case_law"
        session.orchestrator.add_log("→ Case Law Agent started", "dispatch")
        
        # Submit Cantonal Law Agent if needed
        if session.has_cantonal_scope:
            futures[executor.submit(run_cantonal_law_agent_work, session)] = "cantonal_law"
            session.orchestrator.add_log(f"→ Cantonal Law Agent started ({session.canton})", "dispatch")
            
            # Submit Cantonal Case Law Agent (SEPARATE!)
            futures[executor.submit(run_cantonal_case_law_agent_work, session)] = "cantonal_case_law"
            session.orchestrator.add_log(f"→ Cantonal Case Law Agent started ({session.canton})", "dispatch")
        
        yield session
        
        # Wait for results and update as each completes
        completed_count = 0
        total_agents = len(futures)
        
        for future in as_completed(futures):
            agent_type = futures[future]
            completed_count += 1
            
            try:
                result = future.result()
                
                if agent_type == "primary_law":
                    agent = session.primary_law_agent
                    agent.search_results = result["search_results"]
                    agent.llm_response = result["llm_response"]
                    agent.system_prompt = result.get("system_prompt", "")
                    agent.user_prompt = result.get("user_prompt", "")
                    agent.search_query = result.get("search_query", "")
                    agent.planned_queries = result.get("planned_queries", [])
                    agent.planning_duration_ms = result.get("planning_duration_ms", 0)
                    agent.llm_prompt = f"SYSTEM:\n{agent.system_prompt}\n\nUSER:\n{agent.user_prompt}"
                    agent.duration_ms = result["total_duration_ms"]
                    
                    if result["error"]:
                        agent.status = AgentStatus.ERROR
                        agent.error = result["error"]
                        agent.add_log(f"❌ Error: {result['error']}", "error")
                        session.orchestrator.pipeline[0]["status"] = "error"
                    else:
                        agent.status = AgentStatus.COMPLETE
                        agent.current_action = "Complete"
                        if agent.planned_queries:
                            agent.add_log(f"🧠 Planned {len(agent.planned_queries)} searches ({agent.planning_duration_ms:.0f}ms)", "planning")
                        agent.add_log(f"✅ Search: {result['search_duration_ms']:.0f}ms", "complete")
                        agent.add_log(f"✅ LLM: {result['llm_duration_ms']:.0f}ms", "complete")
                        agent.add_log(f"✅ Total: {result['total_duration_ms']:.0f}ms", "complete")
                        session.orchestrator.pipeline[0]["status"] = "complete"
                    
                    session.orchestrator.add_log(f"Primary Law Agent finished ({result['total_duration_ms']:.0f}ms) [{completed_count}/{total_agents}]", "complete")
                
                elif agent_type == "case_law":
                    agent = session.case_law_agent
                    agent.search_results = result["search_results"]
                    agent.llm_response = result["llm_response"]
                    agent.system_prompt = result.get("system_prompt", "")
                    agent.user_prompt = result.get("user_prompt", "")
                    agent.search_query = result.get("search_query", "")
                    agent.planned_queries = result.get("planned_queries", [])
                    agent.planning_duration_ms = result.get("planning_duration_ms", 0)
                    agent.llm_prompt = f"SYSTEM:\n{agent.system_prompt}\n\nUSER:\n{agent.user_prompt}"
                    agent.duration_ms = result["total_duration_ms"]
                    
                    # Find case law index in pipeline (it's after cantonal agents if present)
                    case_idx = 1 if not session.has_cantonal_scope else 3
                    
                    if result["error"]:
                        agent.status = AgentStatus.ERROR
                        agent.error = result["error"]
                        agent.add_log(f"❌ Error: {result['error']}", "error")
                        session.orchestrator.pipeline[case_idx]["status"] = "error"
                    else:
                        agent.status = AgentStatus.COMPLETE
                        agent.current_action = "Complete"
                        if agent.planned_queries:
                            agent.add_log(f"🧠 Planned {len(agent.planned_queries)} searches ({agent.planning_duration_ms:.0f}ms)", "planning")
                        agent.add_log(f"✅ Search: {result['search_duration_ms']:.0f}ms", "complete")
                        agent.add_log(f"✅ LLM: {result['llm_duration_ms']:.0f}ms", "complete")
                        agent.add_log(f"✅ Total: {result['total_duration_ms']:.0f}ms", "complete")
                        session.orchestrator.pipeline[case_idx]["status"] = "complete"
                    
                    session.orchestrator.add_log(f"Case Law Agent finished ({result['total_duration_ms']:.0f}ms) [{completed_count}/{total_agents}]", "complete")
                
                elif agent_type == "cantonal_law":
                    agent = session.cantonal_law_agent
                    agent.search_results = result["search_results"]
                    agent.llm_response = result["llm_response"]
                    agent.system_prompt = result.get("system_prompt", "")
                    agent.user_prompt = result.get("user_prompt", "")
                    agent.search_query = result.get("search_query", "")
                    agent.planned_queries = result.get("planned_queries", [])
                    agent.planning_duration_ms = result.get("planning_duration_ms", 0)
                    agent.llm_prompt = f"SYSTEM:\n{agent.system_prompt}\n\nUSER:\n{agent.user_prompt}"
                    agent.duration_ms = result["total_duration_ms"]
                    
                    if result["error"]:
                        agent.status = AgentStatus.ERROR
                        agent.error = result["error"]
                        agent.add_log(f"❌ Error: {result['error']}", "error")
                        session.orchestrator.pipeline[1]["status"] = "error"
                    else:
                        agent.status = AgentStatus.COMPLETE
                        agent.current_action = "Complete"
                        if agent.planned_queries:
                            agent.add_log(f"🧠 Planned {len(agent.planned_queries)} searches ({agent.planning_duration_ms:.0f}ms)", "planning")
                        agent.add_log(f"✅ Search: {result['search_duration_ms']:.0f}ms", "complete")
                        agent.add_log(f"✅ LLM: {result['llm_duration_ms']:.0f}ms", "complete")
                        agent.add_log(f"✅ Total: {result['total_duration_ms']:.0f}ms", "complete")
                        session.orchestrator.pipeline[1]["status"] = "complete"
                    
                    session.orchestrator.add_log(f"Cantonal Law Agent finished ({result['total_duration_ms']:.0f}ms) [{completed_count}/{total_agents}]", "complete")
                
                elif agent_type == "cantonal_case_law":
                    agent = session.cantonal_case_law_agent
                    agent.search_results = result["search_results"]
                    agent.llm_response = result["llm_response"]
                    agent.system_prompt = result.get("system_prompt", "")
                    agent.user_prompt = result.get("user_prompt", "")
                    agent.search_query = result.get("search_query", "")
                    agent.planned_queries = result.get("planned_queries", [])
                    agent.planning_duration_ms = result.get("planning_duration_ms", 0)
                    agent.llm_prompt = f"SYSTEM:\n{agent.system_prompt}\n\nUSER:\n{agent.user_prompt}"
                    agent.duration_ms = result["total_duration_ms"]
                    
                    if result["error"]:
                        agent.status = AgentStatus.ERROR
                        agent.error = result["error"]
                        agent.add_log(f"❌ Error: {result['error']}", "error")
                        session.orchestrator.pipeline[2]["status"] = "error"
                    else:
                        agent.status = AgentStatus.COMPLETE
                        agent.current_action = "Complete"
                        if agent.planned_queries:
                            agent.add_log(f"🧠 Planned {len(agent.planned_queries)} searches ({agent.planning_duration_ms:.0f}ms)", "planning")
                        agent.add_log(f"✅ Search: {result['search_duration_ms']:.0f}ms", "complete")
                        agent.add_log(f"✅ LLM: {result['llm_duration_ms']:.0f}ms", "complete")
                        agent.add_log(f"✅ Total: {result['total_duration_ms']:.0f}ms", "complete")
                        session.orchestrator.pipeline[2]["status"] = "complete"
                    
                    session.orchestrator.add_log(f"Cantonal Case Law Agent finished ({result['total_duration_ms']:.0f}ms) [{completed_count}/{total_agents}]", "complete")
                
            except Exception as e:
                session.orchestrator.add_log(f"❌ {agent_type} failed: {str(e)}", "error")
                session.errors.append(f"{agent_type}: {str(e)}")
            
            yield session
    
    parallel_duration = (time.time() - parallel_start) * 1000
    session.orchestrator.add_log(f"⚡ All search agents completed in {parallel_duration:.0f}ms (parallel)", "complete")
    yield session
    
    # ========== ORCHESTRATOR: DISPATCH ANALYSIS AGENT ==========
    # Analysis is the last step: index 4 for cantonal (5 agents), index 2 for non-cantonal (3 agents)
    analysis_step_idx = 4 if session.has_cantonal_scope else 2
    session.orchestrator.current_step = "Dispatching Analysis Agent"
    session.orchestrator.pipeline[analysis_step_idx]["status"] = "running"
    session.orchestrator.add_log("Dispatching Analysis Agent", "dispatch")
    
    # Track what orchestrator passes to this agent
    analysis_inputs = {
        "question": session.question,
        "primary_law_agent": f"{len(session.primary_law_agent.llm_response or '')} chars",
        "case_law_agent": f"{len(session.case_law_agent.llm_response or '')} chars",
        "document": f"{len(session.document_text)} chars" if session.document_text else "None",
        "language": session.response_language
    }
    
    # Add document analysis if available
    if session.document_analysis:
        doc_type = session.document_analysis.get("document_type", "")
        problem = session.document_analysis.get("problem", "")
        if doc_type or problem:
            analysis_inputs["document_analysis"] = f"{doc_type}: {problem[:50]}..." if problem else doc_type
    
    if session.has_cantonal_scope:
        analysis_inputs["cantonal_law_agent"] = f"{len(session.cantonal_law_agent.llm_response or '')} chars ({session.canton})"
        analysis_inputs["cantonal_case_law_agent"] = f"{len(session.cantonal_case_law_agent.llm_response or '')} chars ({session.canton})"
        session.orchestrator.add_log("Passing: primary_law, cantonal_law, cantonal_case_law, case_law, question, document", "dispatch")
    else:
        session.orchestrator.add_log("Passing: primary_law, case_law, question, document", "dispatch")
    session.orchestrator.set_agent_input("Analysis Agent", analysis_inputs)
    yield session
    
    # ========== ANALYSIS AGENT ==========
    analysis_start = time.time()
    agent = session.analysis_agent
    agent.status = AgentStatus.RUNNING
    agent.current_action = "Starting"
    agent.add_log("Agent activated by orchestrator", "start")
    agent.add_log("📨 Received: Primary Law Agent results", "input")
    if session.has_cantonal_scope:
        agent.add_log(f"📨 Received: Cantonal Law Agent results ({session.canton})", "input")
        agent.add_log(f"📨 Received: Cantonal Case Law Agent results ({session.canton})", "input")
    agent.add_log("📨 Received: Case Law Agent results", "input")
    if session.document_text:
        agent.add_log(f"📨 Received: User document ({len(session.document_text)} chars)", "input")
    yield session
    
    # Build synthesis prompt - include cantonal results if available
    agent.current_action = "Building synthesis prompt"
    
    # Combine all law research
    primary_law_results = session.primary_law_agent.llm_response or "No results available"
    case_law_results = session.case_law_agent.llm_response or "No results available"
    
    # Add cantonal law results if available
    cantonal_results = ""
    if session.has_cantonal_scope and session.cantonal_law_agent.llm_response:
        cantonal_results = f"\n\n--- CANTONAL LAW ({session.canton}) ---\n{session.cantonal_law_agent.llm_response}"
    
    # Add cantonal case law results if available
    cantonal_case_results = ""
    if session.has_cantonal_scope and session.cantonal_case_law_agent.llm_response:
        cantonal_case_results = f"\n\n--- CANTONAL CASE LAW ({session.canton}) ---\n{session.cantonal_case_law_agent.llm_response}"
    
    # Combine primary and cantonal law for analysis
    combined_primary_law = primary_law_results
    if cantonal_results:
        combined_primary_law = f"{primary_law_results}{cantonal_results}"
    
    # Combine federal and cantonal case law
    combined_case_law = case_law_results
    if cantonal_case_results:
        combined_case_law = f"{case_law_results}{cantonal_case_results}"
    
    # Use prompts from prompts.py with orchestrator context
    agent.system_prompt, agent.user_prompt = get_analysis_prompt(
        primary_law=combined_primary_law,
        case_law=combined_case_law,  # Now includes cantonal case law!
        question=session.question,
        document_text=session.document_text,
        legal_domain=session.legal_domain,
        legal_context=session.legal_context,
        response_language=session.response_language,
        relevant_articles=session.relevant_articles,
        irrelevant_articles=session.irrelevant_articles,
        document_analysis=session.document_analysis  # NEW!
    )
    agent.llm_prompt = f"SYSTEM:\n{agent.system_prompt}\n\nUSER:\n{agent.user_prompt}"
    
    # Track data received (this is what was passed from other agents!)
    agent.data_received = {
        "from_orchestrator": {
            "question": session.question,
            "language": session.response_language,
            "legal_domain": session.legal_domain,
            "legal_context": session.legal_context,
            "relevant_articles": session.relevant_articles,
            "irrelevant_articles": session.irrelevant_articles,
            "document": session.document_text[:200] + "..." if session.document_text else None,
            "document_length": len(session.document_text) if session.document_text else 0
        },
        "from_primary_law_agent": {
            "analysis": session.primary_law_agent.llm_response[:500] + "..." if session.primary_law_agent.llm_response else None,
            "full_length": len(session.primary_law_agent.llm_response) if session.primary_law_agent.llm_response else 0
        },
        "from_case_law_agent": {
            "analysis": session.case_law_agent.llm_response[:500] + "..." if session.case_law_agent.llm_response else None,
            "full_length": len(session.case_law_agent.llm_response) if session.case_law_agent.llm_response else 0
        }
    }
    
    # Add cantonal data if present
    if session.has_cantonal_scope:
        agent.data_received["from_cantonal_law_agent"] = {
            "canton": session.canton,
            "commune": session.commune or None,
            "analysis": session.cantonal_law_agent.llm_response[:500] + "..." if session.cantonal_law_agent.llm_response else None,
            "full_length": len(session.cantonal_law_agent.llm_response) if session.cantonal_law_agent.llm_response else 0
        }
        agent.data_received["from_cantonal_case_law_agent"] = {
            "canton": session.canton,
            "analysis": session.cantonal_case_law_agent.llm_response[:500] + "..." if session.cantonal_case_law_agent.llm_response else None,
            "full_length": len(session.cantonal_case_law_agent.llm_response) if session.cantonal_case_law_agent.llm_response else 0
        }
    
    agent.add_log(f"Built synthesis prompt ({len(agent.llm_prompt)} chars)", "prepare")
    agent.add_log(f"Includes document: {bool(session.document_text)}", "prepare")
    if session.has_cantonal_scope:
        agent.add_log(f"Includes cantonal law: {session.canton}", "prepare")
        agent.add_log(f"Includes cantonal case law: {session.canton}", "prepare")
    yield session
    
    # LLM Synthesis - use stronger model for final analysis
    agent.current_action = "Generating final analysis"
    analysis_llm, analysis_model_name = get_llm("analysis")
    agent.add_log(f"Sending to LLM ({analysis_model_name})", "llm_call")
    agent.add_log(f"System prompt: {len(agent.system_prompt)} chars", "llm_call")
    agent.add_log(f"User prompt: {len(agent.user_prompt)} chars", "llm_call")
    yield session
    
    llm_start = time.time()
    try:
        from langchain_core.messages import SystemMessage, HumanMessage
        messages = [
            SystemMessage(content=agent.system_prompt),
            HumanMessage(content=agent.user_prompt)
        ]
        response = analysis_llm.invoke(messages)
        agent.llm_response = response.content
        llm_duration = (time.time() - llm_start) * 1000
        
        # Track data sent (final output)
        agent.data_sent = {
            "final_analysis": agent.llm_response,
            "length": len(agent.llm_response),
            "destination": "User (Research Output)"
        }
        
        agent.add_log(f"LLM response received ({llm_duration:.0f}ms)", "llm_response")
        agent.add_log(f"Final analysis: {len(agent.llm_response)} chars", "llm_response")
        
        # Set final output directly (no link validation - kept it simple)
        session.final_output = agent.llm_response
        yield session
        
    except Exception as e:
        agent.status = AgentStatus.ERROR
        agent.error = str(e)
        agent.add_log(f"LLM call failed: {str(e)}", "error")
        session.errors.append(f"Analysis LLM: {str(e)}")
        yield session
    
    # Complete
    agent.duration_ms = (time.time() - analysis_start) * 1000
    if agent.status != AgentStatus.ERROR:
        agent.status = AgentStatus.COMPLETE
        agent.current_action = "Complete"
        agent.add_log("Agent complete", "complete")
        agent.add_log("📤 Final output → User (Research Output panel)", "handoff")
    
    session.orchestrator.pipeline[analysis_step_idx]["status"] = "complete" if agent.status == AgentStatus.COMPLETE else "error"
    session.orchestrator.add_log(f"Analysis Agent finished ({agent.duration_ms:.0f}ms)", "complete")
    yield session
    
    # ========== ORCHESTRATOR: FINALIZE ==========
    session.orchestrator.total_duration_ms = (time.time() - pipeline_start) * 1000
    session.orchestrator.status = AgentStatus.COMPLETE
    session.orchestrator.current_step = "Pipeline complete"
    session.orchestrator.add_log(f"All agents complete. Total time: {session.orchestrator.total_duration_ms:.0f}ms", "complete")
    
    # ========== BENCHMARK: Direct LLM comparison ==========
    session.orchestrator.add_log("Running benchmark (direct LLM)...", "info")
    yield session
    
    try:
        benchmark_start = time.time()
        session.benchmark_output = run_benchmark_direct(session.question, session.response_language)
        benchmark_duration = (time.time() - benchmark_start) * 1000
        session.orchestrator.add_log(f"Benchmark complete ({benchmark_duration:.0f}ms)", "complete")
    except Exception as e:
        session.orchestrator.add_log(f"Benchmark failed: {str(e)}", "error")
        session.benchmark_output = f"Benchmark error: {str(e)}"
    
    yield session


# ============================================================
# STREAMLIT UI
# ============================================================

def render_agent_panel(agent: AgentState, expanded: bool = False, handoff_info: dict = None):
    """Render an agent's activity panel with result and handoff information"""
    
    # Status indicator
    status_icons = {
        AgentStatus.IDLE: "⚪",
        AgentStatus.RUNNING: "🔵",
        AgentStatus.COMPLETE: "🟢",
        AgentStatus.ERROR: "🔴"
    }
    
    status_icon = status_icons.get(agent.status, "⚪")
    
    with st.expander(f"{status_icon} {agent.name} — {agent.current_action or 'Idle'}", expanded=expanded):
        # Status bar
        cols = st.columns([2, 1, 1])
        with cols[0]:
            st.caption(f"Status: **{agent.status.value.upper()}**")
        with cols[1]:
            if agent.duration_ms > 0:
                st.caption(f"Duration: **{agent.duration_ms:.0f}ms**")
        with cols[2]:
            if agent.error:
                st.caption(f"❌ Error")
        
        # Tabs for different views
        log_tab, search_tab, prompt_tab, result_tab, dataflow_tab = st.tabs([
            "📋 Log", "🔍 Search", "📝 Prompts", "📊 Result", "🔀 Data Flow"
        ])
        
        with log_tab:
            if agent.logs:
                for log in agent.logs:
                    event_icons = {
                        "start": "🚀",
                        "search": "🔎",
                        "tool_call": "🔧",
                        "tool_result": "📥",
                        "llm_call": "🤖",
                        "llm_response": "💬",
                        "complete": "✅",
                        "error": "❌",
                        "info": "ℹ️",
                        "input": "📨",
                        "prepare": "📝",
                        "handoff": "➡️",
                        "planning": "🧠"  # NEW: for agentic planning
                    }
                    icon = event_icons.get(log["event_type"], "•")
                    st.text(f"[{log['timestamp']}] {icon} {log['message']}")
            else:
                st.caption("No activity yet")
        
        with search_tab:
            # Show planned queries if agentic (NEW!)
            if agent.planned_queries:
                st.caption("**🧠 Agent Planned Searches:**")
                for i, q in enumerate(agent.planned_queries, 1):
                    st.code(f"{i}. {q}", language=None)
                if agent.planning_duration_ms > 0:
                    st.caption(f"Planning took {agent.planning_duration_ms:.0f}ms")
                st.divider()
            
            if agent.search_query:
                st.caption("**Executed Query:**")
                st.code(agent.search_query, language=None)
            if agent.search_results:
                st.caption("**Raw Results from Tavily:**")
                st.code(agent.search_results[:3000] + ("..." if len(agent.search_results) > 3000 else ""), language=None)
            else:
                st.caption("No search performed (Analysis Agent receives data from other agents)")
        
        with prompt_tab:
            if agent.system_prompt:
                st.caption("**🔧 SYSTEM PROMPT** (defines agent behavior):")
                with st.container(height=200):
                    st.code(agent.system_prompt, language=None)
            
            if agent.user_prompt:
                st.caption("**👤 USER PROMPT** (contains the actual data):")
                with st.container(height=250):
                    st.code(agent.user_prompt, language=None)
            
            if not agent.system_prompt and not agent.user_prompt:
                st.caption("Prompts will appear here when the agent runs")
            
            if agent.llm_response:
                st.divider()
                st.caption("**🤖 LLM RESPONSE:**")
                with st.container(height=300):
                    response_with_links = make_links_open_in_new_tab(agent.llm_response)
                    st.markdown(response_with_links, unsafe_allow_html=True)
        
        with result_tab:
            if agent.status == AgentStatus.COMPLETE and agent.llm_response:
                st.success(f"✅ Produced {len(agent.llm_response)} characters of analysis")
                result_with_links = make_links_open_in_new_tab(agent.llm_response)
                st.markdown(result_with_links, unsafe_allow_html=True)
            elif agent.status == AgentStatus.ERROR:
                st.error(f"❌ Agent failed: {agent.error}")
            elif agent.status == AgentStatus.RUNNING:
                st.info("⏳ Agent is still processing...")
            else:
                st.caption("No result yet")
        
        with dataflow_tab:
            st.caption("**📥 DATA RECEIVED:**")
            if agent.data_received:
                st.json(agent.data_received)
            else:
                st.caption("No data received yet")
            
            st.divider()
            
            st.caption("**📤 DATA SENT:**")
            if agent.data_sent:
                st.json(agent.data_sent)
            else:
                st.caption("No data sent yet")
            
            st.divider()
            
            st.caption("**🔀 HANDOFF INFO:**")
            if handoff_info:
                st.caption("Input from:")
                for source in handoff_info.get("received_from", []):
                    st.info(f"📨 {source}")
                
                st.caption("Output to:")
                if handoff_info.get("passed_to"):
                    for target in handoff_info["passed_to"]:
                        st.success(f"➡️ {target}")
                else:
                    st.success("➡️ Final Output (User)")


def render_orchestrator_panel(orch: OrchestratorState, expanded: bool = True):
    """Render the orchestrator panel"""
    
    status_icons = {
        AgentStatus.IDLE: "⚪",
        AgentStatus.RUNNING: "🔵",
        AgentStatus.COMPLETE: "🟢",
        AgentStatus.ERROR: "🔴"
    }
    
    status_icon = status_icons.get(orch.status, "⚪")
    
    with st.expander(f"{status_icon} Orchestrator — {orch.current_step or 'Idle'}", expanded=expanded):
        
        # Tabs for different views
        tab_pipeline, tab_log, tab_handoffs = st.tabs(["📊 Pipeline", "📋 Log", "🔀 Übergaben"])
        
        with tab_pipeline:
            # Pipeline visualization
            cols = st.columns(len(orch.pipeline) if orch.pipeline else 1)
            
            for i, step in enumerate(orch.pipeline):
                with cols[i]:
                    step_icons = {
                        "pending": "⏳",
                        "running": "🔄",
                        "complete": "✅",
                        "error": "❌"
                    }
                    icon = step_icons.get(step["status"], "⏳")
                    st.markdown(f"**Step {step['step']}**")
                    st.caption(f"{icon} {step['agent']}")
            
            if orch.total_duration_ms > 0:
                st.divider()
                st.metric("Total Duration", f"{orch.total_duration_ms/1000:.1f}s")
        
        with tab_log:
            # Activity log
            if orch.logs:
                for log in orch.logs:
                    event_icons = {
                        "start": "🚀",
                        "dispatch": "📤",
                        "complete": "✅",
                        "error": "❌",
                        "info": "ℹ️",
                        "config": "⚙️",
                        "language": "🌐"
                    }
                    icon = event_icons.get(log["event_type"], "•")
                    st.text(f"[{log['timestamp']}] {icon} {log['message']}")
            else:
                st.caption("No activity yet")
        
        with tab_handoffs:
            # Show what orchestrator passes to each agent
            st.caption("**Was der Orchestrator an jeden Agent übergibt:**")
            
            if orch.agent_inputs:
                for agent_name, inputs in orch.agent_inputs.items():
                    with st.container():
                        st.markdown(f"**→ {agent_name}:**")
                        # Format inputs nicely - show ALL values for debugging
                        for key, value in inputs.items():
                            if isinstance(value, list):
                                if value:
                                    st.text(f"  • {key}:")
                                    for item in value[:5]:  # Max 5 items
                                        st.text(f"      - {item}")
                                else:
                                    st.text(f"  • {key}: (empty list)")
                            elif isinstance(value, str) and len(value) > 100:
                                st.text(f"  • {key}: {value[:100]}...")
                            elif value:
                                st.text(f"  • {key}: {value}")
                            else:
                                st.text(f"  • {key}: (not set)")
                        st.divider()
            else:
                st.caption("Pipeline noch nicht gestartet")


def main():
    st.set_page_config(
        page_title="Swiss Legal Research - Dev UI",
        page_icon="🇨🇭",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .stApp { max-width: 1600px; margin: 0 auto; }
    .block-container { padding-top: 2rem; }
    code { font-size: 11px !important; }
    .stExpander { border: 1px solid #333; border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🇨🇭 Swiss Legal Research Assistant")
        st.caption("Developer UI v2 — Full Agent Visibility")
    with col2:
        provider = os.getenv("LLM_PROVIDER", "openai")
        use_claude = os.getenv("USE_CLAUDE_FOR_ANALYSIS", "true").lower() == "true"
        has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
        
        if provider == "openai" and use_claude and has_anthropic:
            st.info("🧠 4o (Orch)\n⚡ 4o-mini (Search)\n🟣 Claude (Analysis)")
        elif provider == "openai":
            st.info("🧠 4o (Orch/Analysis)\n⚡ 4o-mini (Search)")
        elif provider == "anthropic":
            st.info("🟣 Claude Sonnet (All)")
        else:
            st.info(f"LLM: **{provider}**")
    
    # Initialize session state
    if 'research_session' not in st.session_state:
        st.session_state.research_session = None
    if 'is_running' not in st.session_state:
        st.session_state.is_running = False
    if 'reset_counter' not in st.session_state:
        st.session_state.reset_counter = 0
    
    # Sidebar
    with st.sidebar:
        st.header("📝 Research Input")
        
        # Question input - use reset_counter in key to clear on reset
        question = st.text_area(
            "Legal Question",
            placeholder="z.B. Was sind die Kündigungsfristen im Schweizer Arbeitsrecht?",
            height=100,
            key=f"question_input_{st.session_state.reset_counter}"
        )
        
        st.divider()
        
        # Document upload
        st.subheader("📄 Document Upload")
        
        # Use dynamic key for file uploader so Reset can clear it
        uploader_key = f"file_uploader_{st.session_state.reset_counter}"
        uploaded_file = st.file_uploader(
            "Upload document to analyze",
            type=["pdf", "docx", "txt", "md", "jpg", "jpeg", "png", "webp"],
            help="Supported: PDF, DOCX, TXT, MD, Images (JPG/PNG)",
            key=uploader_key
        )
        
        document_text = ""
        document_type = ""
        document_name = ""
        
        if uploaded_file:
            # Reset file position for potential re-read
            uploaded_file.seek(0)
            file_bytes = uploaded_file.read()
            uploaded_file.seek(0)
            
            document_text, document_type = extract_text_from_file(uploaded_file)
            document_name = uploaded_file.name
            
            # Check if we need LLM extraction for scanned PDF
            if document_text == "[PDF_NEEDS_LLM_EXTRACTION]":
                st.info(f"📄 {document_type}: {uploaded_file.name} - Gescanntes PDF erkannt, extrahiere Text mit LLM...")
                
                # Try direct LLM extraction on the PDF
                try:
                    document_text = extract_text_with_llm(file_bytes, "pdf")
                    if document_text and not document_text.startswith("["):
                        st.success(f"✅ {document_type}: {uploaded_file.name} (via LLM Vision)")
                        st.caption(f"Extracted: {len(document_text)} characters")
                    else:
                        st.error("LLM konnte keinen Text extrahieren. Bitte Text manuell eingeben.")
                        document_text = ""
                except Exception as e:
                    st.error(f"LLM Extraktion fehlgeschlagen: {str(e)}")
                    document_text = ""
            
            # Check if extraction was successful
            elif document_text.startswith("["):
                st.warning(f"⚠️ {document_type}: {uploaded_file.name}")
                st.error(document_text)
                document_text = ""
            elif len(document_text.strip()) == 0:
                st.warning(f"⚠️ {document_type}: {uploaded_file.name}")
                st.error("Kein Text extrahiert. Bitte Text manuell eingeben.")
            else:
                st.success(f"✅ {document_type}: {uploaded_file.name}")
                st.caption(f"Extracted: {len(document_text)} characters")
            
            with st.expander("Preview document"):
                if document_text:
                    st.text(document_text[:2000] + ("..." if len(document_text) > 2000 else ""))
                else:
                    st.caption("Kein Text verfügbar")
        
        # Or paste text
        with st.expander("Or paste document text"):
            pasted_doc = st.text_area(
                "Paste document here",
                height=150,
                key=f"pasted_doc_{st.session_state.reset_counter}",
                label_visibility="collapsed"
            )
            if pasted_doc:
                document_text = pasted_doc
                document_type = "PASTED"
                document_name = "Pasted text"
        
        st.divider()
        
        # Run and Reset buttons side by side
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            run_button = st.button(
                "🚀 Run",
                type="primary",
                use_container_width=True,
                disabled=not question or st.session_state.is_running
            )
        with btn_col2:
            if st.button("🔄 Reset", type="secondary", use_container_width=True):
                # Clear session and increment reset counter to force new widget keys
                st.session_state.research_session = None
                st.session_state.is_running = False
                st.session_state.reset_counter += 1
                st.rerun()
        
        if st.session_state.is_running:
            st.warning("Research in progress...")
        
        st.divider()
        
        # Config info
        st.caption("**Configuration:**")
        tavily_ok = bool(os.getenv("TAVILY_API_KEY"))
        st.caption(f"{'✅' if tavily_ok else '❌'} Tavily API")
        
        llm_provider = os.getenv("LLM_PROVIDER", "openai")
        openai_ok = bool(os.getenv("OPENAI_API_KEY"))
        anthropic_ok = bool(os.getenv("ANTHROPIC_API_KEY"))
        use_claude = os.getenv("USE_CLAUDE_FOR_ANALYSIS", "true").lower() == "true"
        
        if llm_provider == "anthropic":
            st.caption(f"{'✅' if anthropic_ok else '❌'} Anthropic API")
        else:
            st.caption(f"{'✅' if openai_ok else '❌'} OpenAI API")
            if use_claude:
                st.caption(f"{'✅' if anthropic_ok else '⚠️'} Claude (Analysis)")
                if not anthropic_ok:
                    st.caption("   ↳ Add ANTHROPIC_API_KEY")
    
    # Main content area
    main_col, agents_col = st.columns([1, 1])
    
    with main_col:
        st.header("📋 Research Output")
        
        output_placeholder = st.empty()
        
        if st.session_state.research_session and st.session_state.research_session.final_output:
            # Convert links to open in new tab
            output_with_links = make_links_open_in_new_tab(st.session_state.research_session.final_output)
            output_placeholder.markdown(output_with_links, unsafe_allow_html=True)
        else:
            output_placeholder.info("Run a research query to see results here.")
        
        # Errors
        if st.session_state.research_session and st.session_state.research_session.errors:
            st.error("**Errors encountered:**")
            for err in st.session_state.research_session.errors:
                st.code(err)
        
        # Benchmark comparison section
        st.divider()
        with st.expander("🆚 **Benchmark: Dieselbe Frage direkt ans LLM (ohne Recherche)**", expanded=False):
            st.caption("Was antwortet das LLM auf exakt dieselbe Frage, aber ohne unsere Agenten-Recherche?")
            
            if st.session_state.research_session and st.session_state.research_session.benchmark_output:
                benchmark_with_links = make_links_open_in_new_tab(st.session_state.research_session.benchmark_output)
                st.markdown(benchmark_with_links, unsafe_allow_html=True)
            elif st.session_state.research_session and st.session_state.research_session.final_output:
                st.info("Benchmark läuft nach dem Research-Run...")
            else:
                st.caption("Nach dem Research-Run erscheint hier der direkte LLM-Vergleich.")
    
    with agents_col:
        st.header("🔧 Agent Activity")
        
        # Check if we have a cantonal scope
        has_cantonal = (st.session_state.research_session and 
                       st.session_state.research_session.has_cantonal_scope)
        
        # Define handoff information for each agent (dynamic based on cantonal scope)
        if has_cantonal:
            primary_handoff = {
                "received_from": ["Orchestrator (user question + document)"],
                "passed_to": ["Cantonal Law Agent", "Analysis Agent"]
            }
            cantonal_handoff = {
                "received_from": ["Orchestrator (user question + canton info)"],
                "passed_to": ["Analysis Agent"]
            }
            case_handoff = {
                "received_from": ["Orchestrator (user question)"],
                "passed_to": ["Analysis Agent"]
            }
            analysis_handoff = {
                "received_from": ["Primary Law Agent", "Cantonal Law Agent", "Case Law Agent", "Orchestrator (document)"],
                "passed_to": []  # End of pipeline
            }
        else:
            primary_handoff = {
                "received_from": ["Orchestrator (user question + document)"],
                "passed_to": ["Analysis Agent"]
            }
            cantonal_handoff = None
            case_handoff = {
                "received_from": ["Orchestrator (user question)"],
                "passed_to": ["Analysis Agent"]
            }
            analysis_handoff = {
                "received_from": ["Primary Law Agent", "Case Law Agent", "Orchestrator (document)"],
                "passed_to": []  # End of pipeline
            }
        
        # Placeholders for agent panels
        orchestrator_placeholder = st.empty()
        primary_placeholder = st.empty()
        cantonal_placeholder = st.empty()  # Always create placeholder
        cantonal_case_placeholder = st.empty()  # Cantonal Case Law Agent
        case_placeholder = st.empty()
        analysis_placeholder = st.empty()
        
        # Render current state
        if st.session_state.research_session:
            session = st.session_state.research_session
            
            with orchestrator_placeholder.container():
                render_orchestrator_panel(
                    session.orchestrator, 
                    expanded=True
                )
            
            with primary_placeholder.container():
                render_agent_panel(
                    session.primary_law_agent, 
                    expanded=session.primary_law_agent.status == AgentStatus.RUNNING,
                    handoff_info=primary_handoff
                )
            
            # Only render cantonal agents if we have cantonal scope
            if session.has_cantonal_scope:
                with cantonal_placeholder.container():
                    render_agent_panel(
                        session.cantonal_law_agent,
                        expanded=session.cantonal_law_agent.status == AgentStatus.RUNNING,
                        handoff_info=cantonal_handoff
                    )
                with cantonal_case_placeholder.container():
                    render_agent_panel(
                        session.cantonal_case_law_agent,
                        expanded=session.cantonal_case_law_agent.status == AgentStatus.RUNNING,
                        handoff_info=cantonal_handoff  # Same handoff info
                    )
            
            with case_placeholder.container():
                render_agent_panel(
                    session.case_law_agent, 
                    expanded=session.case_law_agent.status == AgentStatus.RUNNING,
                    handoff_info=case_handoff
                )
            
            with analysis_placeholder.container():
                render_agent_panel(
                    session.analysis_agent, 
                    expanded=session.analysis_agent.status == AgentStatus.RUNNING,
                    handoff_info=analysis_handoff
                )
        else:
            with orchestrator_placeholder.container():
                st.info("Orchestrator idle — start a research query")
    
    # Run research
    if run_button and question:
        st.session_state.is_running = True
        
        # Create new session
        session = ResearchSession(
            question=question,
            document_text=document_text,
            document_type=document_type,
            document_name=document_name
        )
        st.session_state.research_session = session
        
        # Run pipeline with live updates
        for updated_session in run_research_pipeline(session):
            st.session_state.research_session = updated_session
            
            # Update handoff info based on session state
            if updated_session.has_cantonal_scope:
                primary_handoff = {
                    "received_from": ["Orchestrator (user question + document)"],
                    "passed_to": ["Cantonal Law Agent", "Analysis Agent"]
                }
                cantonal_handoff = {
                    "received_from": ["Orchestrator (user question + canton info)"],
                    "passed_to": ["Analysis Agent"]
                }
                analysis_handoff = {
                    "received_from": ["Primary Law Agent", "Cantonal Law Agent", "Case Law Agent", "Orchestrator (document)"],
                    "passed_to": []
                }
            
            # Update orchestrator panel
            with orchestrator_placeholder.container():
                render_orchestrator_panel(
                    updated_session.orchestrator, 
                    expanded=True
                )
            
            # Update agent panels with handoff info
            with primary_placeholder.container():
                render_agent_panel(
                    updated_session.primary_law_agent,
                    expanded=updated_session.primary_law_agent.status == AgentStatus.RUNNING,
                    handoff_info=primary_handoff
                )
            
            # Render cantonal agents if scope includes canton
            if updated_session.has_cantonal_scope:
                with cantonal_placeholder.container():
                    render_agent_panel(
                        updated_session.cantonal_law_agent,
                        expanded=updated_session.cantonal_law_agent.status == AgentStatus.RUNNING,
                        handoff_info=cantonal_handoff
                    )
                with cantonal_case_placeholder.container():
                    render_agent_panel(
                        updated_session.cantonal_case_law_agent,
                        expanded=updated_session.cantonal_case_law_agent.status == AgentStatus.RUNNING,
                        handoff_info=cantonal_handoff  # Same handoff info
                    )
            
            with case_placeholder.container():
                render_agent_panel(
                    updated_session.case_law_agent,
                    expanded=updated_session.case_law_agent.status == AgentStatus.RUNNING,
                    handoff_info=case_handoff
                )
            
            with analysis_placeholder.container():
                render_agent_panel(
                    updated_session.analysis_agent,
                    expanded=updated_session.analysis_agent.status == AgentStatus.RUNNING,
                    handoff_info=analysis_handoff
                )
            
            # Update output
            if updated_session.final_output:
                output_with_links = make_links_open_in_new_tab(updated_session.final_output)
                output_placeholder.markdown(output_with_links, unsafe_allow_html=True)
            
            time.sleep(0.1)  # Small delay for UI updates
        
        st.session_state.is_running = False
        st.rerun()


if __name__ == "__main__":
    main()

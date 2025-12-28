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

import streamlit as st

# Load environment variables
load_dotenv()

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
from prompts import (
    PRIMARY_LAW_SYSTEM_PROMPT,
    PRIMARY_LAW_USER_PROMPT,
    CASE_LAW_SYSTEM_PROMPT,
    CASE_LAW_USER_PROMPT,
    CANTONAL_LAW_SYSTEM_PROMPT,
    CANTONAL_LAW_USER_PROMPT,
    ANALYSIS_SYSTEM_PROMPT,
    ANALYSIS_USER_PROMPT,
    get_primary_law_prompt,
    get_case_law_prompt,
    get_cantonal_law_prompt,
    get_analysis_prompt
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
    legal_context: str = ""  # Brief explanation of the issue
    relevant_articles: list = field(default_factory=list)  # Articles that ARE relevant
    irrelevant_articles: list = field(default_factory=list)  # Articles to AVOID (wrong domain)
    
    search_topics: list = field(default_factory=list)  # Legacy: Topics to search for
    search_queries: dict = field(default_factory=dict)  # New: Specific queries per agent
    
    # Benchmark
    benchmark_output: str = ""  # Direct ChatGPT response for comparison
    
    orchestrator: OrchestratorState = field(default_factory=OrchestratorState)
    primary_law_agent: AgentState = field(default_factory=lambda: AgentState(name="Primary Law Agent"))
    case_law_agent: AgentState = field(default_factory=lambda: AgentState(name="Case Law Agent"))
    cantonal_law_agent: AgentState = field(default_factory=lambda: AgentState(name="Cantonal Law Agent"))
    analysis_agent: AgentState = field(default_factory=lambda: AgentState(name="Analysis Agent"))
    
    final_output: str = ""
    errors: list = field(default_factory=list)


# ============================================================
# LLM CONFIGURATION
# ============================================================

def get_llm():
    """Get configured LLM"""
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0.1), "claude-sonnet-4-20250514"
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.1), "gpt-4o-mini"


def run_benchmark_direct(question: str, language: str = "German") -> str:
    """
    Run direct LLM query WITHOUT agents for comparison.
    Uses the EXACT same question and output format to ensure fair comparison.
    """
    llm, _ = get_llm()
    
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
# INSTRUMENTED RESEARCH PIPELINE
# ============================================================

def run_research_pipeline(session: ResearchSession) -> Generator[ResearchSession, None, None]:
    """
    Run the full research pipeline with instrumentation.
    Yields updated session after each significant event.
    """
    from langchain_core.prompts import ChatPromptTemplate
    import json
    
    llm, model_name = get_llm()
    pipeline_start = time.time()
    
    # ========== ORCHESTRATOR: LLM-BASED ANALYSIS ==========
    session.orchestrator.status = AgentStatus.RUNNING
    session.orchestrator.current_step = "Analyzing question"
    session.orchestrator.add_log("Analyzing question with LLM...", "start")
    yield session
    
    # Let the LLM analyze the question
    orchestrator_prompt = """You are the orchestrator for a Swiss legal research system. Analyze this question and plan the research.

QUESTION: {question}

YOUR TASKS:
1. Detect the language of the question
2. Identify if a specific Swiss canton is mentioned
3. Identify the SPECIFIC legal domain (Rechtsgebiet)
4. List the RELEVANT articles for this domain
5. Generate search queries

Respond in JSON format ONLY:
{{
    "canton": null,
    "canton_name": null,
    "commune": null, 
    "response_language": "German/French/Italian/English",
    "legal_domain": "specific domain name",
    "legal_context": "brief explanation of the legal issue",
    "relevant_articles": ["Art. X OR", "Art. Y ZGB"],
    "irrelevant_articles": ["Art. Z OR - this is for different domain"],
    "search_queries": {{
        "primary_law": ["specific query 1", "specific query 2"],
        "case_law": ["specific query 1", "specific query 2"]
    }},
    "reasoning": "explanation"
}}

=== LEGAL DOMAIN MAPPING (CRITICAL - DO NOT MIX!) ===

**MIETRECHT (Rental Law)** - Art. 253-274 OR:
- Mietzins/Erhöhung: Art. 269, 269a, 269d OR
- Anfechtung Mietzins: Art. 270, 270a, 270b OR
- Kündigung Mietvertrag: Art. 271, 271a, 272 OR
- Mängel: Art. 259a-259i OR
- VMWG (Verordnung): Art. 11-14 VMWG (Referenzzinssatz)
- NICHT verwechseln mit: Art. 335-336 OR (das ist Arbeitsrecht!)

**ARBEITSRECHT (Employment Law)** - Art. 319-362 OR:
- Kündigung Arbeitsvertrag: Art. 335-335c OR
- Kündigungsschutz: Art. 336-336b OR
- Fristlose Kündigung: Art. 337 OR
- Lohn: Art. 322-324 OR
- NICHT verwechseln mit: Art. 269-274 OR (das ist Mietrecht!)

**FAMILIENRECHT** - ZGB:
- Unterhalt Kinder: Art. 276-277 ZGB
- Unterhalt Berechnung: Art. 285 ZGB
- Ehe: Art. 90-251 ZGB
- Scheidung: Art. 111-134 ZGB

**NACHBARRECHT** - ZGB:
- Immissionen: Art. 684 ZGB
- Grenzabstand: Art. 686-688 ZGB
- Eigentum: Art. 679-698 ZGB

**SACHENRECHT** - ZGB:
- Eigentum: Art. 641-729 ZGB
- Besitz: Art. 919-941 ZGB

**HAFTPFLICHT** - OR:
- Unerlaubte Handlung: Art. 41-61 OR

**ERBRECHT** - ZGB:
- Erbfolge: Art. 457-536 ZGB
- Pflichtteil: Art. 470-480 ZGB

=== LANGUAGE DETECTION ===
- French: "Est-ce que", "loyer", "propriétaire", "bail"
- German: "Kann", "Miete", "Vermieter", "kündigen"
- Italian: "affitto", "locatore", "posso"

=== CANTON DETECTION ===
Set canton to null UNLESS explicitly mentioned by name or abbreviation.
Canton names to detect:
- Appenzell, Appenzell Innerrhoden → "AI"
- Zürich, Zug, Bern, Basel, etc. → Use official abbreviation

=== EXAMPLES ===

Example 1 - MIETRECHT (no canton):
Question: "Kann ich mich gegen die Mietzinserhöhung wehren?"
Response:
{{
    "canton": null,
    "canton_name": null,
    "response_language": "German",
    "legal_domain": "Mietrecht",
    "legal_context": "Anfechtung Mietzinserhöhung",
    "relevant_articles": ["Art. 269d OR (Mietzinserhöhung)", "Art. 270a OR (Anfechtung)", "Art. 270b OR (Fristen)"],
    "irrelevant_articles": ["Art. 335c OR - das ist Arbeitsrecht!"],
    "search_queries": {{
        "primary_law": ["Mietzinserhöhung Art. 269d OR", "Anfechtung Mietzins Art. 270a OR"],
        "case_law": ["BGE Mietzinserhöhung", "BGE missbräuchlicher Mietzins"]
    }},
    "reasoning": "Frage betrifft Mietrecht, nicht Arbeitsrecht"
}}

Example 2 - BAURECHT/NACHBARRECHT (with canton):
Question: "Darf ich in Appenzell einen Zaun um meinen Garten bauen?"
Response:
{{
    "canton": "AI",
    "canton_name": "Appenzell Innerrhoden",
    "response_language": "German",
    "legal_domain": "Nachbarrecht / Baurecht",
    "legal_context": "Einfriedung, Zaun, Grenzabstand",
    "relevant_articles": ["Art. 684 ZGB (Immissionen)", "Art. 686 ZGB (Grenzabstand)", "Kantonales Baugesetz AI"],
    "irrelevant_articles": [],
    "search_queries": {{
        "primary_law": ["Einfriedung Zaun Art. 684 ZGB", "Grenzabstand Zaun Baugesetz Appenzell"],
        "case_law": ["BGE Einfriedung Nachbarrecht", "BGE Zaun Grenze"]
    }},
    "reasoning": "Appenzell explizit erwähnt → Kanton AI. Baurecht/Nachbarrecht betroffen."
}}

Example 3 - ARBEITSRECHT:
Question: "Kann mein Arbeitgeber mich einfach so kündigen?"
Response:
{{
    "canton": null,
    "canton_name": null,
    "response_language": "German",
    "legal_domain": "Arbeitsrecht",
    "legal_context": "Kündigung Arbeitsvertrag",
    "relevant_articles": ["Art. 335 OR", "Art. 335c OR (Fristen)", "Art. 336 OR (missbräuchlich)"],
    "irrelevant_articles": ["Art. 271 OR - das ist Mietrecht!"],
    "search_queries": {{
        "primary_law": ["Kündigung Arbeitsvertrag Art. 335 OR", "Kündigungsschutz Art. 336 OR"],
        "case_law": ["BGE Kündigung Arbeitsvertrag", "BGE missbräuchliche Kündigung"]
    }},
    "reasoning": "Arbeitsrecht, nicht Mietrecht"
}}

Now analyze the question above. Be VERY careful to identify the correct legal domain and ONLY use articles from that domain!"""

    try:
        from langchain_core.messages import HumanMessage
        response = llm.invoke([HumanMessage(content=orchestrator_prompt.format(question=session.question))])
        
        # Parse JSON response
        response_text = response.content.strip()
        session.orchestrator.add_log(f"LLM raw response length: {len(response_text)} chars", "info")
        
        # Extract JSON from markdown code blocks more robustly
        import re
        
        # Try to find JSON in code block first
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(1)
        else:
            # Try to find raw JSON object
            json_match = re.search(r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)
        
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
        session.legal_context = analysis.get("legal_context") or ""
        session.relevant_articles = analysis.get("relevant_articles") or []
        session.irrelevant_articles = analysis.get("irrelevant_articles") or []
        
        # Get search queries (new format) or fall back to search_topics (old format)
        search_queries = analysis.get("search_queries") or {}
        if search_queries:
            session.search_queries = search_queries
            primary_queries = search_queries.get("primary_law") or []
            case_queries = search_queries.get("case_law") or []
            if primary_queries:
                session.orchestrator.add_log(f"🔍 Primary law queries: {', '.join(primary_queries[:2])}", "info")
            if case_queries:
                session.orchestrator.add_log(f"🔍 Case law queries: {', '.join(case_queries[:2])}", "info")
        else:
            # Fallback to old format
            session.search_queries = {"primary_law": [], "case_law": []}
            session.search_topics = analysis.get("search_topics") or []
            if session.search_topics:
                session.orchestrator.add_log(f"🔍 Search topics: {', '.join(session.search_topics)}", "info")
        
        # Log results
        session.orchestrator.add_log(f"🌐 Response language: {session.response_language}", "info")
        if session.legal_domain:
            session.orchestrator.add_log(f"⚖️ Legal domain: {session.legal_domain}", "info")
        if session.legal_context:
            session.orchestrator.add_log(f"📚 Context: {session.legal_context}", "info")
        if session.relevant_articles:
            session.orchestrator.add_log(f"✅ Relevant articles: {', '.join(session.relevant_articles[:4])}", "info")
        if session.irrelevant_articles:
            session.orchestrator.add_log(f"❌ AVOID: {', '.join(session.irrelevant_articles[:2])}", "warning")
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
    if session.has_cantonal_scope:
        session.orchestrator.pipeline = [
            {"step": 1, "agent": "Primary Law Agent", "status": "pending"},
            {"step": 2, "agent": "Cantonal Law Agent", "status": "pending"},
            {"step": 3, "agent": "Case Law Agent", "status": "pending"},
            {"step": 4, "agent": "Analysis Agent", "status": "pending"},
        ]
    else:
        session.orchestrator.pipeline = [
            {"step": 1, "agent": "Primary Law Agent", "status": "pending"},
            {"step": 2, "agent": "Case Law Agent", "status": "pending"},
            {"step": 3, "agent": "Analysis Agent", "status": "pending"},
        ]
    
    session.orchestrator.add_log("Pipeline initialized", "start")
    session.orchestrator.add_log(f"Question: {session.question[:100]}...", "info")
    
    if session.has_cantonal_scope:
        session.orchestrator.add_log("→ Including cantonal law search in pipeline", "info")
    
    if session.document_text:
        session.orchestrator.add_log(f"Document loaded: {session.document_name} ({len(session.document_text)} chars)", "info")
    session.orchestrator.add_log(f"LLM: {model_name}", "config")
    yield session
    
    # ========== ORCHESTRATOR: DISPATCH PRIMARY LAW AGENT ==========
    session.orchestrator.current_step = "Dispatching Primary Law Agent"
    session.orchestrator.pipeline[0]["status"] = "running"
    session.orchestrator.add_log("Dispatching Primary Law Agent", "dispatch")
    
    # Build rich context from orchestrator analysis
    orchestrator_context = {
        "question": session.question,
        "language": session.response_language,
        "legal_domain": session.legal_domain,
        "legal_context": session.legal_context,
        "relevant_articles": session.relevant_articles,
        "irrelevant_articles": session.irrelevant_articles,
        "search_queries": session.search_queries.get("primary_law", []) if session.search_queries else [],
        "document": f"{len(session.document_text)} chars" if session.document_text else None,
        "canton": session.canton if session.canton else None
    }
    
    # Track what orchestrator passes to this agent
    session.orchestrator.set_agent_input("Primary Law Agent", orchestrator_context)
    yield session
    
    # ========== PRIMARY LAW AGENT ==========
    agent = session.primary_law_agent
    agent.status = AgentStatus.RUNNING
    agent.current_action = "Starting"
    agent.add_log("Agent activated by orchestrator", "start")
    if session.legal_domain:
        agent.add_log(f"⚖️ Rechtsgebiet: {session.legal_domain}", "info")
    if session.relevant_articles:
        agent.add_log(f"✅ Relevante Artikel: {', '.join(session.relevant_articles[:3])}", "info")
    if session.irrelevant_articles:
        agent.add_log(f"❌ NICHT verwenden: {', '.join(session.irrelevant_articles[:2])}", "warning")
    yield session
    
    # Search
    agent.current_action = "Searching Fedlex/admin.ch"
    
    # Use specific queries from orchestrator if available
    primary_queries = session.search_queries.get("primary_law", []) if session.search_queries else []
    
    if primary_queries:
        # Use orchestrator's specific queries
        search_query = " OR ".join(primary_queries[:3])
        agent.add_log(f"Using orchestrator queries: {len(primary_queries)} specific queries", "search")
        for q in primary_queries[:3]:
            agent.add_log(f"  → {q}", "search")
    elif session.search_topics:
        # Fallback to old format
        topics_str = " ".join(session.search_topics[:3])
        search_query = f"{session.question} {topics_str}"
    else:
        search_query = session.question
    
    agent.search_query = search_query
    agent.add_log(f"Combined query: {search_query[:100]}...", "search")
    yield session
    
    # Run multiple searches for better coverage
    search_start = time.time()
    try:
        agent.add_log("Calling Tavily API (fedlex.admin.ch, admin.ch)", "tool_call")
        yield session
        
        all_results = []
        
        # Primary search with combined query
        main_results = search_swiss_primary_law(search_query)
        all_results.append(main_results)
        
        # If we have specific queries, also search each one individually
        if primary_queries and len(primary_queries) > 1:
            for i, query in enumerate(primary_queries[:2]):  # Max 2 additional searches
                agent.add_log(f"Additional search {i+1}: {query[:50]}...", "tool_call")
                additional = search_swiss_primary_law(query, max_results=3)
                all_results.append(additional)
        
        agent.search_results = "\n\n---\n\n".join(all_results)
        search_duration = (time.time() - search_start) * 1000
        
        agent.add_log(f"Search complete ({search_duration:.0f}ms)", "tool_result")
        agent.add_log(f"Results: {len(agent.search_results)} characters", "tool_result")
        yield session
        
    except Exception as e:
        agent.status = AgentStatus.ERROR
        agent.error = str(e)
        agent.add_log(f"Search failed: {str(e)}", "error")
        session.errors.append(f"Primary Law Search: {str(e)}")
        yield session
    
    # LLM Analysis
    if agent.status != AgentStatus.ERROR:
        agent.current_action = "Analyzing with LLM"
        
        # Use prompts from prompts.py with orchestrator context
        doc_context = ""
        if session.document_text:
            doc_context = f"DOCUMENT TO CONSIDER:\n{session.document_text[:2000]}..."
        
        agent.system_prompt, agent.user_prompt = get_primary_law_prompt(
            search_results=agent.search_results[:4000],
            question=session.question,
            document_context=doc_context,
            legal_domain=session.legal_domain,
            legal_context=session.legal_context,
            relevant_articles=session.relevant_articles,
            irrelevant_articles=session.irrelevant_articles
        )
        agent.llm_prompt = f"SYSTEM:\n{agent.system_prompt}\n\nUSER:\n{agent.user_prompt}"
        
        # Track data received
        agent.data_received = {
            "question": session.question,
            "legal_domain": session.legal_domain,
            "relevant_articles": session.relevant_articles,
            "document": session.document_text[:500] + "..." if session.document_text else None,
            "search_results_length": len(agent.search_results)
        }
        
        agent.add_log(f"Sending to LLM ({model_name})", "llm_call")
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
            response = llm.invoke(messages)
            agent.llm_response = response.content
            llm_duration = (time.time() - llm_start) * 1000
            
            # Track data to send
            agent.data_sent = {
                "analysis": agent.llm_response,
                "length": len(agent.llm_response)
            }
            
            agent.add_log(f"LLM response received ({llm_duration:.0f}ms)", "llm_response")
            agent.add_log(f"Response length: {len(agent.llm_response)} chars", "llm_response")
            yield session
            
        except Exception as e:
            agent.status = AgentStatus.ERROR
            agent.error = str(e)
            agent.add_log(f"LLM call failed: {str(e)}", "error")
            session.errors.append(f"Primary Law LLM: {str(e)}")
            yield session
    
    # Complete
    agent.duration_ms = (time.time() - pipeline_start) * 1000
    if agent.status != AgentStatus.ERROR:
        agent.status = AgentStatus.COMPLETE
        agent.current_action = "Complete"
        agent.add_log("Agent complete", "complete")
        if session.has_cantonal_scope:
            agent.add_log("Result will be passed to → Cantonal Law Agent → Analysis Agent", "handoff")
        else:
            agent.add_log("Result will be passed to → Analysis Agent", "handoff")
    
    session.orchestrator.pipeline[0]["status"] = "complete" if agent.status == AgentStatus.COMPLETE else "error"
    session.orchestrator.add_log(f"Primary Law Agent finished ({agent.duration_ms:.0f}ms)", "complete")
    yield session
    
    # ========== CANTONAL LAW AGENT (if canton detected) ==========
    cantonal_step_offset = 0
    if session.has_cantonal_scope:
        cantonal_step_offset = 1  # Shifts subsequent step indices by 1
        
        session.orchestrator.current_step = "Dispatching Cantonal Law Agent"
        session.orchestrator.pipeline[1]["status"] = "running"
        session.orchestrator.add_log(f"Dispatching Cantonal Law Agent for {session.canton}", "dispatch")
        
        # Track what orchestrator passes to this agent
        canton_display = session.canton_name.get("de", session.canton) if session.canton_name else session.canton
        session.orchestrator.set_agent_input("Cantonal Law Agent", {
            "question": session.question,
            "canton": f"{session.canton} ({canton_display})",
            "commune": session.commune or "None",
            "language": session.response_language
        })
        yield session
        
        cantonal_start = time.time()
        agent = session.cantonal_law_agent
        agent.status = AgentStatus.RUNNING
        agent.current_action = "Starting"
        agent.add_log("Agent activated by orchestrator", "start")
        agent.add_log(f"Searching cantonal law for: {session.canton}", "info")
        if session.commune:
            agent.add_log(f"Including communal regulations for: {session.commune}", "info")
        yield session
        
        # Search cantonal law
        agent.current_action = f"Searching {session.canton} law"
        agent.search_query = f"{session.question} (Canton {session.canton})"
        agent.add_log(f"Preparing cantonal search", "search")
        yield session
        
        search_start = time.time()
        try:
            canton_display = session.canton_name.get("de", session.canton) if session.canton_name else session.canton
            agent.add_log(f"Calling Tavily API (cantonal sources for {canton_display})", "tool_call")
            yield session
            
            # Search cantonal law
            cantonal_results = search_cantonal_law(session.question, session.canton)
            
            # Also search cantonal case law
            cantonal_case_results = search_cantonal_case_law(session.question, session.canton)
            
            # Combine results
            agent.search_results = f"{cantonal_results}\n\n{cantonal_case_results}"
            
            # If commune is specified, also search communal law
            if session.commune:
                communal_results = search_communal_law(session.question, session.commune, session.canton)
                agent.search_results += f"\n\n{communal_results}"
            
            search_duration = (time.time() - search_start) * 1000
            agent.add_log(f"Cantonal search complete ({search_duration:.0f}ms)", "tool_result")
            agent.add_log(f"Results: {len(agent.search_results)} characters", "tool_result")
            yield session
            
        except Exception as e:
            agent.status = AgentStatus.ERROR
            agent.error = str(e)
            agent.add_log(f"Cantonal search failed: {str(e)}", "error")
            session.errors.append(f"Cantonal Law Search: {str(e)}")
            yield session
        
        # LLM Analysis of cantonal law
        if agent.status != AgentStatus.ERROR:
            agent.current_action = "Analyzing cantonal law with LLM"
            
            canton_display = session.canton_name.get("de", session.canton) if session.canton_name else session.canton
            agent.system_prompt = f"""You are a Swiss cantonal law specialist for {canton_display} ({session.canton}).

YOUR TASK:
Analyze the cantonal law search results and identify relevant cantonal provisions.

CITATION REQUIREMENTS:
- Cite cantonal laws with their official designation
- Include links to cantonal law portals where available
- Note the canton clearly in each citation

LANGUAGE RULES:
- Respond in the same language as the user's question
- Use appropriate cantonal terminology

OUTPUT FORMAT:
1. List relevant cantonal provisions
2. Note any cantonal court decisions
3. If communal regulations were found, list those separately
4. Explain how cantonal law relates to federal law on this topic"""

            commune_info = f"\nCommune: {session.commune}" if session.commune else ""
            agent.user_prompt = f"""CANTONAL LAW SEARCH RESULTS FOR {session.canton}:
{agent.search_results[:6000]}

USER QUESTION:
{session.question}

CANTON: {canton_display} ({session.canton}){commune_info}

Please analyze the cantonal law provisions."""

            agent.llm_prompt = f"SYSTEM:\n{agent.system_prompt}\n\nUSER:\n{agent.user_prompt}"
            
            agent.add_log(f"Sending to LLM ({model_name})", "llm_call")
            yield session
            
            llm_start = time.time()
            try:
                from langchain_core.messages import SystemMessage, HumanMessage
                messages = [
                    SystemMessage(content=agent.system_prompt),
                    HumanMessage(content=agent.user_prompt)
                ]
                response = llm.invoke(messages)
                agent.llm_response = response.content
                llm_duration = (time.time() - llm_start) * 1000
                
                agent.add_log(f"LLM response received ({llm_duration:.0f}ms)", "llm_response")
                yield session
                
            except Exception as e:
                agent.status = AgentStatus.ERROR
                agent.error = str(e)
                agent.add_log(f"LLM call failed: {str(e)}", "error")
                session.errors.append(f"Cantonal Law LLM: {str(e)}")
                yield session
        
        # Complete cantonal agent
        agent.duration_ms = (time.time() - cantonal_start) * 1000
        if agent.status != AgentStatus.ERROR:
            agent.status = AgentStatus.COMPLETE
            agent.current_action = "Complete"
            agent.add_log("Agent complete", "complete")
            agent.add_log("Result will be passed to → Analysis Agent", "handoff")
        
        session.orchestrator.pipeline[1]["status"] = "complete" if agent.status == AgentStatus.COMPLETE else "error"
        session.orchestrator.add_log(f"Cantonal Law Agent finished ({agent.duration_ms:.0f}ms)", "complete")
        yield session
    
    # ========== ORCHESTRATOR: DISPATCH CASE LAW AGENT ==========
    case_step_idx = 1 + cantonal_step_offset
    session.orchestrator.current_step = "Dispatching Case Law Agent"
    session.orchestrator.pipeline[case_step_idx]["status"] = "running"
    session.orchestrator.add_log("Dispatching Case Law Agent", "dispatch")
    
    # Build rich context from orchestrator analysis
    case_queries = session.search_queries.get("case_law", []) if session.search_queries else []
    orchestrator_context = {
        "question": session.question,
        "language": session.response_language,
        "legal_domain": session.legal_domain,
        "legal_context": session.legal_context,
        "search_queries": case_queries
    }
    
    # Track what orchestrator passes to this agent
    session.orchestrator.set_agent_input("Case Law Agent", orchestrator_context)
    yield session
    
    # ========== CASE LAW AGENT ==========
    case_start = time.time()
    agent = session.case_law_agent
    agent.status = AgentStatus.RUNNING
    agent.current_action = "Starting"
    agent.add_log("Agent activated by orchestrator", "start")
    if session.legal_domain:
        agent.add_log(f"⚖️ Rechtsgebiet: {session.legal_domain}", "info")
    yield session
    
    # Search
    agent.current_action = "Searching BGer"
    
    # Use specific queries from orchestrator if available
    if case_queries:
        search_query = " OR ".join(case_queries[:3])
        agent.add_log(f"Using orchestrator queries: {len(case_queries)} specific queries", "search")
        for q in case_queries[:3]:
            agent.add_log(f"  → {q}", "search")
    elif session.search_topics:
        topics_str = " ".join(session.search_topics[:3])
        search_query = f"{session.question} {topics_str}"
    else:
        search_query = session.question
    
    agent.search_query = search_query
    agent.add_log(f"Query: {search_query[:100]}...", "search")
    yield session
    
    search_start = time.time()
    try:
        agent.add_log("Calling Tavily API (bger.ch)", "tool_call")
        yield session
        
        all_results = []
        
        # Primary search
        main_results = search_swiss_case_law(search_query)
        all_results.append(main_results)
        
        # Additional searches for specific queries
        if case_queries and len(case_queries) > 1:
            for i, query in enumerate(case_queries[:2]):
                agent.add_log(f"Additional search {i+1}: {query[:50]}...", "tool_call")
                additional = search_swiss_case_law(query, max_results=3)
                all_results.append(additional)
        
        agent.search_results = "\n\n---\n\n".join(all_results)
        search_duration = (time.time() - search_start) * 1000
        
        agent.add_log(f"Search complete ({search_duration:.0f}ms)", "tool_result")
        agent.add_log(f"Results: {len(agent.search_results)} characters", "tool_result")
        yield session
        
    except Exception as e:
        agent.status = AgentStatus.ERROR
        agent.error = str(e)
        agent.add_log(f"Search failed: {str(e)}", "error")
        session.errors.append(f"Case Law Search: {str(e)}")
        yield session
    
    # LLM Analysis
    if agent.status != AgentStatus.ERROR:
        agent.current_action = "Analyzing with LLM"
        
        agent.system_prompt, agent.user_prompt = get_case_law_prompt(
            search_results=agent.search_results[:4000],
            question=session.question,
            legal_domain=session.legal_domain,
            legal_context=session.legal_context
        )
        agent.llm_prompt = f"SYSTEM:\n{agent.system_prompt}\n\nUSER:\n{agent.user_prompt}"
        
        # Track data received
        agent.data_received = {
            "question": session.question,
            "legal_domain": session.legal_domain,
            "search_results_length": len(agent.search_results)
        }
        
        agent.add_log(f"Sending to LLM ({model_name})", "llm_call")
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
            response = llm.invoke(messages)
            agent.llm_response = response.content
            llm_duration = (time.time() - llm_start) * 1000
            
            # Track data to send
            agent.data_sent = {
                "analysis": agent.llm_response,
                "length": len(agent.llm_response)
            }
            
            agent.add_log(f"LLM response received ({llm_duration:.0f}ms)", "llm_response")
            yield session
            
        except Exception as e:
            agent.status = AgentStatus.ERROR
            agent.error = str(e)
            agent.add_log(f"LLM call failed: {str(e)}", "error")
            session.errors.append(f"Case Law LLM: {str(e)}")
            yield session
    
    # Complete
    agent.duration_ms = (time.time() - case_start) * 1000
    if agent.status != AgentStatus.ERROR:
        agent.status = AgentStatus.COMPLETE
        agent.current_action = "Complete"
        agent.add_log("Agent complete", "complete")
        agent.add_log("Result will be passed to → Analysis Agent", "handoff")
    
    session.orchestrator.pipeline[case_step_idx]["status"] = "complete" if agent.status == AgentStatus.COMPLETE else "error"
    session.orchestrator.add_log(f"Case Law Agent finished ({agent.duration_ms:.0f}ms)", "complete")
    yield session
    
    # ========== ORCHESTRATOR: DISPATCH ANALYSIS AGENT ==========
    analysis_step_idx = 2 + cantonal_step_offset
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
    if session.has_cantonal_scope:
        analysis_inputs["cantonal_law_agent"] = f"{len(session.cantonal_law_agent.llm_response or '')} chars ({session.canton})"
        session.orchestrator.add_log("Passing: primary_law, cantonal_law, case_law, question, document", "dispatch")
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
    agent.add_log("📨 Received: Case Law Agent results", "input")
    if session.document_text:
        agent.add_log(f"📨 Received: User document ({len(session.document_text)} chars)", "input")
    yield session
    
    # Build synthesis prompt - include cantonal results if available
    agent.current_action = "Building synthesis prompt"
    
    # Combine all law research
    primary_law_results = session.primary_law_agent.llm_response or "No results available"
    case_law_results = session.case_law_agent.llm_response or "No results available"
    
    # Add cantonal results if available
    cantonal_results = ""
    if session.has_cantonal_scope and session.cantonal_law_agent.llm_response:
        cantonal_results = f"\n\n--- CANTONAL LAW ({session.canton}) ---\n{session.cantonal_law_agent.llm_response}"
    
    # Combine primary and cantonal law for analysis
    combined_primary_law = primary_law_results
    if cantonal_results:
        combined_primary_law = f"{primary_law_results}{cantonal_results}"
    
    # Use prompts from prompts.py with orchestrator context
    agent.system_prompt, agent.user_prompt = get_analysis_prompt(
        primary_law=combined_primary_law,
        case_law=case_law_results,
        question=session.question,
        document_text=session.document_text,
        legal_domain=session.legal_domain,
        legal_context=session.legal_context,
        response_language=session.response_language,
        relevant_articles=session.relevant_articles,
        irrelevant_articles=session.irrelevant_articles
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
    
    agent.add_log(f"Built synthesis prompt ({len(agent.llm_prompt)} chars)", "prepare")
    agent.add_log(f"Includes document: {bool(session.document_text)}", "prepare")
    if session.has_cantonal_scope:
        agent.add_log(f"Includes cantonal law: {session.canton}", "prepare")
    yield session
    
    # LLM Synthesis
    agent.current_action = "Generating final analysis"
    agent.add_log(f"Sending to LLM ({model_name})", "llm_call")
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
        response = llm.invoke(messages)
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
                        "handoff": "➡️"
                    }
                    icon = event_icons.get(log["event_type"], "•")
                    st.text(f"[{log['timestamp']}] {icon} {log['message']}")
            else:
                st.caption("No activity yet")
        
        with search_tab:
            if agent.search_query:
                st.caption("**Query:**")
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
                    st.markdown(agent.llm_response)
        
        with result_tab:
            if agent.status == AgentStatus.COMPLETE and agent.llm_response:
                st.success(f"✅ Produced {len(agent.llm_response)} characters of analysis")
                st.markdown(agent.llm_response)
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
        if llm_provider == "anthropic":
            llm_ok = bool(os.getenv("ANTHROPIC_API_KEY"))
        else:
            llm_ok = bool(os.getenv("OPENAI_API_KEY"))
        st.caption(f"{'✅' if llm_ok else '❌'} {llm_provider.title()} API")
    
    # Main content area
    main_col, agents_col = st.columns([1, 1])
    
    with main_col:
        st.header("📋 Research Output")
        
        output_placeholder = st.empty()
        
        if st.session_state.research_session and st.session_state.research_session.final_output:
            output_placeholder.markdown(st.session_state.research_session.final_output)
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
                st.markdown(st.session_state.research_session.benchmark_output)
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
            
            # Only render cantonal agent if we have cantonal scope
            if session.has_cantonal_scope:
                with cantonal_placeholder.container():
                    render_agent_panel(
                        session.cantonal_law_agent,
                        expanded=session.cantonal_law_agent.status == AgentStatus.RUNNING,
                        handoff_info=cantonal_handoff
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
            
            # Render cantonal agent if scope includes canton
            if updated_session.has_cantonal_scope:
                with cantonal_placeholder.container():
                    render_agent_panel(
                        updated_session.cantonal_law_agent,
                        expanded=updated_session.cantonal_law_agent.status == AgentStatus.RUNNING,
                        handoff_info=cantonal_handoff
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
                output_placeholder.markdown(updated_session.final_output)
            
            time.sleep(0.1)  # Small delay for UI updates
        
        st.session_state.is_running = False
        st.rerun()


if __name__ == "__main__":
    main()

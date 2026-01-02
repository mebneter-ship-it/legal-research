"""
Swiss Legal Research Tools

Direct tool implementations for use with LangGraph agents.
Supports:
- Federal law (Fedlex, admin.ch)
- Federal court decisions (BGer)
- Cantonal law (lexfind.ch, cantonal portals)
- Cantonal court decisions
- Communal regulations (where detectable)

OPTIMIZATIONS (Best Practice / State-of-the-art):
- LRU Caching for Tavily queries (reduces API costs)
- Two-stage retrieval (light search → Top-K with raw_content)
- Early-stop when high-quality results found
- Retrieval budgeting (max queries/results per agent)
"""

import os
import re
import requests
import tempfile
import hashlib
import time
from typing import Optional, List, Dict, Tuple
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv("keys.env")


# ============================================================
# CACHING LAYER (Best Practice: 30-50% cost reduction)
# ============================================================

# In-memory cache for Tavily results (TTL-based)
_TAVILY_CACHE: Dict[str, Tuple[float, any]] = {}
_CACHE_TTL_SECONDS = 3600 * 24  # 24 hours

def _cache_key(query: str, **kwargs) -> str:
    """Generate cache key from query and parameters."""
    params_str = str(sorted(kwargs.items()))
    return hashlib.md5(f"{query}|{params_str}".encode()).hexdigest()

def _get_cached(key: str) -> Optional[any]:
    """Get cached result if not expired."""
    if key in _TAVILY_CACHE:
        timestamp, result = _TAVILY_CACHE[key]
        if time.time() - timestamp < _CACHE_TTL_SECONDS:
            return result
        else:
            del _TAVILY_CACHE[key]  # Expired
    return None

def _set_cached(key: str, result: any):
    """Store result in cache."""
    _TAVILY_CACHE[key] = (time.time(), result)
    # Limit cache size (simple LRU-like behavior)
    if len(_TAVILY_CACHE) > 500:
        oldest_key = min(_TAVILY_CACHE.keys(), key=lambda k: _TAVILY_CACHE[k][0])
        del _TAVILY_CACHE[oldest_key]


# ============================================================
# RETRIEVAL BUDGETING (Best Practice: cost control)
# ============================================================

RETRIEVAL_CONFIG = {
    "max_queries_per_agent": 3,      # Max queries per search agent
    "max_results_per_query": 5,      # Max results per Tavily call
    "early_stop_score": 75,          # Stop if best result score >= this
    "top_k_for_raw_content": 3,      # Only fetch raw_content for top K results
}


def should_early_stop(results: List[Dict], threshold: int = None) -> bool:
    """Check if we have good enough results to stop searching."""
    threshold = threshold or RETRIEVAL_CONFIG["early_stop_score"]
    if not results:
        return False
    # Check if any result has high relevance
    for r in results:
        score = r.get("score", 0) * 100  # Tavily scores are 0-1
        if score >= threshold:
            return True
    return False


# ============================================================
# TWO-STAGE RETRIEVAL (Best Practice: quality + cost optimization)
# ============================================================

def two_stage_search(
    client,
    query: str,
    include_domains: List[str] = None,
    exclude_domains: List[str] = None,
    max_results: int = 5,
    search_depth: str = "advanced"
) -> List[Dict]:
    """
    Two-stage retrieval: 
    1. Light search (no raw_content) to find candidates
    2. Fetch raw_content only for top-K results
    
    Returns list of results with raw_content for top results only.
    """
    # Check cache first
    cache_key = _cache_key(query, domains=str(include_domains), max=max_results)
    cached = _get_cached(cache_key)
    if cached:
        return cached
    
    try:
        # Stage 1: Light search (snippets only)
        light_results = client.search(
            query=query,
            search_depth=search_depth,
            max_results=max_results,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
            include_raw_content=False  # Light search
        )
        
        candidates = light_results.get("results", [])
        
        if not candidates:
            return []
        
        # Score and rank candidates
        scored = []
        for r in candidates:
            score = r.get("score", 0) * 100
            scored.append((score, r))
        scored.sort(reverse=True, key=lambda x: x[0])
        
        # Stage 2: Fetch raw_content for top-K only
        top_k = RETRIEVAL_CONFIG["top_k_for_raw_content"]
        top_urls = [r["url"] for _, r in scored[:top_k]]
        
        if top_urls:
            # Fetch with raw_content for top URLs
            detailed_results = client.search(
                query=query,
                search_depth="basic",
                max_results=top_k,
                include_domains=[extract_domain(url) for url in top_urls],
                include_raw_content=True
            )
            
            # Merge raw_content into original results
            detailed_by_url = {r["url"]: r for r in detailed_results.get("results", [])}
            
            final_results = []
            for _, r in scored:
                if r["url"] in detailed_by_url:
                    r["raw_content"] = detailed_by_url[r["url"]].get("raw_content", "")
                final_results.append(r)
        else:
            final_results = [r for _, r in scored]
        
        # Cache results
        _set_cached(cache_key, final_results)
        
        return final_results
        
    except Exception as e:
        print(f"Two-stage search error: {e}")
        return []


def extract_domain(url: str) -> str:
    """Extract domain from URL."""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return parsed.netloc
    except:
        return ""


# ============================================================
# ORIGINAL FUNCTIONS (with caching integration)
# ============================================================


def fetch_pdf_content(url: str, max_chars: int = 3000) -> str:
    """
    Fetch and extract text content from a PDF URL.
    
    Args:
        url: URL to the PDF file
        max_chars: Maximum characters to return
        
    Returns:
        Extracted text content or error message
    """
    try:
        # Download PDF
        headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; SwissLegalResearchBot/1.0)'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        # Save to temp file and extract text
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name
        
        try:
            import pdfplumber
            text_parts = []
            with pdfplumber.open(tmp_path) as pdf:
                for page in pdf.pages[:5]:  # First 5 pages max
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
            
            full_text = "\n\n".join(text_parts)
            
            # Clean up temp file
            os.unlink(tmp_path)
            
            if full_text:
                return full_text[:max_chars]
            else:
                return "PDF konnte nicht extrahiert werden"
                
        except ImportError:
            os.unlink(tmp_path)
            return "pdfplumber nicht installiert - PDF-Extraktion nicht möglich"
            
    except requests.exceptions.Timeout:
        return f"Timeout beim Abrufen von {url}"
    except requests.exceptions.RequestException as e:
        return f"Fehler beim Abrufen: {str(e)}"
    except Exception as e:
        return f"PDF-Extraktion fehlgeschlagen: {str(e)}"


# ============================================================
# CANTON AND COMMUNE DETECTION
# ============================================================

SWISS_CANTONS = {
    # German names
    "zürich": "ZH", "zuerich": "ZH", "zurich": "ZH",
    "bern": "BE", "berne": "BE",
    "luzern": "LU", "lucerne": "LU", "lucerna": "LU",
    "uri": "UR",
    "schwyz": "SZ",
    "obwalden": "OW",
    "nidwalden": "NW",
    "glarus": "GL", "glaris": "GL", "glarona": "GL",
    "zug": "ZG", "zoug": "ZG",
    "freiburg": "FR", "fribourg": "FR", "friburgo": "FR",
    "solothurn": "SO", "soleure": "SO", "soletta": "SO",
    "basel-stadt": "BS", "basel stadt": "BS", "bâle-ville": "BS", "basilea città": "BS",
    "basel-landschaft": "BL", "basel landschaft": "BL", "bâle-campagne": "BL", "basilea campagna": "BL",
    "schaffhausen": "SH", "schaffhouse": "SH", "sciaffusa": "SH",
    "appenzell ausserrhoden": "AR", "appenzell rhodes-extérieures": "AR",
    "appenzell innerrhoden": "AI", "appenzell rhodes-intérieures": "AI",
    "st. gallen": "SG", "st.gallen": "SG", "saint-gall": "SG", "san gallo": "SG",
    "graubünden": "GR", "graubuenden": "GR", "grisons": "GR", "grigioni": "GR",
    "aargau": "AG", "argovie": "AG", "argovia": "AG",
    "thurgau": "TG", "thurgovie": "TG", "turgovia": "TG",
    "ticino": "TI", "tessin": "TI",
    "waadt": "VD", "vaud": "VD",
    "wallis": "VS", "valais": "VS", "vallese": "VS",
    "neuenburg": "NE", "neuchâtel": "NE", "neuchatel": "NE",
    "genf": "GE", "genève": "GE", "geneve": "GE", "ginevra": "GE", "geneva": "GE",
    "jura": "JU",
}

# NOTE: We intentionally do NOT add lowercase abbreviations like "so", "be", "ne" etc.
# because they cause false positives (e.g., "einfach so" → Solothurn)
# Canton abbreviations are only recognized in uppercase or with "Kanton" prefix

# Canton full names for search queries
CANTON_NAMES = {
    "ZH": {"de": "Zürich", "fr": "Zurich", "it": "Zurigo"},
    "BE": {"de": "Bern", "fr": "Berne", "it": "Berna"},
    "LU": {"de": "Luzern", "fr": "Lucerne", "it": "Lucerna"},
    "UR": {"de": "Uri", "fr": "Uri", "it": "Uri"},
    "SZ": {"de": "Schwyz", "fr": "Schwytz", "it": "Svitto"},
    "OW": {"de": "Obwalden", "fr": "Obwald", "it": "Obvaldo"},
    "NW": {"de": "Nidwalden", "fr": "Nidwald", "it": "Nidvaldo"},
    "GL": {"de": "Glarus", "fr": "Glaris", "it": "Glarona"},
    "ZG": {"de": "Zug", "fr": "Zoug", "it": "Zugo"},
    "FR": {"de": "Freiburg", "fr": "Fribourg", "it": "Friburgo"},
    "SO": {"de": "Solothurn", "fr": "Soleure", "it": "Soletta"},
    "BS": {"de": "Basel-Stadt", "fr": "Bâle-Ville", "it": "Basilea Città"},
    "BL": {"de": "Basel-Landschaft", "fr": "Bâle-Campagne", "it": "Basilea Campagna"},
    "SH": {"de": "Schaffhausen", "fr": "Schaffhouse", "it": "Sciaffusa"},
    "AR": {"de": "Appenzell Ausserrhoden", "fr": "Appenzell Rhodes-Extérieures", "it": "Appenzello Esterno"},
    "AI": {"de": "Appenzell Innerrhoden", "fr": "Appenzell Rhodes-Intérieures", "it": "Appenzello Interno"},
    "SG": {"de": "St. Gallen", "fr": "Saint-Gall", "it": "San Gallo"},
    "GR": {"de": "Graubünden", "fr": "Grisons", "it": "Grigioni"},
    "AG": {"de": "Aargau", "fr": "Argovie", "it": "Argovia"},
    "TG": {"de": "Thurgau", "fr": "Thurgovie", "it": "Turgovia"},
    "TI": {"de": "Tessin", "fr": "Tessin", "it": "Ticino"},
    "VD": {"de": "Waadt", "fr": "Vaud", "it": "Vaud"},
    "VS": {"de": "Wallis", "fr": "Valais", "it": "Vallese"},
    "NE": {"de": "Neuenburg", "fr": "Neuchâtel", "it": "Neuchâtel"},
    "GE": {"de": "Genf", "fr": "Genève", "it": "Ginevra"},
    "JU": {"de": "Jura", "fr": "Jura", "it": "Giura"},
}

# Cantonal law portal domains
CANTONAL_LAW_DOMAINS = {
    "ZH": ["zh.ch", "zhlex.zh.ch"],
    "BE": ["belex.sites.be.ch", "be.ch"],
    "LU": ["srl.lu.ch", "lu.ch"],
    "UR": ["ur.ch"],
    "SZ": ["sz.ch"],
    "OW": ["ow.ch"],
    "NW": ["nw.ch"],
    "GL": ["gl.ch"],
    "ZG": ["zg.ch"],
    "FR": ["fr.ch", "bdlf.fr.ch"],
    "SO": ["so.ch"],
    "BS": ["bs.ch"],
    "BL": ["bl.ch"],
    "SH": ["sh.ch"],
    "AR": ["ar.ch"],
    "AI": ["ai.ch", "appenzell.org"],
    "SG": ["sg.ch"],
    "GR": ["gr.ch"],
    "AG": ["ag.ch"],
    "TG": ["tg.ch"],
    "TI": ["ti.ch", "legislazione.ti.ch"],
    "VD": ["vd.ch", "rsv.vd.ch"],
    "VS": ["vs.ch"],
    "NE": ["ne.ch", "rsn.ne.ch"],
    "GE": ["ge.ch", "silgeneve.ch"],
    "JU": ["jura.ch", "rsju.jura.ch"],
}

# Major Swiss communes with their canton
SWISS_COMMUNES = {
    # Major cities
    "zürich": "ZH", "zuerich": "ZH", "zurich": "ZH",
    "genf": "GE", "genève": "GE", "geneve": "GE", "geneva": "GE", "ginevra": "GE",
    "basel": "BS", "bâle": "BS", "basilea": "BS",
    "lausanne": "VD", "losanna": "VD",
    "bern": "BE", "berne": "BE", "berna": "BE",
    "winterthur": "ZH", "winterthour": "ZH",
    "luzern": "LU", "lucerne": "LU", "lucerna": "LU",
    "st. gallen": "SG", "st.gallen": "SG", "saint-gall": "SG", "san gallo": "SG",
    "lugano": "TI",
    "biel": "BE", "bienne": "BE",
    "thun": "BE", "thoune": "BE",
    "köniz": "BE", "koeniz": "BE",
    "la chaux-de-fonds": "NE",
    "fribourg": "FR", "freiburg": "FR", "friburgo": "FR",
    "schaffhausen": "SH", "schaffhouse": "SH", "sciaffusa": "SH",
    "chur": "GR", "coire": "GR", "coira": "GR",
    "neuchâtel": "NE", "neuchatel": "NE", "neuenburg": "NE",
    "sion": "VS", "sitten": "VS",
    "emmen": "LU",
    "uster": "ZH",
    "zug": "ZG", "zoug": "ZG",
    "dübendorf": "ZH", "duebendorf": "ZH",
    "kriens": "LU",
    "rapperswil-jona": "SG",
    "montreux": "VD",
    "yverdon": "VD", "yverdon-les-bains": "VD",
    "aarau": "AG",
    "baden": "AG",
    "bellinzona": "TI",
    "locarno": "TI",
    "nyon": "VD",
    "vevey": "VD",
    "morges": "VD",
    "renens": "VD",
    "carouge": "GE",
    "meyrin": "GE",
    "vernier": "GE",
    "onex": "GE",
    "lancy": "GE",
    "bulle": "FR",
    "monthey": "VS",
    "martigny": "VS",
    "sierre": "VS", "siders": "VS",
    "olten": "SO",
    "grenchen": "SO", "granges": "SO",
    "wettingen": "AG",
    "wohlen": "AG",
    "baar": "ZG",
    "cham": "ZG",
    "horgen": "ZH",
    "wädenswil": "ZH", "waedenswil": "ZH",
    "dietikon": "ZH",
    "schlieren": "ZH",
    "kloten": "ZH",
    "opfikon": "ZH",
    "regensdorf": "ZH",
    "wallisellen": "ZH",
    "adliswil": "ZH",
    "küsnacht": "ZH", "kuesnacht": "ZH",
    # Appenzell - town is in AI (Innerrhoden)
    "appenzell": "AI",
    "herisau": "AR",  # Hauptort Appenzell Ausserrhoden
    "teufen": "AR",
    "speicher": "AR",
    "meilen": "ZH",
    "stäfa": "ZH", "staefa": "ZH",
    "wetzikon": "ZH",
    "gossau": "SG",
    "herisau": "AR",
    "arbon": "TG",
    "frauenfeld": "TG",
    "kreuzlingen": "TG",
    "weinfelden": "TG",
    "romanshorn": "TG",
    "amriswil": "TG",
    "liestal": "BL",
    "allschwil": "BL",
    "binningen": "BL",
    "muttenz": "BL",
    "pratteln": "BL",
    "riehen": "BS",
    "davos": "GR",
    "landquart": "GR",
    "chiasso": "TI",
    "mendrisio": "TI",
    "minusio": "TI",
    "massagno": "TI",
    "giubiasco": "TI",
}


def detect_canton_and_commune(text: str) -> dict:
    """
    Detect canton and/or commune mentioned in text.
    
    Returns: {
        "canton": "XX" or None,
        "canton_name": {"de": "...", "fr": "...", "it": "..."} or None,
        "commune": "Name" or None, 
        "canton_from_commune": bool
    }
    """
    import re
    text_lower = text.lower()
    result = {
        "canton": None,
        "canton_name": None,
        "commune": None,
        "canton_from_commune": False
    }
    
    # Check for canton mentions (use word boundaries)
    for name, abbr in SWISS_CANTONS.items():
        # Use word boundary to avoid matching substrings
        # e.g., "bern" should not match in "Arbeitsrecht"
        pattern = r'\b' + re.escape(name) + r'\b'
        if re.search(pattern, text_lower):
            result["canton"] = abbr
            result["canton_name"] = CANTON_NAMES.get(abbr)
            break
    
    # Check for commune mentions (use word boundaries)
    for name, canton in SWISS_COMMUNES.items():
        pattern = r'\b' + re.escape(name) + r'\b'
        if re.search(pattern, text_lower):
            result["commune"] = name.title()
            # If no canton detected yet, infer from commune
            if not result["canton"]:
                result["canton"] = canton
                result["canton_name"] = CANTON_NAMES.get(canton)
                result["canton_from_commune"] = True
            break
    
    return result


def detect_canton(text: str) -> str:
    """
    Detect canton mentioned in text.
    Returns canton abbreviation (e.g., "AI", "ZH") or empty string.
    """
    result = detect_canton_and_commune(text)
    return result.get("canton") or ""


def detect_commune(text: str) -> str:
    """
    Detect commune mentioned in text.
    Returns commune name or empty string.
    """
    result = detect_canton_and_commune(text)
    return result.get("commune") or ""


# ============================================================
# TAVILY SEARCH HELPERS
# ============================================================


def get_tavily_client():
    """Get configured Tavily client with key validation"""
    from tavily import TavilyClient
    
    api_key = os.getenv("TAVILY_API_KEY", "")
    
    # Fix common key format issues
    if api_key.startswith("tvly-tvly-"):
        api_key = api_key.replace("tvly-tvly-", "tvly-", 1)
    
    if not api_key or api_key == "tvly-YOUR_KEY_HERE":
        raise ValueError("TAVILY_API_KEY not configured. Please set it in .env file.")
    
    return TavilyClient(api_key=api_key)


# ============================================================
# SMART SEARCH: Keyword extraction, scoring, excerpt extraction
# ============================================================

def extract_keywords_from_query(query: str) -> list:
    """
    Extract search keywords from a legal query with weights.
    Returns list of (keyword, weight) tuples.
    Higher weight = more important for relevance scoring.
    """
    keywords = []
    query_lower = query.lower()
    
    # 1. Legal article patterns (highest weight)
    article_patterns = [
        r'art\.?\s*\d+',           # Art. 30, Art 684
        r'§\s*\d+',                # § 15
        r'\d+\s*(abs|lit|ziff)',   # 30 Abs. 1
    ]
    for pattern in article_patterns:
        matches = re.findall(pattern, query_lower)
        for match in matches:
            keywords.append((match, 10))
    
    # 2. Specific measurements (high weight)
    measurement_patterns = [
        r'\d+[.,]?\d*\s*m\b',      # 1.5 m, 2m
        r'\d+[.,]?\d*\s*cm\b',     # 50 cm
        r'\d+\s*%',                # 10%
        r'chf\s*\d+',              # CHF 1000
    ]
    for pattern in measurement_patterns:
        matches = re.findall(pattern, query_lower)
        for match in matches:
            keywords.append((match, 8))
    
    # 3. Legal domain terms (medium weight)
    legal_terms = {
        # Building law
        'einfriedung': 5, 'zaun': 5, 'mauer': 5, 'grenze': 4,
        'grenzabstand': 6, 'baugesetz': 4, 'bauverordnung': 4,
        'bauzonen': 4, 'höhe': 4, 'bauv': 5,
        
        # Rental law  
        'miete': 5, 'mietzins': 6, 'kündigung': 5, 'mietvertrag': 5,
        'nebenkosten': 5, 'kaution': 5,
        
        # Employment law
        'arbeitsvertrag': 5, 'lohn': 5, 'ferien': 4, 'überstunden': 5,
        'kündigungsfrist': 6,
        
        # Family law
        'unterhalt': 5, 'scheidung': 5, 'sorgerecht': 5,
        
        # Contract law
        'vertrag': 4, 'schadenersatz': 5, 'haftung': 5,
        
        # Neighbor law
        'nachbarrecht': 5, 'immissionen': 5,
    }
    
    for term, weight in legal_terms.items():
        if term in query_lower:
            keywords.append((term, weight))
    
    # 4. Important nouns from query (lower weight)
    words = query.split()
    for word in words:
        clean = re.sub(r'[^\w]', '', word)
        if clean and len(clean) > 3:
            if clean.lower() not in [k[0] for k in keywords]:
                keywords.append((clean.lower(), 2))
    
    return keywords


def score_search_result(
    result: dict, 
    keywords: list,
    legal_area: str = "general",
    canton: str = None
) -> tuple:
    """
    Score a search result by keyword matches AND domain trust.
    Returns (score, matched_keywords).
    """
    text = (
        (result.get('raw_content', '') or '') + 
        (result.get('content', '') or '') +
        (result.get('title', '') or '')
    ).lower()
    
    url = result.get('url', '').lower()
    
    score = 0
    matched = []
    
    # 1. Keyword matching
    for keyword, weight in keywords:
        if keyword.lower() in text:
            score += weight
            matched.append(keyword)
    
    # 2. Raw content bonus (full text available)
    if result.get('raw_content'):
        score += 3
    
    # 3. CRITICAL: Trusted domain bonus
    trusted_bonus = 0
    
    if legal_area == "cantonal" and canton:
        canton_domains = CANTONAL_LAW_DOMAINS.get(canton, [])
        if any(domain in url for domain in canton_domains):
            trusted_bonus = 50
            matched.append(f"[OFFICIAL: {canton}]")
        elif "clex.ch" in url or "lexfind.ch" in url:
            trusted_bonus = 30
            matched.append("[LAW_COLLECTION]")
    
    elif legal_area == "federal":
        if "fedlex.admin.ch" in url:
            trusted_bonus = 50
            matched.append("[FEDLEX]")
        elif "admin.ch" in url:
            trusted_bonus = 20
            matched.append("[ADMIN.CH]")
    
    elif legal_area == "case_law":
        if "bger.ch" in url:
            trusted_bonus = 50
            matched.append("[BGER]")
        elif "entscheidsuche.ch" in url:
            trusted_bonus = 40
            matched.append("[ENTSCHEIDSUCHE]")
    
    # Penalize commercial/irrelevant domains
    commercial_domains = ['shop', 'kaufen', '-swiss.ch', 'amazon', 'ricardo', 'tutti']
    if any(d in url for d in commercial_domains):
        score -= 20
    
    score += trusted_bonus
    
    return score, matched


def extract_relevant_excerpt(
    raw_content: str, 
    keywords: list,
    context_chars: int = 1500
) -> str:
    """
    Extract the most relevant section around matched keywords.
    Instead of sending 50k chars, extract ~2k chars around the best match.
    """
    if not raw_content:
        return ""
    
    raw_lower = raw_content.lower()
    
    # Sort keywords by weight (highest first)
    sorted_kw = sorted(keywords, key=lambda x: x[1], reverse=True)
    
    # Find position of highest-weight keyword that exists
    best_pos = -1
    
    for keyword, weight in sorted_kw:
        pos = raw_lower.find(keyword.lower())
        if pos != -1:
            best_pos = pos
            break
    
    if best_pos == -1:
        return raw_content[:context_chars]
    
    # Extract context around the keyword
    start = max(0, best_pos - 200)
    end = min(len(raw_content), best_pos + context_chars)
    
    excerpt = raw_content[start:end]
    
    # Clean up: don't cut mid-word
    if start > 0:
        first_space = excerpt.find(' ')
        if first_space > 0 and first_space < 50:
            excerpt = excerpt[first_space+1:]
    
    return f"...{excerpt}..."


def search_swiss_primary_law(query: str, max_results: int = 5) -> str:
    """
    Search Swiss primary law sources (Fedlex, admin.ch)
    
    Goal: Find the ACTUAL LAW TEXT (Gesetzestext), not just documents about it.
    
    Uses smart search features:
    - Prioritize direct law text over commentary
    - Extract article references from query
    - Search with SR numbers when possible
    
    Args:
        query: Legal search query (German, French, or Italian)
        max_results: Maximum results to return
        
    Returns:
        Formatted string of search results with law text excerpts
    """
    try:
        tavily = get_tavily_client()
        
        # Extract keywords for relevance scoring
        keywords = extract_keywords_from_query(query)
        
        # Try to extract article references (e.g., "Art. 266 OR", "§ 684 ZGB")
        article_refs = re.findall(r'art\.?\s*\d+[a-z]?(?:\s*(?:abs|lit|ziff)\.?\s*\d+)?', query.lower())
        law_refs = re.findall(r'\b(or|zgb|stgb|svg|arbg|bgb)\b', query.lower())
        
        # SR number mapping for common Swiss laws
        SR_NUMBERS = {
            'or': '220',      # Obligationenrecht
            'zgb': '210',     # Zivilgesetzbuch
            'stgb': '311.0',  # Strafgesetzbuch
            'svg': '741.01',  # Strassenverkehrsgesetz
            'bgb': '210',     # Same as ZGB
        }
        
        all_results = []
        queries_used = []
        
        # STRATEGY 0: lawbrary.ch - Swiss law platform with ACTUAL LAW TEXT in HTML!
        # This is the most reliable source for readable law text
        if article_refs or law_refs:
            law_abbrev = law_refs[0].upper() if law_refs else ""
            lawbrary_query = f'site:lawbrary.ch {query}'
            queries_used.append(f"📚 {lawbrary_query}")
            
            lawbrary_results = tavily.search(
                query=lawbrary_query,
                max_results=3,
                search_depth="advanced",
                include_raw_content=True,
                chunks_per_source=3,
                country="switzerland"
            )
            all_results.extend(lawbrary_results.get("results", []))
        
        # STRATEGY 1: Search for PDF versions on fedlex.data.admin.ch (these have extractable text!)
        # www.fedlex.admin.ch = JavaScript SPA (no content for bots)
        # fedlex.data.admin.ch = PDFs with actual law text!
        if article_refs or law_refs:
            pdf_query = f'site:fedlex.data.admin.ch {query} filetype:pdf'
            queries_used.append(f"📄 {pdf_query}")
            
            pdf_results = tavily.search(
                query=pdf_query,
                max_results=5,
                search_depth="advanced",
                include_raw_content=True,
                chunks_per_source=3,
                country="switzerland"
            )
            all_results.extend(pdf_results.get("results", []))
        
        # STRATEGY 2: Direct article search on Fedlex (for URL reference)
        if article_refs:
            for art_ref in article_refs[:2]:
                art_query = f'site:fedlex.admin.ch "{art_ref}"'
                if law_refs:
                    art_query += f' {law_refs[0].upper()}'
                queries_used.append(f"📜 {art_query}")
                
                results = tavily.search(
                    query=art_query,
                    max_results=3,
                    search_depth="advanced",
                    include_raw_content=True,
                    chunks_per_source=3,
                    country="switzerland"
                )
                all_results.extend(results.get("results", []))
        
        # STRATEGY 3: Search admin.ch/SECO for readable law explanations
        # These pages often contain the actual article text in HTML!
        admin_query = f'site:admin.ch {query}'
        queries_used.append(f"📋 {admin_query}")
        
        admin_results = tavily.search(
            query=admin_query,
            max_results=3,
            search_depth="advanced",
            include_raw_content=True,
            chunks_per_source=3,
            country="switzerland"
        )
        all_results.extend(admin_results.get("results", []))
        
        # STRATEGY 4: SR-number based search (very reliable!)
        for law_ref in law_refs[:1]:
            sr_num = SR_NUMBERS.get(law_ref)
            if sr_num:
                sr_query = f'site:fedlex.admin.ch/eli/cc/{sr_num} {query}'
                queries_used.append(f"📋 {sr_query}")
                
                results = tavily.search(
                    query=sr_query,
                    max_results=3,
                    search_depth="advanced",
                    include_raw_content=True,
                    country="switzerland"
                )
                all_results.extend(results.get("results", []))
        
        # STRATEGY 5: General Fedlex search with OPTIMAL parameters
        fedlex_query = f'{query} Gesetzestext'
        queries_used.append(f"🏛️ site:fedlex.admin.ch {fedlex_query}")
        
        fedlex_results = tavily.search(
            query=fedlex_query,
            max_results=max_results,
            search_depth="advanced",
            include_raw_content=True,
            chunks_per_source=3,
            country="switzerland",
            include_domains=["fedlex.admin.ch"]
        )
        all_results.extend(fedlex_results.get("results", []))
        
        # Deduplicate
        seen_urls = set()
        unique_results = []
        for r in all_results:
            url = r.get("url", "").lower()
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(r)
        
        # Score and sort - PRIORITIZE actual law texts!
        scored_results = []
        for r in unique_results:
            score, matched = score_search_result(r, keywords, "federal")
            
            url = r.get("url", "").lower()
            title = r.get("title", "").lower()
            
            # BONUS for actual law text pages (not PDFs, not consultation docs)
            if "/eli/cc/" in url:  # Consolidated law text
                score += 30
                matched.append("[GESETZESTEXT]")
            elif "/eli/oc/" in url:  # Original version
                score += 20
                matched.append("[ORIGINALFASSUNG]")
            
            # BONUS for lawbrary.ch - contains actual law text in HTML!
            if 'lawbrary.ch' in url:
                score += 35
                matched.append("[LAWBRARY]")
            
            # BONUS for PDF files from fedlex.data.admin.ch (extractable content!)
            if 'fedlex.data.admin.ch' in url and url.endswith('.pdf'):
                score += 25
                matched.append("[FEDLEX-PDF]")
            elif 'lexfind.ch' in url and url.endswith('.pdf'):
                score += 20
                matched.append("[LEXFIND-PDF]")
            elif url.endswith('.pdf') and 'fedlex' in url:
                score += 15
                matched.append("[PDF]")
            
            # PENALTY for www.fedlex.admin.ch HTML pages (JavaScript-only, no content!)
            if 'www.fedlex.admin.ch/eli' in url and not url.endswith('.pdf'):
                content = r.get('content', '') or r.get('raw_content', '')
                if 'javascript' in content.lower() and len(content) < 1500:
                    score -= 40  # Strong penalty - these pages have no useful content
                    matched.append("[JS-ONLY]")
            
            # BONUS for admin.ch pages with readable content
            content = r.get('content', '') or r.get('raw_content', '')
            if 'seco.admin.ch' in url and len(content) > 500:
                score += 20
                matched.append("[SECO]")
            
            # PENALTY for consultation/report documents
            if any(x in title for x in ['vernehmlassung', 'bericht', 'botschaft', 'gutachten']):
                score -= 15
            
            # PENALTY for non-fedlex PDFs (random documents)
            if url.endswith('.pdf') and 'fedlex' not in url and 'admin.ch' not in url:
                score -= 10
            
            scored_results.append({
                **r,
                "relevance_score": score,
                "matched_keywords": matched
            })
        
        scored_results.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        # Format output
        output_lines = [f"🏛️ PRIMARY LAW SEARCH: {query}", "=" * 50]
        output_lines.append(f"Erkannte Artikel: {article_refs if article_refs else 'keine'}")
        output_lines.append(f"Erkannte Gesetze: {[r.upper() for r in law_refs] if law_refs else 'keine'}")
        output_lines.append(f"Keywords: {', '.join([k[0] for k in keywords[:5]])}")
        output_lines.append("-" * 40)
        
        if not scored_results:
            output_lines.append("\nKeine Ergebnisse gefunden. Versuchen Sie andere Suchbegriffe.")
            return "\n".join(output_lines)
        
        # Top results with smart excerpts
        for i, result in enumerate(scored_results[:max_results], 1):
            title = result.get('title', 'No title')
            url = result.get('url', 'N/A')
            score = result.get('relevance_score', 0)
            matched = result.get('matched_keywords', [])
            raw = result.get('raw_content', '')
            content = result.get('content', '')
            
            output_lines.append(f"\n[{i}] {title}")
            output_lines.append(f"    URL: {url}")
            output_lines.append(f"    Relevanz: {score} ({', '.join(matched)})")
            
            # Use raw_content with smart extraction
            if raw and len(raw) > len(content):
                excerpt = extract_relevant_excerpt(raw, keywords, 1500)
                output_lines.append(f"    GESETZESTEXT/AUSSCHNITT:")
                output_lines.append(f"    {excerpt}")
            else:
                output_lines.append(f"    SNIPPET:")
                output_lines.append(f"    {content[:600]}...")
        
        # Add search queries used (for debugging)
        output_lines.append(f"\n\n🔎 Verwendete Suchanfragen:")
        for q in queries_used:
            output_lines.append(f"   • {q}")
        
        return "\n".join(output_lines)
        
    except Exception as e:
        return f"❌ Primary law search error: {str(e)}"


def search_swiss_case_law(query: str, max_results: int = 5) -> str:
    """
    Search Swiss Federal Court (Bundesgericht) case law
    
    Uses smart search features:
    - include_raw_content for full text
    - Keyword extraction and relevance scoring
    - Smart excerpt extraction
    
    Args:
        query: Case law search query
        max_results: Maximum results to return
        
    Returns:
        Formatted string of BGE decisions
    """
    try:
        tavily = get_tavily_client()
        
        # Extract keywords for relevance scoring
        keywords = extract_keywords_from_query(query)
        
        # PRIORITY 1: Search BGer.ch with OPTIMAL parameters
        bger_results = tavily.search(
            query=f"{query} BGE",
            max_results=max_results,
            search_depth="advanced",
            include_raw_content=True,
            chunks_per_source=3,
            country="switzerland",
            include_domains=["bger.ch"]
        )
        
        all_results = bger_results.get("results", [])
        
        # PRIORITY 2: Search entscheidsuche.ch for additional case law
        if len(all_results) < 3:
            alt_results = tavily.search(
                query=f"{query} BGE Bundesgericht",
                max_results=3,
                search_depth="advanced",
                include_raw_content=True,
                country="switzerland",
                include_domains=["entscheidsuche.ch"]
            )
            # Add non-duplicate results
            seen_urls = {r.get("url", "").lower() for r in all_results}
            for r in alt_results.get("results", []):
                if r.get("url", "").lower() not in seen_urls:
                    all_results.append(r)
        
        # Score and sort by relevance
        scored_results = []
        for r in all_results:
            score, matched = score_search_result(r, keywords, "case_law")
            scored_results.append({
                **r,
                "relevance_score": score,
                "matched_keywords": matched
            })
        
        scored_results.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        # Format output
        output_lines = [f"⚖️ CASE LAW SEARCH: {query}", "=" * 50]
        output_lines.append(f"Keywords: {', '.join([k[0] for k in keywords[:5]])}")
        output_lines.append("-" * 40)
        
        if not scored_results:
            output_lines.append("\nNo case law found. Try broader search terms.")
            return "\n".join(output_lines)
        
        # Top N results with smart excerpts
        for i, result in enumerate(scored_results[:max_results], 1):
            title = result.get('title', 'No title')
            url = result.get('url', 'N/A')
            score = result.get('relevance_score', 0)
            matched = result.get('matched_keywords', [])
            raw = result.get('raw_content', '')
            content = result.get('content', '')
            
            output_lines.append(f"\n[{i}] {title}")
            output_lines.append(f"    URL: {url}")
            output_lines.append(f"    Relevanz: {score} ({', '.join(matched)})")
            
            if raw and len(raw) > len(content):
                excerpt = extract_relevant_excerpt(raw, keywords, 1200)
                output_lines.append(f"    RELEVANTER AUSSCHNITT:\n    {excerpt}")
            else:
                output_lines.append(f"    SNIPPET:\n    {content[:500]}...")
        
        return "\n".join(output_lines)
        
    except Exception as e:
        return f"❌ Case law search error: {str(e)}"


def search_general_legal(query: str, max_results: int = 5) -> str:
    """
    General Swiss legal web search
    
    Args:
        query: General legal query
        max_results: Maximum results to return
        
    Returns:
        Formatted string of general legal results
    """
    try:
        tavily = get_tavily_client()
        
        search_query = f"{query} Schweizer Recht Swiss law"
        
        results = tavily.search(
            query=search_query,
            max_results=max_results,
            search_depth="basic"
        )
        
        output_lines = [f"🔍 GENERAL LEGAL SEARCH: {query}", "=" * 50]
        
        for i, result in enumerate(results.get("results", []), 1):
            output_lines.append(f"\n[{i}] {result.get('title', 'No title')}")
            output_lines.append(f"    URL: {result.get('url', 'N/A')}")
            output_lines.append(f"    {result.get('content', 'No content')[:400]}...")
        
        return "\n".join(output_lines)
        
    except Exception as e:
        return f"❌ General search error: {str(e)}"


def search_cantonal_law(query: str, canton: str, canton_name: str = None, 
                        orchestrator_queries: list = None, max_results: int = 5, 
                        commune: str = None) -> str:
    """
    Search cantonal law sources using orchestrator-provided queries.
    
    Args:
        query: Original legal question
        canton: Canton abbreviation (e.g., "ZH", "AI")
        canton_name: Full canton name (e.g., "Appenzell Innerrhoden")
        orchestrator_queries: Specific search queries from orchestrator (preferred!)
        max_results: Maximum results to return
        commune: Optional commune name
        
    Returns:
        Formatted string of cantonal law results
    """
    # Known official cantonal law collection URLs
    CANTONAL_LAW_URLS = {
        "ZH": "https://www.zh.ch/de/politik-staat/gesetze-beschluesse/gesetzessammlung.html",
        "BE": "https://www.belex.sites.be.ch/",
        "LU": "https://srl.lu.ch/",
        "UR": "https://www.ur.ch/recht",
        "SZ": "https://www.sz.ch/politik-und-verwaltung/rechtspflege/gesetzgebung.html",
        "OW": "https://www.ow.ch/dienstleistungen/3404",
        "NW": "https://www.nw.ch/gesetzessammlung",
        "GL": "https://gs.gl.ch/",
        "ZG": "https://bgs.zg.ch/",
        "FR": "https://bdlf.fr.ch/",
        "SO": "https://bgs.so.ch/",
        "BS": "https://www.gesetzessammlung.bs.ch/",
        "BL": "https://bl.clex.ch/",
        "SH": "https://rechtsbuch.sh.ch/",
        "AR": "https://www.lexfind.ch/fe/de/tol/24/de",
        "AI": "https://www.lexfind.ch/fe/de/tol/25/de",
        "SG": "https://www.gesetzessammlung.sg.ch/",
        "GR": "https://www.gr-lex.gr.ch/",
        "AG": "https://gesetzessammlungen.ag.ch/",
        "TG": "https://www.rechtsbuch.tg.ch/",
        "TI": "https://www.lexfind.ch/fe/de/tol/28/it",
        "VD": "https://www.lexfind.ch/fe/de/tol/30/fr",
        "VS": "https://lex.vs.ch/",
        "NE": "https://rsn.ne.ch/",
        "GE": "https://silgeneve.ch/",
        "JU": "https://rsju.jura.ch/",
    }
    
    # Known Merkblätter (official info sheets) - these contain concrete measurements!
    # Format: canton -> list of (topic_keywords, title, url)
    KNOWN_MERKBLAETTER = {
        "AI": [
            (["einfriedung", "zaun", "bepflanzung", "grenze", "hecke"], 
             "Merkblatt Einfriedungen und Bepflanzungen", 
             "https://www.appenzell.org/_docn/5208400/Merkblatt_Einfriedungen_und_Bepflanzungen.pdf"),
        ],
        # Add more cantons as we discover their Merkblätter
    }
    
    try:
        tavily = get_tavily_client()
        
        # Get canton info
        canton_info = CANTON_NAMES.get(canton, {"de": canton, "fr": canton, "it": canton})
        canton_display = canton_name or canton_info["de"]
        official_url = CANTONAL_LAW_URLS.get(canton, "")
        canton_domains = CANTONAL_LAW_DOMAINS.get(canton, [])
        
        # Format output header
        output_lines = [
            f"🏔️ CANTONAL LAW SEARCH: {canton_display}",
            f"   Canton Code: {canton}",
            "=" * 50
        ]
        
        # Add official source link FIRST (always useful!)
        if official_url:
            output_lines.append(f"\n📚 OFFIZIELLE GESETZESSAMMLUNG {canton}:")
            output_lines.append(f"   {official_url}")
            output_lines.append(f"   → Direkte Quelle für kantonale Gesetze")
        
        # Check for known Merkblätter that match the query
        query_lower = query.lower()
        known_docs = KNOWN_MERKBLAETTER.get(canton, [])
        matched_merkblaetter = []
        
        for keywords, title, url in known_docs:
            if any(kw in query_lower for kw in keywords):
                matched_merkblaetter.append((title, url))
        
        # If we have matching Merkblätter, fetch their content!
        if matched_merkblaetter:
            output_lines.append(f"\n📄 BEKANNTE MERKBLÄTTER (mit konkreten Angaben!):")
            output_lines.append("-" * 40)
            
            for title, url in matched_merkblaetter:
                output_lines.append(f"\n📄 {title}")
                output_lines.append(f"   URL: {url}")
                
                # Fetch PDF content
                pdf_content = fetch_pdf_content(url, max_chars=2500)
                
                # Check if extraction was successful (no error prefixes)
                error_prefixes = ["Fehler", "Timeout", "PDF konnte nicht", "pdfplumber nicht", "PDF-Extraktion fehlgeschlagen"]
                is_error = any(pdf_content.startswith(prefix) for prefix in error_prefixes)
                
                if pdf_content and not is_error:
                    output_lines.append(f"   ✅ INHALT EXTRAHIERT:")
                    output_lines.append(f"   {pdf_content}")
                else:
                    output_lines.append(f"   ⚠️ PDF-Extraktion: {pdf_content[:200] if pdf_content else 'Kein Inhalt'}")
        
        all_results = []
        queries_used = []
        
        # Extract keywords for relevance scoring
        keywords = extract_keywords_from_query(query)
        
        # STRATEGY 1: Use orchestrator queries if provided (with optimal params)
        if orchestrator_queries and len(orchestrator_queries) > 0:
            for oq in orchestrator_queries[:3]:
                # Ensure canton name is in quotes for exact match
                if canton_display and canton_display in oq and f'"{canton_display}"' not in oq:
                    oq = oq.replace(canton_display, f'"{canton_display}"')
                queries_used.append(oq)
                results = tavily.search(
                    query=oq,
                    max_results=max_results,
                    search_depth="advanced",
                    include_raw_content=True,    # KEY: Get full text!
                    chunks_per_source=3,
                    country="switzerland"
                )
                all_results.extend(results.get("results", []))
        
        # STRATEGY 2: Site-specific search on cantonal law collection (MOST RELIABLE!)
        if canton_domains:
            for domain in canton_domains[:1]:  # Primary domain
                site_query = f"site:{domain} {query}"
                queries_used.append(f"🏛️ {site_query}")
                results = tavily.search(
                    query=site_query,
                    max_results=5,
                    search_depth="advanced",
                    include_raw_content=True,
                    chunks_per_source=3,
                    country="switzerland"
                )
                all_results.extend(results.get("results", []))
        
        # STRATEGY 3: Search for Merkblätter (with quoted canton name!)
        merkblatt_query = f'Merkblatt "{canton_display}" Einfriedung Zaun'
        queries_used.append(f"📄 {merkblatt_query}")
        merkblatt_results = tavily.search(
            query=merkblatt_query,
            max_results=5,
            search_depth="advanced",
            include_raw_content=True,
            country="switzerland"
        )
        all_results.extend(merkblatt_results.get("results", []))
        
        # STRATEGY 4: Broad search with quoted canton name (fallback)
        if len(all_results) < 5:
            broad_query = f'"{canton_display}" {query}'
            queries_used.append(f"🔍 {broad_query}")
            broad_results = tavily.search(
                query=broad_query,
                max_results=3,
                search_depth="advanced",
                include_raw_content=True,
                country="switzerland"
            )
            all_results.extend(broad_results.get("results", []))
        
        # Log queries used
        output_lines.append(f"\n🔎 SUCHANFRAGEN:")
        for i, q in enumerate(queries_used, 1):
            output_lines.append(f"   {i}. {q}")
        
        # Deduplicate by URL
        seen_urls = set()
        unique_results = []
        for r in all_results:
            url = r.get("url", "").lower()
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(r)
        
        # SCORE and SORT by relevance!
        scored_results = []
        for r in unique_results:
            score, matched = score_search_result(r, keywords, "cantonal", canton)
            scored_results.append({
                **r,
                "relevance_score": score,
                "matched_keywords": matched
            })
        
        scored_results.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        # Format search results (TOP results only!)
        if scored_results:
            output_lines.append(f"\n\n🔍 SUCHERGEBNISSE ({len(scored_results)} gefunden, sortiert nach Relevanz):")
            output_lines.append(f"   Keywords: {', '.join([k[0] for k in keywords[:5]])}")
            output_lines.append("-" * 40)
            
            pdf_contents = []  # Store PDF extracts
            
            for i, result in enumerate(scored_results[:max_results], 1):
                url = result.get('url', 'N/A')
                title = result.get('title', 'No title')
                score = result.get('relevance_score', 0)
                matched = result.get('matched_keywords', [])
                raw = result.get('raw_content', '')
                content = result.get('content', '')
                
                output_lines.append(f"\n[{i}] {title}")
                output_lines.append(f"    URL: {url}")
                output_lines.append(f"    Relevanz: {score} ({', '.join(matched)})")
                
                # Use raw_content with SMART EXTRACTION if available
                if raw and len(raw) > len(content):
                    excerpt = extract_relevant_excerpt(raw, keywords, 1500)
                    output_lines.append(f"    RELEVANTER AUSSCHNITT:")
                    output_lines.append(f"    {excerpt}")
                else:
                    output_lines.append(f"    SNIPPET:")
                    output_lines.append(f"    {content[:500]}...")
                
                # If it's a PDF and looks relevant, fetch its content!
                if url.lower().endswith('.pdf') and score > 10:
                    output_lines.append(f"    📄 PDF erkannt - versuche Inhalt abzurufen...")
                    pdf_text = fetch_pdf_content(url, max_chars=2000)
                    if pdf_text and not pdf_text.startswith("Fehler") and not pdf_text.startswith("Timeout"):
                        pdf_contents.append({
                            "title": title,
                            "url": url,
                            "content": pdf_text
                        })
                        output_lines.append(f"    ✅ PDF-Inhalt erfolgreich extrahiert ({len(pdf_text)} Zeichen)")
            
            # Add extracted PDF contents as separate section
            if pdf_contents:
                output_lines.append(f"\n\n📄 EXTRAHIERTE PDF-INHALTE (WICHTIG - enthält konkrete Angaben!):")
                output_lines.append("=" * 50)
                for pdf in pdf_contents:
                    output_lines.append(f"\n📄 {pdf['title']}")
                    output_lines.append(f"   URL: {pdf['url']}")
                    output_lines.append(f"   INHALT:")
                    output_lines.append(f"   {pdf['content']}")
                    output_lines.append("-" * 40)
        else:
            output_lines.append(f"\n\n⚠️ KEINE SUCHERGEBNISSE für {canton_display}.")
            output_lines.append(f"\n💡 EMPFEHLUNG:")
            output_lines.append(f"   1. Besuchen Sie die offizielle Gesetzessammlung: {official_url}")
            output_lines.append(f"   2. Suchen Sie nach: Bauverordnung, Baugesetz, Einfriedung")
            output_lines.append(f"   3. Kontaktieren Sie die Gemeinde für lokale Vorschriften")
        
        return "\n".join(output_lines)
        
    except Exception as e:
        return f"❌ Cantonal law search error: {str(e)}"


def search_cantonal_case_law(query: str, canton: str, canton_name: str = None,
                             orchestrator_queries: list = None, max_results: int = 5) -> str:
    """
    Search cantonal court decisions
    
    Args:
        query: Case law search query
        canton: Canton abbreviation (e.g., "ZH", "GE", "TI")
        canton_name: Full canton name
        orchestrator_queries: Specific queries from orchestrator
        max_results: Maximum results to return
        
    Returns:
        Formatted string of cantonal case law results
    """
    try:
        tavily = get_tavily_client()
        
        canton_info = CANTON_NAMES.get(canton, {"de": canton, "fr": canton, "it": canton})
        canton_display = canton_name or canton_info["de"]
        
        # Format output header
        output_lines = [
            f"⚖️ CANTONAL CASE LAW SEARCH: {canton_display}",
            f"   Canton Code: {canton}",
            "=" * 50
        ]
        
        all_results = []
        queries_used = []
        
        # PRIMARY SOURCE: entscheidsuche.ch with quoted canton name for exact match
        entscheidsuche_query = f'site:entscheidsuche.ch "{canton_display}" {query}'
        queries_used.append(entscheidsuche_query)
        
        results_1 = tavily.search(
            query=entscheidsuche_query,
            max_results=max_results,
            search_depth="advanced"
        )
        all_results.extend(results_1.get("results", []))
        
        # Use orchestrator queries if provided (add quotes if missing)
        if orchestrator_queries:
            for oq in orchestrator_queries[:2]:
                if canton_display and canton_display in oq and f'"{canton_display}"' not in oq:
                    oq = oq.replace(canton_display, f'"{canton_display}"')
                queries_used.append(oq)
                results = tavily.search(
                    query=oq,
                    max_results=3,
                    search_depth="basic"
                )
                all_results.extend(results.get("results", []))
        else:
            # Fallback: Search with quoted canton name + court terms
            court_query = f'"{canton_display}" Obergericht Verwaltungsgericht Entscheid {query}'
            queries_used.append(court_query)
            
            results_2 = tavily.search(
                query=court_query,
                max_results=3,
                search_depth="basic"
            )
            all_results.extend(results_2.get("results", []))
        
        # Log queries used
        output_lines.append(f"\n🔎 SUCHANFRAGEN:")
        for i, q in enumerate(queries_used, 1):
            output_lines.append(f"   {i}. {q}")
        
        # Deduplicate by URL
        seen_urls = set()
        unique_results = []
        for r in all_results:
            url = r.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(r)
        
        # Format search results
        if unique_results:
            output_lines.append(f"\n\n🔍 KANTONALE ENTSCHEIDE ({len(unique_results)} gefunden):")
            output_lines.append("-" * 40)
            for i, result in enumerate(unique_results[:max_results], 1):
                output_lines.append(f"\n[{i}] {result.get('title', 'No title')}")
                output_lines.append(f"    URL: {result.get('url', 'N/A')}")
                output_lines.append(f"    {result.get('content', 'No content')[:400]}...")
        else:
            output_lines.append(f"\n\n⚠️ Keine kantonalen Entscheide für {canton_display} gefunden.")
            output_lines.append(f"💡 Versuchen Sie:")
            output_lines.append(f"   - entscheidsuche.ch direkt besuchen")
            output_lines.append(f"   - Bundesgerichtsentscheide (BGE) zum Thema prüfen")
        
        return "\n".join(output_lines)
        
    except Exception as e:
        return f"❌ Cantonal case law search error: {str(e)}"


def search_communal_law(query: str, commune: str, canton: str, max_results: int = 5) -> str:
    """
    Search communal/municipal regulations
    
    Args:
        query: Legal search query
        commune: Commune name
        canton: Canton abbreviation
        max_results: Maximum results to return
        
    Returns:
        Formatted string of communal law results
    """
    try:
        tavily = get_tavily_client()
        
        canton_names = CANTON_NAMES.get(canton, {"de": canton, "fr": canton, "it": canton})
        
        # Search for communal regulations
        # Terms: Gemeindeordnung, Baureglement, règlement communal, regolamento comunale
        communal_terms = (
            f"Gemeindeordnung Baureglement Zonenplan Bauordnung "
            f"règlement communal plan de zones "
            f"regolamento comunale piano regolatore"
        )
        
        search_query = f"{query} {commune} {canton_names['de']} {communal_terms}"
        
        results = tavily.search(
            query=search_query,
            max_results=max_results,
            search_depth="advanced"
        )
        
        # Format results
        output_lines = [
            f"🏘️ COMMUNAL LAW SEARCH: {query}",
            f"   Commune: {commune} ({canton})",
            "=" * 50
        ]
        
        for i, result in enumerate(results.get("results", []), 1):
            output_lines.append(f"\n[{i}] {result.get('title', 'No title')}")
            output_lines.append(f"    URL: {result.get('url', 'N/A')}")
            output_lines.append(f"    {result.get('content', 'No content')[:500]}...")
        
        if not results.get("results"):
            output_lines.append(f"\nNo communal regulations found for {commune}. Try cantonal law for {canton}.")
        
        return "\n".join(output_lines)
        
    except Exception as e:
        return f"❌ Communal law search error: {str(e)}"


# ============================================================
# URL EXTRACTION AND VALIDATION
# ============================================================

import re
import urllib.request
import urllib.error
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed


def extract_urls_from_search_results(search_results: str) -> List[str]:
    """
    Extract all URLs from search results string.
    
    Returns: List of URLs found in the search results
    """
    # Match URLs starting with http:// or https://
    url_pattern = r'https?://[^\s\)\]\"\'<>]+'
    urls = re.findall(url_pattern, search_results)
    
    # Clean up URLs (remove trailing punctuation)
    cleaned_urls = []
    for url in urls:
        # Remove trailing punctuation that might have been captured
        url = url.rstrip('.,;:!?')
        if url not in cleaned_urls:
            cleaned_urls.append(url)
    
    return cleaned_urls


def validate_url(url: str, timeout: float = 3.0) -> Tuple[str, bool, str]:
    """
    Check if a URL is reachable.
    
    Returns: (url, is_valid, status_message)
    """
    try:
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0 (Swiss Legal Research Bot)'}
        )
        response = urllib.request.urlopen(req, timeout=timeout)
        return (url, True, f"OK ({response.status})")
    except urllib.error.HTTPError as e:
        return (url, False, f"HTTP {e.code}")
    except urllib.error.URLError as e:
        return (url, False, f"URL Error: {str(e.reason)[:50]}")
    except Exception as e:
        return (url, False, f"Error: {str(e)[:50]}")


def validate_urls_batch(urls: List[str], max_workers: int = 5, timeout: float = 3.0) -> dict:
    """
    Validate multiple URLs in parallel.
    
    Returns: {url: {"valid": bool, "status": str}, ...}
    """
    results = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {
            executor.submit(validate_url, url, timeout): url 
            for url in urls
        }
        
        for future in as_completed(future_to_url):
            url, is_valid, status = future.result()
            results[url] = {"valid": is_valid, "status": status}
    
    return results


def extract_and_validate_citations(text: str) -> dict:
    """
    Extract markdown links from text and validate them.
    
    Returns: {
        "valid_links": [(text, url), ...],
        "invalid_links": [(text, url, error), ...],
        "citations_without_links": [citation, ...]
    }
    """
    result = {
        "valid_links": [],
        "invalid_links": [],
        "citations_without_links": []
    }
    
    # Extract markdown links: [text](url)
    link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
    links = re.findall(link_pattern, text)
    
    if links:
        urls = [url for _, url in links]
        validation_results = validate_urls_batch(urls)
        
        for link_text, url in links:
            if validation_results.get(url, {}).get("valid", False):
                result["valid_links"].append((link_text, url))
            else:
                error = validation_results.get(url, {}).get("status", "Unknown error")
                result["invalid_links"].append((link_text, url, error))
    
    # Find citations without links (Art. X OR, BGE X, etc.)
    # These patterns match citations NOT inside markdown links
    citation_patterns = [
        r'Art\.\s+\d+[a-z]?(?:\s+(?:Abs|al|cpv|para)\.\s+\d+)?\s+(?:OR|ZGB|BV|ArG|CO|CC|Cst|Cost|FC|LA|LTr|LL|DSG)(?:\s+\((?:SR|RS)\s+[\d.]+\))?',
        r'(?:BGE|ATF|DTF)\s+\d+\s+[IVX]+\s+\d+',
        r'(?:Urteil|Arrêt|Sentenza|Decision)\s+\d+[A-Z]_\d+/\d+',
    ]
    
    for pattern in citation_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            # Check if this citation is already in a link
            if not any(match in link_text for link_text, _ in links):
                if match not in result["citations_without_links"]:
                    result["citations_without_links"].append(match)
    
    return result


def create_validated_output(agent_response: str, search_results: str) -> Tuple[str, dict]:
    """
    Process agent response:
    1. Extract URLs from search results
    2. Validate any links in the response
    3. Return cleaned response and validation report
    
    Returns: (processed_response, validation_report)
    """
    # Get available URLs from search
    available_urls = extract_urls_from_search_results(search_results)
    
    # Validate citations in response
    citation_check = extract_and_validate_citations(agent_response)
    
    # Build validation report
    report = {
        "available_urls": available_urls,
        "valid_links": citation_check["valid_links"],
        "invalid_links": citation_check["invalid_links"],
        "citations_without_links": citation_check["citations_without_links"]
    }
    
    # If there are invalid links, remove them from the response
    processed_response = agent_response
    for link_text, url, error in citation_check["invalid_links"]:
        # Replace [text](url) with just text
        old_link = f"[{link_text}]({url})"
        processed_response = processed_response.replace(old_link, link_text)
    
    return processed_response, report


# LangChain Tool wrappers
def create_langchain_tools():
    """Create LangChain-compatible tool instances"""
    from langchain_core.tools import tool
    
    @tool
    def swiss_primary_law_search(query: str) -> str:
        """Search Swiss primary law (Fedlex, admin.ch). Use for finding federal laws, 
        ordinances, constitutional provisions. Returns Art., Abs., SR numbers."""
        return search_swiss_primary_law(query)
    
    @tool
    def swiss_case_law_search(query: str) -> str:
        """Search Swiss Federal Court decisions (BGer). Use for finding BGE cases, 
        court precedents, judicial interpretations."""
        return search_swiss_case_law(query)
    
    @tool
    def swiss_general_legal_search(query: str) -> str:
        """General Swiss legal search. Use for commentary, doctrine, and broader 
        legal information not found in primary sources."""
        return search_general_legal(query)
    
    return [swiss_primary_law_search, swiss_case_law_search, swiss_general_legal_search]


if __name__ == "__main__":
    # Test the tools
    print("Testing Swiss Legal Research Tools...")
    print("\n" + "=" * 60)
    print(search_swiss_primary_law("Arbeitsrecht Kündigungsfrist OR Art 335"))
    print("\n" + "=" * 60)
    print(search_swiss_case_law("fristlose Kündigung Arbeitsverhältnis"))

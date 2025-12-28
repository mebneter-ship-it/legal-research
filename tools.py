"""
Swiss Legal Research Tools

Direct tool implementations for use with LangGraph agents.
Supports:
- Federal law (Fedlex, admin.ch)
- Federal court decisions (BGer)
- Cantonal law (lexfind.ch, cantonal portals)
- Cantonal court decisions
- Communal regulations (where detectable)
"""

import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


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
    "AI": ["ai.ch"],
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


def search_swiss_primary_law(query: str, max_results: int = 5) -> str:
    """
    Search Swiss primary law sources (Fedlex, admin.ch)
    
    Args:
        query: Legal search query (German, French, or Italian)
        max_results: Maximum results to return
        
    Returns:
        Formatted string of search results
    """
    try:
        tavily = get_tavily_client()
        
        # Construct targeted query for Swiss federal law
        search_query = f"{query} site:fedlex.admin.ch OR site:admin.ch"
        
        results = tavily.search(
            query=search_query,
            max_results=max_results,
            search_depth="advanced"
        )
        
        # Format results
        output_lines = [f"🏛️ PRIMARY LAW SEARCH: {query}", "=" * 50]
        
        for i, result in enumerate(results.get("results", []), 1):
            output_lines.append(f"\n[{i}] {result.get('title', 'No title')}")
            output_lines.append(f"    URL: {result.get('url', 'N/A')}")
            output_lines.append(f"    {result.get('content', 'No content')[:500]}...")
        
        if not results.get("results"):
            output_lines.append("\nNo results found. Try different search terms.")
        
        return "\n".join(output_lines)
        
    except Exception as e:
        return f"❌ Primary law search error: {str(e)}"


def search_swiss_case_law(query: str, max_results: int = 5) -> str:
    """
    Search Swiss Federal Court (Bundesgericht) case law
    
    Args:
        query: Case law search query
        max_results: Maximum results to return
        
    Returns:
        Formatted string of BGE decisions
    """
    try:
        tavily = get_tavily_client()
        
        # Construct targeted query for Swiss case law
        search_query = f"{query} site:bger.ch BGE"
        
        results = tavily.search(
            query=search_query,
            max_results=max_results,
            search_depth="advanced"
        )
        
        # Format results
        output_lines = [f"⚖️ CASE LAW SEARCH: {query}", "=" * 50]
        
        for i, result in enumerate(results.get("results", []), 1):
            output_lines.append(f"\n[{i}] {result.get('title', 'No title')}")
            output_lines.append(f"    URL: {result.get('url', 'N/A')}")
            output_lines.append(f"    {result.get('content', 'No content')[:500]}...")
        
        if not results.get("results"):
            output_lines.append("\nNo case law found. Try broader search terms.")
        
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


def search_cantonal_law(query: str, canton: str, max_results: int = 5) -> str:
    """
    Search cantonal law sources
    
    Args:
        query: Legal search query
        canton: Canton abbreviation (e.g., "ZH", "GE", "TI")
        max_results: Maximum results to return
        
    Returns:
        Formatted string of cantonal law results
    """
    try:
        tavily = get_tavily_client()
        
        # Get canton-specific domains
        domains = CANTONAL_LAW_DOMAINS.get(canton, [])
        canton_names = CANTON_NAMES.get(canton, {"de": canton, "fr": canton, "it": canton})
        
        # Also search aggregators and legal portals
        domains.extend([
            "lexfind.ch",      # Aggregates cantonal law
            "fedlex.admin.ch", # Sometimes has cantonal concordats
        ])
        
        # Build canton-specific query (without strict site restriction for broader results)
        canton_terms = f"{canton_names['de']} OR {canton_names['fr']} OR {canton_names['it']}"
        search_query = f"{query} Kanton {canton} ({canton_terms}) Gesetz Verordnung Reglement"
        
        results = tavily.search(
            query=search_query,
            max_results=max_results,
            search_depth="advanced",
            include_domains=domains  # Prefer these but don't exclude others
        )
        
        # Format results
        output_lines = [
            f"🏔️ CANTONAL LAW SEARCH: {query}",
            f"   Canton: {canton} ({canton_names['de']} / {canton_names['fr']} / {canton_names['it']})",
            "=" * 50
        ]
        
        for i, result in enumerate(results.get("results", []), 1):
            output_lines.append(f"\n[{i}] {result.get('title', 'No title')}")
            output_lines.append(f"    URL: {result.get('url', 'N/A')}")
            output_lines.append(f"    {result.get('content', 'No content')[:500]}...")
        
        if not results.get("results"):
            output_lines.append(f"\nNo cantonal law results found for {canton}. Try federal law or different terms.")
        
        return "\n".join(output_lines)
        
    except Exception as e:
        return f"❌ Cantonal law search error: {str(e)}"


def search_cantonal_case_law(query: str, canton: str, max_results: int = 5) -> str:
    """
    Search cantonal court decisions
    
    Args:
        query: Case law search query
        canton: Canton abbreviation (e.g., "ZH", "GE", "TI")
        max_results: Maximum results to return
        
    Returns:
        Formatted string of cantonal case law results
    """
    try:
        tavily = get_tavily_client()
        
        canton_names = CANTON_NAMES.get(canton, {"de": canton, "fr": canton, "it": canton})
        domains = CANTONAL_LAW_DOMAINS.get(canton, [])
        
        # Build search query for cantonal courts
        # Many cantons publish decisions on their main portal
        site_query = " OR ".join([f"site:{d}" for d in domains]) if domains else ""
        
        # Terms for cantonal courts
        court_terms = {
            "de": f"Obergericht Verwaltungsgericht Kantonsgericht {canton_names['de']} Entscheid Urteil",
            "fr": f"Tribunal cantonal Cour {canton_names['fr']} arrêt décision",
            "it": f"Tribunale cantonale {canton_names['it']} sentenza decisione"
        }
        
        search_query = f"{query} {court_terms['de']} {court_terms['fr']} {court_terms['it']}"
        if site_query:
            search_query = f"{query} ({site_query}) Gericht Tribunal Tribunale"
        
        results = tavily.search(
            query=search_query,
            max_results=max_results,
            search_depth="advanced"
        )
        
        # Format results
        output_lines = [
            f"⚖️ CANTONAL CASE LAW SEARCH: {query}",
            f"   Canton: {canton} ({canton_names['de']} / {canton_names['fr']} / {canton_names['it']})",
            "=" * 50
        ]
        
        for i, result in enumerate(results.get("results", []), 1):
            output_lines.append(f"\n[{i}] {result.get('title', 'No title')}")
            output_lines.append(f"    URL: {result.get('url', 'N/A')}")
            output_lines.append(f"    {result.get('content', 'No content')[:500]}...")
        
        if not results.get("results"):
            output_lines.append(f"\nNo cantonal case law found for {canton}. Try federal court (BGer) or different terms.")
        
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

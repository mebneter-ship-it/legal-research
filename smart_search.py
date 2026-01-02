"""
Swiss Legal Search - Generalized Smart Search Module

Principles learned:
1. include_raw_content=True for full text
2. Relevance scoring based on query keywords
3. Smart extraction around keywords (not full text)
4. Domain prioritization per legal area
5. Top N results only (reduce noise)

Usage:
    from smart_search import SmartLegalSearch
    
    search = SmartLegalSearch()
    results = search.search(
        query="Wie hoch darf ein Zaun sein?",
        legal_area="cantonal",
        canton="AI"
    )
"""

import os
import re
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

load_dotenv("keys.env")


# ============================================================
# CONFIGURATION: Domain mappings per legal area
# ============================================================

TRUSTED_DOMAINS = {
    "federal": [
        "fedlex.admin.ch",      # Official federal law
    ],
    "cantonal": {
        # Cantonal law collections (clex.ch pattern)
        "AI": ["ai.clex.ch"],
        "AR": ["ar.clex.ch"],
        "ZH": ["zhlex.zh.ch", "zh.ch"],
        "BE": ["belex.sites.be.ch"],
        "LU": ["srl.lu.ch"],
        "SG": ["gesetzessammlung.sg.ch"],
        "AG": ["gesetzessammlungen.ag.ch"],
        "TG": ["rechtsbuch.tg.ch"],
        "GR": ["gr-lex.gr.ch"],
        "TI": ["legislazione.ti.ch"],
        "VD": ["rsv.vd.ch"],
        "GE": ["silgeneve.ch"],
        # Fallback for unknown cantons
        "DEFAULT": ["lexfind.ch"],
    },
    "case_law": [
        "bger.ch",              # Federal Supreme Court
        "entscheidsuche.ch",    # Case law aggregator
    ],
    "general": [
        "admin.ch",
        "ch.ch",
    ]
}

CANTON_NAMES = {
    "AI": "Appenzell Innerrhoden",
    "AR": "Appenzell Ausserrhoden",
    "ZH": "Zürich",
    "BE": "Bern",
    "LU": "Luzern",
    "SG": "St. Gallen",
    "AG": "Aargau",
    "TG": "Thurgau",
    "GR": "Graubünden",
    "TI": "Ticino",
    "VD": "Vaud",
    "GE": "Genève",
    # Add more as needed
}


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class SearchResult:
    """A single search result with relevance info"""
    title: str
    url: str
    content: str  # Snippet or extracted text
    raw_content: str = ""  # Full content if available
    relevance_score: float = 0.0
    matched_keywords: list = field(default_factory=list)
    is_trusted_domain: bool = False


@dataclass 
class SearchOutput:
    """Complete search output ready for LLM"""
    query: str
    results: list  # List of SearchResult
    formatted_text: str  # Ready for LLM prompt
    keywords_used: list = field(default_factory=list)
    domains_searched: list = field(default_factory=list)


# ============================================================
# SMART LEGAL SEARCH
# ============================================================

class SmartLegalSearch:
    """
    Generalized smart search for Swiss legal research.
    
    Features:
    - Automatic keyword extraction from query
    - Domain prioritization per legal area
    - Relevance scoring and ranking
    - Smart excerpt extraction around keywords
    """
    
    def __init__(self):
        self.tavily = self._get_tavily_client()
    
    def _get_tavily_client(self):
        from tavily import TavilyClient
        api_key = os.getenv("TAVILY_API_KEY", "")
        if api_key.startswith("tvly-tvly-"):
            api_key = api_key.replace("tvly-tvly-", "tvly-", 1)
        return TavilyClient(api_key=api_key)
    
    # --------------------------------------------------------
    # KEYWORD EXTRACTION
    # --------------------------------------------------------
    
    def extract_keywords(self, query: str, legal_area: str = None) -> list:
        """
        Extract search keywords from a legal query.
        
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
            'bauzonen': 4, 'höhe': 4,
            
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
        }
        
        for term, weight in legal_terms.items():
            if term in query_lower:
                keywords.append((term, weight))
        
        # 4. Important nouns from query (lower weight)
        # Simple extraction of capitalized words (German nouns)
        words = query.split()
        for word in words:
            clean = re.sub(r'[^\w]', '', word)
            if clean and clean[0].isupper() and len(clean) > 3:
                if clean.lower() not in [k[0] for k in keywords]:
                    keywords.append((clean.lower(), 2))
        
        return keywords
    
    # --------------------------------------------------------
    # QUERY BUILDING
    # --------------------------------------------------------
    
    def build_queries(
        self, 
        query: str, 
        legal_area: str,
        canton: str = None
    ) -> list:
        """
        Build optimized search queries for Tavily.
        
        Returns list of (query_string, domains) tuples.
        """
        queries = []
        
        if legal_area == "cantonal" and canton:
            # Get canton-specific domain
            domains = TRUSTED_DOMAINS["cantonal"].get(
                canton, 
                TRUSTED_DOMAINS["cantonal"]["DEFAULT"]
            )
            canton_name = CANTON_NAMES.get(canton, canton)
            
            # Strategy 1: Site-specific search (most reliable)
            for domain in domains[:1]:  # Primary domain only
                queries.append((f"site:{domain} {query}", None))
            
            # Strategy 2: Quoted canton name (avoid confusion)
            queries.append((f'"{canton_name}" {query}', None))
            
        elif legal_area == "federal":
            # Fedlex search
            queries.append((query, TRUSTED_DOMAINS["federal"]))
            queries.append((f"site:fedlex.admin.ch {query}", None))
            
        elif legal_area == "case_law":
            # BGer and entscheidsuche
            queries.append((f"BGE {query}", TRUSTED_DOMAINS["case_law"]))
            queries.append((f"site:bger.ch {query}", None))
            queries.append((f"site:entscheidsuche.ch {query}", None))
            
        else:
            # General search
            queries.append((query, None))
            queries.append((f"{query} Schweiz Recht", None))
        
        return queries
    
    # --------------------------------------------------------
    # RELEVANCE SCORING
    # --------------------------------------------------------
    
    def score_result(
        self, 
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
        
        # 3. CRITICAL: Trusted domain bonus (much higher for official sources!)
        trusted_bonus = 0
        
        if legal_area == "cantonal" and canton:
            # For cantonal: official law collection is ESSENTIAL
            canton_domains = TRUSTED_DOMAINS["cantonal"].get(canton, [])
            if any(domain in url for domain in canton_domains):
                trusted_bonus = 50  # Very high bonus for official source!
                matched.append(f"[OFFICIAL: {canton}]")
            elif "clex.ch" in url or "lexfind.ch" in url:
                trusted_bonus = 30  # Other law collections
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
        commercial_domains = ['shop', 'kaufen', 'swiss.ch', 'amazon', 'ricardo']
        if any(d in url for d in commercial_domains):
            score -= 20  # Penalty for commercial sites
        
        score += trusted_bonus
        
        return score, matched
    
    # --------------------------------------------------------
    # SMART EXTRACTION
    # --------------------------------------------------------
    
    def extract_relevant_section(
        self, 
        raw_content: str, 
        keywords: list,
        context_chars: int = 1500
    ) -> str:
        """
        Extract the most relevant section around matched keywords.
        
        Instead of sending 50k chars, extract ~2k chars around
        the most important keyword match.
        """
        if not raw_content:
            return ""
        
        raw_lower = raw_content.lower()
        
        # Sort keywords by weight (highest first)
        sorted_kw = sorted(keywords, key=lambda x: x[1], reverse=True)
        
        # Find position of highest-weight keyword that exists
        best_pos = -1
        best_keyword = None
        
        for keyword, weight in sorted_kw:
            pos = raw_lower.find(keyword.lower())
            if pos != -1:
                best_pos = pos
                best_keyword = keyword
                break
        
        if best_pos == -1:
            # No keyword found, return beginning
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
    
    # --------------------------------------------------------
    # MAIN SEARCH METHOD
    # --------------------------------------------------------
    
    def search(
        self,
        query: str,
        legal_area: str = "general",  # federal, cantonal, case_law, general
        canton: str = None,
        max_results_per_query: int = 5,
        top_n: int = 3,  # Only return top N most relevant
    ) -> SearchOutput:
        """
        Execute a smart legal search.
        
        Args:
            query: The legal question
            legal_area: Type of law to search
            canton: Canton code (e.g., "AI") for cantonal searches
            max_results_per_query: Results per Tavily query
            top_n: Number of top results to return
            
        Returns:
            SearchOutput with formatted text ready for LLM
        """
        # 1. Extract keywords for relevance scoring
        keywords = self.extract_keywords(query, legal_area)
        
        # 2. Build optimized queries
        queries = self.build_queries(query, legal_area, canton)
        
        # 3. Execute searches with optimal Tavily parameters
        all_results = []
        domains_searched = []
        
        for query_str, include_domains in queries:
            try:
                params = {
                    "query": query_str,
                    "search_depth": "advanced",
                    "max_results": max_results_per_query,
                    "include_raw_content": True,  # KEY: Get full text!
                    "chunks_per_source": 3,
                    "country": "switzerland",
                }
                if include_domains:
                    params["include_domains"] = include_domains
                    domains_searched.extend(include_domains)
                
                results = self.tavily.search(**params)
                all_results.extend(results.get("results", []))
                
            except Exception as e:
                print(f"Search error for '{query_str}': {e}")
        
        # 4. Deduplicate by URL
        seen_urls = set()
        unique_results = []
        for r in all_results:
            url = r.get("url", "").lower()
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(r)
        
        # 5. Score and sort by relevance
        scored_results = []
        for r in unique_results:
            score, matched = self.score_result(r, keywords, legal_area, canton)
            
            # Check if from trusted domain
            url = r.get("url", "").lower()
            is_trusted = any(
                domain in url 
                for domains in TRUSTED_DOMAINS.values()
                for domain in (domains if isinstance(domains, list) else [])
            )
            
            scored_results.append({
                **r,
                "relevance_score": score,
                "matched_keywords": matched,
                "is_trusted": is_trusted
            })
        
        # Sort by score (and trust as tiebreaker)
        scored_results.sort(
            key=lambda x: (x["relevance_score"], x["is_trusted"]), 
            reverse=True
        )
        
        # 6. Take top N and format for LLM
        top_results = scored_results[:top_n]
        
        formatted_lines = [f"=== SUCHERGEBNISSE (Top {top_n}, sortiert nach Relevanz) ===\n"]
        
        for i, r in enumerate(top_results, 1):
            title = r.get("title", "Ohne Titel")
            url = r.get("url", "")
            raw = r.get("raw_content", "")
            content = r.get("content", "")
            score = r.get("relevance_score", 0)
            matched = r.get("matched_keywords", [])
            
            formatted_lines.append(f"[{i}] {title}")
            formatted_lines.append(f"    URL: {url}")
            formatted_lines.append(f"    Relevanz: {score} (Keywords: {', '.join(matched)})")
            
            # Smart extraction: use raw_content if available
            if raw and len(raw) > len(content):
                excerpt = self.extract_relevant_section(raw, keywords)
                formatted_lines.append(f"    RELEVANTER AUSSCHNITT:\n    {excerpt}")
            else:
                formatted_lines.append(f"    SNIPPET:\n    {content[:500]}")
            
            formatted_lines.append("")
        
        formatted_text = "\n".join(formatted_lines)
        
        # 7. Build output
        return SearchOutput(
            query=query,
            results=[
                SearchResult(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    content=r.get("content", ""),
                    raw_content=r.get("raw_content", ""),
                    relevance_score=r.get("relevance_score", 0),
                    matched_keywords=r.get("matched_keywords", []),
                    is_trusted_domain=r.get("is_trusted", False)
                )
                for r in top_results
            ],
            formatted_text=formatted_text,
            keywords_used=[k[0] for k in keywords],
            domains_searched=list(set(domains_searched))
        )


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def search_cantonal(query: str, canton: str, top_n: int = 3) -> SearchOutput:
    """Quick search for cantonal law"""
    searcher = SmartLegalSearch()
    return searcher.search(query, legal_area="cantonal", canton=canton, top_n=top_n)

def search_federal(query: str, top_n: int = 3) -> SearchOutput:
    """Quick search for federal law"""
    searcher = SmartLegalSearch()
    return searcher.search(query, legal_area="federal", top_n=top_n)

def search_case_law(query: str, top_n: int = 3) -> SearchOutput:
    """Quick search for case law (BGE)"""
    searcher = SmartLegalSearch()
    return searcher.search(query, legal_area="case_law", top_n=top_n)


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧪 SMART LEGAL SEARCH - TEST")
    print("="*70)
    
    # Test 1: Cantonal (our proven case)
    print("\n📍 TEST 1: Cantonal Search (Appenzell Zaun)")
    result = search_cantonal(
        query="Wie hoch darf ein Zaun um die Grenzen sein?",
        canton="AI"
    )
    print(f"   Keywords extracted: {result.keywords_used}")
    print(f"   Results found: {len(result.results)}")
    print(f"   Top result score: {result.results[0].relevance_score if result.results else 0}")
    print("\n" + result.formatted_text[:2000])
    
    # Test 2: Federal
    print("\n" + "-"*70)
    print("\n📍 TEST 2: Federal Search (Mietrecht Kündigung)")
    result = search_federal(
        query="Kündigungsfrist Mietvertrag Wohnung"
    )
    print(f"   Keywords extracted: {result.keywords_used}")
    print(f"   Results found: {len(result.results)}")
    print("\n" + result.formatted_text[:2000])
    
    # Test 3: Case Law
    print("\n" + "-"*70)
    print("\n📍 TEST 3: Case Law Search (Nachbarrecht)")
    result = search_case_law(
        query="Einfriedung Nachbarrecht Grenzabstand"
    )
    print(f"   Keywords extracted: {result.keywords_used}")
    print(f"   Results found: {len(result.results)}")
    print("\n" + result.formatted_text[:2000])
    
    print("\n" + "="*70)
    print("✅ TESTS COMPLETE")
    print("="*70)

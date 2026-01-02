"""
Swiss Legal Agent - Direct Test (no menu)

Just run: python test_cantonal.py
"""

import os
from dotenv import load_dotenv

load_dotenv("keys.env")

# ============================================================
# SETUP
# ============================================================

def get_tavily_client():
    from tavily import TavilyClient
    api_key = os.getenv("TAVILY_API_KEY", "")
    if api_key.startswith("tvly-tvly-"):
        api_key = api_key.replace("tvly-tvly-", "tvly-", 1)
    return TavilyClient(api_key=api_key)

def get_openai_client():
    from openai import OpenAI
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ============================================================
# TEST CONFIG - CHANGE THESE!
# ============================================================

QUESTION = "Wie hoch darf ein Zaun um die Grenzen sein?"
CANTON = "AI"
CANTON_NAME = "Appenzell Innerrhoden"


# ============================================================
# RUN TEST
# ============================================================

print("\n" + "="*70)
print("🏔️  CANTONAL LAW AGENT TEST")
print("="*70)
print(f"\n📝 Question: {QUESTION}")
print(f"🏛️  Canton: {CANTON} ({CANTON_NAME})")

# ---- STEP 1: SEARCH ----
print("\n" + "-"*50)
print("STEP 1: TAVILY SEARCH")
print("-"*50)

tavily = get_tavily_client()

# PRIORITÄT 1: Direkt auf kantonaler Gesetzessammlung suchen!
# Mit OPTIMALEN Tavily-Parametern für vollen Content
queries = [
    # Direkt auf ai.clex.ch (offizielle Gesetzessammlung AI)
    "site:ai.clex.ch Einfriedung",
    "site:ai.clex.ch BauV 700.010",
]

all_results = []
for q in queries:
    print(f"\n🔎 Query: {q}")
    
    # OPTIMALE PARAMETER für Rechtsrecherche:
    results = tavily.search(
        query=q,
        search_depth="advanced",          # Mehrere Chunks pro URL
        max_results=5,
        include_raw_content=True,         # VOLLER Seiteninhalt! 🎯
        chunks_per_source=3,              # Bis zu 3 relevante Abschnitte
        country="switzerland",            # Boost für CH-Resultate
    )
    
    found = results.get('results', [])
    print(f"   Found: {len(found)} results")
    
    for r in found:
        title = r.get('title', '')[:60]
        url = r.get('url', '')
        has_raw = "raw_content" in r and r["raw_content"]
        raw_len = len(r.get("raw_content", "") or "") if has_raw else 0
        
        # Markiere ob URL zum richtigen Kanton gehört
        is_relevant = any(x in url.lower() for x in ['ai.clex.ch', 'ai.ch', 'appenzell', 'entscheidsuche'])
        marker = "✅" if is_relevant else "⚠️"
        raw_info = f"[RAW: {raw_len} chars]" if has_raw else "[no raw]"
        print(f"   {marker} {title}")
        print(f"      {url} {raw_info}")
    
    all_results.extend(found)

# Dedupe AND filter for relevance
seen = set()
unique = []
irrelevant_domains = ['tourismus', 'booking', 'tripadvisor', 'wikipedia', 'wwfost.ch']

for r in all_results:
    url = r.get("url", "").lower()
    
    # Skip if already seen
    if url in seen:
        continue
    seen.add(url)
    
    # Skip irrelevant domains
    if any(d in url for d in irrelevant_domains):
        print(f"   ❌ Filtered out (irrelevant): {url[:60]}")
        continue
    
    unique.append(r)

print(f"\n📊 Total unique relevant results: {len(unique)}")

# ---- SORT BY RELEVANCE (keyword matches) ----
keywords_to_find = ['art. 30', 'art.30', '1.5 m', '1,5 m', 'grenzeinfriedung', 'einfriedung']

def score_result(r):
    """Score results by how many relevant keywords they contain"""
    raw = (r.get('raw_content', '') or '').lower()
    content = (r.get('content', '') or '').lower()
    text = raw + content
    
    score = 0
    for kw in keywords_to_find:
        if kw in text:
            # Art. 30 and specific measurements are most important
            if 'art. 30' in kw or 'art.30' in kw:
                score += 10
            elif '1.5' in kw or '1,5' in kw:
                score += 5
            else:
                score += 1
    
    # Bonus for raw_content (full text available)
    if raw:
        score += 2
        
    return score

# Sort by relevance score (highest first)
unique_sorted = sorted(unique, key=score_result, reverse=True)

print("\n📊 Results sorted by relevance:")
for i, r in enumerate(unique_sorted[:5], 1):
    url = r.get('url', '')[:60]
    score = score_result(r)
    print(f"   {i}. Score {score}: {url}...")

# ---- SHOW RAW CONTENT (if available) ----
print("\n" + "-"*50)
print("RAW CONTENT CHECK (from include_raw_content=True)")
print("-"*50)

# Only check top 3 most relevant
for r in unique_sorted[:3]:
    url = r.get('url', '')
    raw = r.get('raw_content', '')
    
    if raw:
        # Search for relevant keywords in raw content
        keywords = ['einfriedung', 'art. 30', 'art.30', '1.5 m', '1,5 m', 'grenzeinfriedung']
        found_kw = [kw for kw in keywords if kw.lower() in raw.lower()]
        
        print(f"\n📄 {url}")
        print(f"   Raw content length: {len(raw)} chars")
        print(f"   Found keywords: {found_kw}")
        
        # Show relevant excerpt if Art. 30 found
        if 'art. 30' in raw.lower() or 'art.30' in raw.lower():
            # Find position of Art. 30
            pos = raw.lower().find('art. 30')
            if pos == -1:
                pos = raw.lower().find('art.30')
            
            # Extract context around it
            start = max(0, pos - 50)
            end = min(len(raw), pos + 500)
            excerpt = raw[start:end]
            print(f"\n   📍 EXCERPT around Art. 30:")
            print(f"   {excerpt}")
    else:
        print(f"\n📄 {url}")
        print(f"   ⚠️ No raw_content returned")

# Format for LLM - USE RAW CONTENT if available!
# Only use TOP 3 most relevant results to avoid noise
search_text_lines = []
search_text_lines.append("=== GESETZESTEXTE (sortiert nach Relevanz) ===\n")

for i, r in enumerate(unique_sorted[:3], 1):  # TOP 3 ONLY!
    title = r.get('title', 'No title')
    url = r.get('url', 'N/A')
    raw = r.get('raw_content', '')
    content = r.get('content', '')
    
    search_text_lines.append(f"[{i}] {title}")
    search_text_lines.append(f"    URL: {url}")
    
    # Prefer raw_content (full text) over content (snippet)
    if raw and len(raw) > len(content):
        # SMART EXTRACTION: Find the most relevant part around Art. 30 / Einfriedung
        raw_lower = raw.lower()
        
        # Find best position to extract from
        best_pos = -1
        for search_term in ['art. 30 einfriedung', 'art.30 einfriedung', 'grenzeinfriedung', '1.5 m', '1,5 m']:
            pos = raw_lower.find(search_term)
            if pos != -1:
                best_pos = pos
                break
        
        if best_pos != -1:
            # Extract 2000 chars around the best match
            start = max(0, best_pos - 200)
            end = min(len(raw), best_pos + 1800)
            excerpt = raw[start:end]
            search_text_lines.append(f"    RELEVANTER AUSSCHNITT:")
            search_text_lines.append(f"    ...{excerpt}...")
        else:
            # Fallback: just first part
            text = ' '.join(raw.split())[:2000]
            search_text_lines.append(f"    VOLLTEXT (Anfang):\n    {text}")
    else:
        text = ' '.join(content.split())[:500]
        search_text_lines.append(f"    SNIPPET:\n    {text}")
    
    search_text_lines.append("")
    
search_results_text = "\n".join(search_text_lines)


# ---- STEP 2: BUILD PROMPTS ----
print("\n" + "-"*50)
print("STEP 2: PROMPTS")
print("-"*50)

system_prompt = f"""Du bist Spezialist für kantonales Schweizer Recht, speziell für {CANTON_NAME}.

DEINE AUFGABE:
Beantworte die Frage basierend auf den Suchergebnissen.

⚠️ KRITISCHE REGELN:
1. "VOLLTEXT" Abschnitte enthalten den ECHTEN Gesetzestext - HÖCHSTE PRIORITÄT!
2. Suche nach konkreten Artikeln (z.B. "Art. 30") und Zahlen (z.B. "1.5 m")
3. Zitiere den Gesetzestext WÖRTLICH wenn möglich
4. URLs mit "ai.clex.ch" sind die OFFIZIELLE Gesetzessammlung von {CANTON_NAME}
5. ERFINDE KEINE Zahlen oder Gesetze!

OUTPUT-FORMAT:
1. **Relevanter Gesetzesartikel**: [Artikelnummer + exakter Wortlaut]
2. **Quelle**: [URL]
3. **Antwort**: [Direkte Antwort auf die Frage]

Antworte auf Deutsch."""

user_prompt = f"""KANTON: {CANTON} ({CANTON_NAME})

SUCHERGEBNISSE:
{search_results_text}

FRAGE: {QUESTION}"""

print("\n🔧 SYSTEM PROMPT:")
print("-"*30)
print(system_prompt)

print("\n👤 USER PROMPT:")
print("-"*30)
print(user_prompt[:1500])
if len(user_prompt) > 1500:
    print("... [truncated]")


# ---- STEP 3: LLM CALL ----
print("\n" + "-"*50)
print("STEP 3: LLM RESPONSE (GPT-4o-mini)")
print("-"*50)

openai = get_openai_client()
response = openai.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    temperature=0.1
)

llm_response = response.choices[0].message.content

print("\n🤖 LLM RESPONSE:")
print("-"*30)
print(llm_response)

print("\n" + "="*70)
print("✅ TEST COMPLETE")
print("="*70)

"""
Swiss Legal Agent - Isolated Testing Tool

Test each agent individually with full visibility into:
- Search queries sent to Tavily
- Raw search results
- System prompt
- User prompt  
- LLM response

Usage:
    python test_agents.py
"""

import os
import json
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

def get_anthropic_client():
    from anthropic import Anthropic
    return Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


# ============================================================
# SIMPLE SEARCH FUNCTIONS (no complex processing)
# ============================================================

def search_tavily(query: str, max_results: int = 5, domains: list = None) -> dict:
    """Simple Tavily search with full result visibility"""
    tavily = get_tavily_client()
    
    params = {
        "query": query,
        "max_results": max_results,
        "search_depth": "advanced"
    }
    if domains:
        params["include_domains"] = domains
    
    results = tavily.search(**params)
    return results


def format_search_results(results: dict) -> str:
    """Format search results for LLM consumption"""
    lines = []
    for i, r in enumerate(results.get("results", []), 1):
        lines.append(f"[{i}] {r.get('title', 'No title')}")
        lines.append(f"    URL: {r.get('url', 'N/A')}")
        lines.append(f"    {r.get('content', '')[:500]}")
        lines.append("")
    return "\n".join(lines)


# ============================================================
# TEST: CANTONAL LAW AGENT
# ============================================================

def test_cantonal_law_agent(question: str, canton: str, canton_name: str):
    """Test cantonal law agent in isolation"""
    
    print("\n" + "="*70)
    print("🏔️  CANTONAL LAW AGENT TEST")
    print("="*70)
    print(f"\n📝 Question: {question}")
    print(f"🏛️  Canton: {canton} ({canton_name})")
    
    # ---- STEP 1: SEARCH ----
    print("\n" + "-"*50)
    print("STEP 1: TAVILY SEARCH")
    print("-"*50)
    
    # Build queries with quotes for exact match
    queries = [
        f'"{canton_name}" Baugesetz Einfriedung Zaun',
        f'site:entscheidsuche.ch "{canton_name}" Einfriedung',
    ]
    
    all_results = []
    for q in queries:
        print(f"\n🔎 Query: {q}")
        results = search_tavily(q, max_results=3)
        print(f"   Found: {len(results.get('results', []))} results")
        
        for r in results.get("results", []):
            print(f"   • {r.get('title', '')[:60]}")
            print(f"     {r.get('url', '')}")
        
        all_results.extend(results.get("results", []))
    
    # Dedupe
    seen = set()
    unique = []
    for r in all_results:
        url = r.get("url", "")
        if url not in seen:
            seen.add(url)
            unique.append(r)
    
    search_results_text = format_search_results({"results": unique})
    
    print(f"\n📊 Total unique results: {len(unique)}")
    
    # ---- STEP 2: BUILD PROMPTS ----
    print("\n" + "-"*50)
    print("STEP 2: PROMPTS")
    print("-"*50)
    
    system_prompt = f"""Du bist Spezialist für kantonales Schweizer Recht.

DEINE AUFGABE:
Analysiere die Suchergebnisse und identifiziere relevante kantonale Bestimmungen.

WICHTIG:
- Zitiere NUR was in den Suchergebnissen steht
- Erfinde KEINE Gesetze oder Zahlen
- Wenn keine konkreten Angaben gefunden: sage das ehrlich
- Alle URLs vollständig übernehmen

OUTPUT-FORMAT:
1. **Gefundene Gesetze**: [Mit URLs]
2. **Konkrete Angaben**: [NUR aus Quellen!]
3. **Empfehlung**: Offizielle Quelle konsultieren

Antworte auf Deutsch."""

    user_prompt = f"""KANTON: {canton} ({canton_name})

SUCHERGEBNISSE:
{search_results_text}

FRAGE: {question}"""

    print("\n🔧 SYSTEM PROMPT:")
    print("-"*30)
    print(system_prompt[:500] + "..." if len(system_prompt) > 500 else system_prompt)
    
    print("\n👤 USER PROMPT:")
    print("-"*30)
    print(user_prompt[:1000] + "..." if len(user_prompt) > 1000 else user_prompt)
    
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
    print("✅ CANTONAL LAW AGENT TEST COMPLETE")
    print("="*70)
    
    return {
        "search_results": search_results_text,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "llm_response": llm_response
    }


# ============================================================
# TEST: PRIMARY LAW AGENT
# ============================================================

def test_primary_law_agent(question: str):
    """Test primary law agent in isolation"""
    
    print("\n" + "="*70)
    print("🏛️  PRIMARY LAW AGENT TEST")
    print("="*70)
    print(f"\n📝 Question: {question}")
    
    # ---- STEP 1: SEARCH ----
    print("\n" + "-"*50)
    print("STEP 1: TAVILY SEARCH (Fedlex)")
    print("-"*50)
    
    query = f"{question}"
    print(f"\n🔎 Query: {query}")
    print(f"   Domain: fedlex.admin.ch")
    
    results = search_tavily(query, max_results=5, domains=["fedlex.admin.ch"])
    
    print(f"   Found: {len(results.get('results', []))} results")
    for r in results.get("results", []):
        print(f"   • {r.get('title', '')[:60]}")
        print(f"     {r.get('url', '')}")
    
    search_results_text = format_search_results(results)
    
    # ---- STEP 2: BUILD PROMPTS ----
    print("\n" + "-"*50)
    print("STEP 2: PROMPTS")
    print("-"*50)
    
    system_prompt = """Du bist Spezialist für Schweizer Bundesrecht.

DEINE AUFGABE:
Identifiziere die relevanten Gesetzesbestimmungen für die Frage.

HYBRID-ANSATZ:
1. Recherche zuerst: Prüfe was in den Suchergebnissen steht
2. Eigenes Wissen ergänzt: Wenn wichtige Artikel fehlen, ergänze
3. Recherche hat Vorrang: Bei Widersprüchen gilt die Recherche

WICHTIG:
- Alle URLs aus Suchergebnissen übernehmen
- Kennzeichne: "Aus Recherche:" vs "Ergänzend relevant:"
- ERFINDE KEINE Artikel

OUTPUT:
1. **Aus der Recherche**: [Artikel mit URLs]
2. **Ergänzend relevant**: [Weitere Artikel aus Fachwissen]

Antworte auf Deutsch."""

    user_prompt = f"""SUCHERGEBNISSE:
{search_results_text}

FRAGE: {question}"""

    print("\n🔧 SYSTEM PROMPT:")
    print("-"*30)
    print(system_prompt[:500] + "...")
    
    print("\n👤 USER PROMPT:")
    print("-"*30)
    print(user_prompt[:800] + "..." if len(user_prompt) > 800 else user_prompt)
    
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
    print("✅ PRIMARY LAW AGENT TEST COMPLETE")
    print("="*70)
    
    return {
        "search_results": search_results_text,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "llm_response": llm_response
    }


# ============================================================
# TEST: CASE LAW AGENT
# ============================================================

def test_case_law_agent(question: str):
    """Test case law agent in isolation"""
    
    print("\n" + "="*70)
    print("⚖️  CASE LAW AGENT TEST")
    print("="*70)
    print(f"\n📝 Question: {question}")
    
    # ---- STEP 1: SEARCH ----
    print("\n" + "-"*50)
    print("STEP 1: TAVILY SEARCH (BGer)")
    print("-"*50)
    
    queries = [
        (f"BGE {question}", ["bger.ch"]),
        (f"Bundesgericht {question}", ["entscheidsuche.ch"]),
    ]
    
    all_results = []
    for query, domains in queries:
        print(f"\n🔎 Query: {query}")
        print(f"   Domains: {domains}")
        
        results = search_tavily(query, max_results=3, domains=domains)
        print(f"   Found: {len(results.get('results', []))} results")
        
        for r in results.get("results", []):
            print(f"   • {r.get('title', '')[:60]}")
        
        all_results.extend(results.get("results", []))
    
    search_results_text = format_search_results({"results": all_results})
    
    # ---- STEP 2: BUILD PROMPTS ----
    print("\n" + "-"*50)
    print("STEP 2: PROMPTS")
    print("-"*50)
    
    system_prompt = """Du bist Spezialist für Schweizer Bundesgerichts-Rechtsprechung.

DEINE AUFGABE:
Identifiziere relevante Rechtsprechung für die Frage.

WICHTIG - BGE-NUMMERN:
- BGE-Nummern NUR zitieren wenn sie in der Recherche vorkommen
- Wenn keine BGE gefunden: Beschreibe die allgemeine Rechtsprechungslinie OHNE konkrete Nummern
- ERFINDE NIEMALS BGE-Nummern!

OUTPUT:
1. **Aus der Recherche**: [BGE/Urteile mit URLs]
2. **Allgemeine Rechtsprechung**: [Generelle Linie ohne konkrete Nummern wenn nicht aus Recherche]

Antworte auf Deutsch."""

    user_prompt = f"""SUCHERGEBNISSE:
{search_results_text}

FRAGE: {question}"""

    print("\n🔧 SYSTEM PROMPT:")
    print("-"*30)
    print(system_prompt[:500] + "...")
    
    print("\n👤 USER PROMPT:")
    print("-"*30)
    print(user_prompt[:800] + "..." if len(user_prompt) > 800 else user_prompt)
    
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
    print("✅ CASE LAW AGENT TEST COMPLETE")
    print("="*70)
    
    return {
        "search_results": search_results_text,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "llm_response": llm_response
    }


# ============================================================
# INTERACTIVE MENU
# ============================================================

def main():
    print("\n" + "="*70)
    print("🇨🇭 SWISS LEGAL AGENT - ISOLATED TESTING")
    print("="*70)
    
    while True:
        print("\n📋 MENU:")
        print("1. Test Cantonal Law Agent (Appenzell Zaun)")
        print("2. Test Primary Law Agent (Einfriedung ZGB)")
        print("3. Test Case Law Agent (Einfriedung BGE)")
        print("4. Custom Cantonal Law Test")
        print("5. Custom Primary Law Test")
        print("6. Custom Case Law Test")
        print("q. Quit")
        
        choice = input("\nWähle Option: ").strip().lower()
        
        if choice == "1":
            test_cantonal_law_agent(
                question="Wie hoch darf ein Zaun um die Grenzen sein?",
                canton="AI",
                canton_name="Appenzell Innerrhoden"
            )
        
        elif choice == "2":
            test_primary_law_agent(
                question="Einfriedung Zaun Grenzabstand Nachbarrecht"
            )
        
        elif choice == "3":
            test_case_law_agent(
                question="Einfriedung Zaun Nachbarrecht"
            )
        
        elif choice == "4":
            question = input("Frage: ")
            canton = input("Kanton (z.B. ZH): ").upper()
            canton_name = input("Kantonsname (z.B. Zürich): ")
            test_cantonal_law_agent(question, canton, canton_name)
        
        elif choice == "5":
            question = input("Frage: ")
            test_primary_law_agent(question)
        
        elif choice == "6":
            question = input("Frage: ")
            test_case_law_agent(question)
        
        elif choice == "q":
            print("\n👋 Bye!")
            break
        
        else:
            print("❌ Ungültige Option")


if __name__ == "__main__":
    main()

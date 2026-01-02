# Swiss Legal Research Agent - Handover Documentation

## 📅 Last Updated: January 2025

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                               │
│                      Gradio Web UI (ui.py)                          │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         ORCHESTRATOR                                 │
│                          (GPT-4o)                                   │
│                                                                     │
│  Aufgaben:                                                          │
│  • Spracherkennung (DE/FR/IT/EN)                                   │
│  • Kanton/Gemeinde-Erkennung                                        │
│  • Rechtsgebiet-Klassifikation                                      │
│  • Generierung von: key_terms, related_domains, search_hints        │
│  • Relevante Artikel identifizieren                                 │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐
│  PRIMARY LAW AGENT  │ │   CASE LAW AGENT    │ │ CANTONAL LAW AGENT  │
│   (GPT-4o-mini)     │ │   (GPT-4o-mini)     │ │   (GPT-4o-mini)     │
│                     │ │                     │ │                     │
│ 1. PLAN: Generate   │ │ 1. PLAN: Generate   │ │ 1. PLAN: Generate   │
│    search queries   │ │    search queries   │ │    search queries   │
│                     │ │                     │ │                     │
│ 2. EXECUTE: Tavily  │ │ 2. EXECUTE: Tavily  │ │ 2. EXECUTE: Tavily  │
│    search           │ │    search           │ │    + PDF extraction │
│                     │ │                     │ │                     │
│ Output: RAW results │ │ Output: RAW results │ │ Output: RAW results │
└─────────────────────┘ └─────────────────────┘ └─────────────────────┘
                    │               │               │
                    └───────────────┼───────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        ANALYSIS AGENT                                │
│                          (Claude)                                   │
│                                                                     │
│  Empfängt: RAW search results von allen Agents                     │
│  Aufgabe: Synthese zur finalen strukturierten Antwort              │
│  Output: Kurze Antwort, Rechtliche Grundlagen, BGE, Empfehlungen   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📁 File Structure

```
swiss-legal-agent/
├── ui.py                 # Main Gradio UI + Pipeline (2200+ lines)
├── prompts.py            # All prompts for agents (670+ lines)
├── tools.py              # Tavily search functions, PDF extraction
├── smart_search.py       # Enhanced search with Merkblatt detection
├── agents.py             # Legacy agent definitions (mostly unused)
├── main.py               # CLI entry point
├── mcp_server.py         # MCP server for Claude Desktop
├── requirements.txt      # Dependencies
├── keys.env.template     # API key template
└── HANDOVER.md          # This file
```

---

## 🔑 Key Components

### 1. Orchestrator (ui.py, lines ~980-1180)

**Model:** GPT-4o (strong model for critical analysis)

**Input:** User question + optional document

**Output JSON:**
```json
{
    "canton": "AI",
    "canton_name": "Appenzell Innerrhoden",
    "response_language": "German",
    "legal_domain": "Erbrecht",
    "related_domains": ["Pflichtteilsrecht", "Herabsetzungsklage", "Steuerrecht"],
    "legal_context": "...",
    "relevant_articles": ["Art. 626 ZGB", "Art. 627 ZGB"],
    "key_terms": ["Erbvorbezug", "Ausgleichung", "Pflichtteil"],
    "synonyms": {"Erbvorbezug": ["Vorempfang"]},
    "search_hints": {
        "primary_law": "Art. 626-628 ZGB Ausgleichung",
        "case_law": "BGE Erbvorbezug Ausgleichung",
        "cantonal_law": "..."
    }
}
```

### 2. Search Agents (ui.py, lines ~480-860)

**Model:** GPT-4o-mini (fast/cheap for query generation)

**Flow:**
1. **PLANNING:** LLM generates 2-3 search queries based on orchestrator context
2. **EXECUTE:** Run Tavily searches
3. **OUTPUT:** Raw results (NO LLM analysis - saves 50% of LLM calls!)

**Agent-Specific Contexts (token-optimized):**
```python
# PRIMARY LAW AGENT receives:
{
    "question": "...",
    "legal_domain": "Erbrecht",
    "related_domains": "Pflichtteilsrecht, Herabsetzungsklage",  # String!
    "relevant_articles": "Art. 626, 627, 522, 560 ZGB",
    "key_terms": "Erbvorbezug, Ausgleichung, Pflichtteil",
    "search_hint": "Art. 626-628 ZGB Ausgleichung"
}

# CANTONAL LAW AGENT additionally receives:
{
    "canton": "AI",
    "canton_name": "Appenzell Innerrhoden",
    "commune": "..."
}
```

### 3. Analysis Agent (ui.py, lines ~1470-1600)

**Model:** Claude (via Anthropic API)

**Input:** Raw search results from all agents + orchestrator context

**Output:** Structured legal answer in user's language

---

## 🔄 Recent Optimizations (January 2025)

### 1. Removed Step 3 LLM Analysis from Agents

**Before:** Each agent had 2 LLM calls (Planning + Analysis)
**After:** Each agent has 1 LLM call (Planning only)

**Savings:**
- 50% fewer GPT-4o-mini calls (4 instead of 8)
- ~90% fewer tokens per agent
- Claude does ALL synthesis (was doing it anyway)

### 2. Lists → Strings in Agent Contexts

**Before:**
```python
"relevant_articles": ["Art. 626 ZGB", "Art. 627 ZGB", "Art. 522 ZGB"]
```

**After:**
```python
"relevant_articles": "Art. 626, 627, 522 ZGB"
```

### 3. Added `related_domains` Field

Orchestrator now identifies related legal domains for comprehensive search:
- "Erbrecht" → ["Pflichtteilsrecht", "Herabsetzungsklage", "Steuerrecht"]
- "Nachbarrecht" → ["Kantonales Baurecht"]

### 4. Simplified Orchestrator Prompt

Removed 150+ lines of detailed legal mappings. Now trusts GPT-4o's knowledge:
```
Du bist ein erfahrener SCHWEIZER RECHTSRECHERCHE-SPEZIALIST.
Deine Aufgabe: Analysiere die Rechtsfrage und liefere optimalen Kontext.
```

---

## ⚠️ Known Issues

### 1. Fedlex Returns JS-Only Pages
Many Fedlex results show "JavaScript-fähigen Browser" instead of content.
**Workaround:** tools.py tries to extract from PDFs when available.

### 2. Prompts Tab Empty in UI
The "Prompts" tab in agent panels may not display correctly after Step 3 removal.
**Root cause:** `system_prompt` and `user_prompt` fields are now empty since agents only do planning.
**Fix needed:** Either remove the Prompts tab or show the planning prompt instead.

### 3. Variable Search Quality
Tavily results vary between runs. Same question can get different BGE.

### 4. Bewertungszeitpunkt Confusion
For "Erbvorbezug" questions, the system sometimes confuses:
- Art. 630 ZGB: Bewertung zum Zeitpunkt des **Erbgangs** (correct for Ausgleichung)
- vs. Zeitpunkt der Übertragung

---

## 🧪 Testing

### Run UI locally:
```bash
cd swiss-legal-agent
source venv/bin/activate  # or create venv
pip install -r requirements.txt
cp keys.env.template keys.env
# Edit keys.env with your API keys
python ui.py
```

### Test Questions:
1. **Kantonal:** "Wie hoch darf ein Zaun in Appenzell sein?"
   - Should detect: canton=AI, search cantonal Bauverordnung

2. **Erbrecht:** "Was muss ich bei einem Erbvorbezug einer Liegenschaft beachten?"
   - Should detect: related_domains including Pflichtteilsrecht

3. **Sozialversicherung:** "Muss ich für eine gelegentliche Putzhilfe AHV abrechnen?"
   - Should find: Bagatellgrenze CHF 2'300

4. **Multilingual:** "Puis-je résilier mon bail?"
   - Should respond in French

---

## 📊 Cost Analysis

### Per Query (typical):

| Component | Model | Calls | Est. Tokens | Est. Cost |
|-----------|-------|-------|-------------|-----------|
| Orchestrator | GPT-4o | 1 | ~2000 | ~$0.02 |
| Primary Law Planning | GPT-4o-mini | 1 | ~300 | ~$0.0001 |
| Case Law Planning | GPT-4o-mini | 1 | ~300 | ~$0.0001 |
| Cantonal Law Planning | GPT-4o-mini | 1 | ~300 | ~$0.0001 |
| Cantonal Case Law Planning | GPT-4o-mini | 1 | ~300 | ~$0.0001 |
| Analysis | Claude | 1 | ~8000 | ~$0.08 |
| Tavily | - | 8-12 | - | ~$0.01 |
| **Total** | | | | **~$0.11** |

### Optimization Impact:
- **Before:** ~$0.15-0.20 per query (8 agent LLM calls)
- **After:** ~$0.11 per query (4 agent LLM calls)
- **Savings:** ~30-40%

---

## 🚀 Future Improvements

### High Priority:
1. **Fix Fedlex scraping** - Use API or better PDF extraction
2. **Cache common queries** - Erbvorbezug, Kündigung, etc.
3. **Validate BGE citations** - Check if cited BGE actually exists
4. **Fix UI Prompts tab** - Show planning prompts or remove tab

### Medium Priority:
5. **Add confidence scores** - How sure is the system?
6. **Source highlighting** - Show which source said what
7. **Multi-turn conversations** - Follow-up questions

### Low Priority:
8. **French/Italian optimization** - Test with Romandie/Ticino questions
9. **Document analysis** - Better contract/letter parsing
10. **API endpoint** - REST API for integration

---

## 🔧 Configuration

### Environment Variables (keys.env):
```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
TAVILY_API_KEY=tvly-...
```

### Model Configuration (ui.py, get_llm function):
```python
def get_llm(role: str = "agent"):
    if role == "orchestrator":
        return ChatOpenAI(model="gpt-4o", temperature=0.1)
    elif role == "agent":
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
```

---

## 📝 Prompt Locations

| Prompt | File | Function/Variable |
|--------|------|-------------------|
| Orchestrator | ui.py | `orchestrator_prompt` (line ~980) |
| Primary Law Planning | prompts.py | `PRIMARY_LAW_PLANNING_PROMPT` |
| Case Law Planning | prompts.py | `CASE_LAW_PLANNING_PROMPT` |
| Cantonal Law Planning | prompts.py | `CANTONAL_LAW_PLANNING_PROMPT` |
| Cantonal Case Law Planning | prompts.py | `CANTONAL_CASE_LAW_PLANNING_PROMPT` |
| Analysis | prompts.py | `get_analysis_prompt()` |

---

## 🔍 Data Flow Detail

### 1. User Input
```
Question: "Was muss ich bei einem Erbvorbezug beachten?"
Document: (optional PDF/image)
```

### 2. Orchestrator Analysis
```python
{
    "legal_domain": "Erbrecht",
    "related_domains": ["Pflichtteilsrecht", "Herabsetzungsklage"],
    "relevant_articles": ["Art. 626 ZGB", "Art. 627 ZGB", "Art. 522 ZGB"],
    "key_terms": ["Erbvorbezug", "Ausgleichung", "Pflichtteil"],
    "search_hints": {
        "primary_law": "Art. 626-628 ZGB Ausgleichung",
        "case_law": "BGE Erbvorbezug Ausgleichung Pflichtteil"
    }
}
```

### 3. Agent Planning (each agent)
```
Input: Orchestrator context as formatted string
Output: 
SEARCH_QUERIES:
1. Erbvorbezug Liegenschaften site:fedlex.admin.ch
2. Art. 626 ZGB Ausgleichung
3. Pflichtteil Herabsetzung site:admin.ch
```

### 4. Search Execution
```python
# Each query → Tavily search → Raw results
# Results include: URL, title, snippet, relevance score
```

### 5. Analysis Agent Input
```
=== ORCHESTRATOR KONTEXT ===
⚖️ RECHTSGEBIET: Erbrecht
✅ RELEVANTE ARTIKEL: Art. 626, 627, 522 ZGB
=== ENDE ORCHESTRATOR KONTEXT ===

PRIMARY LAW FINDINGS:
[Raw search results - 10-20k chars]

CASE LAW FINDINGS:
[Raw search results - 10-20k chars]

USER QUESTION: Was muss ich bei einem Erbvorbezug beachten?
```

### 6. Final Output
```
Kurze Antwort: ...
Rechtliche Grundlagen: Art. 626 ZGB...
Relevante Rechtsprechung: BGE 133 III 416...
Erläuterung: ...
Empfehlung: ...
Quellen: ...
```

---

## 🤝 Handover Notes

### What Works Well:
- ✅ Canton detection (Appenzell → AI, Tessin → TI)
- ✅ Key terms generation by orchestrator
- ✅ Parallel agent execution (4 agents run simultaneously)
- ✅ Tavily search with domain prioritization
- ✅ Claude synthesis produces good structured output
- ✅ Multilingual support (DE/FR/IT/EN)

### What Needs Attention:
- ⚠️ Raw search results sometimes contain irrelevant content
- ⚠️ Step 3 removal may have broken UI prompts display
- ⚠️ Bewertungszeitpunkt for Erbvorbezug needs legal review
- ⚠️ Fedlex JavaScript-only pages not properly handled

### Architecture Decisions Explained:
- **GPT-4o for Orchestrator:** Critical for accurate legal domain detection
- **GPT-4o-mini for Agents:** Only generates search queries, doesn't need intelligence
- **Claude for Analysis:** Best at synthesis, handles multilingual output well
- **Raw results to Claude:** Avoids double-processing, reduces cost significantly
- **No synonyms in agent contexts:** Removed for token savings, LLM handles this

---

## 📞 Support

For questions about this codebase, check:
1. This HANDOVER.md
2. Comments in ui.py (especially around pipeline functions)
3. Test files: test_orchestrator.py, test_agents.py, test_cantonal.py

---

*Document generated: January 2025*

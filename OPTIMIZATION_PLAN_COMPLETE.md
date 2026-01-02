# Swiss Legal Research Agent - Optimierungsplan

## 📊 Analyse-Grundlage

**Analysierte Quellen:**
1. HANDOVER.md - Architektur, Known Issues, Data Flow
2. Best Practices Feedback - State-of-the-art Empfehlungen  
3. Recherche-Beispiel - Konkrete Probleme identifiziert
4. Code: ui.py, tools.py, prompts.py, smart_search.py

---

## 🎯 Design-Prinzipien

**Alle Optimierungen MÜSSEN:**
- ✅ **Generisch** sein - auf alle Rechtsgebiete anwendbar
- ✅ **LLM-basiert** - das Sprachmodell seine juristische Expertise nutzen lassen
- ✅ **Skalierbar** - keine Einzelfall-Instruktionen
- ❌ **NICHT rule-based** - keine hardcodierten juristischen Regeln für spezifische Rechtsgebiete

---

## 🔴 KRITISCH - Implementiert

### 1. UI: Prompts Tab zeigt jetzt Inhalte

**Problem:** `"Prompts will appear here when the agent runs"`

**Root Cause (aus HANDOVER Known Issue #2):**
Nach Entfernung von Step 3 wurden `system_prompt` und `user_prompt` nicht mehr gesetzt.

**Fix:** Alle 4 Agent-Funktionen setzen jetzt beide Prompts:
- `run_primary_law_agent_work()` 
- `run_case_law_agent_work()`
- `run_cantonal_law_agent_work()`
- `run_cantonal_case_law_agent_work()`

### 2. Generische Präzisions-Instruktionen (prompts.py)

**Problem:** LLM verwechselte manchmal Grundregel mit Ausnahme aus BGE-Zitaten.

**Fix (GENERISCH, nicht rule-based):**
```
=== RECHTLICHE PRÄZISION ===
1. GRUNDREGEL vs. AUSNAHME unterscheiden:
   - Identifiziere zuerst die gesetzliche GRUNDREGEL
   - BGE-Entscheide behandeln oft AUSNAHMEN oder Spezialfälle
   - Stelle klar, was Regel und was Ausnahme ist
   
2. ZEITPUNKTE und FRISTEN präzise angeben:
   - Welcher Zeitpunkt ist rechtlich massgebend?
   - Gibt es Ausnahmen von diesem Zeitpunkt?

3. QUELLENKONTEXT beachten:
   - BGE-Zitate im Kontext lesen
   - Nicht Ausnahme-Rechtsprechung als Grundregel darstellen
```

**Warum generisch:** Funktioniert für ALLE Rechtsgebiete (Mietrecht, Arbeitsrecht, Erbrecht, etc.)

### 3. Claim-Evidence-Mapping (prompts.py)

**Fix:**
```
Jede rechtliche Aussage einer Quelle zuordnen:
• Aus Recherche: "Gemäss Art. X..." / "Laut BGE Y..."
• Aus Rechtswissen: Als allgemeines Wissen kennzeichnen
• Bei Unsicherheit: Ehrlich kommunizieren
```

### 4. Citation-Validierung mit Warnings (ui.py)

**Neu:** Nach Analysis Agent werden Zitate validiert:
- Warnung bei Zitaten ohne Link aus Recherche
- Warnung bei ungültigen Links
- Logs im Orchestrator-Panel

---

## 🟡 HOCH - Diese Woche

### 5. Zweistufiges Retrieval (aus Best Practices)

**Aktuell:** Alle Suchen mit `include_raw_content=True` → Token-Verschwendung

**Empfehlung:**
```python
# Stufe A: Light search (nur Snippets)
light_results = tavily.search(query, max_results=10, include_raw_content=False)

# Stufe B: Score, dann nur Top-3 mit raw_content
scored = score_results(light_results)
top_urls = [r['url'] for r in scored[:3]]

detailed = tavily.search(query, include_raw_content=True, 
                         include_domains=extract_domains(top_urls))
```

**Erwartete Einsparung:** 30-50% weniger Tokens

### 6. Retrieval Budgeting + Early Stop

```python
MAX_QUERIES = 3
MAX_RESULTS_PER_QUERY = 5
EARLY_STOP_SCORE = 80

for query in planned_queries[:MAX_QUERIES]:
    results = search(query, max_results=MAX_RESULTS_PER_QUERY)
    best_score = max(r['relevance_score'] for r in results)
    if best_score >= EARLY_STOP_SCORE:
        break  # Genug gute Ergebnisse
```

### 7. Fedlex JS-Only Filter verbessern

**Problem:** Viele Fedlex-Ergebnisse zeigen nur "JavaScript-fähigen Browser"

**Fix in tools.py:**
```python
# Früher filtern - vor Scoring
if "javascript" in content.lower() and len(content) < 500:
    continue  # Skip JS-only results
```

---

## 🟢 MITTEL - Nächste Woche

### 8. Caching Layer

```python
# Level 1: Tavily Query Cache (24h TTL)
@lru_cache(maxsize=200)
def cached_search(query_hash: str, params_hash: str):
    return tavily.search(...)

# Level 2: Content Cache (7d TTL)
URL_CONTENT_CACHE = {}
```

### 9. Kanton/Gemeinde-Gating (aus Best Practices)

**Aktuell:** Kommunale Suche bei jeder Gemeinde-Erkennung

**Besser:** Nur wenn Rechtsgebiet typisch kommunal:
```python
COMMUNAL_DOMAINS = ["Baurecht", "Planungsrecht", "Polizeiverordnung"]

if session.commune and session.legal_domain in COMMUNAL_DOMAINS:
    run_communal_search()
```

---

## 🔵 NIEDRIG - Später

### 10. BGE-Format-Validierung
- Prüfen ob Format stimmt: BGE [Band] [Abteilung] [Seite]
- Keine inhaltliche Validierung (wäre rule-based)

### 11. Confidence Scores (LLM-basiert)
- LLM selbst einschätzen lassen wie sicher die Antwort ist
- Basierend auf: Quellenqualität, Übereinstimmung

### 12. Pipeline-Konsolidierung
- `agents.py` ist "Legacy" → entfernen oder mit ui.py vereinen

---

## 📁 Geänderte Dateien

| Datei | Änderungen |
|-------|------------|
| `prompts.py` | Generische Präzisions-Instruktionen, Claim-Evidence |
| `ui.py` | system_prompts in Agents, Citation-Validierung |

---

## 🧪 Test-Empfehlung

1. **Beliebige komplexe Rechtsfrage testen** - prüfen ob Grundregel/Ausnahme klar unterschieden wird
2. **UI Prompts Tab** - alle Agent-Panels sollten Inhalte zeigen
3. **Orchestrator Log** - Citation-Validierung sollte erscheinen

---

## ❌ Was wir NICHT machen

- Keine spezifischen Instruktionen für einzelne Rechtsgebiete
- Keine hardcodierten juristischen Regeln
- Keine Einzelfall-Fixes die nicht skalieren
- Das LLM (Claude) hat juristische Expertise - wir geben generische Leitplanken, nicht Detailwissen

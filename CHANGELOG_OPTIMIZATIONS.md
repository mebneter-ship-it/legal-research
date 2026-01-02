# Swiss Legal Agent - Optimierungen (Best Practice / State-of-the-art)

## Version: Januar 2025

### 🎯 Design-Prinzipien

**Alle Änderungen sind:**
- ✅ Generisch - funktionieren für alle Rechtsgebiete
- ✅ LLM-basiert - vertrauen auf GPT-4o/Claude Expertise
- ✅ Skalierbar - keine Einzelfall-Instruktionen oder Beispiele
- ❌ NICHT rule-based - keine hardcodierten juristischen Regeln

---

## 🚀 Implementierte Optimierungen

### 1. CACHING LAYER (tools.py)

**Aus Best Practices:** "30-50% Kostenersparnis möglich"

```python
# In-memory Cache für Tavily-Suchen (24h TTL)
_TAVILY_CACHE: Dict[str, Tuple[float, any]] = {}
_CACHE_TTL_SECONDS = 3600 * 24

# Funktionen: _cache_key(), _get_cached(), _set_cached()
```

**Effekt:** Wiederholte Suchen werden aus Cache bedient.

### 2. RETRIEVAL BUDGETING (tools.py)

**Aus Best Practices:** "Kostenkontrolle durch Limits"

```python
RETRIEVAL_CONFIG = {
    "max_queries_per_agent": 3,
    "max_results_per_query": 5,
    "early_stop_score": 75,
    "top_k_for_raw_content": 3,
}
```

**Effekt:** Verhindert Token-Explosion bei komplexen Fragen.

### 3. TWO-STAGE RETRIEVAL (tools.py)

**Aus Best Practices:** "Light search → Top-K raw_content"

```python
def two_stage_search(client, query, ...):
    # Stage 1: Light search (nur Snippets)
    # Stage 2: raw_content nur für Top-K Results
```

**Effekt:** Gleiche Qualität bei weniger Tokens.

### 4. EARLY-STOP LOGIC (tools.py)

**Aus Best Practices:** "Stop wenn gute Ergebnisse gefunden"

```python
def should_early_stop(results, threshold=75):
    # True wenn beste Relevanz >= Schwelle
```

**Effekt:** Schnellere Recherchen bei klaren Fragen.

### 5. CLAIM-EVIDENCE-MAPPING (prompts.py)

**Aus Best Practices:** "Jede Aussage muss belegt sein"

```
=== CLAIM-EVIDENCE-MAPPING (PFLICHT!) ===
Jede rechtliche Aussage MUSS einer Quelle zugeordnet sein:
• Aus Recherche: "Gemäss Art. X [LINK]..."
• Aus Rechtswissen: "[Allgemeines Rechtswissen]"

VERBOTEN:
• Rechtliche Aussagen ohne Quellenangabe
• Erfundene BGE-Nummern oder Links
```

**Effekt:** Transparente, nachprüfbare Antworten.

### 6. RECHTLICHE PRÄZISION (prompts.py)

**Generische Instruktionen für alle Rechtsgebiete:**

```
1. GESETZESTEXT HAT VORRANG
2. ZEITPUNKTE KRITISCH PRÜFEN
3. QUELLEN GENAU LESEN

WARNUNG: Verwechsle NIEMALS Ausnahme-Rechtsprechung mit der Grundregel!
```

**Effekt:** Bessere Unterscheidung von Regel vs. Ausnahme.

### 7. KEINE site:-OPERATOREN (prompts.py)

**Problem:** Tavily unterstützt keine Google-Operatoren wie `site:fedlex.admin.ch`

**Fix:** Alle Planning Prompts enthalten jetzt:
```
WICHTIG: Verwende KEINE site:-Operatoren! Nur natürliche Suchbegriffe.
```

**Effekt:** Bessere Suchergebnisse.

### 8. UI PROMPTS TAB (ui.py)

Alle 4 Search-Agents setzen jetzt `system_prompt` UND `user_prompt`:

- 🏛️ Primary Law Agent
- ⚖️ Case Law Agent
- 🏔️ Cantonal Law Agent
- ⚖️🏔️ Cantonal Case Law Agent

**Effekt:** Transparenz über Agent-Verhalten in UI.

### 9. UI FIXES (ui.py)

**CSS Overflow Fix:**
- Document Preview überläuft nicht mehr seinen Container
- `st.code()` statt `st.text()` für scrollbare Vorschau

**Session State Stabilität:**
- Agent-Panels werden direkt aus Session State gerendert
- Keine `st.empty()` Platzhalter mehr (waren instabil bei Reruns)
- Ergebnisse bleiben nach Expander-Klicks erhalten

---

## 📁 Geänderte Dateien

| Datei | Änderungen |
|-------|------------|
| `tools.py` | +150 Zeilen (Caching, Two-Stage, Early-Stop, Budgeting) |
| `prompts.py` | +40 Zeilen (Claim-Evidence, Präzision, keine site:-Operatoren) |
| `ui.py` | +80 Zeilen (system_prompts, CSS Fix, Session State Stabilität) |

---

## ⚠️ Bekannte Limitationen

### Bewertungszeitpunkt-Problem
Das LLM verwechselt manchmal die Grundregel (Art. 630 ZGB: Zeitpunkt des Erbgangs) 
mit Ausnahme-Rechtsprechung (BGE: Zeitpunkt der Übertragung bei vorzeitiger Veräusserung).

**Warum nicht gelöst:** 
- Spezifische Instruktionen für Erbrecht würden nicht skalieren
- Generische Instruktionen werden nicht zuverlässig befolgt
- Das ist ein fundamentales LLM-Synthese-Problem

**Empfehlung:** Disclaimer für rechtliche Beratung hinzufügen.

---

## 🧪 Test-Empfehlung

1. **Wiederholte Suchen** - prüfen ob Caching greift
2. **UI Prompts Tab** - sollte jetzt Inhalte zeigen
3. **Expander klicken** - Ergebnisse sollten erhalten bleiben
4. **Document Preview** - kein Overflow mehr
5. **Verschiedene Rechtsgebiete** - Mietrecht, Arbeitsrecht, Erbrecht

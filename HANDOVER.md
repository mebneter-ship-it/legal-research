# Swiss Legal Research Assistant - Übergabedokument

## Projektübersicht

Ein Multi-Agenten-System für die Recherche im Schweizer Recht, gebaut mit LangGraph, LangChain und Tavily Search. Das System durchsucht offizielle Schweizer Rechtsquellen (Fedlex, BGer, kantonale Portale) und synthetisiert die Ergebnisse zu umfassenden Rechtsanalysen.

**Hauptfunktionen:**
- Multi-Agenten-Pipeline (Bundesrecht → Kantonales Recht → Rechtsprechung → Analyse)
- Viersprachige Unterstützung (Deutsch, Französisch, Italienisch, Englisch)
- Korrekte Zitat-Formate (Art. X OR, BGE XXX III XXX, etc.)
- Dokumenten-Upload für rechtliche Prüfung (PDF, DOCX, TXT)
- Echtzeit-Entwickler-UI mit vollständiger Agenten-Sichtbarkeit
- MCP-Server für Claude Desktop Integration

---

## Architektur

### Systemübersicht

```
┌─────────────────────────────────────────────────────────────────┐
│                      BENUTZER-EINGABE                            │
│              (Rechtsfrage + optionales Dokument)                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        ORCHESTRATOR (LLM)                        │
│  • Analysiert Frage mit LLM                                     │
│  • Erkennt Kanton intelligent (nicht per Regex!)                │
│  • Bestimmt Antwort-Sprache                                     │
│  • Verwaltet Pipeline-Ausführung                                │
│  • Vermeidet falsche Erkennungen ("einfach so" ≠ Solothurn)    │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│ BUNDESRECHT   │     │ KANTONALES    │     │ BUNDES-       │
│ AGENT         │     │ RECHT AGENT   │     │ GERICHT AGENT │
│               │     │ (optional)    │     │               │
│ • Tavily      │     │               │     │ • Tavily      │
│   fedlex.admin│     │ • Kantonale   │     │   bger.ch     │
│   .ch         │     │   Gesetze     │     │ • BGE/ATF/DTF │
│ • OR, ZGB, BV │     │ • Kantonale   │     │   Extraktion  │
│   Extraktion  │     │   Gerichte    │     │               │
│               │     │ • Gemeinde-   │     │               │
│               │     │   ordnungen   │     │               │
└───────┬───────┘     └───────┬───────┘     └───────┬───────┘
        │                     │                     │
        └──────────┬──────────┴──────────┬──────────┘
                   ▼                     │
┌────────────────────────────────────────┴────────────────────────┐
│                       ANALYSE AGENT                              │
│  • Empfängt: Bundesrecht + Kantonales Recht + BGer + Dokument   │
│  • Methodik: Allgemeiner Rahmen → Sonderbestimmungen → Anwendung│
│  • Output: Strukturierte Analyse mit klickbaren Zitaten         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FINALE AUSGABE                              │
│  • Sprachspezifische Überschriften und Zitate                   │
│  • Korrekte Referenzformate (Art. X OR, BGE XXX III XXX)        │
│  • Strukturiert: Zusammenfassung → Rahmen → Anwendung           │
└─────────────────────────────────────────────────────────────────┘
```

### Dynamische Pipeline

Die Pipeline passt sich automatisch an die Frage an:

**Ohne Kanton/Gemeinde:**
```
Schritt 1          Schritt 2          Schritt 3
Bundesrecht   →    Bundesgericht  →   Analyse
Agent              Agent              Agent
```

**Mit Kanton/Gemeinde erkannt:**
```
Schritt 1          Schritt 2          Schritt 3          Schritt 4
Bundesrecht   →    Kantonales     →   Bundesgericht  →   Analyse
Agent              Recht Agent        Agent              Agent
```

---

## Prozess- und Datenfluss

### 1. Eingabeverarbeitung

```
Benutzer-Eingabe
      │
      ▼
┌─────────────────────────────────────┐
│ 1. Spracherkennung                  │
│    detect_language(question)        │
│    → "German" / "French" / etc.     │
├─────────────────────────────────────┤
│ 2. Kanton/Gemeinde-Erkennung        │
│    detect_canton_and_commune(q)     │
│    → {"canton": "ZH", "commune":    │
│       "Zürich"} oder None           │
├─────────────────────────────────────┤
│ 3. Pipeline-Konfiguration           │
│    → 3 Agents (ohne Kanton)         │
│    → 4 Agents (mit Kanton)          │
└─────────────────────────────────────┘
```

### 2. Bundesrecht Agent - Datenfluss

```
EINGANG:
├── Frage (string)
├── Dokument (optional, string)
└── Erkannte Sprache (string)

VERARBEITUNG:
├── 1. Tavily API Aufruf
│      Query: "{frage} site:fedlex.admin.ch OR site:admin.ch"
│      → Rohe Suchergebnisse (HTML-Snippets, URLs)
│
├── 2. Prompt-Konstruktion
│      System: PRIMARY_LAW_SYSTEM_PROMPT
│      User: Suchergebnisse + Frage + Sprache
│
└── 3. LLM Aufruf (GPT-4o-mini / Claude)
       → Strukturierte Analyse mit Art.-Zitaten

AUSGANG:
├── search_results (string): Rohe Tavily-Ergebnisse
├── llm_response (string): Analysiertes Bundesrecht
└── data_sent: {analysis: "...", length: N}
       ↓
       Weitergabe an → Analyse Agent
```

### 3. Kantonales Recht Agent - Datenfluss (falls aktiviert)

```
EINGANG:
├── Frage (string)
├── Kanton (z.B. "ZH")
├── Gemeinde (optional, z.B. "Zürich")
└── Erkannte Sprache (string)

VERARBEITUNG:
├── 1. Kantonale Gesetzessuche
│      search_cantonal_law(frage, "ZH")
│      → Domains: zh.ch, zhlex.zh.ch, lexfind.ch
│
├── 2. Kantonale Rechtsprechung
│      search_cantonal_case_law(frage, "ZH")
│      → Obergericht, Verwaltungsgericht
│
├── 3. Gemeinderecht (falls Gemeinde erkannt)
│      search_communal_law(frage, "Zürich", "ZH")
│      → Gemeindeordnung, Baureglement, Zonenplan
│
└── 4. LLM Analyse
       → Kantonale Bestimmungen extrahiert

AUSGANG:
├── search_results (string): Kombinierte kantonale Quellen
├── llm_response (string): Kantonale Rechtsanalyse
└── data_sent → Analyse Agent
```

### 4. Bundesgericht Agent - Datenfluss

```
EINGANG:
├── Frage (string)
└── Erkannte Sprache (string)

VERARBEITUNG:
├── 1. Tavily API Aufruf
│      Query: "{frage} site:bger.ch BGE"
│      → BGE/ATF/DTF Entscheide
│
├── 2. Prompt-Konstruktion
│      System: CASE_LAW_SYSTEM_PROMPT
│      User: Suchergebnisse + Frage + Sprache
│
└── 3. LLM Aufruf
       → Rechtsprechungsanalyse mit BGE-Zitaten

AUSGANG:
├── search_results (string): BGer-Suchergebnisse
├── llm_response (string): Rechtsprechungsanalyse
└── data_sent → Analyse Agent
```

### 5. Analyse Agent - Datenfluss

```
EINGANG (aggregiert):
├── from_primary_law_agent:
│   └── llm_response: "Art. 335c OR..."
├── from_cantonal_law_agent (falls vorhanden):
│   └── llm_response: "§ 123 PBG ZH..."
├── from_case_law_agent:
│   └── llm_response: "BGE 142 III 579..."
├── from_orchestrator:
│   ├── question: "Kann ich..."
│   └── document: "Vertrag vom..."
└── detected_language: "German"

VERARBEITUNG:
├── 1. Prompt-Konstruktion
│      Kombiniert alle Eingaben
│      Fügt Methodik-Anweisungen hinzu
│
└── 2. LLM Synthese
       → Strukturierte Gesamtanalyse

AUSGANG:
└── final_output (string):
    ## ZUSAMMENFASSUNG
    ## ALLGEMEINER RECHTSRAHMEN
    ## BESONDERE BESTIMMUNGEN
    ## ANWENDUNG AUF IHREN FALL
    ## EMPFEHLUNGEN
    ## QUELLENVERZEICHNIS
```

### 6. Kompletter Datenfluss (Übersicht)

```
                    ┌──────────────┐
                    │   Benutzer   │
                    │   Eingabe    │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │ Orchestrator │
                    │              │
                    │ • Sprache    │
                    │ • Kanton     │
                    │ • Pipeline   │
                    └──────┬───────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
         ▼                 ▼                 ▼
   ┌───────────┐    ┌───────────┐    ┌───────────┐
   │ Bundesrecht│    │ Kantonal  │    │   BGer    │
   │   Agent   │    │   Agent   │    │   Agent   │
   └─────┬─────┘    └─────┬─────┘    └─────┬─────┘
         │                │                │
         │   ┌────────────┼────────────┐   │
         │   │            │            │   │
         ▼   ▼            ▼            ▼   ▼
   ┌─────────────────────────────────────────────┐
   │              ANALYSE AGENT                   │
   │                                              │
   │  Eingaben:                                   │
   │  ├── Bundesrecht-Analyse (2000 chars)       │
   │  ├── Kantonales Recht (1500 chars)          │
   │  ├── BGer-Analyse (1800 chars)              │
   │  └── Dokument (falls vorhanden)             │
   │                                              │
   │  Verarbeitung:                              │
   │  └── GPT-4o-mini / Claude Synthese          │
   │                                              │
   │  Ausgabe:                                    │
   │  └── Strukturierte Rechtsanalyse            │
   └──────────────────────┬──────────────────────┘
                          │
                          ▼
                  ┌───────────────┐
                  │ Finale Ausgabe│
                  │               │
                  │ • Markdown    │
                  │ • Zitate      │
                  │   (Art. X OR) │
                  └───────────────┘
```

---

## Risiken und Mitigationsstrategien

### 1. Risiken bei den Tools (Tavily Search)

| Risiko | Beschreibung | Schweregrad | Mitigation |
|--------|--------------|-------------|------------|
| **Unvollständige Suchergebnisse** | Tavily findet nicht alle relevanten Quellen auf Fedlex oder BGer | Hoch | Mehrere Suchqueries verwenden; Domain-spezifische Suchen kombinieren; Benutzer auf mögliche Lücken hinweisen |
| **Veraltete Ergebnisse** | Gecachte oder alte Seiten werden zurückgegeben | Mittel | Suchergebnisse mit Datum anzeigen; Benutzer auf mögliche Veraltung hinweisen; Fedlex-Links zeigen immer aktuelle Fassung |
| **Rate Limiting** | Tavily API hat Anfragelimits (1000/Monat im Free-Tier) | Mittel | API-Quotas überwachen; Caching für häufige Anfragen implementieren; bei Bedarf auf kostenpflichtigen Plan upgraden |
| **Nicht indizierte Quellen** | Kantonale Quellen sind oft schlecht von Suchmaschinen indiziert | Hoch | lexfind.ch als Aggregator zusätzlich nutzen; direkte kantonale Portal-Domains einbeziehen |
| **Sprachliche Einschränkungen** | Suchbegriffe müssen zur Quellsprache passen (FR-Frage findet DE-Quelle nicht) | Mittel | Mehrsprachige Suchqueries generieren; Kantonssprache berücksichtigen |

**Empfohlene Mitigationen:**
```python
# 1. Mehrere Suchstrategien kombinieren
results = []
results.append(search_swiss_primary_law(query))      # Fedlex
if canton:
    results.append(search_cantonal_law(query, canton))  # Kantonal
results.append(search_general_legal(query))          # Allgemein als Fallback

# 2. Retry-Logik bei Fehlern
@retry(max_attempts=3, backoff_seconds=2)
def search_with_retry(query):
    return tavily.search(query)

# 3. Caching häufiger Anfragen
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_search(query_hash):
    return tavily.search(query)
```

### 2. Risiken bei den Modellen (LLM)

| Risiko | Beschreibung | Schweregrad | Mitigation |
|--------|--------------|-------------|------------|
| **Halluzinationen** | LLM erfindet nicht existierende Artikel oder BGE-Nummern | Kritisch | Explizite Prompt-Anweisungen "NEVER invent citations"; nur aus Suchergebnissen zitieren; Zitat-Validierung implementieren |
| **Falsche Zitate** | Artikel-Nummern oder BGE-Referenzen sind falsch formatiert oder existieren nicht | Kritisch | Format-Validierung (BGE \d+ [IVX]+ \d+); Benutzer zur Verifikation über Links auffordern |
| **Inkonsistente Sprache** | Output mischt Sprachen (z.B. deutsche Überschriften mit französischem Inhalt) | Mittel | Dreifache Sprach-Emphasis in Prompts; Ausgabesprache explizit am Anfang und Ende des Prompts betonen |
| **Kontext-Überlauf** | Zu viele Suchergebnisse sprengen das Kontext-Fenster | Mittel | Suchergebnisse auf 4000-6000 Zeichen begrenzen; Relevanz-Ranking vor Übergabe |
| **API-Ausfälle** | OpenAI oder Anthropic API nicht erreichbar | Mittel | Provider-Fallback implementieren (OpenAI ↔ Anthropic); Fehlerbehandlung mit klarer Benutzermeldung |
| **Kosten** | Hohe API-Kosten bei vielen Anfragen | Mittel | gpt-4o-mini statt gpt-4 verwenden; Token-Verbrauch monitoren; Caching von Ergebnissen |

**Empfohlene Mitigationen:**
```python
# 1. Zitat-Validierung
import re

def validate_bge_citation(citation):
    """Prüft ob BGE-Format plausibel ist"""
    pattern = r'BGE \d{2,3} [IVX]+ \d+'
    return bool(re.match(pattern, citation))

def validate_article_citation(citation):
    """Prüft ob Artikel-Format plausibel ist"""
    pattern = r'Art\. \d+[a-z]?( Abs\. \d+)?'
    return bool(re.search(pattern, citation))

# 2. Provider-Fallback
def get_llm_with_fallback():
    try:
        return ChatOpenAI(model="gpt-4o-mini")
    except Exception:
        return ChatAnthropic(model="claude-3-haiku-20240307")

# 3. Sprach-Konsistenz prüfen
def check_language_consistency(response, expected_lang):
    detected = detect_language(response[:500])
    if detected != expected_lang:
        logging.warning(f"Sprachmismatch: erwartet {expected_lang}, erkannt {detected}")
```

### 3. Risiken bei den Resultaten

| Risiko | Beschreibung | Schweregrad | Mitigation |
|--------|--------------|-------------|------------|
| **Rechtliche Fehler** | Analyse ist juristisch falsch oder irreführend | Kritisch | Klaren Disclaimer in jede Ausgabe; keine Rechtsberatung; Anwaltsempfehlung |
| **Veraltetes Recht** | Zitiertes Recht wurde zwischenzeitlich geändert | Hoch | Fedlex-Links zeigen automatisch aktuelle Fassung; Recherche-Datum anzeigen |
| **Fehlende Relevanz** | Wichtige Bestimmungen werden nicht gefunden | Hoch | Mehrere Suchstrategien; Benutzer explizit auf mögliche Lücken hinweisen |
| **Falsche Anwendung** | Allgemeines Recht wird falsch auf Spezialfall angewandt | Hoch | Methodik General→Spezial→Anwendung strikt einhalten; Unsicherheiten klar kommunizieren |
| **Jurisdiktions-Fehler** | Falsches kantonales Recht zitiert (z.B. ZH statt BE) | Mittel | Kanton-Erkennung verbessern; bei Unsicherheit Benutzer nach Kanton fragen |
| **Übervertrauen** | Benutzer verlässt sich blind auf die Analyse | Kritisch | Prominenter Disclaimer; alle Zitate als klickbare Links zur Verifikation |

**Empfohlene Mitigationen:**

```markdown
## Automatischer Disclaimer (in jeder Ausgabe)

⚠️ **Wichtiger Hinweis:**
Diese Analyse dient ausschliesslich zu Informationszwecken und stellt 
keine Rechtsberatung dar.

- Verifizieren Sie alle Zitate über die verlinkten Originalquellen
- Konsultieren Sie für verbindliche Auskünfte einen Rechtsanwalt
- Stand der Recherche: [aktuelles Datum]
- Die Rechtslage kann sich seit der Recherche geändert haben
```

### 4. Risiko-Matrix (Gesamtübersicht)

```
                        SCHWEREGRAD
                    Niedrig    Mittel    Hoch      Kritisch
              ┌──────────┬──────────┬──────────┬──────────┐
    Hoch      │          │ Rate     │ Unvoll-  │          │
              │          │ Limiting │ ständige │          │
              │          │          │ Suche    │          │
WAHRSCHEIN-   ├──────────┼──────────┼──────────┼──────────┤
LICHKEIT      │ Sprach-  │ API-     │ Veraltete│ Halluzi- │
    Mittel    │ mixing   │ Kosten   │ Quellen  │ nationen │
              │          │          │          │          │
              ├──────────┼──────────┼──────────┼──────────┤
    Niedrig   │          │ API-     │ Falsche  │ Rechtl.  │
              │          │ Ausfall  │ Jurisd.  │ Fehler   │
              └──────────┴──────────┴──────────┴──────────┘
```

### 5. Kritische Mitigationen (Must-Have)

1. **Disclaimer in jeder Ausgabe** (bereits implementiert)
   ```python
   DISCLAIMER = """
   ⚠️ Diese Analyse ist keine Rechtsberatung. 
   Verifizieren Sie alle Angaben über die Links und konsultieren Sie 
   bei Bedarf einen Rechtsanwalt.
   """
   ```

2. **Link-Validierung** (neu implementiert)
   - URLs werden aus Suchergebnissen extrahiert (nicht konstruiert)
   - Jeder Link wird auf Erreichbarkeit geprüft
   - Ungültige Links werden automatisch entfernt
   - Validierungsbericht im Orchestrator sichtbar

3. **Keine erfundenen Zitate** (in Prompts verankert)
   ```
   CRITICAL: NEVER invent or fabricate citations. 
   Only cite sources that appear in the search results.
   ```

4. **Fehler-Logging**
   ```python
   import logging
   logging.basicConfig(level=logging.INFO)
   
   # Bei jedem API-Aufruf
   logging.info(f"Tavily search: {query}")
   logging.info(f"LLM call: {model}, tokens: {usage}")
   ```

---

## Dateistruktur

```
swiss-legal-agent/
├── .env.template          # Umgebungsvariablen-Vorlage
├── .env                   # Eigene Konfiguration (nicht im Git)
├── requirements.txt       # Python-Abhängigkeiten
├── README.md             # Benutzer-Dokumentation (Englisch)
├── HANDOVER.md           # Dieses Dokument (Deutsch)
│
├── prompts.py            # Alle Agenten-Prompts (editierbar)
├── tools.py              # Tavily-Suchwrapper + Kanton-Erkennung
├── agents.py             # LangGraph-Agenten (für CLI)
├── main.py               # Kommandozeilen-Interface
├── ui.py                 # Streamlit Entwickler-UI
├── mcp_server.py         # MCP-Server für Claude Desktop
│
└── claude_desktop_config.json  # Beispiel MCP-Konfiguration
```

---

## Wichtige Dateien erklärt

### `prompts.py`
Zentrale Sammlung aller Agenten-Prompts (vereinfacht):

**Kernprinzipien:**
- NUR zitieren was in Suchergebnissen vorkommt
- NIEMALS Artikel oder BGE-Nummern erfinden
- Lieber "keine Informationen gefunden" als Halluzinationen
- Vollständige Referenzen mit SR-Nummern

**Prompts:**
- `PRIMARY_LAW_SYSTEM_PROMPT` - Bundesrecht-Recherche
- `CASE_LAW_SYSTEM_PROMPT` - Rechtsprechungs-Recherche
- `ANALYSIS_SYSTEM_PROMPT` - Synthese und Analyse
- `detect_language()` - Spracherkennung

### `tools.py`
Tavily-Suchwrapper und Hilfsfunktionen:
- `search_swiss_primary_law()` → fedlex.admin.ch, admin.ch
- `search_swiss_case_law()` → bger.ch
- `search_cantonal_law()` → Kantonale Portale
- `search_cantonal_case_law()` → Kantonale Gerichte
- `search_communal_law()` → Gemeindeordnungen
- `detect_canton_and_commune()` → Kanton/Gemeinde-Erkennung (inkl. Appenzell AI/AR)
- `extract_and_validate_citations()` → Link-Validierung
- `create_validated_output()` → Entfernt ungültige Links

### `ui.py`
Streamlit Entwickler-UI mit drei Bereichen:
- **Sidebar:** Frage-Eingabe, Dokument-Upload, Run/Reset
- **Hauptbereich:** Research-Output mit Markdown
- **Agent Activity:** Live-Panels für jeden Agenten + Link-Validierung

---

## Sprachunterstützung

### Zitat-Formate nach Sprache

Jedes Zitat enthält die vollständige Referenznummer (SR/RS für Bundesrecht):

| Gesetz | DE | FR | IT |
|--------|----|----|----| 
| Obligationenrecht | Art. X Abs. Y OR (SR 220) | Art. X al. Y CO (RS 220) | Art. X cpv. Y CO (RS 220) |
| Zivilgesetzbuch | Art. X ZGB (SR 210) | Art. X CC (RS 210) | Art. X CC (RS 210) |
| Bundesverfassung | Art. X BV (SR 101) | Art. X Cst. (RS 101) | Art. X Cost. (RS 101) |
| Arbeitsgesetz | Art. X ArG (SR 822.11) | Art. X LTr (RS 822.11) | Art. X LL (RS 822.11) |
| Datenschutzgesetz | Art. X DSG (SR 235.1) | Art. X LPD (RS 235.1) | Art. X LPD (RS 235.1) |

| Rechtsprechung | DE | FR | IT |
|----------------|----|----|----| 
| Leitentscheid | BGE 142 III 579 E. 4.2 | ATF 142 III 579 consid. 4.2 | DTF 142 III 579 consid. 4.2 |
| Nicht publiziert | Urteil 4A_123/2020 vom 15.3.2021 | Arrêt 4A_123/2020 du 15.3.2021 | Sentenza 4A_123/2020 del 15.3.2021 |

### Kantonales Recht
Kantonale Gesetze werden mit der kantonalen Sammlungsnummer zitiert:
- Zürich: § 123 PBG (LS 700.1) - Planungs- und Baugesetz
- Bern: Art. 15 BauG (BSG 721.0) - Baugesetz

### Link-Validierung

Der Orchestrator validiert alle Links in der Ausgabe:
1. **URLs aus Suchergebnissen extrahieren** - Nur tatsächlich gefundene URLs werden verwendet
2. **HTTP-Prüfung** - Jeder Link wird auf Erreichbarkeit geprüft (3s Timeout)
3. **Ungültige Links entfernen** - Nicht erreichbare Links werden durch Klartext ersetzt
4. **Validierungsbericht** - Im "🔗 Links" Tab des Orchestrators sichtbar

Die Agents konstruieren keine URLs selbst - sie verwenden nur URLs die in den Tavily-Suchergebnissen erscheinen.

---

## Kanton/Gemeinde-Erkennung

```python
detect_canton_and_commune("Kann ich in Zürich bauen?")
# → {"canton": "ZH", "commune": "Zürich"}

detect_canton_and_commune("Baurecht Kanton Luzern")
# → {"canton": "LU", "commune": None}

detect_canton_and_commune("Was sind die Kündigungsfristen?")
# → {"canton": None, "commune": None}  # Keine kantonale Suche
```

Unterstützt: Alle 26 Kantone + ~100 grössere Gemeinden

---

## Installation

```bash
# 1. Umgebung einrichten
cd ~/legal-research-agent
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Konfiguration
cp .env.template .env
# .env bearbeiten mit API-Keys

# 3. Ausführen
streamlit run ui.py
```

---

## Bekannte Probleme

| Problem | Lösung |
|---------|--------|
| Doppelte API-Key-Präfixe (`tvly-tvly-`) | Wird automatisch korrigiert |
| OpenAI Quota überschritten | Credits hinzufügen oder zu Anthropic wechseln |
| Kanton nicht erkannt | Gemeinde-Mapping in `SWISS_COMMUNES` erweitern |
| Falsche Spracherkennung | Wortlisten in `detect_language()` erweitern |

---

## Versionshistorie

| Datum | Änderungen |
|-------|------------|
| 2024-12-28 | Initiale Entwicklung |
| 2024-12-28 | Streamlit UI mit Agenten-Sichtbarkeit |
| 2024-12-28 | Mehrsprachige Unterstützung (DE/FR/IT/EN) |
| 2024-12-28 | Kantonale/kommunale Rechtssuche |
| 2024-12-28 | Appenzell AI/AR Erkennung |
| 2024-12-28 | Zurück zu einfachen Original-Prompts |
| 2024-12-28 | LLM Orchestrator für Kanton/Sprache-Erkennung |
| 2024-12-28 | Reset-Button löscht alle Eingaben |
| 2024-12-28 | **Orchestrator erweitert:** legal_context + search_topics für breiteren Kontext |
| 2024-12-28 | **Benchmark:** Direkter LLM-Vergleich ohne Agenten |
| 2024-12-28 | **Tavily:** Breitere Suche für kantonales Recht |

---

*Letzte Aktualisierung: 28. Dezember 2024*

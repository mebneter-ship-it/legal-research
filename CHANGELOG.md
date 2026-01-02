# Swiss Legal Agent - Änderungsprotokoll

## Version: Optimierung Januar 2025

### 🎯 Design-Prinzip

**Alle Änderungen sind:**
- ✅ Generisch - funktionieren für alle Rechtsgebiete
- ✅ LLM-basiert - vertrauen auf GPT-4o/Claude Expertise
- ✅ Skalierbar - keine Einzelfall-Instruktionen
- ❌ NICHT rule-based - keine hardcodierten juristischen Regeln

---

### 🔴 Implementierte Fixes

#### 1. Few-Shot Examples komplett entfernt (ui.py)

**Problem:** GPT-4o kopierte Beispiele 1:1 statt selbst zu analysieren

**Warum Examples schlecht sind:**
- Veraltet - moderne LLMs verstehen JSON ohne Beispiele
- Riskant - LLMs kopieren statt denken
- Token-Verschwendung - ~500 Tokens für nichts
- Nicht skalierbar - kann nicht alle Rechtsgebiete abdecken

**Fix:** Orchestrator-Prompt ist jetzt rein instruktionsbasiert:
- Keine Examples mehr
- Klare Aufgaben-Liste
- JSON-Schema als einzige Format-Vorgabe
- Vertraut auf GPT-4o's juristische Expertise

#### 2. UI Prompts Tab zeigt jetzt Inhalte (ui.py)

Alle 4 Search-Agents setzen jetzt `system_prompt` UND `user_prompt`.

#### 3. Generische Präzisions-Instruktionen (prompts.py)

```
=== RECHTLICHE PRÄZISION ===
1. GRUNDREGEL vs. AUSNAHME unterscheiden
2. ZEITPUNKTE und FRISTEN präzise angeben  
3. QUELLENKONTEXT beachten
```

#### 4. Claim-Evidence-Mapping (prompts.py)

Jede rechtliche Aussage muss einer Quelle zugeordnet werden.

#### 5. Citation-Validierung (ui.py)

Warnings bei Zitaten ohne Link aus Recherche.

---

### 📁 Geänderte Dateien

| Datei | Änderungen |
|-------|------------|
| `ui.py` | Examples entfernt, system_prompts, Citation-Validierung |
| `prompts.py` | Generische Präzision, Claim-Evidence |

---

### 🧪 Test-Empfehlung

1. **Erbvorbezug-Frage erneut testen** - `related_domains` sollten jetzt INDIVIDUELL sein
2. **Andere Rechtsgebiete testen** - Mietrecht, Arbeitsrecht, etc.
3. **Orchestrator-Output prüfen** - Keine kopierten Beispiele mehr

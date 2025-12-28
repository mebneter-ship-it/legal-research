"""
Swiss Legal Research - Agent Prompts

Simple, effective prompts that work.
"""

# ============================================================
# PRIMARY LAW AGENT PROMPT
# ============================================================

PRIMARY_LAW_SYSTEM_PROMPT = """Du bist Spezialist für Schweizer Bundesrecht.

DEINE AUFGABE:
Identifiziere die relevanten Gesetzesbestimmungen für die Frage.

HYBRID-ANSATZ:
1. **Recherche zuerst**: Prüfe was in den Suchergebnissen steht
2. **Eigenes Wissen ergänzt**: Wenn wichtige Artikel fehlen, ergänze aus deinem Wissen
3. **Recherche hat Vorrang**: Bei Widersprüchen gilt was die Recherche sagt

WICHTIG:
- Kennzeichne: "Aus Recherche: Art. X" vs "Ergänzend relevant: Art. Y"
- ERFINDE KEINE Artikel - nur echte Schweizer Gesetze
- Antworte in der Sprache der Frage

ZITAT-FORMAT je nach Sprache:
- Deutsch: Art. 335c Abs. 1 OR (SR 220), Art. 684 ZGB (SR 210)
- Français: Art. 335c al. 1 CO (RS 220), Art. 684 CC (RS 210)
- Italiano: Art. 335c cpv. 1 CO (RS 220), Art. 684 CC (RS 210)

OUTPUT:
1. **Aus der Recherche**: [Artikel die in den Suchergebnissen vorkommen]
2. **Ergänzend relevant**: [Weitere einschlägige Artikel aus deinem Fachwissen]
3. **Lücken**: [Was noch fehlt]"""


PRIMARY_LAW_USER_PROMPT = """SEARCH RESULTS:
{search_results}

USER QUESTION:
{question}

{document_context}"""


# ============================================================
# CANTONAL LAW AGENT PROMPT
# ============================================================

CANTONAL_LAW_SYSTEM_PROMPT = """Du bist Spezialist für kantonales Schweizer Recht.

DEINE AUFGABE:
Identifiziere die relevanten kantonalen Bestimmungen für die Frage.

ANSATZ:
1. **Recherche zuerst**: Was steht in den Suchergebnissen?
2. **Allgemeine Hinweise**: Welche Art von kantonalem Recht könnte relevant sein?

WICHTIG - Kantonales Recht variiert stark:
- Konkrete Paragraphen/Nummern NUR aus der Recherche zitieren
- Ohne Recherche-Treffer: Allgemein beschreiben welche Art Gesetz relevant sein könnte
- Beispiel: "Das kantonale Baugesetz regelt typischerweise..." (ohne konkrete §-Nummer)
- ERFINDE KEINE kantonalen Gesetzesnummern!

OUTPUT:
1. **Aus der Recherche**: [Konkrete kantonale Bestimmungen mit Fundstelle]
2. **Typischerweise relevant**: [Art von kantonalem Recht das gelten könnte]
3. **Empfehlung**: [Wo der Nutzer nachschauen sollte]

Antworte in der Sprache der Frage."""


CANTONAL_LAW_USER_PROMPT = """CANTON: {canton} ({canton_name})
{commune_info}

SEARCH RESULTS:
{search_results}

USER QUESTION:
{question}"""


# ============================================================
# CASE LAW AGENT PROMPT
# ============================================================

CASE_LAW_SYSTEM_PROMPT = """Du bist Spezialist für Schweizer Bundesgerichts-Rechtsprechung.

DEINE AUFGABE:
Identifiziere relevante Rechtsprechung für die Frage.

HYBRID-ANSATZ:
1. **Recherche zuerst**: Zitiere BGE/Urteile die in den Suchergebnissen vorkommen
2. **Allgemeine Rechtsprechung**: Beschreibe die generelle Rechtsprechungslinie
3. **Recherche hat Vorrang**: Konkrete BGE aus der Suche sind verlässlicher

WICHTIG - BGE-NUMMERN:
- BGE-Nummern NUR zitieren wenn sie in der Recherche vorkommen
- Wenn keine BGE gefunden: Beschreibe die allgemeine Rechtsprechungslinie OHNE konkrete Nummern
- Beispiel: "Das Bundesgericht hat wiederholt entschieden, dass..." (ohne BGE-Nummer wenn nicht aus Recherche)
- ERFINDE NIEMALS BGE-Nummern!

ZITAT-FORMAT je nach Sprache:
- Deutsch: BGE 123 III 456 E. 4.2, Urteil 4A_123/2020
- Français: ATF 123 III 456 consid. 4.2, Arrêt 4A_123/2020
- Italiano: DTF 123 III 456 consid. 4.2, Sentenza 4A_123/2020

OUTPUT:
1. **Aus der Recherche**: [BGE/Urteile mit konkreten Referenzen]
2. **Allgemeine Rechtsprechung**: [Generelle Linie des Bundesgerichts zu diesem Thema - ohne konkrete Nummern wenn nicht aus Recherche]

Antworte in der Sprache der Frage."""


CASE_LAW_USER_PROMPT = """SEARCH RESULTS:
{search_results}

USER QUESTION:
{question}"""


# ============================================================
# ANALYSIS AGENT PROMPT
# ============================================================

ANALYSIS_SYSTEM_PROMPT = """Du bist ein Schweizer Rechtsberater. Deine Aufgabe ist es, die konkrete Rechtsfrage klar, verständlich und PRÄZISE zu beantworten.

HYBRID-ANSATZ - Nutze BEIDE Quellen:
1. **Recherche-Ergebnisse** (PRIORITÄT): Was in den Suchergebnissen steht, hat Vorrang
2. **Dein juristisches Wissen**: Ergänze mit deinem Wissen über Schweizer Recht

WICHTIGE REGELN:
- Wenn die Recherche konkrete Artikel/BGE findet → diese IMMER zitieren
- Wenn die Recherche NICHTS findet → nutze dein Wissen über einschlägige Artikel
- Kennzeichne klar was woher kommt:
  - "Gemäss der Recherche..." oder "Die Suche ergab..." → aus Recherche
  - "Nach Schweizer Recht gilt..." oder "Einschlägig ist..." → aus deinem Wissen
- Bei Widersprüchen: Recherche-Ergebnisse haben Vorrang
- ERFINDE KEINE BGE-Nummern - wenn du eine BGE-Nummer nennst, muss sie echt sein

QUALITÄTSKRITERIEN:
- Konkrete Artikelnummern mit SR-Nummern (z.B. Art. 684 ZGB, SR 210)
- BGE-Referenzen nur wenn sicher korrekt
- Praktische, umsetzbare Empfehlungen
- Ehrlich wenn Unsicherheit besteht

STRUKTUR (wähle die Sprache der Frage):

Deutsch:
## Kurze Antwort
## Rechtliche Grundlagen
## Relevante Rechtsprechung
## Erläuterung
## Empfehlung
## Quellen

Français:
## Réponse courte
## Base juridique
## Jurisprudence pertinente
## Explications
## Recommandation
## Sources

Italiano:
## Risposta breve
## Base giuridica
## Giurisprudenza rilevante
## Spiegazioni
## Raccomandazione
## Fonti

English:
## Short Answer
## Legal Basis
## Relevant Case Law
## Explanation
## Recommendation
## Sources

---
Bei "Quellen" unterscheide:
- **Aus Recherche**: [Artikel/BGE die in den Suchergebnissen vorkamen]
- **Allgemeines Rechtswissen**: [Artikel die du aus deinem Training kennst]"""


ANALYSIS_USER_PROMPT = """PRIMARY LAW FINDINGS:
{primary_law}

{cantonal_law_section}

CASE LAW FINDINGS:
{case_law}

USER QUESTION:
{question}

{document_section}"""


# ============================================================
# DOCUMENT ANALYSIS SECTION
# ============================================================

DOCUMENT_SECTION_TEMPLATE = """
USER DOCUMENT TO ANALYZE:
\"\"\"
{document_text}
\"\"\"
"""


# ============================================================
# LANGUAGE DETECTION
# ============================================================

GERMAN_WORDS = {
    "der", "die", "das", "und", "ist", "von", "mit", "für", "auf", "des",
    "dem", "nicht", "sich", "bei", "auch", "nach", "werden", "aus", "hat",
    "sind", "noch", "wie", "einer", "über", "einem", "wenn", "kann", "aber",
    "arbeitsvertrag", "kündigung", "kündigungsfrist", "arbeitsrecht", "vertrag",
    "schweizer", "recht", "gesetz", "arbeitnehmer", "arbeitgeber", "darf",
    "ich", "einen", "meinen", "garten", "zaun", "bauen", "machen"
}

FRENCH_WORDS = {
    "le", "la", "les", "de", "du", "des", "et", "est", "un", "une",
    "que", "en", "pour", "dans", "qui", "au", "sur", "par", "avec", "son",
    "sont", "ont", "ce", "cette", "ne", "pas", "plus", "mais", "ou", "aux",
    "contrat", "travail", "résiliation", "délai", "droit", "suisse", "employé"
}

ITALIAN_WORDS = {
    "il", "la", "di", "che", "è", "e", "un", "una", "per", "non",
    "sono", "da", "con", "si", "del", "della", "le", "al", "dei", "alla",
    "più", "ha", "anche", "come", "questo", "gli", "nel", "essere", "suo",
    "contratto", "lavoro", "disdetta", "termine", "diritto", "svizzero",
    "posso", "costruire", "casa", "senza", "autorizzazione"
}


def detect_language(text: str) -> str:
    """
    Detect the language of the input text.
    Returns: 'German', 'French', 'Italian', or 'English'
    """
    text_lower = text.lower()
    words = set(text_lower.split())
    
    german_count = len(words & GERMAN_WORDS)
    french_count = len(words & FRENCH_WORDS)
    italian_count = len(words & ITALIAN_WORDS)
    
    if german_count >= french_count and german_count >= italian_count and german_count > 0:
        return "German"
    elif french_count >= italian_count and french_count > 0:
        return "French"
    elif italian_count > 0:
        return "Italian"
    else:
        return "English"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_primary_law_prompt(search_results: str, question: str, document_context: str = "", 
                          legal_domain: str = "", legal_context: str = "",
                          relevant_articles: list = None, irrelevant_articles: list = None) -> tuple:
    """Returns (system_prompt, user_prompt) for primary law agent"""
    
    # Build orchestrator context section
    orchestrator_section = ""
    if legal_domain or relevant_articles or irrelevant_articles:
        orchestrator_section = "\n=== ORCHESTRATOR KONTEXT (WICHTIG!) ===\n"
        
        if legal_domain:
            orchestrator_section += f"⚖️ RECHTSGEBIET: {legal_domain}\n"
        
        if legal_context:
            orchestrator_section += f"📋 Kontext: {legal_context}\n"
        
        if relevant_articles:
            orchestrator_section += f"\n✅ RELEVANTE ARTIKEL für dieses Rechtsgebiet:\n"
            for art in relevant_articles[:5]:
                orchestrator_section += f"   • {art}\n"
        
        if irrelevant_articles:
            orchestrator_section += f"\n❌ NICHT VERWENDEN (falsches Rechtsgebiet!):\n"
            for art in irrelevant_articles[:3]:
                orchestrator_section += f"   • {art}\n"
        
        orchestrator_section += "\n→ Nutze NUR Artikel aus dem richtigen Rechtsgebiet!\n"
        orchestrator_section += "=== ENDE ORCHESTRATOR KONTEXT ===\n\n"
    
    user_prompt = PRIMARY_LAW_USER_PROMPT.format(
        search_results=search_results,
        question=question,
        document_context=document_context
    )
    
    # Prepend orchestrator context to user prompt
    if orchestrator_section:
        user_prompt = orchestrator_section + user_prompt
    
    return PRIMARY_LAW_SYSTEM_PROMPT, user_prompt


def get_cantonal_law_prompt(search_results: str, question: str, canton: str, 
                            canton_name: str, commune: str = None) -> tuple:
    """Returns (system_prompt, user_prompt) for cantonal law agent"""
    commune_info = f"COMMUNE: {commune}" if commune else ""
    user_prompt = CANTONAL_LAW_USER_PROMPT.format(
        canton=canton,
        canton_name=canton_name,
        commune_info=commune_info,
        search_results=search_results,
        question=question
    )
    return CANTONAL_LAW_SYSTEM_PROMPT, user_prompt


def get_case_law_prompt(search_results: str, question: str, 
                        legal_domain: str = "", legal_context: str = "") -> tuple:
    """Returns (system_prompt, user_prompt) for case law agent"""
    
    # Build orchestrator context section
    orchestrator_section = ""
    if legal_domain or legal_context:
        orchestrator_section = "\n=== ORCHESTRATOR KONTEXT ===\n"
        if legal_domain:
            orchestrator_section += f"⚖️ RECHTSGEBIET: {legal_domain}\n"
        if legal_context:
            orchestrator_section += f"📋 Kontext: {legal_context}\n"
        orchestrator_section += "→ Suche BGE-Entscheide die zu diesem Rechtsgebiet passen.\n"
        orchestrator_section += "=== ENDE ORCHESTRATOR KONTEXT ===\n\n"
    
    user_prompt = CASE_LAW_USER_PROMPT.format(
        search_results=search_results,
        question=question
    )
    
    if orchestrator_section:
        user_prompt = orchestrator_section + user_prompt
    
    return CASE_LAW_SYSTEM_PROMPT, user_prompt


def get_analysis_prompt(primary_law: str, case_law: str, question: str,
                        document_text: str = "", cantonal_law: str = "",
                        legal_domain: str = "", legal_context: str = "", 
                        response_language: str = "German",
                        relevant_articles: list = None, irrelevant_articles: list = None) -> tuple:
    """Returns (system_prompt, user_prompt) for analysis agent"""
    document_section = ""
    if document_text:
        document_section = DOCUMENT_SECTION_TEMPLATE.format(document_text=document_text)
    
    cantonal_law_section = ""
    if cantonal_law:
        cantonal_law_section = f"CANTONAL LAW FINDINGS:\n{cantonal_law}"
    
    # Build orchestrator context
    orchestrator_section = f"""
=== ORCHESTRATOR KONTEXT (WICHTIG!) ===
⚖️ RECHTSGEBIET: {legal_domain or 'Allgemein'}
📋 Kontext: {legal_context or 'Keine spezifische Einordnung'}
🌐 Antwortsprache: {response_language}
"""
    
    if relevant_articles:
        orchestrator_section += f"\n✅ RELEVANTE ARTIKEL für dieses Rechtsgebiet:\n"
        for art in relevant_articles[:5]:
            orchestrator_section += f"   • {art}\n"
    
    if irrelevant_articles:
        orchestrator_section += f"\n❌ NICHT VERWENDEN (falsches Rechtsgebiet!):\n"
        for art in irrelevant_articles[:3]:
            orchestrator_section += f"   • {art}\n"
    
    orchestrator_section += f"""
→ Antworte KOMPLETT auf {response_language}, inklusive aller Überschriften!
→ Verwende NUR Artikel aus dem richtigen Rechtsgebiet!
=== ENDE ORCHESTRATOR KONTEXT ===
"""
    
    user_prompt = ANALYSIS_USER_PROMPT.format(
        primary_law=primary_law,
        cantonal_law_section=cantonal_law_section,
        case_law=case_law,
        question=question,
        document_section=document_section
    )
    
    user_prompt = orchestrator_section + "\n" + user_prompt
    
    return ANALYSIS_SYSTEM_PROMPT, user_prompt

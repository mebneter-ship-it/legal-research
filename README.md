# 🇨🇭 Swiss Legal Research Assistant

A multi-agent AI system for Swiss legal research, built with LangGraph and Streamlit.

## Features

- **Multi-Agent Architecture**: Orchestrator → Primary Law Agent → (Cantonal Law Agent) → Case Law Agent → Analysis Agent
- **Hybrid Approach**: Combines web research with LLM legal knowledge (research has priority)
- **Legal Domain Recognition**: Automatically identifies the correct legal domain (Mietrecht, Arbeitsrecht, Nachbarrecht, etc.)
- **Canton Detection**: Recognizes when cantonal law applies (e.g., "in Appenzell" → AI)
- **Article Mapping**: Identifies relevant AND irrelevant articles to prevent domain mixing
- **Multilingual**: German, French, Italian, English
- **Source Transparency**: Clearly separates research results from general legal knowledge
- **Document Analysis**: Upload PDFs, DOCX, images for context

## Architecture

```
┌─────────────────┐
│   Orchestrator  │  Analyzes question, identifies legal domain,
│                 │  generates search queries, detects canton
└────────┬────────┘
         │ Passes: legal_domain, relevant_articles, 
         │         irrelevant_articles, search_queries, canton
         ▼
┌─────────────────┐
│ Primary Law     │  Searches Fedlex for federal law
│ Agent           │  + LLM knowledge for gaps
└────────┬────────┘
         │
         ▼ (if canton detected)
┌─────────────────┐
│ Cantonal Law    │  Searches cantonal law portals
│ Agent           │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Case Law        │  Searches BGer for court decisions
│ Agent           │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Analysis        │  Synthesizes all findings into
│ Agent           │  structured legal advice
└─────────────────┘
```

## Quick Start

### Prerequisites
- Python 3.11+
- Tavily API key (for web search)
- OpenAI API key (for LLM)

### Installation

```bash
# Clone repository
git clone https://github.com/mebneter-ship-it/legal-research.git
cd legal-research

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.template .env
# Edit .env with your API keys
```

### Running the UI

```bash
streamlit run ui.py
```

### CLI Mode

```bash
# Interactive
python main.py

# Direct question
python main.py -q "Was sind die Kündigungsfristen im Arbeitsrecht?"

# With document
python main.py -q "Analyse this contract" -d contract.txt
```

## Configuration

Edit `.env`:
```
TAVILY_API_KEY=your_tavily_key
OPENAI_API_KEY=your_openai_key
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
```

## Project Structure

```
├── ui.py              # Streamlit UI + Pipeline orchestration
├── prompts.py         # All agent prompts
├── agents.py          # Agent state classes
├── tools.py           # Search tools (Tavily)
├── main.py            # CLI interface
├── mcp_server.py      # MCP server for Claude Desktop
├── requirements.txt   # Dependencies
├── .env.template      # Environment template
├── .gitignore         # Git ignore rules
└── HANDOVER.md        # Technical documentation
```

## Key Design Decisions

### 1. Hybrid Approach
- Research results have **priority** over LLM knowledge
- LLM knowledge fills gaps when research doesn't find relevant articles
- Sources are clearly labeled: "Aus Recherche" vs "Allgemeines Rechtswissen"

### 2. Legal Domain Mapping
The orchestrator maintains strict domain separation:

| Domain | Articles | NOT to confuse with |
|--------|----------|---------------------|
| **Mietrecht** | Art. 253-274 OR | Art. 335c OR (Arbeitsrecht!) |
| **Arbeitsrecht** | Art. 319-362 OR | Art. 271 OR (Mietrecht!) |
| **Nachbarrecht** | Art. 679-698 ZGB | - |
| **Familienrecht** | Art. 276-277 ZGB | - |

### 3. Orchestrator Context
Rich context passed to all agents:
- `legal_domain`: "Mietrecht", "Arbeitsrecht", etc.
- `relevant_articles`: Articles that apply to this domain
- `irrelevant_articles`: Articles to AVOID (wrong domain)
- `search_queries`: Specific queries for each agent
- `canton`: Detected canton code (e.g., "AI" for Appenzell)

### 4. Honest Uncertainty
Better to say "Die relevanten Bestimmungen konnten nicht gefunden werden" than hallucinate wrong articles.

## UI Features

### Developer UI (ui.py)
- **Research Output**: Final synthesized answer
- **Agent Activity**: Real-time pipeline progress
  - Pipeline tab: Visual workflow
  - Log tab: Detailed execution log
  - Übergaben tab: What orchestrator passes to each agent
- **Benchmark**: Compare with direct LLM (no research)
- **Per-Agent Details**: Search queries, prompts, raw results

### Document Upload
Supports: PDF, DOCX, TXT, MD, JPG, PNG, WEBP
- Text PDFs: Extracted with pypdf
- Scanned PDFs: Extracted with LLM vision
- Images: Text extracted with LLM vision

## MCP Integration (Claude Desktop)

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "swiss-legal-research": {
      "command": "python",
      "args": ["/path/to/legal-research/mcp_server.py"],
      "env": {
        "TAVILY_API_KEY": "your-key"
      }
    }
  }
}
```

## Supported Legal Sources

| Source | URL | Content |
|--------|-----|---------|
| Fedlex | fedlex.admin.ch | Federal laws, ordinances, SR collection |
| BGer | bger.ch | Federal Court decisions (BGE) |
| Cantonal | varies | Cantonal law portals |

## Limitations

- **Not legal advice**: Research assistance only
- **Search quality**: Depends on Tavily's indexing
- **BGE numbers**: Only cite when found in research (no hallucination)
- **Cantonal law**: Highly variable, recommend checking official sources

## License

MIT License

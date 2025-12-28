# 🇨🇭 Swiss Multi-Agent Legal Research Assistant

A powerful legal research tool that uses multiple AI agents to search and analyze Swiss law. Built with **LangGraph** for agent orchestration and **MCP (Model Context Protocol)** for tool integration.

## Features

- **Primary Law Search**: Searches Fedlex and admin.ch for Swiss federal laws, ordinances, and constitutional provisions
- **Case Law Search**: Searches BGer (Federal Court) for BGE decisions and precedents
- **Multi-Agent Architecture**: Specialized agents for different research tasks
- **Document Analysis**: Optional analysis of user-provided legal documents
- **Bilingual Support**: Works with German, French, Italian, and English queries
- **MCP Integration**: Can be used as an MCP server with Claude Desktop

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Query                               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   LangGraph Orchestrator                     │
└─────────────────────────────────────────────────────────────┘
          │                   │                    │
          ▼                   ▼                    ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────────┐
│  Primary Law    │ │   Case Law      │ │     Analysis        │
│     Agent       │ │     Agent       │ │       Agent         │
│                 │ │                 │ │                     │
│ • Fedlex search │ │ • BGer search   │ │ • Synthesizes       │
│ • Art./Abs./SR  │ │ • BGE citations │ │ • Risk analysis     │
│ • Federal laws  │ │ • Precedents    │ │ • Recommendations   │
└─────────────────┘ └─────────────────┘ └─────────────────────┘
          │                   │                    │
          └───────────────────┴────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Tavily Search API                         │
│         (site-specific queries to Swiss legal sources)       │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Clone/Download the Project

```bash
# Create project directory
mkdir -p ~/swiss-legal-agent
cd ~/swiss-legal-agent

# Copy all files here
```

### 2. Set Up Python Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # macOS/Linux
# OR: venv\Scripts\activate  # Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure API Keys

```bash
# Copy template
cp .env.template .env

# Edit with your keys
nano .env  # or use any text editor
```

Required keys:
- **TAVILY_API_KEY**: Get from [tavily.com](https://tavily.com) (free tier available)
- **OPENAI_API_KEY**: Get from [platform.openai.com](https://platform.openai.com)
- OR **ANTHROPIC_API_KEY**: Get from [console.anthropic.com](https://console.anthropic.com)

**Important**: If you're using OpenAI Project keys:
- Keys should start with `sk-proj-` (not `sk-sk-proj-`)
- Set `LLM_PROVIDER=openai` in your .env

### 4. Run the Assistant

```bash
# Interactive mode
python main.py

# Direct question
python main.py -q "What are the notice periods in Swiss employment law?"

# Test run
python main.py --test
```

## Usage Examples

### Basic Legal Query

```bash
python main.py
> Enter your legal question:
> Was sind die Kündigungsfristen im Schweizer Arbeitsrecht?
```

### Document Analysis

```bash
python main.py
> Enter your legal question:
> Prüfe diesen Arbeitsvertrag auf rechtliche Risiken
> 
> Paste document to analyze:
> [paste your contract text here]
> [empty line to finish]
```

### Command Line Mode

```bash
# Quick query
python main.py -q "Welche Formvorschriften gelten für Mietverträge?"

# With document file
python main.py -q "Analyse this contract" -d contract.txt
```

## MCP Integration (Claude Desktop)

To use with Claude Desktop:

1. Locate your Claude Desktop config:
   - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - Windows: `%APPDATA%\Claude\claude_desktop_config.json`

2. Add the server configuration:

```json
{
  "mcpServers": {
    "swiss-legal-research": {
      "command": "python",
      "args": ["/full/path/to/swiss-legal-agent/mcp_server.py"],
      "env": {
        "TAVILY_API_KEY": "your-actual-tavily-key"
      }
    }
  }
}
```

3. Restart Claude Desktop

4. You'll now have Swiss legal research tools available in Claude!

## File Structure

```
swiss-legal-agent/
├── main.py              # CLI application
├── agents.py            # LangGraph multi-agent system
├── tools.py             # Search tool implementations
├── mcp_server.py        # MCP server for Claude Desktop
├── requirements.txt     # Python dependencies
├── .env                 # Your API keys (create from template)
├── .env.template        # Template for environment variables
└── README.md            # This file
```

## Troubleshooting

### "TAVILY_API_KEY not configured"
- Make sure your `.env` file exists and contains valid keys
- Check for double prefixes: key should be `tvly-xxx` not `tvly-tvly-xxx`

### "OPENAI_API_KEY not configured"
- Project keys start with `sk-proj-` (not `sk-sk-proj-`)
- Make sure `LLM_PROVIDER=openai` is set

### Import errors
```bash
# Make sure venv is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Rate limits
- Tavily free tier: 1000 searches/month
- Consider upgrading if you hit limits

## Supported Legal Sources

| Source | URL | Content |
|--------|-----|---------|
| Fedlex | fedlex.admin.ch | Federal laws, ordinances, SR collection |
| admin.ch | admin.ch | Government publications, official texts |
| BGer | bger.ch | Federal Court decisions (BGE) |

## Limitations

- **Not legal advice**: This tool is for research assistance only
- **Search quality**: Results depend on Tavily's indexing of Swiss sources
- **Language**: Best results with German queries; French/Italian supported
- **Currency**: May not have the most recent law changes

## Contributing

Feel free to submit issues or PRs for:
- Additional Swiss legal sources
- Improved search queries
- Better citation formatting
- Additional language support

## License

MIT License - Use freely with attribution.

#!/usr/bin/env python3
"""
Swiss Multi-Agent Legal Research Assistant

A CLI tool for researching Swiss law using multiple specialized AI agents.
Supports both OpenAI and Anthropic (Claude) as the underlying LLM.

Usage:
    python main.py                    # Interactive mode
    python main.py --question "..."   # Direct question mode
    python main.py --test             # Run test query
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from keys.env file
env_path = Path(__file__).parent / "keys.env"
if env_path.exists():
    load_dotenv(env_path)
else:
    # Fallback to .env
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()


def print_banner():
    """Print the application banner"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║     🇨🇭 Swiss Multi-Agent Legal Research Assistant 🇨🇭      ║
║                                                              ║
║  Powered by: LangGraph + MCP                                 ║
║  Sources: Fedlex, admin.ch, BGer (Federal Court)            ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def check_configuration():
    """Check that required API keys are configured"""
    errors = []
    
    # Check Tavily
    tavily_key = os.getenv("TAVILY_API_KEY", "")
    if not tavily_key or tavily_key == "tvly-YOUR_KEY_HERE":
        errors.append("❌ TAVILY_API_KEY not configured")
    elif tavily_key.startswith("tvly-tvly-"):
        print("⚠️  Warning: TAVILY_API_KEY has double prefix, will auto-fix")
    else:
        print("✅ Tavily API configured")
    
    # Check LLM provider
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    
    if provider == "anthropic":
        anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not anthropic_key or anthropic_key == "sk-ant-YOUR_KEY_HERE":
            errors.append("❌ ANTHROPIC_API_KEY not configured")
        else:
            print("✅ Anthropic API configured")
    else:
        openai_key = os.getenv("OPENAI_API_KEY", "")
        if not openai_key or openai_key == "sk-proj-YOUR_KEY_HERE":
            errors.append("❌ OPENAI_API_KEY not configured")
        elif openai_key.startswith("sk-sk-"):
            print("⚠️  Warning: OPENAI_API_KEY has double prefix, will auto-fix")
        else:
            print("✅ OpenAI API configured")
    
    print(f"📡 LLM Provider: {provider}")
    
    if errors:
        print("\n" + "\n".join(errors))
        print("\nPlease configure your .env file. See .env.template for reference.")
        return False
    
    return True


def get_multiline_input(prompt: str) -> str:
    """Get multi-line input from user (empty line to finish)"""
    print(prompt)
    lines = []
    while True:
        try:
            line = input()
            if line.strip() == "":
                break
            lines.append(line)
        except EOFError:
            break
    return "\n".join(lines)


def format_output(result: dict) -> str:
    """Format the research results for display"""
    output = []
    
    output.append("\n" + "=" * 60)
    output.append("📜 PRIMARY LAW RESEARCH")
    output.append("=" * 60)
    output.append(result.get("primary_law", "No results"))
    
    output.append("\n" + "=" * 60)
    output.append("⚖️ CASE LAW RESEARCH")
    output.append("=" * 60)
    output.append(result.get("case_law", "No results"))
    
    output.append("\n" + "=" * 60)
    output.append("📋 LEGAL ANALYSIS")
    output.append("=" * 60)
    output.append(result.get("analysis", "No analysis generated"))
    
    if result.get("errors"):
        output.append("\n" + "=" * 60)
        output.append("⚠️ ERRORS ENCOUNTERED")
        output.append("=" * 60)
        for error in result["errors"]:
            output.append(f"  • {error}")
    
    return "\n".join(output)


def run_interactive():
    """Run in interactive mode"""
    from agents import run_legal_research
    
    print_banner()
    
    if not check_configuration():
        sys.exit(1)
    
    print("\n" + "-" * 60)
    
    # Get the legal question
    question = input("\n📝 Enter your legal question:\n> ").strip()
    
    if not question:
        print("No question provided. Exiting.")
        sys.exit(0)
    
    # Optionally get a document to analyze
    print("\n📄 Paste document to analyze (optional, empty line to skip):")
    document = get_multiline_input("")
    
    # Show processing message
    print("\n🔄 Researching... (this may take 30-60 seconds)")
    print("   → Searching primary law sources...")
    print("   → Searching case law...")
    print("   → Generating analysis...")
    
    # Run the research
    try:
        result = run_legal_research(question, document)
        print(format_output(result))
    except Exception as e:
        print(f"\n❌ Error during research: {str(e)}")
        sys.exit(1)


def run_direct_question(question: str, document: str = ""):
    """Run with a direct question (non-interactive)"""
    from agents import run_legal_research
    
    if not check_configuration():
        sys.exit(1)
    
    print(f"\n🔄 Researching: {question[:50]}...")
    
    try:
        result = run_legal_research(question, document)
        print(format_output(result))
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)


def run_test():
    """Run a test query to verify setup"""
    print_banner()
    print("🧪 Running test query...")
    
    test_question = "Was sind die Kündigungsfristen gemäss Schweizer Arbeitsrecht (OR)?"
    run_direct_question(test_question)


def main():
    parser = argparse.ArgumentParser(
        description="Swiss Multi-Agent Legal Research Assistant"
    )
    parser.add_argument(
        "-q", "--question",
        type=str,
        help="Direct legal question to research"
    )
    parser.add_argument(
        "-d", "--document",
        type=str,
        help="Path to document file to analyze"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run a test query to verify setup"
    )
    
    args = parser.parse_args()
    
    if args.test:
        run_test()
    elif args.question:
        document = ""
        if args.document and os.path.exists(args.document):
            with open(args.document, "r") as f:
                document = f.read()
        run_direct_question(args.question, document)
    else:
        run_interactive()


if __name__ == "__main__":
    main()

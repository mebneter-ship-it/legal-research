"""
Test the Smart Orchestrator

Tests:
1. Cantonal question → Should detect canton, search cantonal law
2. Federal question → Should focus on Fedlex/BGer
3. Mixed question → Should search both

Run: python test_orchestrator.py
"""

import os
from dotenv import load_dotenv

load_dotenv("keys.env")

from agents import run_legal_research, run_orchestrator_analyze, ResearchState


# ============================================================
# TEST CASES
# ============================================================

TEST_CASES = [
    {
        "name": "Kantonal: Zaunhöhe Appenzell",
        "question": "Wie hoch darf ein Zaun in Appenzell sein?",
        "expected_canton": "AI",
        "expected_domain": "Baurecht",
    },
    {
        "name": "Kantonal: Mietrecht Zürich",
        "question": "Welche Kündigungsfristen gelten für Mietwohnungen in Zürich?",
        "expected_canton": "ZH",
        "expected_domain": "Mietrecht",
    },
    {
        "name": "Bundesrecht: Arbeitsrecht",
        "question": "Wie viele Ferientage hat ein Arbeitnehmer in der Schweiz?",
        "expected_canton": "",
        "expected_domain": "Arbeitsrecht",
    },
    {
        "name": "Bundesrecht: Nachbarrecht",
        "question": "Was sagt Art. 684 ZGB über Immissionen?",
        "expected_canton": "",
        "expected_domain": "Nachbarrecht",
    },
]


def test_orchestrator_analyze():
    """Test only the analyze step (no API calls)"""
    print("\n" + "="*70)
    print("🧠 TESTING ORCHESTRATOR ANALYZE (no API calls)")
    print("="*70)
    
    for test in TEST_CASES:
        print(f"\n--- {test['name']} ---")
        print(f"Question: {test['question']}")
        
        # Create minimal state
        state: ResearchState = {
            "question": test["question"],
            "document": "",
            "detected_canton": "",
            "detected_canton_name": "",
            "detected_commune": "",
            "legal_domain": "",
            "enriched_queries": {},
            "primary_law_results": "",
            "cantonal_law_results": "",
            "case_law_results": "",
            "final_analysis": "",
            "current_stage": "started",
            "errors": []
        }
        
        # Run orchestrator analyze
        result = run_orchestrator_analyze(state)
        
        # Check results
        canton_ok = result["detected_canton"] == test["expected_canton"]
        domain_ok = result["legal_domain"] == test["expected_domain"]
        
        print(f"\nResults:")
        print(f"  Canton: {result['detected_canton']} {'✅' if canton_ok else '❌ expected ' + test['expected_canton']}")
        print(f"  Domain: {result['legal_domain']} {'✅' if domain_ok else '❌ expected ' + test['expected_domain']}")
        print(f"  Enriched Queries: {list(result['enriched_queries'].keys())}")
        
        if result.get("enriched_queries", {}).get("cantonal_search_queries"):
            print(f"  Cantonal Queries:")
            for q in result["enriched_queries"]["cantonal_search_queries"]:
                print(f"    - {q}")


def test_full_pipeline(test_index=0):
    """Test full pipeline with API calls"""
    print("\n" + "="*70)
    print("🚀 TESTING FULL PIPELINE (with API calls)")
    print("="*70)
    
    # Pick test by index
    test = TEST_CASES[test_index]
    
    print(f"\n📋 TEST {test_index + 1}: {test['name']}")
    print(f"Question: {test['question']}")
    print("-"*70)
    
    result = run_legal_research(test["question"])
    
    print("\n" + "="*70)
    print("📊 RESULTS SUMMARY")
    print("="*70)
    print(f"Detected Canton: {result.get('detected_canton', 'None')} ({result.get('detected_canton_name', '')})")
    print(f"Expected Canton: {test['expected_canton'] or 'None (Bundesrecht)'}")
    print(f"Legal Domain: {result.get('legal_domain', 'Unknown')}")
    print(f"Expected Domain: {test['expected_domain']}")
    print(f"\nPrimary Law: {len(result.get('primary_law', ''))} chars")
    print(f"Cantonal Law: {len(result.get('cantonal_law', ''))} chars")
    print(f"Case Law: {len(result.get('case_law', ''))} chars")
    print(f"Errors: {result.get('errors', [])}")
    
    # Validation
    canton_ok = result.get('detected_canton', '') == test['expected_canton']
    domain_ok = result.get('legal_domain', '') == test['expected_domain']
    print(f"\n✅ Canton correct: {canton_ok}")
    print(f"✅ Domain correct: {domain_ok}")
    
    print("\n" + "="*70)
    print("📝 FINAL ANALYSIS")
    print("="*70)
    print(result.get("analysis", "No analysis"))
    
    return result


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        
        if arg == "all":
            # Run all tests
            for i in range(len(TEST_CASES)):
                test_full_pipeline(i)
                if i < len(TEST_CASES) - 1:
                    input("\n[Press Enter for next test...]")
        elif arg.isdigit():
            # Run specific test by number (1-based)
            idx = int(arg) - 1
            if 0 <= idx < len(TEST_CASES):
                test_full_pipeline(idx)
            else:
                print(f"Invalid test number. Choose 1-{len(TEST_CASES)}")
        else:
            print("Usage:")
            print("  python test_orchestrator.py       - Test analyze step only (fast)")
            print("  python test_orchestrator.py 1     - Run test 1 (Kantonal: Appenzell)")
            print("  python test_orchestrator.py 2     - Run test 2 (Kantonal: Zürich)")
            print("  python test_orchestrator.py 3     - Run test 3 (Bundesrecht: Arbeitsrecht)")
            print("  python test_orchestrator.py 4     - Run test 4 (Bundesrecht: Nachbarrecht)")
            print("  python test_orchestrator.py all   - Run all tests")
    else:
        print("Usage:")
        print("  python test_orchestrator.py       - Show this help")
        print("  python test_orchestrator.py 1     - Run test 1 (Kantonal: Appenzell)")
        print("  python test_orchestrator.py 2     - Run test 2 (Kantonal: Zürich)")
        print("  python test_orchestrator.py 3     - Run test 3 (Bundesrecht: Arbeitsrecht)")
        print("  python test_orchestrator.py 4     - Run test 4 (Bundesrecht: Nachbarrecht)")
        print("  python test_orchestrator.py all   - Run all tests")
        print()
        print("Available tests:")
        for i, t in enumerate(TEST_CASES):
            print(f"  {i+1}. {t['name']}")
        print()
        test_orchestrator_analyze()

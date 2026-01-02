"""
Swiss Legal Agent - Primary Law Agent Test

Tests the federal law search (Fedlex, admin.ch)

Just run: python test_primary.py
"""

import os
from dotenv import load_dotenv

load_dotenv("keys.env")

# Import the updated search function
from tools import search_swiss_primary_law, search_swiss_case_law

# ============================================================
# TEST CASES
# ============================================================

TEST_CASES = [
    {
        "name": "Mietrecht - Kündigungsfrist",
        "query": "Art. 266c OR Kündigungsfrist Wohnung",
        "expected_keywords": ["kündigung", "art. 266"],
        "should_find": ["GESETZESTEXT", "lawbrary"],  # lawbrary has readable text
    },
    {
        "name": "Nachbarrecht - Immissionen",
        "query": "Art. 684 ZGB Immissionen Nachbar",
        "expected_keywords": ["art. 684", "immissionen"],
        "should_find": ["GESETZESTEXT", "lawbrary"],
    },
    {
        "name": "Arbeitsrecht - Ferien",
        "query": "Art. 329a OR Ferienanspruch Arbeitnehmer",
        "expected_keywords": ["ferien", "art. 329"],
        "should_find": ["GESETZESTEXT", "lawbrary"],
    },
]


# ============================================================
# RUN TESTS
# ============================================================

def run_tests():
    print("\n" + "="*70)
    print("🏛️  PRIMARY LAW AGENT TEST")
    print("="*70)
    
    for i, test in enumerate(TEST_CASES, 1):
        print(f"\n\n{'='*70}")
        print(f"TEST {i}: {test['name']}")
        print(f"Query: {test['query']}")
        print(f"Expected keywords: {test['expected_keywords']}")
        print("="*70)
        
        # Run search
        result = search_swiss_primary_law(test['query'], max_results=3)
        
        # Show result
        print("\n📋 RESULT:")
        print("-"*50)
        print(result)
        
        # Check if expected elements are found
        result_lower = result.lower()
        found_keywords = [kw for kw in test['expected_keywords'] if kw in result_lower]
        
        print("\n" + "-"*50)
        print(f"✅ Found keywords: {found_keywords}")
        print(f"📊 Score: {len(found_keywords)}/{len(test['expected_keywords'])}")
        
        # Check for official sources
        has_fedlex = "fedlex" in result_lower
        has_admin = "admin.ch" in result_lower
        print(f"🏛️  Fedlex found: {'✅' if has_fedlex else '❌'}")
        print(f"📋 Admin.ch found: {'✅' if has_admin else '⚠️'}")
        
        # NEW: Check for actual law text indicators
        should_find = test.get('should_find', [])
        if should_find:
            print(f"\n📜 LAW TEXT CHECK:")
            for indicator in should_find:
                found = indicator.lower() in result_lower
                print(f"   {indicator}: {'✅ FOUND' if found else '❌ NOT FOUND'}")
        
        input("\n[Press Enter to continue to next test...]")
    
    print("\n" + "="*70)
    print("✅ ALL TESTS COMPLETE")
    print("="*70)


def run_case_law_test():
    """Quick test for case law search"""
    print("\n" + "="*70)
    print("⚖️  CASE LAW AGENT TEST")
    print("="*70)
    
    query = "Nachbarrecht Einfriedung Grenzabstand BGE"
    print(f"\nQuery: {query}")
    
    result = search_swiss_case_law(query, max_results=3)
    print("\n📋 RESULT:")
    print("-"*50)
    print(result)
    
    # Check for official sources
    result_lower = result.lower()
    has_bger = "bger" in result_lower
    has_bge = "bge" in result_lower
    print("\n" + "-"*50)
    print(f"🏛️  BGer found: {'✅' if has_bger else '❌'}")
    print(f"📜 BGE citations: {'✅' if has_bge else '❌'}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "case":
        run_case_law_test()
    else:
        print("Usage:")
        print("  python test_primary.py       - Test primary law search")
        print("  python test_primary.py case  - Test case law search")
        print()
        
        choice = input("Run primary law tests? [Y/n]: ").strip().lower()
        if choice != 'n':
            run_tests()
        else:
            run_case_law_test()

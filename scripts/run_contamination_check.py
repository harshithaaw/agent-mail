"""
Run contamination check: how often Enron/fallback data outranks real_user data
"""
import json
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.retrieve import retrieve_similar

def main():
    # Use simulated incoming email queries (realistic scenarios needing replies)
    print("Using simulated incoming email queries for contamination check")
    
    queries = [
        "Can you send me the report by end of day?",
        "Meeting scheduled for tomorrow at 3pm - please confirm attendance",
        "Thanks for your application - we'd like to schedule an interview",
        "Please review the attached document and provide feedback",
        "Quick question about the deadline extension request",
        "Reminder: Assignment due this Friday at 11:59 PM",
        "Are you available for dinner this Saturday evening?",
        "The team is waiting for your approval on the budget"
    ]
    
    print(f"\nRunning contamination check with {len(queries)} queries (NO source filter)")
    print("=" * 80)
    
    contamination_count = 0
    total_analyzed = 0
    
    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: '{query}'")
        print("-" * 80)
        
        # Query WITHOUT source_filter (full mixed collection)
        results = retrieve_similar(query, k=3, source_filter=None)
        
        if not results:
            print("  No results found")
            continue
        
        print(f"  Top 3 results (NO source filter):")
        
        for j, result in enumerate(results, 1):
            distance = result.get('distance', float('inf'))
            source = result.get('metadata', {}).get('source', 'unknown')
            subject = result.get('metadata', {}).get('subject', 'N/A')
            sender = result.get('metadata', {}).get('sender', 'N/A')
            
            print(f"    {j}. Distance: {distance:.4f}")
            print(f"       Source: {source}")
            print(f"       Subject: {subject[:60]}...")
            print(f"       Sender: {sender}")
            
            # Check if Enron/fallback outranks real_user in top-3
            if source != 'real_user':
                contamination_count += 1
            
            total_analyzed += 1
    
    print("\n" + "=" * 80)
    print("CONTAMINATION CHECK SUMMARY")
    print("=" * 80)
    
    print(f"Total top-3 positions analyzed: {total_analyzed}")
    print(f"Enron/fallback results in top-3: {contamination_count}")
    print(f"real_user results in top-3: {total_analyzed - contamination_count}")
    print(f"Contamination rate: {contamination_count/total_analyzed*100:.1f}%")
    
    print("\nINTERPRETATION:")
    print(f"- When source_filter is OFF, {contamination_count}/{total_analyzed} top-3 positions are occupied by Enron/fallback data")
    print(f"- This means Enron/fallback data outranks real_user data in {contamination_count/total_analyzed*100:.1f}% of top-3 positions")

if __name__ == "__main__":
    main()
"""
Output raw threshold data for human review - no judgment, no filtering
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.retrieve import retrieve_similar

def main():
    # Use the same 8 simulated incoming email queries
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
    
    for i, query in enumerate(queries, 1):
        print(f"QUERY {i}: {query}")
        print("=" * 80)
        
        # Query with source_filter="real_user"
        results = retrieve_similar(query, k=3, source_filter="real_user")
        
        if not results:
            print("No results found")
            print()
            continue
        
        for j, result in enumerate(results, 1):
            distance = result.get('distance', float('inf'))
            document = result.get('document', '')
            
            print(f"HIT {j} - Distance: {distance:.4f}")
            print("DOCUMENT:")
            print(document)
            print("-" * 80)
        
        print()

if __name__ == "__main__":
    main()
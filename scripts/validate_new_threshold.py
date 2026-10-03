"""
Validate new 0.60 threshold with the same 8 queries using validate_context()
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.retrieve import retrieve_similar
from agents.reply.context import validate_context

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
    
    print("VALIDATING NEW THRESHOLD (0.60) WITH validate_context()")
    print("=" * 80)
    
    for i, query in enumerate(queries, 1):
        print(f"\nQUERY {i}: {query}")
        print("-" * 80)
        
        # Retrieve with source_filter="real_user"
        retrieved_examples = retrieve_similar(query, k=3, source_filter="real_user")
        
        # Validate using the new threshold
        validation_result = validate_context(retrieved_examples)
        
        print(f"sufficient: {validation_result['sufficient']}")
        print(f"usable_examples count: {len(validation_result['usable_examples'])}")
        print(f"total_retrieved: {validation_result['total_retrieved']}")
        print(f"best_distance: {validation_result['best_distance']}")
        print(f"should_retry: {validation_result['should_retry']}")

if __name__ == "__main__":
    main()
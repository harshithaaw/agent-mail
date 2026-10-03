"""
Task/action extraction module using spaCy sentence segmentation and pattern matching.
Extracts structured task/action items from email text.
"""
import spacy
import re
from typing import List, Dict, Any

# Load spaCy model once at import time
nlp = spacy.load("en_core_web_sm")

# Action/imperative patterns that indicate tasks or action items
ACTION_PATTERNS = [
    # Direct imperatives (must start sentence or be clearly task-oriented)
    r"\bplease\s+(?:submit|send|reply|respond|write|create|update|complete|finish|confirm|verify|make\s+sure|prepare|bring|review)\b",
    r"\byou\s+(?:need\s+to|must|should)\s+(?:complete|finish|submit|send|do|confirm|verify|prepare)\b",
    r"\b(?:submit|send|reply|respond|write|create|update|complete|finish|prepare)\s+(?:your|the|this)\b",
    r"\bmake\s+sure\s+to\b",
    
    # Deadlines and due dates (in task context)
    r"\b(?:submit|send|complete|finish|reply)\s+(?:by|before|on|due)\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    r"\b(?:submit|send|complete|finish|reply)\s+(?:by|before|on|due)\s+\d{1,2}(?:st|nd|rd|th)?\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)\b",
    r"\b(?:submit|send|complete|finish|reply)\s+(?:by|before|on|due)\s+\d{1,2}/\d{1,2}/\d{2,4}\b",
    r"\bdeadline\s+(?:is|:|for)\b",
    
    # Priority indicators (in task context)
    r"\burgent\b.*(?:action|response|reply|submit|send)\b",
    r"\basap\b.*(?:action|response|reply|submit|send)\b",
    r"\bimmediately\b.*(?:action|response|reply|submit|send)\b",
    r"\baction\s+required\b",
    
    # Task-related phrases
    r"\b(?:don't\s+forget|remember\s+to)\s+(?:submit|send|complete|finish|reply|do)\b",
    r"\b(?:task|action\s+item)\b",
]


def extract_tasks(email_text: str) -> List[Dict[str, Any]]:
    """
    Extract structured task/action items from email text.
    
    Args:
        email_text: The email text to analyze (should include subject and body)
    
    Returns:
        List of task dicts with keys:
        - task_text: str - the extracted task text
        - sentence: str - the full sentence containing the task
        - is_urgent: bool - whether the task appears urgent
        - has_deadline: bool - whether the task has a deadline
        - deadline_mention: str or None - deadline text if found
    """
    doc = nlp(email_text)
    tasks = []
    
    # Process each sentence
    for sent in doc.sents:
        sentence_text = sent.text.strip()
        if not sentence_text:
            continue
        
        # Check if sentence contains action patterns
        task_found = False
        is_urgent = False
        has_deadline = False
        deadline_mention = None
        
        for pattern in ACTION_PATTERNS:
            if re.search(pattern, sentence_text, re.IGNORECASE):
                task_found = True
                
                # Check for urgency
                if re.search(r"\burgent\b|\basap\b|\bimmediately\b|\baction\s+required\b", sentence_text, re.IGNORECASE):
                    is_urgent = True
                
                # Check for deadline (only match clear temporal expressions with prepositions)
                deadline_match = re.search(r"(?:by|before|on|due)\s+(?:next\s+)?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|\d{1,2}(?:st|nd|rd|th)?\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)|\d{1,2}/\d{1,2}/\d{2,4}|tomorrow|today|tonight|(?:the\s+)?end\s+of\s+the\s+day)", sentence_text, re.IGNORECASE)
                if deadline_match:
                    has_deadline = True
                    deadline_mention = deadline_match.group(0)
                
                # Extract task text (first part of sentence)
                task_text = sentence_text[:100]  # Limit to 100 chars for task text
                if len(sentence_text) > 100:
                    task_text += "..."
                
                tasks.append({
                    "task_text": task_text,
                    "sentence": sentence_text,
                    "is_urgent": is_urgent,
                    "has_deadline": has_deadline,
                    "deadline_mention": deadline_mention
                })
                break  # Only count each sentence once
    
    return tasks


if __name__ == "__main__":
    # Smoke test with example strings
    print("Running task extraction smoke test...")
    
    examples = [
        "Please submit your assignment by Friday at 5 PM.",
        "URGENT: Action required - respond immediately.",
        "Just a reminder about our meeting tomorrow.",
        "You need to complete the project by next Monday.",
        "Hope you're doing well! Let's catch up soon.",
        "Make sure to confirm your attendance for the conference.",
        "Don't forget to send the report by the end of the day.",
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\nExample {i}: {example}")
        tasks = extract_tasks(example)
        print(f"Tasks found: {len(tasks)}")
        for task in tasks:
            print(f"  - {task['task_text']}")
            print(f"    Urgent: {task['is_urgent']}, Deadline: {task['has_deadline']}")
    
    print("\nSmoke test complete.")

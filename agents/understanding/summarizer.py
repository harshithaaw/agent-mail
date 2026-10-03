"""
Email summarization module.
Loads a local HF summarization model and provides summarization function.
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import warnings

# Suppress warnings about model loading
warnings.filterwarnings("ignore", message=".*The given model is not a summary model.*")

# Load model once at module level - same pattern as spacy in prioritization.py
try:
    model_name = "sshleifer/distilbart-cnn-12-6"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()  # Set to evaluation mode
    
    # Use CPU if CUDA is not available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    SUMMARIZER_AVAILABLE = True
except Exception as e:
    print(f"Error loading summarization model: {e}")
    model = None
    tokenizer = None
    SUMMARIZER_AVAILABLE = False


def summarize_email(text: str) -> str | None:
    """
    Summarize an email text using the loaded summarization model.
    
    Args:
        text: The email text to summarize
    
    Returns:
        Summarized text as a string, or None if summarization fails
        If input text is very short (<20 words), returns original text unchanged
    """
    if model is None or tokenizer is None:
        return None
    
    # Guard: skip summarization for very short emails
    # CNN/DailyMail-tuned models degrade on very short/casual text
    if len(text.split()) < 20:
        return text
    
    try:
        # Get device
        device = next(model.parameters()).device
        
        # Tokenize input
        inputs = tokenizer(text, max_length=1024, truncation=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate summary
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=150,
            min_length=30,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        
        # Decode summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
            
    except Exception as e:
        # Return None on failure instead of raising - matches detect_deadline() pattern
        # This ensures one failing email can't crash a batch
        return None


if __name__ == "__main__":
    # Smoke test with example strings
    print("Running summarization smoke test...")
    
    examples = [
        # Short email (should return unchanged)
        "Hey, just checking in. Are we still on for dinner tomorrow?",
        
        # Medium-length email
        """Subject: Project Update
Hi Team,
I wanted to give you a quick update on our project progress. We've completed the first phase of development and are now moving into testing. The initial results look promising, but we've identified a few bugs that need to be addressed before launch. Please review the attached report and let me know if you have any questions. We're aiming to release by the end of the month.
Best regards,
Alex""",
        
        # Long email
        """Subject: Quarterly Review and Planning Meeting
Dear Team,
I hope this email finds you well. As we approach the end of Q3, I wanted to schedule our quarterly review and planning meeting. This meeting will be crucial for setting our direction for the next quarter and reviewing our performance over the past three months.

During this meeting, we will cover:
1. Review of Q3 performance metrics and KPIs
2. Analysis of market trends and competitive landscape
3. Budget planning for Q4
4. Resource allocation and team capacity planning
5. Strategic initiatives and priority projects
6. Risk assessment and mitigation strategies

Please come prepared with your team's performance reports and any insights on challenges faced or opportunities identified. I would also like each department head to present a brief 10-minute overview of their key achievements and areas for improvement.

The meeting is scheduled for next Friday, October 15th, from 9 AM to 12 PM in the main conference room. Please confirm your attendance by replying to this email by Wednesday.

If you have any specific topics you'd like to add to the agenda, please let me know in advance so we can allocate appropriate time.

Looking forward to a productive session.
Best regards,
Sarah Johnson
Director of Operations"""
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\nExample {i}:")
        print(f"Original: {example[:100]}...")
        
        summary = summarize_email(example)
        
        if summary is None:
            print("Summary: None (model failed or not available)")
        elif summary == example:
            print("Summary: [Original text returned - too short to summarize]")
        else:
            print(f"Summary: {summary}")
    
    print("\nSmoke test complete.")
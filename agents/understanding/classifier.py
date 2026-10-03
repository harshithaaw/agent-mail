"""
Email category classification module.
Loads trained category classifier and provides classification function.
"""

import joblib
from utils.preprocessing import clean_text
from agents.understanding.categories import CATEGORIES, Category

# Load models once at module level
try:
    clf = joblib.load("models/category/category_classifier.pkl")
    vectorizer = joblib.load("models/category/category_tfidf_vectorizer.pkl")
except Exception as e:
    print(f"Error loading classifier models: {e}")
    clf = None
    vectorizer = None


def classify_email(text: str) -> str:
    """
    Classify an email text into one of the categories.
    
    Args:
        text: The email text to classify (should include subject and body)
    
    Returns:
        The predicted label as a string: "Academic", "Career", "Personal", or "Promotional"
    
    Raises:
        RuntimeError: If the classifier models failed to load
    """
    if clf is None or vectorizer is None:
        raise RuntimeError("Classifier models not loaded")
    
    # Clean the text using the same pipeline as training
    cleaned_text, _ = clean_text(text)
    
    # Skip if cleaning produced empty text
    if len(cleaned_text) == 0:
        # Return a default classification for empty text
        return "Personal"
    
    # Transform and predict
    text_vec = vectorizer.transform([cleaned_text])
    prediction = clf.predict(text_vec)[0]
    # Keep the classifier's output inside the shared taxonomy.
    return prediction if prediction in CATEGORIES else Category.PERSONAL.value


if __name__ == "__main__":
    # Smoke test with example strings
    print("Running classification smoke test...")
    
    examples = [
        "Subject: Formal Employment Offer - Lead Architect Dear Jordan, On behalf of Apex Solutions, we are thrilled to offer you the Lead Architect position. Please sign the attached offer letter and complete your onboarding forms.",
        "Subject: Board game night at my place! Hey folks, Hosting a board game evening next Tuesday starting around 6 PM. I will order pizza, so let me know if you have any dietary restrictions or preferences.",
        "Subject: Office Hours Schedule Change & Midterm Syllabus Notes Dear Students, Please note that Dr. Aris will hold virtual office hours on Wednesday instead of Thursday this week to cover questions regarding chapter 4 of the textbook."
    ]
    
    for i, example in enumerate(examples, 1):
        classification = classify_email(example)
        print(f"Example {i}: {classification}")
    
    print("Smoke test complete.")

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pandas as pd
import joblib

from scripts.data_loading import build_training_set

# Load and label all emails as (cleaned_text, label) tuples
data = build_training_set()
texts = [text for text, label in data]
labels = [label for text, label in data]

# Stratified split preserves the 1:5 spam:ham ratio in both sets
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# ngram_range=(1,2) captures both single words and two-word phrases (e.g. "click here")
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)   # fit only on train — no test leakage
X_test_vec = vectorizer.transform(X_test)          # test reuses train's vocabulary

# class_weight="balanced" penalizes missed spam more heavily during training
clf = LogisticRegression(max_iter=1000, class_weight="balanced")
clf.fit(X_train_vec, y_train)

y_pred = clf.predict(X_test_vec)
print(classification_report(y_test, y_pred))

# Inspect errors before deciding whether this version is good enough to save
results = pd.DataFrame({"text": X_test, "actual": y_test, "predicted": y_pred})

missed_spam = results[(results["actual"] == "spam") & (results["predicted"] == "ham")]
print("Missed spam count:", len(missed_spam))
print(missed_spam["text"].head(5).tolist())

false_positives = results[(results["actual"] == "ham") & (results["predicted"] == "spam")]
print("False positive count:", len(false_positives))
print(false_positives["text"].head(5).tolist())

# Save model + vectorizer for reuse in the live screening pipeline
joblib.dump(clf, "models/spam/spam_classifier.pkl")
joblib.dump(vectorizer, "models/spam/tfidf_vectorizer.pkl")
print("Saved models/spam/spam_classifier.pkl and models/spam/tfidf_vectorizer.pkl")


from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

# Rebuild vectorizer + classifier as one pipeline so each CV fold
# fits TF-IDF only on that fold's training data — avoids leakage across folds
cv_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
])

cv_scores = cross_val_score(cv_pipeline, texts, labels, cv=5, scoring="f1_macro")
print("Cross-val F1 (macro) per fold:", cv_scores)
print("Mean:", cv_scores.mean(), "Std:", cv_scores.std())
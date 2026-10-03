"""
Train email category classifier (Career/Personal/Promotional).
Follows the same pattern as train_spam_classifier.py.
"""

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
import pandas as pd
import joblib

# Load the combined dataset
print("=" * 80)
print("LOADING DATASET")
print("=" * 80)

df = pd.read_csv("data/datasets/combined_dataset.csv")
print(f"Total row count: {len(df)}")
print("\nLabel value_counts():")
print(df["label"].value_counts())

# Extract texts and labels
texts = df["text"].tolist()
labels = df["label"].tolist()

# Stratified 80/20 train/test split
print("\n" + "=" * 80)
print("STRATIFIED TRAIN/TEST SPLIT")
print("=" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

print(f"Train set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print("\nTrain label distribution:")
print(pd.Series(y_train).value_counts())
print("\nTest label distribution:")
print(pd.Series(y_test).value_counts())

# Vectorize with TF-IDF
print("\n" + "=" * 80)
print("VECTORIZATION")
print("=" * 80)

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
print(f"TF-IDF feature count: {X_train_vec.shape[1]}")

# Train classifier
print("\n" + "=" * 80)
print("TRAINING CLASSIFIER")
print("=" * 80)

clf = LogisticRegression(max_iter=1000, class_weight="balanced")
clf.fit(X_train_vec, y_train)
print("Training complete")

# Evaluate on test set
print("\n" + "=" * 80)
print("TEST SET EVALUATION")
print("=" * 80)

y_pred = clf.predict(X_test_vec)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Explicit macro and weighted F1
macro_f1 = f1_score(y_test, y_pred, average="macro")
weighted_f1 = f1_score(y_test, y_pred, average="weighted")
print(f"\nMacro F1: {macro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")

# Confusion matrix with class names
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
cm_df = pd.DataFrame(cm, index=clf.classes_, columns=clf.classes_)
print(cm_df)

# Cross-validation on full dataset
print("\n" + "=" * 80)
print("5-FOLD CROSS-VALIDATION ON FULL DATASET")
print("=" * 80)

cv_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
])

cv_scores = cross_val_score(cv_pipeline, texts, labels, cv=5, scoring="f1_macro")
print("Cross-val F1 (macro) per fold:", cv_scores)
print("Mean:", cv_scores.mean(), "Std:", cv_scores.std())

# Save model and vectorizer
print("\n" + "=" * 80)
print("SAVING MODEL")
print("=" * 80)

joblib.dump(clf, "models/category/category_classifier.pkl")
joblib.dump(vectorizer, "models/category/category_tfidf_vectorizer.pkl")
print("Saved models/category/category_classifier.pkl and models/category/category_tfidf_vectorizer.pkl")

print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)

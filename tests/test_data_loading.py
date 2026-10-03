from scripts.data_loading import load_eml_body
from utils.preprocessing import clean_text
import os


from scripts.data_loading import build_training_set

data = build_training_set()
print("Total samples:", len(data))
print("Spam count:", sum(1 for _, label in data if label == "spam"))
print("Ham count:", sum(1 for _, label in data if label == "ham"))
print("Example:", data[0])
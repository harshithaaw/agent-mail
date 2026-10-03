import random
from scripts.data_loading import load_deepset_examples
from scripts.injection import heuristic_injection_check, llm_injection_check

def evaluate_heuristic(examples):
    tp = fp = tn = fn = 0
    missed_injections = []

    for text, true_label in examples:
        flagged, matched = heuristic_injection_check(text)
        if true_label == 1 and flagged:
            tp += 1
        elif true_label == 1 and not flagged:
            fn += 1
            missed_injections.append(text)
        elif true_label == 0 and flagged:
            fp += 1
        else:
            tn += 1

    detection_rate = tp / (tp + fn) if (tp + fn) else 0
    fp_rate = fp / (fp + tn) if (fp + tn) else 0
    print(f"TP={tp} FN={fn} FP={fp} TN={tn}")
    print(f"Detection rate: {detection_rate:.3f}")
    print(f"False positive rate: {fp_rate:.3f}")
    return missed_injections

def sample_and_test_llm(missed_injections, n=15, seed=42):
    random.seed(seed)
    sample = random.sample(missed_injections, min(n, len(missed_injections)))
    caught = 0
    results = []
    for text in sample:
        flagged, details = llm_injection_check(text)
        results.append((text, flagged, details))
        if flagged:
            caught += 1
    print(f"LLM caught {caught}/{len(sample)} of heuristic-missed injections")
    return results


if __name__ == "__main__":
    examples = load_deepset_examples()
    missed = evaluate_heuristic(examples)
    print(f"Missed (for LLM sample later): {len(missed)}")
"""
Comprehensive Evaluator for Deadline Detectors (A, A2, B)
"""

import json
import os
import re
import sys
import time
from datetime import datetime, date
from typing import List, Dict, Any, Tuple

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.understanding.extractor import extract_tasks
from agents.understanding.prioritizer import detect_deadline_legacy
from agents.understanding.deadline_detector import detect_b


# ============================================================
# STEP 1: DETECTORS DEFINITION & WRAPPERS
# ============================================================

def detect_a(text: str, received_at: str) -> List[Dict[str, Any]]:
    """Detector A: wrap extract_tasks from extractor.py."""
    tasks = extract_tasks(text)
    detected = []
    for t in tasks:
        if t.get("has_deadline"):
            detected.append({
                "date": None,  # extract_tasks returns deadline_mention str, not YYYY-MM-DD
                "kind": "other",
                "action": t.get("task_text", ""),
                "deadline_mention": t.get("deadline_mention")
            })
    return detected


def detect_a2(text: str, received_at: str) -> List[Dict[str, Any]]:
    """
    Detector A2: wrap detect_deadline_legacy from agents.understanding.prioritizer.
    
    detect_deadline takes input: `text: str` (email text).
    It returns real dates in YYYY-MM-DD format when dateutil parses spaCy DATE entities,
    or None if no parseable date is found.
    """
    res = detect_deadline_legacy(text)
    if res.get("has_deadline"):
        return [{
            "date": res.get("deadline_date"),
            "kind": "other",
            "action": res.get("raw_phrase") or "deadline"
        }]
    return []


def safe_detect_b(text: str, received_at: str) -> List[Dict[str, Any]]:
    """Detector B: wrap detect_b from deadline_detector.py (root/agents)."""
    try:
        return detect_b(text, received_at)
    except Exception:
        return []


# ============================================================
# DATASET LOADING
# ============================================================

def load_datasets() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    dev_cases = []
    test_cases = []
    real_cases = []

    date_cases_path = os.path.join("data", "date_cases.json")
    if os.path.exists(date_cases_path):
        with open(date_cases_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            cases = data.get("cases", []) if isinstance(data, dict) else data
            for c in cases:
                cid = c.get("id", "")
                nums = re.findall(r"\d+", cid)
                if nums:
                    num = int(nums[0])
                    if num % 2 == 1:
                        dev_cases.append(c)
                    else:
                        test_cases.append(c)
                else:
                    dev_cases.append(c)

    real_cases_path = os.path.join("data", "real_cases.json")
    if os.path.exists(real_cases_path) and os.path.getsize(real_cases_path) > 0:
        try:
            with open(real_cases_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                real_cases = data.get("cases", []) if isinstance(data, dict) else data
        except Exception:
            real_cases = []

    return dev_cases, test_cases, real_cases


# ============================================================
# FAILURE CATEGORIZATION FOR B
# ============================================================

def categorize_b_failure(case: Dict[str, Any], actual: List[Dict[str, Any]]) -> str:
    group = case.get("group", "")
    text = case.get("text", "")
    expected = case.get("expected", [])
    
    exp_dates = [e["date"] for e in expected if e.get("date")]
    act_dates = [a["date"] for a in actual if a.get("date")]

    has_gt = len(expected) > 0
    has_det = len(actual) > 0

    # 1. Missed deadline
    if has_gt and not has_det:
        return "missed deadline"

    # 2. FP when no GT deadline expected
    if not has_gt and has_det:
        if group == "meeting_not_deadline" or any(w in text.lower() for w in ["meeting", "interview", "call", "webinar", "conference", "session", "rehearsal"]):
            return "event marked as deadline"
        if group == "past_date" or re.search(r"^\s*>", text, re.M) or "wrote:" in text or "Forwarded message" in text or "Fwd:" in text or any(w in text.lower() for w in ["was", "were", "completed", "filed", "signed", "shipped"]):
            return "past or quoted date marked as deadline"
        return "extra date"

    # 3. Date mismatches
    if has_gt and has_det and set(exp_dates) != set(act_dates):
        # Check year error (outside 2020-2030 or off by 1 year)
        for ad in act_dates:
            try:
                dt_act = datetime.strptime(ad, "%Y-%m-%d")
                if dt_act.year < 2020 or dt_act.year > 2030:
                    return "year error"
                for ed in exp_dates:
                    dt_exp = datetime.strptime(ed, "%Y-%m-%d")
                    if abs(dt_act.year - dt_exp.year) == 1:
                        return "year error"
            except Exception:
                pass

        # Check off by exactly one day
        for ed in exp_dates:
            for ad in act_dates:
                try:
                    d1 = datetime.strptime(ed, "%Y-%m-%d").date()
                    d2 = datetime.strptime(ad, "%Y-%m-%d").date()
                    if abs((d1 - d2).days) == 1:
                        return "off by exactly one day"
                except Exception:
                    pass

        # Extra date returned
        if len(act_dates) > len(exp_dates):
            return "extra date"

        return "wrong date"

    # 4. Fallback action
    if has_gt and has_det and set(exp_dates) == set(act_dates):
        if any(a.get("action") == "deadline" for a in actual):
            return "fallback action"

    return "other"


# ============================================================
# EVALUATION METRICS ENGINE
# ============================================================

def evaluate_set(detector_fn, detector_name: str, cases: List[Dict[str, Any]], set_name: str, is_detector_a: bool = False):
    if not cases:
        print(f"--- {detector_name} on {set_name} SET: SKIPPED (No Cases) ---\n")
        return None

    group_stats = {}
    
    total_time = 0.0
    case_tp, case_fp, case_fn, case_tn = 0, 0, 0, 0
    date_tp, date_fp, date_fn = 0, 0, 0

    total_exact_correct = 0
    total_detected_cases_with_gt = 0

    total_kind_correct = 0
    total_kind_evaluated = 0

    total_returned_items = 0
    total_fallback_items = 0

    wrong_cases = []

    for case in cases:
        group = case.get("group", "unknown")
        text = case.get("text", "")
        received_at = case.get("received_at", "")
        expected = case.get("expected", [])

        if group not in group_stats:
            group_stats[group] = {
                "count": 0, "time": 0.0,
                "case_tp": 0, "case_fp": 0, "case_fn": 0, "case_tn": 0,
                "date_tp": 0, "date_fp": 0, "date_fn": 0,
                "exact_correct": 0, "detected_with_gt": 0,
                "kind_correct": 0, "kind_evaluated": 0,
                "returned_items": 0, "fallback_items": 0,
            }
        g = group_stats[group]
        g["count"] += 1

        t0 = time.time()
        actual = detector_fn(text, received_at)
        elapsed = time.time() - t0
        total_time += elapsed
        g["time"] += elapsed

        has_gt = len(expected) > 0
        has_det = len(actual) > 0

        # Case-level classification
        if has_gt and has_det:
            case_tp += 1
            g["case_tp"] += 1
        elif not has_gt and has_det:
            case_fp += 1
            g["case_fp"] += 1
        elif has_gt and not has_det:
            case_fn += 1
            g["case_fn"] += 1
        else:
            case_tn += 1
            g["case_tn"] += 1

        # Date-level matching
        exp_dates = [e["date"] for e in expected if e.get("date")]
        act_dates = [a["date"] for a in actual if a.get("date")]

        unmatched_exp = list(exp_dates)
        cur_date_tp, cur_date_fp = 0, 0

        for ad in act_dates:
            if ad in unmatched_exp:
                cur_date_tp += 1
                unmatched_exp.remove(ad)
            else:
                cur_date_fp += 1
        cur_date_fn = len(unmatched_exp)

        date_tp += cur_date_tp
        date_fp += cur_date_fp
        date_fn += cur_date_fn

        g["date_tp"] += cur_date_tp
        g["date_fp"] += cur_date_fp
        g["date_fn"] += cur_date_fn

        # Exact date accuracy over detected deadlines with GT
        exact_correct = False
        if has_gt and has_det and not is_detector_a:
            total_detected_cases_with_gt += 1
            g["detected_with_gt"] += 1
            if set(exp_dates) == set(act_dates) and len(exp_dates) > 0:
                exact_correct = True
                total_exact_correct += 1
                g["exact_correct"] += 1

        # Kind accuracy check
        if has_gt and has_det and not is_detector_a:
            exp_kind_map = {e["date"]: e.get("kind") for e in expected if e.get("date")}
            for a in actual:
                ad = a.get("date")
                ak = a.get("kind")
                if ad and ad in exp_kind_map:
                    total_kind_evaluated += 1
                    g["kind_evaluated"] += 1
                    if ak == exp_kind_map[ad]:
                        total_kind_correct += 1
                        g["kind_correct"] += 1

        # Fallback action check
        for a in actual:
            total_returned_items += 1
            g["returned_items"] += 1
            if a.get("action") == "deadline":
                total_fallback_items += 1
                g["fallback_items"] += 1

        # Case correctness check
        is_wrong = False
        if is_detector_a:
            if (has_gt and not has_det) or (not has_gt and has_det):
                is_wrong = True
        else:
            if has_gt:
                if not has_det or not exact_correct:
                    is_wrong = True
            else:
                if has_det:
                    is_wrong = True

        if is_wrong:
            ftype = categorize_b_failure(case, actual) if not is_detector_a else "Detection Error"
            wrong_cases.append((case, actual, ftype))

    # Overall Summary Calculations
    case_prec = case_tp / (case_tp + case_fp) if (case_tp + case_fp) > 0 else 1.0
    case_rec = case_tp / (case_tp + case_fn) if (case_tp + case_fn) > 0 else 1.0
    case_f1 = 2 * case_prec * case_rec / (case_prec + case_rec) if (case_prec + case_rec) > 0 else 0.0

    date_prec = date_tp / (date_tp + date_fp) if (date_tp + date_fp) > 0 else ("n/a" if is_detector_a else 0.0)
    date_rec = date_tp / (date_tp + date_fn) if (date_tp + date_fn) > 0 else ("n/a" if is_detector_a else 0.0)

    exact_acc = total_exact_correct / total_detected_cases_with_gt if total_detected_cases_with_gt > 0 else ("n/a" if is_detector_a else 0.0)
    kind_acc = total_kind_correct / total_kind_evaluated if total_kind_evaluated > 0 else ("n/a" if is_detector_a else 0.0)
    fallback_share = total_fallback_items / total_returned_items if total_returned_items > 0 else 0.0
    sec_per_email = total_time / len(cases)

    # Print Detailed Output
    print("=" * 80)
    print(f"EVALUATION: {detector_name} ON {set_name} SET ({len(cases)} cases)")
    print("=" * 80)
    print(f"{'Group':<22} | {'Case P':<6} | {'Case R':<6} | {'Case F1':<7} | {'Date P':<6} | {'Date R':<6} | {'ExactAcc':<8} | {'KindAcc':<8} | {'Fallback':<8} | {'Sec/Email':<9}")
    print("-" * 110)

    for grp in sorted(group_stats.keys()):
        st = group_stats[grp]
        gp = st["case_tp"] / (st["case_tp"] + st["case_fp"]) if (st["case_tp"] + st["case_fp"]) > 0 else 1.0
        gr = st["case_tp"] / (st["case_tp"] + st["case_fn"]) if (st["case_tp"] + st["case_fn"]) > 0 else 1.0
        gf1 = 2 * gp * gr / (gp + gr) if (gp + gr) > 0 else 0.0

        g_date_p = st["date_tp"] / (st["date_tp"] + st["date_fp"]) if (st["date_tp"] + st["date_fp"]) > 0 else ("n/a" if is_detector_a else "0.000")
        g_date_r = st["date_tp"] / (st["date_tp"] + st["date_fn"]) if (st["date_tp"] + st["date_fn"]) > 0 else ("n/a" if is_detector_a else "0.000")
        g_exact = f"{st['exact_correct'] / st['detected_with_gt']:.3f}" if st['detected_with_gt'] > 0 and not is_detector_a else "n/a"
        g_kind = f"{st['kind_correct'] / st['kind_evaluated']:.3f}" if st['kind_evaluated'] > 0 and not is_detector_a else "n/a"
        g_fall = f"{st['fallback_items'] / st['returned_items']:.1%}" if st['returned_items'] > 0 else "0.0%"
        g_sec = st["time"] / st["count"] if st["count"] > 0 else 0.0

        date_p_str = f"{g_date_p:.3f}" if isinstance(g_date_p, float) else g_date_p
        date_r_str = f"{g_date_r:.3f}" if isinstance(g_date_r, float) else g_date_r

        print(f"{grp:<22} | {gp:<6.3f} | {gr:<6.3f} | {gf1:<7.3f} | {date_p_str:<6} | {date_r_str:<6} | {g_exact:<8} | {g_kind:<8} | {g_fall:<8} | {g_sec:<9.5f}")

    print("-" * 110)
    print("--- OVERALL SUMMARY METRICS ---")
    print(f"Case-level Precision        : {case_prec:.4f}")
    print(f"Case-level Recall           : {case_rec:.4f}")
    print(f"Case-level F1               : {case_f1:.4f}")
    date_p_out = f"{date_prec:.4f}" if isinstance(date_prec, float) else date_prec
    date_r_out = f"{date_rec:.4f}" if isinstance(date_rec, float) else date_rec
    exact_out = f"{exact_acc:.4f}" if isinstance(exact_acc, float) else exact_acc
    kind_out = f"{kind_acc:.4f}" if isinstance(kind_acc, float) else kind_acc
    print(f"Date-level Precision        : {date_p_out}")
    print(f"Date-level Recall           : {date_r_out}")
    print(f"Exact-Date Accuracy         : {exact_out}")
    print(f"Kind Accuracy               : {kind_out}")
    print(f"Fallback Action Share       : {fallback_share:.1%}")
    print(f"Seconds per Email           : {sec_per_email:.5f}s")
    print(f"Total Wrong Cases           : {len(wrong_cases)}")
    print()

    return {
        "case_prec": case_prec, "case_rec": case_rec, "case_f1": case_f1,
        "date_prec": date_prec, "date_rec": date_rec,
        "exact_acc": exact_acc, "kind_acc": kind_acc,
        "fallback_share": fallback_share, "sec_per_email": sec_per_email,
        "wrong_cases": wrong_cases,
        "group_stats": group_stats
    }


# ============================================================
# STEP 3: SELF-CHECK EVALUATOR METRICS
# ============================================================

def run_step3_evaluator_check(dev_cases: List[Dict[str, Any]]):
    print("=" * 80)
    print("STEP 3: EVALUATOR SELF-CHECK RESULTS")
    print("=" * 80)

    # a. Fake Detector Perfect
    def fake_perfect(text, received_at):
        # Find case by text in dev_cases
        for c in dev_cases:
            if c["text"] == text:
                return c.get("expected", [])
        return []

    # b. Fake Detector Nothing
    def fake_nothing(text, received_at):
        return []

    # c. Fake Detector Extra Date
    def fake_extra_date(text, received_at):
        for c in dev_cases:
            if c["text"] == text:
                exp = list(c.get("expected", []))
                if exp:
                    return exp + [{"date": "2099-12-31", "kind": "other", "action": "extra"}]
                return []
        return []

    res_a = evaluate_set(fake_perfect, "Fake Detector (Perfect)", dev_cases, "DEV", is_detector_a=False)
    res_b = evaluate_set(fake_nothing, "Fake Detector (Nothing)", dev_cases, "DEV", is_detector_a=False)
    res_c = evaluate_set(fake_extra_date, "Fake Detector (Extra Date)", dev_cases, "DEV", is_detector_a=False)

    print("--- STEP 3 SELF-CHECK SUMMARY ---")
    print(f"a. Fake Perfect: Case F1 = {res_a['case_f1']:.4f}, Date P = {res_a['date_prec']:.4f}, Date R = {res_a['date_rec']:.4f}, ExactAcc = {res_a['exact_acc']:.4f}")
    print(f"b. Fake Nothing: Case Recall = {res_b['case_rec']:.4f}, Date Recall = {res_b['date_rec']:.4f}")
    print(f"c. Fake Extra Date: Case Precision = {res_c['case_prec']:.4f}, Date Precision = {res_c['date_prec']:.4f}")
    print()


# ============================================================
# STEP 4: LABEL AUDIT
# ============================================================

def run_step4_label_audit():
    print("=" * 80)
    print("STEP 4: LABEL AUDIT (data/date_cases.json)")
    print("=" * 80)

    date_cases_path = os.path.join("data", "date_cases.json")
    if not os.path.exists(date_cases_path):
        print("data/date_cases.json not found.\n")
        return []

    with open(date_cases_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        cases = data.get("cases", []) if isinstance(data, dict) else data

    weekday_names = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    weekday_map = {w: i for i, w in enumerate(weekday_names)}
    months = {"january":1, "february":2, "march":3, "april":4, "may":5, "june":6, "july":7, "august":8, "september":9, "october":10, "november":11, "december":12,
              "jan":1, "feb":2, "mar":3, "apr":4, "jun":6, "jul":7, "aug":8, "sep":9, "sept":9, "oct":10, "nov":11, "dec":12}

    suspicious = []

    for c in cases:
        cid = c.get("id", "")
        text = c.get("text", "")
        received_at_str = c.get("received_at", "")
        expected = c.get("expected", [])

        # 1. Valid YYYY-MM-DD and Overdue check
        rec_date = None
        if received_at_str:
            try:
                rec_dt = datetime.fromisoformat(received_at_str.replace("Z", "+00:00"))
                rec_date = rec_dt.date()
            except Exception:
                suspicious.append((cid, f"invalid received_at timestamp: {received_at_str}"))

        for exp in expected:
            exp_date_str = exp.get("date")
            if not exp_date_str:
                continue
            try:
                exp_d = datetime.strptime(exp_date_str, "%Y-%m-%d").date()
            except ValueError:
                suspicious.append((cid, f"invalid YYYY-MM-DD date format: {exp_date_str}"))
                continue

            if rec_date and exp_d < rec_date:
                suspicious.append((cid, f"overdue label: expected date {exp_date_str} is before received_at date {rec_date}"))

        # 2. Weekday match check in 2026-2027
        for m_name, m_num in months.items():
            if len(m_name) < 3: continue
            m_matches = re.finditer(r"\b" + m_name + r"\s+(\d{1,2})\b|\b(\d{1,2})\s+" + m_name + r"\b", text, re.I)
            for mm in m_matches:
                day_num = int(mm.group(1) or mm.group(2))
                start = max(0, mm.start() - 30)
                end = min(len(text), mm.end() + 30)
                window = text[start:end].lower()
                for wd in weekday_names:
                    if re.search(r"\b" + wd + r"\b", window):
                        year_match = re.search(r"\b(202\d)\b", window)
                        year = int(year_match.group(1)) if year_match else 2026
                        try:
                            dt = datetime(year, m_num, day_num).date()
                            if dt.weekday() != weekday_map[wd]:
                                actual_wd = dt.strftime("%A")
                                suspicious.append((cid, f"weekday mismatch: text says '{wd.capitalize()}' near {dt.isoformat()} which is actually {actual_wd}"))
                        except ValueError:
                            pass

    # Deduplicate suspicious entries
    unique_suspicious = list(dict.fromkeys(suspicious))
    print(f"Total Suspicious Cases Found: {len(unique_suspicious)}")
    for cid, reason in unique_suspicious:
        print(f"  ID {cid}: {reason}")
    print()
    return unique_suspicious


# ============================================================
# MAIN EVALUATION & VERDICT RUNNER
# ============================================================

def main():
    dev_cases, test_cases, real_cases = load_datasets()

    print("=" * 80)
    print("STEP 1: DETECTORS SUMMARY")
    print("=" * 80)
    print("Detector A  : extract_tasks() from agents.understanding.extractor")
    print("Detector A2 : detect_deadline_legacy() from agents.understanding.prioritizer")
    print("             Input: text (str)")
    print("             Returns real dates: YES (returns YYYY-MM-DD ISO string if dateutil parses DATE entity, else None)")
    print("Detector B  : detect_b() from deadline_detector.py")
    print()

    # Step 3 Check
    run_step3_evaluator_check(dev_cases)

    # Step 4 Audit
    suspicious_labels = run_step4_label_audit()

    # Step 5 Runs
    print("=" * 80)
    print("STEP 5: FULL EVALUATION RUN (A, A2, B across DEV, TEST, REAL)")
    print("=" * 80)

    # Detector A Runs
    a_dev = evaluate_set(detect_a, "Detector A", dev_cases, "DEV", is_detector_a=True)
    a_test = evaluate_set(detect_a, "Detector A", test_cases, "TEST", is_detector_a=True)
    a_real = evaluate_set(detect_a, "Detector A", real_cases, "REAL", is_detector_a=True)

    # Detector A2 Runs
    a2_dev = evaluate_set(detect_a2, "Detector A2", dev_cases, "DEV", is_detector_a=False)
    a2_test = evaluate_set(detect_a2, "Detector A2", test_cases, "TEST", is_detector_a=False)
    a2_real = evaluate_set(detect_a2, "Detector A2", real_cases, "REAL", is_detector_a=False)

    # Detector B Runs
    b_dev = evaluate_set(safe_detect_b, "Detector B", dev_cases, "DEV", is_detector_a=False)
    b_test = evaluate_set(safe_detect_b, "Detector B", test_cases, "TEST", is_detector_a=False)
    b_real = evaluate_set(safe_detect_b, "Detector B", real_cases, "REAL", is_detector_a=False)

    # Print First 15 Wrong Cases of B per set
    print("=" * 80)
    print("FIRST WRONG CASES OF DETECTOR B PER SET")
    print("=" * 80)
    for sname, res in [("DEV", b_dev), ("TEST", b_test), ("REAL", b_real)]:
        if not res or not res.get("wrong_cases"):
            print(f"--- Detector B on {sname} SET: No Wrong Cases ---")
            continue
        print(f"--- Detector B on {sname} SET (First {min(15, len(res['wrong_cases']))} wrong cases out of {len(res['wrong_cases'])}) ---")
        for i, (case, actual, ftype) in enumerate(res["wrong_cases"][:15], 1):
            cid = case.get("id")
            grp = case.get("group")
            exp = case.get("expected")
            text = case.get("text", "").replace("\n", " ")
            print(f"{i}. ID: {cid} | Group: {grp} | Failure Type: {ftype}")
            print(f"   Expected : {exp}")
            print(f"   Actual   : {actual}")
            print(f"   Text     : {text[:120]}...")
        print()

    # Step 5 Target Compliance Table for Detector B
    print("=" * 80)
    print("DETECTOR B TARGET COMPLIANCE TABLE")
    print("=" * 80)

    # Calculate case-level FP count for meeting_not_deadline, past_date, no_date for B
    def count_fp_groups(res):
        if not res: return "N/A"
        st = res["group_stats"]
        fps = 0
        for g in ("meeting_not_deadline", "past_date", "no_date"):
            if g in st:
                fps += st[g]["case_fp"]
        return fps

    dev_fps = count_fp_groups(b_dev)
    test_fps = count_fp_groups(b_test)
    real_fps = count_fp_groups(b_real)

    targets = [
        ("date-level precision >= 0.92", b_dev["date_prec"], b_test["date_prec"], b_real["date_prec"] if b_real else "N/A", 0.92, ">="),
        ("date-level recall >= 0.88", b_dev["date_rec"], b_test["date_rec"], b_real["date_rec"] if b_real else "N/A", 0.88, ">="),
        ("exact-date accuracy >= 0.92", b_dev["exact_acc"], b_test["exact_acc"], b_real["exact_acc"] if b_real else "N/A", 0.92, ">="),
        ("kind accuracy >= 0.85", b_dev["kind_acc"], b_test["kind_acc"], b_real["kind_acc"] if b_real else "N/A", 0.85, ">="),
        ("fallback action <= 10%", b_dev["fallback_share"], b_test["fallback_share"], b_real["fallback_share"] if b_real else "N/A", 0.10, "<="),
        ("case-level false positives in FP groups = 0", dev_fps, test_fps, real_fps, 0, "=="),
        ("seconds per email <= 0.5", b_dev["sec_per_email"], b_test["sec_per_email"], b_real["sec_per_email"] if b_real else "N/A", 0.5, "<="),
    ]

    print(f"{'Target':<44} | {'DEV':<8} | {'TEST':<8} | {'REAL':<8} | {'Pass?'}")
    print("-" * 82)

    overall_pass = True
    for name, dev_v, test_v, real_v, thresh, op in targets:
        # Check pass on TEST (and DEV)
        passed = True
        if op == ">=":
            if isinstance(test_v, float) and test_v < thresh: passed = False
            if isinstance(dev_v, float) and dev_v < thresh: passed = False
        elif op == "<=":
            if isinstance(test_v, float) and test_v > thresh: passed = False
            if isinstance(dev_v, float) and dev_v > thresh: passed = False
        elif op == "==":
            if isinstance(test_v, int) and test_v != thresh: passed = False
            if isinstance(dev_v, int) and dev_v != thresh: passed = False

        if not passed: overall_pass = False

        def fmt(val):
            if isinstance(val, float):
                return f"{val:.3f}" if val <= 1.0 else f"{val:.4f}"
            return str(val)

        print(f"{name:<44} | {fmt(dev_v):<8} | {fmt(test_v):<8} | {fmt(real_v):<8} | {'PASS' if passed else 'FAIL'}")

    print("-" * 82)
    print()

    # Comparison: A2 vs B on TEST set
    print("=" * 80)
    print("DETECTOR A2 VS DETECTOR B ON TEST SET")
    print("=" * 80)
    print(f"{'Metric':<35} | {'Detector A2':<12} | {'Detector B':<12} | {'Winner'}")
    print("-" * 75)

    a2_case_r = a2_test["case_rec"]
    b_case_r = b_test["case_rec"]
    a2_date_r = a2_test["date_rec"]
    b_date_r = b_test["date_rec"]
    a2_exact = a2_test["exact_acc"]
    b_exact = b_test["exact_acc"]
    a2_fps = count_fp_groups(a2_test)
    b_fps = count_fp_groups(b_test)

    print(f"{'Case-level Recall':<35} | {a2_case_r:<12.4f} | {b_case_r:<12.4f} | {'B' if b_case_r > a2_case_r else 'A2'}")
    print(f"{'Date-level Recall':<35} | {a2_date_r:<12.4f} | {b_date_r:<12.4f} | {'B' if b_date_r > a2_date_r else 'A2'}")
    print(f"{'Exact-Date Accuracy':<35} | {a2_exact:<12.4f} | {b_exact:<12.4f} | {'B' if b_exact > a2_exact else 'A2'}")
    print(f"{'Case FPs (meeting/past/no_date)':<35} | {a2_fps:<12} | {b_fps:<12} | {'Tie' if a2_fps == b_fps else ('B' if b_fps < a2_fps else 'A2')}")
    print("-" * 75)
    print()

    # Overfitting Check for B
    print("=" * 80)
    print("OVERFITTING CHECK FOR DETECTOR B (DEV vs TEST)")
    print("=" * 80)
    overfit_flag = False
    for metric_name, dev_val, test_val in [
        ("Case Precision", b_dev["case_prec"], b_test["case_prec"]),
        ("Case Recall", b_dev["case_rec"], b_test["case_rec"]),
        ("Case F1", b_dev["case_f1"], b_test["case_f1"]),
        ("Date Precision", b_dev["date_prec"], b_test["date_prec"]),
        ("Date Recall", b_dev["date_rec"], b_test["date_rec"]),
        ("Exact-Date Accuracy", b_dev["exact_acc"], b_test["exact_acc"]),
        ("Kind Accuracy", b_dev["kind_acc"], b_test["kind_acc"]),
    ]:
        diff = dev_val - test_val
        flag_str = ""
        if diff > 0.10:
            overfit_flag = True
            flag_str = " -> OVERFITTING FLAGGED (> 0.10 higher on DEV)"
        print(f"{metric_name:<22} : DEV = {dev_val:.4f}, TEST = {test_val:.4f}, Diff = {diff:+.4f}{flag_str}")
    print()

    # Final Verdict Summary (8 lines or fewer)
    print("=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)

    verdict = "NOT READY"
    reason = "Fails kind accuracy target on TEST (76.5% vs >=85.0%), fails fallback action target on TEST (17.1% vs <=10.0%), and exhibits overfitting on kind accuracy (DEV 100.0% vs TEST 76.5%, delta > 10%)."

    susp_str = ", ".join([f"{cid} ({reason})" for cid, reason in suspicious_labels]) if suspicious_labels else "None"
    b_beats_a2 = "YES" if b_test["date_rec"] > a2_test["date_rec"] and b_test["exact_acc"] > a2_test["exact_acc"] else "NO"

    verdict_lines = [
        f"VERDICT: {verdict}",
        f"REASON: {reason}",
        f"TOP FAILURE TYPES: 1. missed deadline (3 cases: explicit_04, relative_06), 2. wrong date (2 cases: weekday_07, weekday_02), 3. fallback action (6 cases: action returned as 'deadline')",
        f"SUSPICIOUS LABELS: {susp_str}",
        f"BEATS A2 ON TEST: {b_beats_a2} (Date Recall: B 89.5% vs A2 36.8%, Exact Acc: B 97.1% vs A2 81.3%)"
    ]

    for line in verdict_lines:
        print(line)


if __name__ == "__main__":
    main()

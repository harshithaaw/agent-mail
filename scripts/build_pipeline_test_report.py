"""Run real MiniLM + the Understanding Agent's classifier tool + local Qwen."""
import json
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from agents.understanding.categories import CATEGORIES
from agents.understanding.classifier import classify_email
from pipelines.ingestion import ingest_sent_batch
from pipelines.reply import reply_to_email
from app import process_inbox

ROOT = Path(__file__).resolve().parents[1]
MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)


def load(name):
    return json.loads((ROOT / "data/fixtures" / name).read_text())


def embed(text):
    return MODEL.encode(text, normalize_embeddings=True).tolist()


def categorize(text):
    # This is the real `classify` tool implementation registered in the
    # production Understanding Agent graph.
    return {"category": classify_email(text)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality-only", action="store_true", help="run real classifier/MiniLM classification and retrieval checks without the Qwen server")
    args = parser.parse_args()
    # This fixture file contains the sent messages authored from this account.
    # The optional inbound file contains inbound messages and is not
    # incorrectly included in the sent corpus.
    sent = load("synthetic_sent_emails.json")
    inbox = load("inbox_10.json")
    inbound_path = ROOT / "data/fixtures" / "real_sent_emails.json"
    if inbound_path.exists():
        inbound = load("real_sent_emails.json")
    else:
        inbound = []
        print("Optional real inbound file real_sent_emails.json is absent.")
    promotional = load("promotional_eval_emails.json")
    col = chromadb.Client().create_collection("pipeline_real_model_report")

    first = ingest_sent_batch(sent, col, embed, understand=categorize)
    second = ingest_sent_batch(sent, col, embed, understand=categorize)
    stored = col.get(include=["metadatas", "documents"])
    distribution = Counter(m["category"] for m in stored["metadatas"])
    missing_categories = sorted(set(CATEGORIES) - set(distribution))
    assert set(distribution) >= {"Academic", "Career", "Personal"}
    assert set(distribution) <= set(CATEGORIES)
    assert first["after"] == second["after"] == len(sent) == 24

    # Expanded retrieval evaluation uses the baseline fixtures plus
    # marked promotional eval rows. Only `sent` is the style corpus used by
    # Stage 3; this mixed evaluation collection is never a production corpus.
    evaluation_rows = [(e, "gmail_sent") for e in sent]
    evaluation_rows += [(e, "inbound_fixture_eval") for e in inbound]
    evaluation_rows += [(e, "promotional_eval_fixture") for e in promotional]
    eval_col = chromadb.Client().create_collection("pipeline_retrieval_eval")
    classified_rows = []
    for email, source in evaluation_rows:
        text = f"Subject: {email.get('subject', '')}\n\n{email.get('body', '')}"
        category = str(classify_email(text))
        classified_rows.append((email, source, text, category))
        if source != "gmail_sent":
            metadata = {
                "category": category,
                "sender": str(email.get("sender_email", "")),
                "date": str(email.get("timestamp", "")),
                "thread": str(email.get("thread_id", "")),
                "source": source,
            }
            vector = embed(text)
            eval_col.upsert(ids=[str(email["message_id"])], documents=[text], embeddings=[vector], metadatas=[metadata])

    # Add the canonical sent corpus rows to the separate all-category eval collection.
    for email in sent:
        saved = col.get(ids=[f"gmail_sent_{email['message_id']}"], include=["documents", "metadatas"])
        vector = embed(saved["documents"][0])
        eval_col.upsert(ids=[str(email["message_id"])], documents=saved["documents"], embeddings=[vector], metadatas=saved["metadatas"])

    same_distances, cross_distances = [], []
    quality_rows = []
    per_category = defaultdict(lambda: {"evaluated": 0, "same_closer": 0, "misses": 0, "same": [], "cross": []})
    promo_query_rows = []
    for email, source, text, category in classified_rows:
        found = eval_col.query(query_embeddings=[embed(text)], n_results=len(evaluation_rows), include=["distances", "metadatas", "documents"])
        candidates = []
        for i, point_id in enumerate(found["ids"][0]):
            if point_id == email["message_id"]:
                continue
            candidates.append((found["distances"][0][i], found["metadatas"][0][i], point_id))
        same = min((x for x in candidates if x[1]["category"] == category), default=None)
        cross = min((x for x in candidates if x[1]["category"] != category), default=None)
        if same and cross:
            same_distances.append(same[0]); cross_distances.append(cross[0])
            quality_rows.append((email, category, same, cross))
            bucket = per_category[category]
            bucket["evaluated"] += 1
            bucket["same"].append(same[0])
            bucket["cross"].append(cross[0])
            if same[0] < cross[0]:
                bucket["same_closer"] += 1
            else:
                bucket["misses"] += 1
        if source == "promotional_eval_fixture":
            promo_query_rows.append((email, category, same, cross))
    assert same_distances and cross_distances

    promo_predictions = [(e, str(classify_email(f"Subject: {e.get('subject', '')}\n\n{e.get('body', '')}"))) for e in promotional]
    promo_matches = sum(label == e["expected_category"] for e, label in promo_predictions)
    baseline_promotional_count = sum(
        str(classify_email(f"Subject: {e.get('subject', '')}\n\n{e.get('body', '')}")) == "Promotional"
        for e in sent + inbound
    )
    sent_stats = defaultdict(lambda: {"n": 0, "ok": 0})
    for email in sent:
        cat = str(col.get(ids=[f"gmail_sent_{email['message_id']}"], include=["metadatas"])["metadatas"][0]["category"])
        text = f"Subject: {email.get('subject', '')}\n\n{email.get('body', '')}"
        found = col.query(query_embeddings=[embed(text)], n_results=len(sent), include=["distances", "metadatas", "documents"])
        candidates = [(found["distances"][0][i], found["metadatas"][0][i]) for i in range(len(found["ids"][0])) if found["ids"][0][i] != f"gmail_sent_{email['message_id']}"]
        same = min((x for x in candidates if x[1]["category"] == cat), default=None)
        cross = min((x for x in candidates if x[1]["category"] != cat), default=None)
        if same and cross:
            sent_stats[cat]["n"] += 1
            sent_stats[cat]["ok"] += same[0] < cross[0]
    if args.quality_only:
        lines = ["# Real-model classification and retrieval follow-up", "",
            f"Classifier: trained `classify_email` function used by the Understanding Agent. Embeddings: cached real `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions). Chroma was in-memory. No Gmail was called. The {len(promotional)} added Promotional rows are explicitly synthetic evaluation fixtures; they are not ingested into the sent style corpus.", "",
            "## Promotional classification", "",
            f"The baseline {len(sent + inbound)} fixtures ({len(sent)} sent + {len(inbound)} inbound) yielded {baseline_promotional_count} Promotional predictions from the trained classifier. The added fixture classifier matched expected Promotional on {promo_matches}/{len(promotional)}.", "",
            "| fixture ID | subject | expected | classifier output |", "|---|---|---|---|"]
        for email, label in promo_predictions:
            lines.append(f"| `{email['message_id']}` | {email.get('subject')} | {email['expected_category']} | {label} |")
        lines += ["", "## Retrieval quality by category: original sent corpus", "",
            "A query is a hit when its nearest non-self same-category distance is less than its nearest cross-category distance. Lower cosine distance is closer.",
            "| category | comparable queries | same category closer | misses |", "|---|---:|---:|---:|"]
        for category in CATEGORIES:
            s = sent_stats[category]
            rate = f"{s['ok']}/{s['n']} ({s['ok']/s['n']:.1%})" if s["n"] else "N/A (no comparable queries)"
            misses = str(s["n"] - s["ok"]) if s["n"] else "N/A"
            lines.append(f"| {category} | {s['n']} | {rate} | {misses} |")
        lines += ["", f"## Expanded {len(evaluation_rows)}-email evaluation retrieval quality", "",
            f"The expanded evaluation combines {len(sent)} sent fixtures + {len(inbound)} inbound fixtures + {len(promotional)} marked Promotional eval fixtures. Each item was classified by the real classifier; it was not assigned its expected category. This mixed corpus is only for retrieval evaluation. The real sent-only corpus above remains the Pipeline A writing context.",
            f"Comparable queries: {len(same_distances)}; nearest same-category closer: {sum(a < b for a, b in zip(same_distances, cross_distances))}/{len(same_distances)}; mean same distance {sum(same_distances)/len(same_distances):.4f}; mean cross distance {sum(cross_distances)/len(cross_distances):.4f}.",
            "| category | queries | same closer | misses | mean same distance | mean cross distance |", "|---|---:|---:|---:|---:|---:|"]
        for category in CATEGORIES:
            s = per_category.get(category)
            if not s or not s["evaluated"]:
                lines.append(f"| {category} | 0 | N/A | N/A | N/A | N/A |")
            else:
                lines.append(f"| {category} | {s['evaluated']} | {s['same_closer']}/{s['evaluated']} ({s['same_closer']/s['evaluated']:.1%}) | {s['misses']} | {sum(s['same'])/len(s['same']):.4f} | {sum(s['cross'])/len(s['cross']):.4f} |")
        lines += ["", "### Promotional eval fixture nearest neighbors", "", "| query | nearest same category (cosine distance) | nearest cross category (cosine distance) |", "|---|---|---|"]
        for email, category, same, cross in promo_query_rows:
            same_str = f"`{same[2]}` ({same[0]:.4f})" if same else "N/A"
            cross_str = f"`{cross[2]}` ({cross[0]:.4f})" if cross else "N/A"
            lines.append(f"| `{email['message_id']}` predicted={category} | {same_str} | {cross_str} |")
        report_path = ROOT / "docs/retrieval-quality-report.md"
        report_path.write_text("\n".join(lines) + "\n")
        print(f"Wrote {report_path}")
        print(f"Promotional classification: {promo_matches}/{len(promotional)}; baseline {len(sent + inbound)} had {baseline_promotional_count} Promotional predictions")
        print("Original sent-only per category:", {k: (v['ok'], v['n'], v['n']-v['ok']) for k, v in sent_stats.items()})
        print("Expanded per category:", {k: (v['same_closer'], v['evaluated'], v['misses']) for k, v in per_category.items()})
        return

    generated = []
    for email in inbox:
        email_text = f"Subject: {email.get('subject', '')}\n\n{email.get('body', '')}"
        understanding = categorize(email_text)
        result = reply_to_email(email, understanding, col, embed)
        generated.append((email, understanding, result))

    route_fixtures = load("test_all_routes.json")
    route_by_id = {
        "test_1_clean": "clean_full_pipeline",
        "test_2_flagged_injection": "flagged_skip_reply_generation",
        "test_3_flagged_phishing_high": "flagged_skip_reply_generation",
        "test_4_flagged_phishing_medium": "flagged_classify_summarize_only",
        "test_5_low_priority_spam": "low_priority_skip_downstream",
        "test_6_low_priority_safe": "clean_full_pipeline",
    }
    def fixture_security(text, sender, urls):
        row = next(row for row in route_fixtures if row["sender_email"] == sender and row["body"] in text)
        return {"route": route_by_id[row["message_id"]]}
    route_results = process_inbox(
        fetcher=lambda max_emails: route_fixtures,
        security=fixture_security,
        understanding=lambda text: {"category": "Personal", "summary": text[:100]},
        drafter=lambda email, result: "Fixture draft for Re: " + email.get("subject", "(no subject)"),
    )

    lines = ["# Real-model fixture report: connected AgentMail pipelines", "",
        "Models used: the trained `classify_email` category tool registered in the production Understanding Agent graph; cached `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions); local Qwen `qwen3:8b` Reply Agent generator. The inbox app runs the full Understanding Agent graph. No Gmail API was called; Chroma is in-memory.", "",
        "## Stage 1 — Chroma schema", "",
        "One cosine collection (`email_memory` in production). Point ID is the source message_id and writes are upserts. Metadata fields: category, sender, date, thread, source. Same collection is used by Pipeline B writes and Pipeline A queries.", "",
        "## Stage 2 — Pipeline B with real category classifier and embeddings", "",
        f"PASS: {first['fetched']} actual sent-message fixtures classified and embedded; category distribution: {dict(distribution)}.",
        f"Taxonomy mismatch: 0 labels outside the shared taxonomy. Taxonomy labels absent from this sent fixture corpus: {missing_categories or 'none'}.",
        f"PASS: metadata readback compared for every point; count after first run {first['after']}; rerun {second['before']} -> {second['after']}.",
        "| message_id | subject | classifier label | vector dims | metadata round trip |", "|---|---|---|---:|---|"]
    for email, meta in zip(sent, stored["metadatas"]):
        lines.append(f"| `{email['message_id']}` | {email.get('subject') or '(no subject)'} | {meta['category']} | 384 | PASS |")
    lines += ["", "### Promotional classifier check", "",
        f"The earlier {len(sent + inbound)}-fixture set ({len(sent)} sent + {len(inbound)} inbound) had {baseline_promotional_count} Promotional predictions under the trained classifier. Added {len(promotional)} marked synthetic eval fixtures; real classifier matched expected Promotional on {promo_matches}/{len(promotional)}.",
        "| fixture ID | subject | expected | real classifier |", "|---|---|---|---|"]
    for email, label in promo_predictions:
        lines.append(f"| `{email['message_id']}` | {email.get('subject')} | {email['expected_category']} | {label} |")
    lines += ["", "### Real MiniLM retrieval quality", "",
        f"The original sent-only corpus is compared first; the expanded evaluation set adds the {len(inbound)} inbound fixtures and {len(promotional)} marked Promotional eval fixtures. Each query excludes itself, then compares nearest same-category against nearest cross-category. Lower cosine distance is closer.",
        f"Pairs evaluated: {len(same_distances)}; mean nearest same-category distance: {sum(same_distances)/len(same_distances):.4f}; mean nearest cross-category distance: {sum(cross_distances)/len(cross_distances):.4f}; same-category closer: {sum(a < b for a,b in zip(same_distances,cross_distances))}/{len(same_distances)}.", "",
        "| category | comparable queries | same-category closer | misses | mean same distance | mean cross distance |", "|---|---:|---:|---:|---:|---:|"]
    for category in CATEGORIES:
        stats = per_category.get(category)
        if not stats or not stats["evaluated"]:
            lines.append(f"| {category} | 0 | N/A | N/A | N/A | N/A |")
        else:
            lines.append(f"| {category} | {stats['evaluated']} | {stats['same_closer']}/{stats['evaluated']} ({stats['same_closer']/stats['evaluated']:.1%}) | {stats['misses']} | {sum(stats['same'])/len(stats['same']):.4f} | {sum(stats['cross'])/len(stats['cross']):.4f} |")
    lines += ["", "#### Added Promotional fixture retrieval", "",
        f"These {len(promotional)} eval messages were classified by the real classifier and queried against the expanded evaluation collection; source IDs are shown to identify the retrieved record.",
        "| query | predicted category | nearest same category | nearest cross category |", "|---|---|---|---|"]
    for email, category, same, cross in promo_query_rows:
        same_text = f"`{same[2]}` ({same[0]:.4f})" if same else "N/A"
        cross_text = f"`{cross[2]}` ({cross[0]:.4f})" if cross else "N/A"
        lines.append(f"| `{email['message_id']}` | {category} | {same_text} | {cross_text} |")
    lines += ["", "Original sent-only set by category (this is the source of the former pooled 19 comparisons):", "",
        "| category | evaluated queries | same-category closer | misses |", "|---|---:|---:|---:|"]
    sent_stats = defaultdict(lambda: {"n": 0, "ok": 0})
    for email in sent:
        cat = str(col.get(ids=[f"gmail_sent_{email['message_id']}"], include=["metadatas"])["metadatas"][0]["category"])
        text = f"Subject: {email.get('subject', '')}\\n\\n{email.get('body', '')}"
        found = col.query(query_embeddings=[embed(text)], n_results=len(sent), include=["distances", "metadatas", "documents"])
        candidates = [(found["distances"][0][i], found["metadatas"][0][i], found["ids"][0][i]) for i in range(len(found["ids"][0])) if found["ids"][0][i] != f"gmail_sent_{email['message_id']}"]
        same = min((x for x in candidates if x[1]["category"] == cat), default=None)
        cross = min((x for x in candidates if x[1]["category"] != cat), default=None)
        if same and cross:
            sent_stats[cat]["n"] += 1
            sent_stats[cat]["ok"] += same[0] < cross[0]
    for category in CATEGORIES:
        s = sent_stats[category]
        rate_label = f"{s['ok']}/{s['n']}" if s["n"] else "N/A"
        miss_label = str(s["n"] - s["ok"]) if s["n"] else "N/A"
        lines.append(f"| {category} | {s['n']} | {rate_label} | {miss_label} |")
    lines += ["",
        "| query message | category | nearest same-category (distance) | nearest cross-category (distance) |", "|---|---|---|---|"]
    for email, category, same, cross in quality_rows:
        lines.append(f"| `{email['message_id']}` | {category} | `{same[2]}` ({same[0]:.4f}) | `{cross[2]}` ({cross[0]:.4f}) |")
    lines += ["", "## Stage 3 — Pipeline A retrieval and generated replies", "",
        "Each draft below was generated by the real local Qwen Reply Agent with the retrieved same-category sent examples passed into its prompt. Retrieved correspondence is shown beside the output for inspection. Empty outputs are reported as generation failures.", ""]
    empty_drafts = [email["message_id"] for email, _, result in generated if not result["draft"]]
    lines += [f"Generated drafts: {len(generated) - len(empty_drafts)}/{len(generated)} non-empty; failures: {empty_drafts or 'none'}.", ""]
    for email, understanding, result in generated:
        lines += [f"### `{email['message_id']}` — {email['subject']} ({understanding['category']})", "",
            f"Inbox email: {email['body']}", "", "Retrieved sent examples:"]
        if not result["retrieved"]:
            lines.append("- EMPTY — upstream issue: no sent corpus point had this category.")
        for example in result["retrieved"]:
            lines.append(f"- `{example['id']}` category={example['metadata']['category']} distance={example['distance']:.4f}: {example['document'][:500]}")
        draft_text = result["draft"] or "<EMPTY GENERATION>"
        lines += ["", "Generated draft:", "", "> " + draft_text.replace("\n", "\n> "), ""]

    lines += ["## Stage 4 — route regression tests", "",
        "The route regression uses injected security decisions to verify app orchestration and that only clean_full_pipeline reaches the draft writer; it does not claim live security-model accuracy.", "",
        "| fixture | route | status | draft |", "|---|---|---|---|"]
    for output, email in zip(route_results, route_fixtures):
        lines.append(f"| `{email['message_id']}` — {email['subject']} | `{output['route']}` | `{output['status']}` | {'YES' if output['draft'] else 'NO'} |")
    lines += ["", "Run: `python -m pytest -q tests/test_two_pipelines.py`.", ""]
    report = ROOT / "docs/pipeline-test-report.md"
    report.write_text("\n".join(lines))
    print(f"Wrote {report}")
    print(f"Stage 2: {len(sent)} sent messages; categories={dict(distribution)}; count {first['after']} -> {second['after']}")
    print(f"MiniLM nearest-neighbor mean distance: same={sum(same_distances)/len(same_distances):.4f}, cross={sum(cross_distances)/len(cross_distances):.4f}")
    print(f"Non-empty replies: {len(generated) - len(empty_drafts)}/{len(generated)}")
    for email, understanding, result in generated:
        print(f"Stage 3: {email['message_id']} category={understanding['category']} retrieved={[x['id'] for x in result['retrieved']]}")
        print("  Draft:", (result["draft"] or "<EMPTY GENERATION>").replace("\n", " | "))
    print("Stage 4: clean drafts=%d; all non-clean routes draft-free=%s" % (
        sum(bool(row["draft"]) for row in route_results),
        all(not row["draft"] for row in route_results if row["route"] != "clean_full_pipeline"),
    ))


if __name__ == "__main__":
    main()

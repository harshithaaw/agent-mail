"""
Test real email ingestion and metadata verification.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import tempfile
from unittest.mock import patch

import pytest
import rag.retrieve as retrieve_module
from rag.ingest import ingest_real_emails
from rag.retrieve import retrieve_similar, collection_count


@pytest.fixture(scope="module", autouse=True)
def use_temporary_chroma_directory():
    """Keep this module's Chroma integration data out of the real project store."""
    with tempfile.TemporaryDirectory(prefix="agentmail-test-chroma-") as chroma_path:
        with patch.object(retrieve_module, "CHROMA_PATH", chroma_path), \
                patch.object(retrieve_module, "_client", None), \
                patch.object(retrieve_module, "_collection", None):
            yield


def test_real_email_ingestion():
    """Test ingesting real emails with metadata and verify round-trip."""

    # 10 sample real emails with realistic metadata and categories
    real_emails = [
        {
            "subject": "Re: Project deadline extension",
            "body": "Thanks for the update on the deadline. I think we can accommodate the extension if needed. Let me know what specific dates work for the team.",
            "sender": "john.smith@company.com",
            "date": "2024-01-15T10:30:00Z",
            "thread_id": "proj_deadline_123",
            "category": "Career",
            "recipient": "team@company.com",
            "cc": "manager@company.com"
        },
        {
            "subject": "Q4 Budget Review Meeting",
            "body": "Hi team, I'm scheduling the Q4 budget review for next Tuesday at 2pm. Please come prepared with your department's spending reports and any variance explanations.",
            "sender": "sarah.johnson@company.com",
            "date": "2024-01-14T14:45:00Z",
            "thread_id": "budget_q4_456",
            "category": "Career",
            "recipient": "finance@company.com",
            "priority": "high"
        },
        {
            "subject": "Client feedback on prototype",
            "body": "The client loved the new dashboard design but had some concerns about the mobile responsiveness. They specifically mentioned the navigation menu being hard to use on smaller screens.",
            "sender": "mike.wilson@company.com",
            "date": "2024-01-13T09:15:00Z",
            "thread_id": "client_feedback_789",
            "category": "Career",
            "recipient": "dev-team@company.com",
            "client": "Acme Corp"
        },
        {
            "subject": "Office closure tomorrow",
            "body": "Due to the severe weather forecast, the office will be closed tomorrow. All employees should work remotely if possible. Stay safe and check your email for updates.",
            "sender": "hr@company.com",
            "date": "2024-01-12T16:00:00Z",
            "thread_id": "weather_closure_001",
            "category": "Personal",
            "recipient": "all@company.com",
            "urgency": "urgent"
        },
        {
            "subject": "Training materials for new hires",
            "body": "I've compiled the training materials for the upcoming onboarding session. The folder includes the employee handbook, IT setup guide, and team introductions.",
            "sender": "lisa.chen@company.com",
            "date": "2024-01-11T11:20:00Z",
            "thread_id": "onboarding_2024_002",
            "category": "Career",
            "recipient": "training@company.com",
            "attachment_count": 3
        },
        {
            "subject": "Re: API integration issues",
            "body": "I've identified the problem with the API integration. It seems the authentication token was expiring earlier than expected. I've updated the token refresh logic and it should be stable now.",
            "sender": "dev.lead@company.com",
            "date": "2024-01-10T15:30:00Z",
            "thread_id": "api_integration_003",
            "category": "Career",
            "recipient": "engineering@company.com",
            "error_type": "auth_token_expiry"
        },
        {
            "subject": "Team lunch reservation",
            "body": "I've made a reservation for 12 people at Italian Gardens for Friday at noon. Please let me know if you have any dietary restrictions I should mention to the restaurant.",
            "sender": "admin@company.com",
            "date": "2024-01-09T13:00:00Z",
            "thread_id": "team_lunch_004",
            "category": "Personal",
            "recipient": "team@company.com",
            "restaurant": "Italian Gardens"
        },
        {
            "subject": "Performance review schedule",
            "body": "Performance reviews are scheduled for next month. Please complete your self-assessments by the 15th and schedule review meetings with your direct reports.",
            "sender": "manager@company.com",
            "date": "2024-01-08T10:00:00Z",
            "thread_id": "perf_review_2024_005",
            "category": "Career",
            "recipient": "dept@company.com",
            "deadline": "2024-02-15"
        },
        {
            "subject": "Software license renewal",
            "body": "Our software licenses for the design tools are up for renewal next month. I need approval for the budget which is $15,000 for the annual subscription.",
            "sender": "it.procurement@company.com",
            "date": "2024-01-07T14:15:00Z",
            "thread_id": "license_renewal_006",
            "category": "Promotional",
            "recipient": "finance@company.com",
            "amount": 15000
        },
        {
            "subject": "New security policy implementation",
            "body": "Starting next week, we'll be implementing the new security policy. This includes mandatory password changes every 90 days and two-factor authentication for all systems.",
            "sender": "security@company.com",
            "date": "2024-01-06T09:45:00Z",
            "thread_id": "security_policy_007",
            "category": "Career",
            "recipient": "all@company.com",
            "policy_type": "password_and_2fa"
        }
    ]

    print("Step 1: Ingesting 10 real emails with metadata")
    print("=" * 60)

    # Show metadata for first 3 emails before ingestion
    print("\nMetadata for email 1 (before ingestion):")
    print(json.dumps({
        "source": "real_user",
        "sender": real_emails[0]["sender"],
        "date": real_emails[0]["date"],
        "thread_id": real_emails[0]["thread_id"],
        "category": real_emails[0]["category"],
        "recipient": real_emails[0]["recipient"],
        "cc": real_emails[0]["cc"]
    }, indent=2))

    print("\nMetadata for email 2 (before ingestion):")
    print(json.dumps({
        "source": "real_user",
        "sender": real_emails[1]["sender"],
        "date": real_emails[1]["date"],
        "thread_id": real_emails[1]["thread_id"],
        "category": real_emails[1]["category"],
        "recipient": real_emails[1]["recipient"],
        "priority": real_emails[1]["priority"]
    }, indent=2))

    print("\nMetadata for email 3 (before ingestion):")
    print(json.dumps({
        "source": "real_user",
        "sender": real_emails[2]["sender"],
        "date": real_emails[2]["date"],
        "thread_id": real_emails[2]["thread_id"],
        "category": real_emails[2]["category"],
        "recipient": real_emails[2]["recipient"],
        "client": real_emails[2]["client"]
    }, indent=2))

    # Delete existing real_user emails to re-ingest with categories
    from rag.retrieve import get_or_create_collection, collection_count
    collection = get_or_create_collection()
    try:
        collection.delete(where={"source": "real_user"})
        print(f"\nDeleted existing real_user emails")
        print(f"Collection count after deletion: {collection_count()}")
    except Exception as e:
        print(f"Error deleting real_user emails: {e}")

    # Ingest the emails with categories
    print(f"\nCollection count before ingestion: {collection_count()}")
    ingest_real_emails(real_emails)
    print(f"Collection count after ingestion: {collection_count()}")

    # Verify category metadata is present after ingestion
    print("\n" + "=" * 60)
    print("Verifying category metadata after ingestion")
    print("=" * 60)

    query = "project deadline extension"
    results = retrieve_similar(query, k=3, source_filter="real_user")

    print(f"\nQuery: '{query}' (source_filter='real_user')")
    for i, result in enumerate(results, 1):
        metadata = result["metadata"]
        category = metadata.get("category", "NOT FOUND")
        print(f"Result {i}: category={category}, full_metadata={json.dumps(metadata, indent=2)}")

    return real_emails


def test_metadata_roundtrip():
    """Test that metadata survives the round-trip through ChromaDB."""
    print("\n" + "=" * 60)
    print("Step 2: Verifying metadata round-trip through ChromaDB")
    print("=" * 60)

    # Query for something related to the ingested emails
    query = "project deadline extension"
    results = retrieve_similar(query, k=5)

    print(f"\nQuery: '{query}'")
    print(f"Found {len(results)} results")

    # Check if any of our real emails are in the results
    real_email_found = False
    for i, result in enumerate(results):
        metadata = result["metadata"]
        source = metadata.get("source", "unknown")

        print(f"\nResult {i+1}:")
        print(f"  Source: {source}")
        print(f"  Distance: {result['distance']:.6f}")
        print(f"  Document preview: {result['document'][:80]}...")

        if source == "real_user":
            real_email_found = True
            print(f"  Full metadata: {json.dumps(metadata, indent=4)}")

            # Verify key fields are present
            required_fields = ["source", "sender", "date", "thread_id"]
            missing_fields = [f for f in required_fields if f not in metadata]
            if missing_fields:
                print(f"  WARNING: Missing fields: {missing_fields}")
            else:
                print(f"  All required fields present: {required_fields}")

    if not real_email_found:
        print("\nNo real_user emails found in top results. Trying different query...")
        query2 = "budget review meeting"
        results2 = retrieve_similar(query2, k=5)
        print(f"\nQuery: '{query2}'")
        for i, result in enumerate(results2):
            metadata = result["metadata"]
            if metadata.get("source") == "real_user":
                print(f"\nFound real_user email in result {i+1}:")
                print(f"  Full metadata: {json.dumps(metadata, indent=4)}")

                # Verify key fields
                required_fields = ["source", "sender", "date", "thread_id"]
                missing_fields = [f for f in required_fields if f not in metadata]
                if missing_fields:
                    print(f"  WARNING: Missing fields: {missing_fields}")
                else:
                    print(f"  All required fields present: {required_fields}")
                break


def test_source_filter():
    """Test that source_filter correctly excludes non-matching sources."""
    print("\n" + "=" * 60)
    print("Step 3: Testing source_filter functionality")
    print("=" * 60)

    query = "project deadline extension"

    # Query WITHOUT filter (mixed results)
    print(f"\nQuery WITHOUT source_filter: '{query}'")
    results_unfiltered = retrieve_similar(query, k=5)
    print(f"Total results: {len(results_unfiltered)}")

    sources_unfiltered = [r["metadata"].get("source", "unknown") for r in results_unfiltered]
    print(f"Sources found: {sources_unfiltered}")

    for i, result in enumerate(results_unfiltered):
        print(f"  Result {i+1}: source={result['metadata'].get('source', 'unknown')}, distance={result['distance']:.6f}")

    # Query WITH filter (real_user only)
    print(f"\nQuery WITH source_filter='real_user': '{query}'")
    results_filtered = retrieve_similar(query, k=5, source_filter="real_user")
    print(f"Total results: {len(results_filtered)}")

    sources_filtered = [r["metadata"].get("source", "unknown") for r in results_filtered]
    print(f"Sources found: {sources_filtered}")

    for i, result in enumerate(results_filtered):
        print(f"  Result {i+1}: source={result['metadata'].get('source', 'unknown')}, distance={result['distance']:.6f}")

    # Verify filter worked
    non_real_user = [r for r in results_filtered if r["metadata"].get("source") != "real_user"]
    if non_real_user:
        print(f"\nWARNING: Found {len(non_real_user)} non-real_user results despite filter!")
    else:
        print(f"\nSUCCESS: All filtered results are from real_user source")

    # Show before/after comparison
    print(f"\nBefore filter: {sources_unfiltered}")
    print(f"After filter:  {sources_filtered}")
    print(f"Excluded sources: {set(sources_unfiltered) - set(sources_filtered)}")


def test_real_distances():
    """Test real distance values with source_filter restricted to real_user data."""
    print("\n" + "=" * 60)
    print("Step 4: Getting real distance numbers for threshold analysis")
    print("=" * 60)

    # 8 different real queries related to our ingested emails
    queries = [
        "Can we extend the project deadline?",
        "When is the budget review meeting?",
        "The client has feedback on the prototype",
        "Is the office closed due to weather?",
        "I need training materials for new hires",
        "There are issues with the API integration",
        "Team lunch reservation for Friday",
        "Performance review schedule this month"
    ]

    DISTANCE_THRESHOLD = 0.35  # From context.py

    print(f"\nCurrent DISTANCE_THRESHOLD from context.py: {DISTANCE_THRESHOLD}")
    print(f"Testing {len(queries)} queries with source_filter='real_user'\n")

    relevant_above_threshold = 0
    relevant_below_threshold = 0
    total_relevant = 0

    for i, query in enumerate(queries, 1):
        print(f"Query {i}: '{query}'")
        results = retrieve_similar(query, k=3, source_filter="real_user")

        if not results:
            print("  No results found")
            continue

        print(f"  Top-3 distances:")
        for j, result in enumerate(results, 1):
            distance = result["distance"]
            relevance = "RELEVANT" if j == 1 else "less relevant"
            print(f"    {j}. {distance:.6f} ({relevance})")

            # Count obviously relevant (top-1) vs threshold
            if j == 1:
                total_relevant += 1
                if distance < DISTANCE_THRESHOLD:
                    relevant_below_threshold += 1
                else:
                    relevant_above_threshold += 1

    print(f"\n" + "=" * 60)
    print("THRESHOLD ANALYSIS RESULTS")
    print("=" * 60)
    print(f"Total obviously relevant queries (top-1): {total_relevant}")
    print(f"Relevant queries below threshold ({DISTANCE_THRESHOLD}): {relevant_below_threshold}")
    print(f"Relevant queries above threshold ({DISTANCE_THRESHOLD}): {relevant_above_threshold}")
    print(f"Percentage of relevant queries below threshold: {100 * relevant_below_threshold / total_relevant if total_relevant > 0 else 0:.1f}%")
    print(f"Percentage of relevant queries above threshold: {100 * relevant_above_threshold / total_relevant if total_relevant > 0 else 0:.1f}%")


def test_enron_contamination():
    """Test Enron contamination when source_filter is OFF."""
    print("\n" + "=" * 60)
    print("Step 5: Reporting Enron contamination (source_filter OFF)")
    print("=" * 60)

    # Same 8 queries from Step 4
    queries = [
        "Can we extend the project deadline?",
        "When is the budget review meeting?",
        "The client has feedback on the prototype",
        "Is the office closed due to weather?",
        "I need training materials for new hires",
        "There are issues with the API integration",
        "Team lunch reservation for Friday",
        "Performance review schedule this month"
    ]

    print(f"Testing {len(queries)} queries WITHOUT source_filter (mixed collection)\n")

    enron_outranks_real = 0
    total_queries = len(queries)
    enron_in_top3_count = 0

    for i, query in enumerate(queries, 1):
        print(f"Query {i}: '{query}'")
        results = retrieve_similar(query, k=3)  # No source_filter

        if not results:
            print("  No results found")
            continue

        print(f"  Top-3 results (source, distance):")
        enron_in_top3 = 0
        real_user_in_top3 = 0
        first_real_user_position = None
        first_enron_position = None

        for j, result in enumerate(results, 1):
            source = result["metadata"].get("source", "unknown")
            distance = result["distance"]
            print(f"    {j}. {source}: {distance:.6f}")

            if source == "enron":
                enron_in_top3 += 1
                if first_enron_position is None:
                    first_enron_position = j
            elif source == "real_user":
                real_user_in_top3 += 1
                if first_real_user_position is None:
                    first_real_user_position = j

        # Check if Enron outranks real_user (Enron appears before any real_user)
        if first_enron_position is not None and first_real_user_position is not None:
            if first_enron_position < first_real_user_position:
                enron_outranks_real += 1
                print(f"  → Enron outranks real_user (Enron at position {first_enron_position}, real_user at {first_real_user_position})")
        elif first_enron_position is not None and first_real_user_position is None:
            enron_outranks_real += 1
            print(f"  → Enron outranks real_user (Enron present, no real_user in top-3)")

        if enron_in_top3 > 0:
            enron_in_top3_count += 1
            print(f"  → Enron contamination: {enron_in_top3} Enron results in top-3")

    print(f"\n" + "=" * 60)
    print("ENRON CONTAMINATION REPORT")
    print("=" * 60)
    print(f"Total queries tested: {total_queries}")
    print(f"Queries with Enron in top-3: {enron_in_top3_count} ({100 * enron_in_top3_count / total_queries:.1f}%)")
    print(f"Queries where Enron outranks real_user (top-1): {enron_outranks_real} ({100 * enron_outranks_real / total_queries:.1f}%)")
    print(f"Queries with only real_user in top-3: {total_queries - enron_in_top3_count} ({100 * (total_queries - enron_in_top3_count) / total_queries:.1f}%)")


def test_category_filtering():
    """Test category filtering with fallback behavior."""
    print("\n" + "=" * 60)
    print("Step 3: Testing category filtering capability")
    print("=" * 60)

    # Test 1: Query with correct category
    print("\nTest 1: Query with correct category (Career)")
    query = "project deadline extension"
    results_career = retrieve_similar(query, k=3, source_filter="real_user", category="Career")
    print(f"Query: '{query}' with category='Career'")
    print(f"Results: {len(results_career)}")
    for i, result in enumerate(results_career, 1):
        category = result["metadata"].get("category", "unknown")
        print(f"  {i}. category={category}, distance={result['distance']:.6f}")

    # Test 2: Query with category that has zero matches
    print("\nTest 2: Query with category that has zero matches (Academic)")
    results_academic = retrieve_similar(query, k=3, source_filter="real_user", category="Academic")
    print(f"Query: '{query}' with category='Academic'")
    print(f"Results: {len(results_academic)}")
    for i, result in enumerate(results_academic, 1):
        category = result["metadata"].get("category", "unknown")
        print(f"  {i}. category={category}, distance={result['distance']:.6f}")

    # Test 3: Query with category=None (should behave like source_filter-only)
    print("\nTest 3: Query with category=None (source_filter-only behavior)")
    results_none = retrieve_similar(query, k=3, source_filter="real_user", category=None)
    print(f"Query: '{query}' with category=None")
    print(f"Results: {len(results_none)}")
    for i, result in enumerate(results_none, 1):
        category = result["metadata"].get("category", "unknown")
        print(f"  {i}. category={category}, distance={result['distance']:.6f}")

    # Test 4: Query with Personal category
    print("\nTest 4: Query with Personal category")
    query_personal = "team lunch reservation"
    results_personal = retrieve_similar(query_personal, k=3, source_filter="real_user", category="Personal")
    print(f"Query: '{query_personal}' with category='Personal'")
    print(f"Results: {len(results_personal)}")
    for i, result in enumerate(results_personal, 1):
        category = result["metadata"].get("category", "unknown")
        print(f"  {i}. category={category}, distance={result['distance']:.6f}")

    print("\n" + "=" * 60)
    print("CATEGORY FILTERING TEST RESULTS")
    print("=" * 60)
    print(f"Test 1 (Career category): {len(results_career)} results - {'SUCCESS' if len(results_career) > 0 else 'FAILED'}")
    print(f"Test 2 (Academic category - zero matches): {len(results_academic)} results - {'FALLBACK_WORKED' if len(results_academic) > 0 else 'FALLBACK_FAILED'}")
    print(f"Test 3 (category=None): {len(results_none)} results - {'SUCCESS' if len(results_none) > 0 else 'FAILED'}")
    print(f"Test 4 (Personal category): {len(results_personal)} results - {'SUCCESS' if len(results_personal) > 0 else 'FAILED'}")


if __name__ == "__main__":
    # Always re-ingest to add category metadata
    real_emails = test_real_email_ingestion()

    test_metadata_roundtrip()
    test_source_filter()
    test_real_distances()
    test_enron_contamination()
    test_category_filtering()

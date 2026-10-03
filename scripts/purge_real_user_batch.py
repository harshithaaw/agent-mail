"""
Purge the previous mislabeled real_user batch (inbox emails) from Chroma
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.retrieve import get_or_create_collection

def main():
    collection = get_or_create_collection()
    
    # Check collection count before purge
    before_count = collection.count()
    print(f"Collection count before purge: {before_count}")
    
    # Get all documents with source="real_user"
    try:
        results = collection.get(
            where={"source": "real_user"},
            include=["metadatas", "documents"]
        )
        real_user_ids = results.get("ids", [])
        real_user_metadatas = results.get("metadatas", [])
    except Exception as e:
        print(f"Error with full get, trying simpler approach: {e}")
        # Fallback: just get IDs
        results = collection.get(where={"source": "real_user"})
        real_user_ids = results.get("ids", [])
        real_user_metadatas = []
    
    print(f"Found {len(real_user_ids)} documents with source='real_user'")
    
    if real_user_ids:
        # Show sample of what we're about to delete
        print("\nSample of documents to be deleted:")
        for i in range(min(3, len(real_user_ids))):
            metadata = real_user_metadatas[i] if i < len(real_user_metadatas) else {}
            sender = metadata.get("sender", "N/A")
            print(f"  {i+1}. ID: {real_user_ids[i]}, Sender: {sender}")
        
        # Delete all real_user documents
        collection.delete(ids=real_user_ids)
        print(f"\nDeleted {len(real_user_ids)} documents with source='real_user'")
    
    # Check collection count after purge
    after_count = collection.count()
    print(f"Collection count after purge: {after_count}")
    print(f"Removed {before_count - after_count} documents total")
    
    # Verify no real_user documents remain
    verification = collection.get(
        where={"source": "real_user"}
    )
    remaining_ids = verification.get("ids", [])
    
    if remaining_ids:
        print(f"WARNING: {len(remaining_ids)} real_user documents still remain!")
    else:
        print("Verification: No real_user documents remain in collection")

if __name__ == "__main__":
    main()
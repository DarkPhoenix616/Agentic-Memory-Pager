import os
import uuid
from typing import List
from mcp.server.fastmcp import FastMCP
import chromadb

# ---------------------------------------------------------------------------
# Path setup — anchor storage to <project_root>/storage/chroma_db regardless
# of which directory this script is run from.
#
# This file lives at: <project_root>/server/amp_memory_server.py
# So project_root = parent of this file's directory.
# ---------------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
CHROMA_DB_PATH = os.path.join(PROJECT_ROOT, "storage", "chroma_db")

os.makedirs(CHROMA_DB_PATH, exist_ok=True)

mcp = FastMCP("amp-memory", host="localhost", port=8001)

chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_or_create_collection("amp_memory")
print(collection._embedding_function)

@mcp.tool()
def store_memory(content: str, tags: List[str] = []) -> str:
    """
    Store a piece of content into long-term memory with optional tags.

    Args:
        content: The text content to remember
        tags: Optional list of tags/labels for categorization

    Returns:
        The generated memory ID, or a rejection message if input is invalid
    """
    # ---------------------------------------------------------------
    # GUARDRAILS HOOK — Pydantic validation slots in here.
    # Expected pattern once wired in:
    #
    #   try:
    #       record = MemoryRecord(content=content, tags=tags)
    #   except ValidationError as e:
    #       return f"Rejected: {e}"
    #
    # Minimal manual check below until that's ready, so the tool
    # never silently accepts broken input in the meantime.
    # ---------------------------------------------------------------
    content = content.strip()
    if not content:
        return "Rejected: content cannot be empty."

    memory_id = f"mem_{uuid.uuid4().hex[:8]}"

    collection.add(
        documents=[content],
        ids=[memory_id],
        metadatas=[{"tags": ",".join(tags) if tags else ""}]
    )

    return f"Stored as {memory_id}"


@mcp.tool()
def search_vault(query: str, top_k: int = 5) -> List[dict]:
    """
    Search long-term memory for content semantically related to the query.

    Args:
        query: The search query
        top_k: Maximum number of results to return (default: 5)

    Returns:
        List of matching memory records, each with id, content, distance, tags
    """
    query = query.strip()
    if not query:
        return []

    # Guard against asking for more results than exist in the collection —
    # ChromaDB errors if n_results exceeds the total stored document count.
    count = collection.count()
    if count == 0:
        return []
    effective_k = min(top_k, count)

    results = collection.query(
        query_texts=[query],
        n_results=effective_k
    )

    matches = []
    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    for i in range(len(ids)):
        matches.append({
            "id": ids[i],
            "content": documents[i],
            "distance": distances[i],
            "tags": metadatas[i].get("tags", "") if metadatas[i] else ""
        })

    return matches


@mcp.tool()
def delete_memory(memory_id: str) -> str:
    """
    Delete a specific memory record by ID.

    Args:
        memory_id: The ID of the memory to delete

    Returns:
        Confirmation message
    """
    try:
        collection.delete(ids=[memory_id])
        return f"Deleted {memory_id}"
    except Exception as e:
        return f"Error deleting {memory_id}: {str(e)}"


@mcp.tool()
def list_all_memories() -> List[dict]:
    """
    List every memory currently stored (useful for debugging/inspection).

    Returns:
        List of all stored memory records with id, content, tags
    """
    all_data = collection.get()
    memories = []
    ids = all_data.get("ids", [])
    documents = all_data.get("documents", [])
    metadatas = all_data.get("metadatas", [])

    for i in range(len(ids)):
        memories.append({
            "id": ids[i],
            "content": documents[i],
            "tags": metadatas[i].get("tags", "") if metadatas[i] else ""
        })

    return memories


if __name__ == "__main__":
    print(f"ChromaDB persisting to: {CHROMA_DB_PATH}")
    mcp.run(transport="streamable-http")
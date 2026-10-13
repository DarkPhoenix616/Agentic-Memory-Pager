import os
import uuid
import tiktoken
from datetime import datetime, timezone
from typing import List
from mcp.server.fastmcp import FastMCP
import chromadb

encoder=tiktoken.get_encoding("cl100k_base") # gpt-4 one to count the number of tokens in code or dialogue. Isko hum token limit ke liye use karenge.
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


@mcp.tool()
def store_memory(content: str, tags: List[str] = [], session_id: str = "default", source_type: str = "dialogue") -> str:
    """
    Store a piece of content into long-term memory with optional tags.

    Args:
        content: The text content to remember
        tags: Optional list of tags/labels for categorization
        session_id: Which session/workspace this memory belongs to
        source_type: The type of the source content (default: "dialogue")

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
    source_type: str="dialogue"  # default source type

    content = content.strip()
    if not content:
        return "Rejected: content cannot be empty."

    if source_type not in ["dialogue", "code"]:
        return "Rejected: Source type must be 'dialogue' or 'code'."

    memory_id = f"mem_{uuid.uuid4().hex[:8]}"
    token_count=len(encoder.encode(content))
    timestamp = datetime.now(timezone.utc).isoformat()

    collection.add(
        documents=[content],
        ids=[memory_id],
        metadatas=[{"tags": ",".join(tags) if tags else ""
                , "session_id": session_id
                , "source_type": source_type,
                "token_count": token_count,
                "timestamp": timestamp  
            }]
    )
    
    return f"Stored as {memory_id}"


@mcp.tool()
def search_vault(query: str, top_k: int = 5, session_id: str = "") -> List[dict]:
    """
    Search long-term memory for content semantically related to the query.

    Args:
        query: The search query
        top_k: Maximum number of results to return (default: 5)
        session_id: Filter results to a specific session/workspace
        source_type: Filter results by source type (default: "dialogue")
    Returns:
        List of matching memory records, each with id, content, distance, tags
        time stamp, and source_type. Empty list if no matches found.
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

    where = {"session_id": session_id} if session_id else None
    results = collection.query(
        query_texts=[query],
        n_results=effective_k,
        where=where
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
            "tags": metadatas[i].get("tags", "") if metadatas[i] else "",
            "session_id": metadatas[i].get("session_id", "") if metadatas[i] else "",
            "source_type": metadatas[i].get("source_type", "") if metadatas[i] else "",
            "token_count": metadatas[i].get("token_count", 0) if metadatas[i] else 0,
            "timestamp": metadatas[i].get("timestamp", "") if metadatas[i] else ""
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
        session_id, source_type, token_count, and timestamp.
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
            "tags": metadatas[i].get("tags", "") if metadatas[i] else "",
            "session_id": metadatas[i].get("session_id", "") if metadatas[i] else "",
            "source_type": metadatas[i].get("source_type", "") if metadatas[i] else "",
            "token_count": metadatas[i].get("token_count", 0) if metadatas[i] else 0,
            "timestamp": metadatas[i].get("timestamp", "") if metadatas[i] else ""
        })

    return memories


if __name__ == "__main__":
    print(f"ChromaDB persisting to: {CHROMA_DB_PATH}")
    mcp.run(transport="streamable-http")
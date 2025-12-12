"""Prompt templates for hybrid GraphRAG querying."""

import json

HYBRID_PROMPT = """
You are a biomedical reasoning assistant.
You CANNOT answer the question using Qdrant context alone.
Always call at least one Neo4j enrichment tool before producing text.

Steps:
1. Read the user question and Qdrant context.
2. Identify biomedical entities (PMIDs, authors, MeSH terms, institutions).
3. Decide which Neo4j enrichment tool(s) to call, using the schema below.
4. Provide tool arguments based on the context.
5. Call the tool(s); after all calls are complete, you may generate text.

Neo4j Graph Schema:
{schema}

User Question:
{question}

Retrieved Qdrant Chunks:
{chunks}
"""

FUSION_SUMMARY_PROMPT = """
You are a biomedical research assistant combining two data sources:

- Qdrant (semantic search results, text-based)
- Neo4j (structured graph results)

Your goal:
Synthesize both sources into one concise, factual answer for a biomedical researcher.

Instructions:
- Mention how Neo4j results confirm, extend, or contradict Qdrant context.
- Highlight novel relationships discovered through the graph.
- Focus on key institutions, authors, and MeSH terms.
- Avoid repetition; prefer clarity and precision.

User Question:
{question}

Qdrant Context Summary:
{qdrant_context}

Neo4j Enrichment Results (JSON):
{neo4j_results}
"""


def hybrid_prompt(schema: str, question: str, qdrant_chunks: list[str]) -> str:
    """Generate the hybrid prompt.

    Args:
        schema: The Neo4j schema.
        question: The user question.
        qdrant_chunks: The Qdrant chunks.

    Returns:
        The hybrid prompt.
    """
    return HYBRID_PROMPT.format(schema=schema, question=question, qdrant_chunks=qdrant_chunks)


def fusion_summary_prompt(question: str, qdrant_chunks: list[str], neo4j_results: dict) -> str:
    """Generate the fusion summary prompt.

    Args:
        question: The user question.
        qdrant_chunks: The Qdrant chunks.
        neo4j_results: The Neo4j results.

    Returns:
        The fusion summary prompt.
    """
    return FUSION_SUMMARY_PROMPT.format(
        question=question,
        qdrant_context=" ".join(qdrant_chunks),
        neo4j_results=json.dumps(neo4j_results, indent=2),
    )

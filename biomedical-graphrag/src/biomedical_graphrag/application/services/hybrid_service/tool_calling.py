import json
from typing import Any

import google.generativeai as genai

from biomedical_graphrag.application.services.hybrid_service.neo4j_query import Neo4jGraphQuery
from biomedical_graphrag.application.services.hybrid_service.prompts.hybrid_prompts import (
    HYBRID_PROMPT,
    fusion_summary_prompt,
)
from biomedical_graphrag.application.services.hybrid_service.tools.enrichment_tools import (
    ENRICHMENT_TOOLS,
)
from biomedical_graphrag.config import settings
from biomedical_graphrag.utils.logger_util import setup_logging

logger = setup_logging()

genai.configure(api_key=settings.gemini.api_key.get_secret_value())
gemini_model = genai.GenerativeModel(settings.gemini.model)


def get_neo4j_schema() -> str:
    """Retrieve the Neo4j schema dynamically."""
    neo4j = Neo4jGraphQuery()
    schema = neo4j.get_schema()
    logger.info(f"Retrieved Neo4j schema length: {len(schema)}")
    return schema


# --------------------------------------------------------------------
# Phase 1 — Tool selection + execution
# --------------------------------------------------------------------
def run_graph_enrichment(question: str, qdrant_chunks: list[str]) -> dict[str, Any]:
    """Run graph enrichment.

    Args:
        question: The user question.
        qdrant_chunks: The Qdrant chunks.

    Returns:
        The Neo4j results.
    """
    schema = get_neo4j_schema()
    logger.info(f"Neo4j schema: {schema}")
    neo4j = Neo4jGraphQuery()

    prompt = HYBRID_PROMPT.format(schema=schema, question=question, chunks="---".join(qdrant_chunks))

    # Convert OpenAI-style tools to Gemini format
    gemini_tools = []
    for tool in ENRICHMENT_TOOLS:
        gemini_tool = genai.protos.Tool(
            function_declarations=[
                genai.protos.FunctionDeclaration(
                    name=tool["name"],
                    description=tool["description"],
                    parameters=genai.protos.Schema(
                        type=genai.protos.Type.OBJECT,
                        properties={
                            prop_name: genai.protos.Schema(
                                type=genai.protos.Type.STRING if prop_def["type"] == "string"
                                else genai.protos.Type.INTEGER if prop_def["type"] == "integer"
                                else genai.protos.Type.BOOLEAN if prop_def["type"] == "boolean"
                                else genai.protos.Type.ARRAY if prop_def["type"] == "array"
                                else genai.protos.Type.OBJECT,
                                description=prop_def.get("description", ""),
                                items=genai.protos.Schema(type=genai.protos.Type.STRING) if prop_def.get("type") == "array" else None,
                            )
                            for prop_name, prop_def in tool["parameters"]["properties"].items()
                        },
                        required=tool["parameters"].get("required", []),
                    ),
                )
            ]
        )
        gemini_tools.append(gemini_tool)

    # Create model with function calling
    model_with_tools = genai.GenerativeModel(
        model_name=settings.gemini.model,
        tools=gemini_tools,
    )

    response = model_with_tools.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=settings.gemini.temperature,
            max_output_tokens=settings.gemini.max_tokens,
        )
    )

    results = {}
    
    # Check if Gemini made any function calls
    if response.candidates and response.candidates[0].content.parts:
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'function_call') and part.function_call:
                function_call = part.function_call
                name = function_call.name
                # Convert args from protobuf to dict
                args = {key: val for key, val in function_call.args.items()}
                
                logger.info(f"Executing Neo4j function: {name} with args: {args}")
                
                func = getattr(neo4j, name, None)
                if func:
                    try:
                        results[name] = func(**args)
                        logger.info(f"Function {name} executed successfully")
                    except Exception as e:
                        results[name] = f"Error: {e}"
                        logger.error(f"Error executing function {name}: {e}")
                else:
                    logger.warning(f"Function {name} not found in Neo4j query class")
    else:
        logger.info(f"No function calls made. Gemini response: {response.text if hasattr(response, 'text') else 'No text response'}")
    
    logger.info(f"Neo4j tool results: {results}")
    return results


# --------------------------------------------------------------------
# Phase 2 — Fusion summarization
# --------------------------------------------------------------------
def summarize_fused_results(question: str, qdrant_chunks: list[str], neo4j_results: dict) -> str:
    """Fuse semantic and graph evidence into one final biomedical summary.

    Args:
        question: The user question.
        qdrant_chunks: The Qdrant chunks.
        neo4j_results: The Neo4j results.

    Returns:
        The summarized results.
    """
    prompt = fusion_summary_prompt(question, qdrant_chunks, neo4j_results)
    resp = gemini_model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=settings.gemini.temperature,
            max_output_tokens=settings.gemini.max_tokens,
        )
    )
    return resp.text.strip()


# --------------------------------------------------------------------
# Unified helper
# --------------------------------------------------------------------
def run_graph_enrichment_and_summarize(question: str, qdrant_chunks: list[str]) -> str:
    """Run graph enrichment and summarize the results.

    Args:
        question: The user question.
        qdrant_chunks: The Qdrant chunks.

    Returns:
        The summarized results.
    """
    neo4j_results = run_graph_enrichment(question, qdrant_chunks)
    return summarize_fused_results(question, qdrant_chunks, neo4j_results)

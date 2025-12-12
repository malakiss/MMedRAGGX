"""
Enrichment tools definition for Neo4j graph queries.

This module defines the tools that can be called by the LLM to enrich
biomedical queries with graph-based information from Neo4j.
"""

# ----------------------------
# Enrichment tools definition
# ----------------------------
ENRICHMENT_TOOLS = [
    {
        "type": "function",
        "name": "get_collaborators_with_topics",
        "description": "Get collaborators for an author filtered by MeSH topics.",
        "parameters": {
            "type": "object",
            "properties": {
                "author_name": {"type": "string"},
                "topics": {"type": "array", "items": {"type": "string"}},
                "require_all": {"type": "boolean"},
            },
            "required": ["author_name", "topics"],
        },
    },
    {
        "type": "function",
        "name": "get_collaborating_institutions",
        "description": "Get institutions that collaborate frequently.",
        "parameters": {
            "type": "object",
            "properties": {"min_collaborations": {"type": "integer"}},
            "required": ["min_collaborations"],
        },
    },
    {
        "type": "function",
        "name": "get_related_papers_by_mesh",
        "description": "Get papers related by MeSH terms to a given PMID.",
        "parameters": {
            "type": "object",
            "properties": {"pmid": {"type": "string"}},
            "required": ["pmid"],
        },
    },
    {
        "type": "function",
        "name": "get_genes_in_same_papers",
        "description": (
            "Find genes that co-occur in the same papers as a specified target gene, "
            "optionally filtered by a MeSH topic (e.g., 'cancer', 'HIV'). "
            "This reveals potential biological associations based on co-mention frequency."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "target_gene": {
                    "type": "string",
                    "description": (
                        "Name or alias of the target gene (e.g., 'TP53', 'CCR5', 'gag'). "
                        "Search is case-insensitive and supports partial matches."
                    ),
                },
                "mesh_filter": {
                    "type": "string",
                    "description": (
                        "Optional MeSH term substring to filter relevant papers "
                        "(e.g., 'cancer', 'immunity', 'HIV')."
                    ),
                },
            },
            "required": ["target_gene"],
        },
    },
    {
        "type": "function",
        "name": "get_similar_images_by_finding",
        "description": (
            "Find radiology images that share similar findings. "
            "Useful for finding cases with similar radiological presentations."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "finding": {
                    "type": "string",
                    "description": (
                        "Radiological finding name (e.g., 'pneumothorax', 'cardiomegaly', 'opacity'). "
                        "Search is case-insensitive."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of images to return (default: 10).",
                },
            },
            "required": ["finding"],
        },
    },
    {
        "type": "function",
        "name": "get_image_report",
        "description": "Get the full radiology report for a specific image.",
        "parameters": {
            "type": "object",
            "properties": {
                "image_id": {
                    "type": "string",
                    "description": "Unique identifier for the radiology image.",
                }
            },
            "required": ["image_id"],
        },
    },
    {
        "type": "function",
        "name": "get_images_by_modality",
        "description": "Get radiology images filtered by imaging modality (e.g., chest X-ray, CT, MRI).",
        "parameters": {
            "type": "object",
            "properties": {
                "modality": {
                    "type": "string",
                    "description": "Imaging modality (e.g., 'chest_xray', 'ct', 'mri').",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of images to return (default: 20).",
                },
            },
            "required": ["modality"],
        },
    },
]

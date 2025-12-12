import json

from biomedical_graphrag.config import settings


def load_pubmed_json() -> dict:
    """
    Load the pubmed_dataset.json file from the data directory.
    Returns:
        dict: Parsed JSON data.
    """
    json_path = settings.json_data.pubmed_json_path
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def load_gene_json() -> dict:
    """
    Load the gene_dataset.json file from the data directory.
    Returns:
        dict: Parsed JSON data.
    """
    json_path = settings.json_data.gene_json_path
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def load_iuxray_json() -> dict:
    """Load the iuxray_dataset.json file from the data directory."""
    json_path = settings.json_data.iuxray_json_path
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)
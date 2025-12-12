"""Script to create and populate Neo4j graph from biomedical papers dataset."""

import json
from pathlib import Path

from biomedical_graphrag.domain.dataset import GeneDataset, PaperDataset
from biomedical_graphrag.domain.radiology import (
    RadiologyDataset,
    RadiologyDatasetMetadata,
    RadiologyImage,
)
from biomedical_graphrag.infrastructure.neo4j_db.neo4j_client import AsyncNeo4jClient
from biomedical_graphrag.infrastructure.neo4j_db.neo4j_graph_schema import Neo4jGraphIngestion
from biomedical_graphrag.infrastructure.neo4j_db.neo4j_radiology_ingestion import (
    Neo4jRadiologyIngestion,
)
from biomedical_graphrag.utils.logger_util import setup_logging

logger = setup_logging()


def load_paper_dataset(file_path: str | Path) -> PaperDataset:
    """
    Load dataset from JSON file.

    Args:
        file_path: Path to the JSON dataset file

    Returns:
        Dataset object with papers and citation network
    """
    with open(file_path) as f:
        data = json.load(f)

    return PaperDataset(**data)


def load_gene_dataset(file_path: str | Path) -> GeneDataset:
    """
    Load gene dataset from JSON file.

    Args:
        file_path: Path to the JSON dataset file

    Returns:
        Dataset object with genes
    """
    with open(file_path) as f:
        data = json.load(f)
    return GeneDataset(**data)


def load_radiology_dataset(file_path: str | Path) -> RadiologyDataset:
    """
    Load radiology dataset from IU X-ray annotation.json file.

    Args:
        file_path: Path to the annotation.json file

    Returns:
        RadiologyDataset with images
    """
    logger.info(f"Loading radiology data from {file_path}")
    with open(file_path) as f:
        data = json.load(f)
    
    # Extract train split images (you can change to 'val' or 'test' if needed)
    train_data = data.get("train", [])
    
    images = []
    for item in train_data:
        # IU X-ray annotation format
        image = RadiologyImage(
            image_id=item.get("id", ""),
            image_path=item.get("image", ""),  # List of image paths
            image_root=str(Path(file_path).parent / "images"),  # Root to images folder
            report=item.get("report", ""),
            modality="chest_xray",
        )
        images.append(image)
    
    metadata = RadiologyDatasetMetadata(
        collection_date="2025-12-12",
        total_images=len(images),
        dataset_name="IU X-Ray",
        modality="chest_xray",
    )
    
    return RadiologyDataset(metadata=metadata, images=images)


async def create_graph(
    dataset_path: str | Path | None = None,
    radiology_dataset_path: str | Path | None = None,
    gene_dataset_path: str | Path | None = None,
) -> None:
    """
    Create a new graph in the Neo4j graph database from biomedical papers and radiology dataset.

    Args:
        dataset_path: Optional path to PubMed dataset JSON file.
                     Defaults to data/pubmed_dataset.json in project root.
        radiology_dataset_path: Optional path to IU X-ray annotation.json file.
                               Defaults to /home/m.ismail/MMed-RAG/iu_xray/iu_xray/annotation.json
        gene_dataset_path: Optional path to gene dataset JSON (deprecated, prefer radiology).

    Returns:
        None
    """
    # Use default paths if none provided
    if dataset_path is None:
        project_root = Path(__file__).parents[4]
        dataset_path = project_root / "data" / "pubmed_dataset.json"
    
    if radiology_dataset_path is None:
        radiology_dataset_path = Path("/home/m.ismail/MMed-RAG/iu_xray/iu_xray/annotation.json")
    
    if gene_dataset_path is None:
        project_root = Path(__file__).parents[4]
        gene_dataset_path = project_root / "data" / "gene_dataset.json"

    # Load dataset
    logger.info(f"Loading paper dataset from {dataset_path}...")
    dataset = load_paper_dataset(dataset_path)
    logger.info(f"Loaded {len(dataset.papers)} papers with {len(dataset.citation_network)} citations")

    # Load radiology dataset (optional)
    radiology_dataset: RadiologyDataset | None = None
    if Path(radiology_dataset_path).exists():
        logger.info(f"Loading radiology dataset from {radiology_dataset_path}...")
        try:
            radiology_dataset = load_radiology_dataset(radiology_dataset_path)
            logger.info(f"Loaded {len(radiology_dataset.images)} radiology images")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to load radiology dataset: {e}")
    else:
        logger.warning(f"No radiology dataset found at {radiology_dataset_path}; skipping radiology ingestion")
    
    # Load gene dataset (optional, deprecated)
    gene_dataset: GeneDataset | None = None
    if Path(gene_dataset_path).exists():
        logger.info(f"Loading gene dataset from {gene_dataset_path}...")
        try:
            gene_dataset = load_gene_dataset(gene_dataset_path)
            logger.info(f"Loaded {len(gene_dataset.genes)} genes")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to load gene dataset: {e}")
    else:
        logger.warning(f"No gene dataset found at {gene_dataset_path}; skipping gene ingestion")

    # Initialize Neo4j client
    client = await AsyncNeo4jClient.create()

    try:
        # Initialize ingestion handler
        ingestion = Neo4jGraphIngestion(client)

        # Ingest complete paper dataset
        await ingestion.ingest_paper_dataset(dataset)

        # Ingest radiology images if available
        if radiology_dataset is not None:
            radiology_ingestion = Neo4jRadiologyIngestion(client)
            await radiology_ingestion.ingest_radiology_dataset(radiology_dataset)
        
        # Ingest genes if available (deprecated)
        if gene_dataset is not None:
            await ingestion.ingest_genes(gene_dataset)

        logger.info("✅ Graph created!")
        logger.info(f"   - {len(dataset.papers)} papers")
        logger.info(f"   - {dataset.metadata.total_authors} authors")
        logger.info(f"   - {dataset.metadata.total_mesh_terms} MeSH terms")
        logger.info("   - Citation network relationships created")
        if radiology_dataset is not None:
            logger.info(f"   - {len(radiology_dataset.images)} radiology images and findings created")
        if gene_dataset is not None:
            logger.info(f"   - {len(gene_dataset.genes)} genes and gene-paper relationships created")

    finally:
        await client.close()


if __name__ == "__main__":
    import asyncio

    asyncio.run(create_graph())

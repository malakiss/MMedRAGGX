"""Ingest radiology dataset with embeddings into Neo4j.

This script loads the radiology dataset (with pre-computed embeddings)
and populates the Neo4j graph database.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from biomedical_graphrag.config import settings
from biomedical_graphrag.domain.radiology import (
    RadiologyDataset,
    RadiologyDatasetMetadata,
    RadiologyImage,
)
from biomedical_graphrag.infrastructure.neo4j_db.neo4j_client import AsyncNeo4jClient
from biomedical_graphrag.infrastructure.neo4j_db.neo4j_radiology_ingestion import (
    Neo4jRadiologyIngestion,
)
from biomedical_graphrag.utils.logger_util import setup_logging

logger = setup_logging()


async def load_dataset_with_embeddings(json_path: str) -> RadiologyDataset:
    """Load radiology dataset from JSON file with embeddings.
    
    Args:
        json_path: Path to JSON file with embeddings
        
    Returns:
        RadiologyDataset with images and embeddings
    """
    logger.info(f"Loading dataset from {json_path}")
    
    with open(json_path, "r") as f:
        data = json.load(f)
    
    images = []
    for item in data:
        image = RadiologyImage(
            image_id=item["image_id"],
            image_path=item["image_path"],
            image_root=item["image_root"],
            report=item["report"],
            modality=item.get("modality", "chest_xray"),
            findings=item.get("findings", []),
            image_embedding=item.get("image_embedding", []),
            report_embedding=item.get("report_embedding", []),
        )
        images.append(image)
    
    metadata = RadiologyDatasetMetadata(
        collection_date="2025-12-11",
        total_images=len(images),
        dataset_name="IU X-Ray",
        modality="chest_xray",
    )
    
    dataset = RadiologyDataset(metadata=metadata, images=images)
    logger.info(f"✅ Loaded {len(images)} images from dataset")
    
    return dataset


async def ingest_radiology_data(
    dataset_path: str,
    neo4j_uri: str,
    neo4j_username: str,
    neo4j_password: str,
    batch_size: int = 100
) -> None:
    """Ingest radiology dataset into Neo4j.
    
    Args:
        dataset_path: Path to JSON file with embeddings
        neo4j_uri: Neo4j database URI
        neo4j_username: Neo4j username
        neo4j_password: Neo4j password
        batch_size: Batch size for ingestion
    """
    # Load dataset
    dataset = await load_dataset_with_embeddings(dataset_path)
    
    # Initialize Neo4j client
    logger.info("Connecting to Neo4j...")
    client = AsyncNeo4jClient(
        uri=neo4j_uri,
        username=neo4j_username,
        password=neo4j_password,
        database="neo4j"
    )
    
    try:
        # Initialize ingestion service
        ingestion = Neo4jRadiologyIngestion(
            client=client,
            concurrency_limit=25,
            batch_size=batch_size
        )
        
        # Ingest dataset
        logger.info("Starting ingestion...")
        await ingestion.ingest_radiology_dataset(dataset)
        
        logger.info("✅ Ingestion complete!")
        
        # Print statistics
        logger.info("\n" + "="*60)
        logger.info("INGESTION STATISTICS")
        logger.info("="*60)
        logger.info(f"Total images ingested: {dataset.metadata.total_images}")
        
        with_image_emb = sum(1 for img in dataset.images if img.image_embedding)
        with_report_emb = sum(1 for img in dataset.images if img.report_embedding)
        with_findings = sum(1 for img in dataset.images if img.findings)
        
        logger.info(f"Images with visual embeddings: {with_image_emb}/{len(dataset.images)}")
        logger.info(f"Images with report embeddings: {with_report_emb}/{len(dataset.images)}")
        logger.info(f"Images with findings: {with_findings}/{len(dataset.images)}")
        logger.info("="*60)
        
    finally:
        await client.close()


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest radiology dataset into Neo4j")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/home/m.ismail/MMed-RAG/biomedical-graphrag/data/radiology_with_embeddings.json",
        help="Path to dataset JSON with embeddings"
    )
    parser.add_argument(
        "--neo4j_uri",
        type=str,
        default=None,
        help="Neo4j URI (default: from config)"
    )
    parser.add_argument(
        "--neo4j_username",
        type=str,
        default=None,
        help="Neo4j username (default: from config)"
    )
    parser.add_argument(
        "--neo4j_password",
        type=str,
        default=None,
        help="Neo4j password (default: from config)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Batch size for ingestion"
    )
    
    args = parser.parse_args()
    
    # Use settings from config if not provided
    neo4j_uri = args.neo4j_uri or settings.neo4j.uri
    neo4j_username = args.neo4j_username or settings.neo4j.username
    neo4j_password = args.neo4j_password or settings.neo4j.password.get_secret_value()
    
    # Validate dataset exists
    if not Path(args.dataset_path).exists():
        logger.error(f"Dataset not found: {args.dataset_path}")
        logger.error("Please run extract_embeddings.py first to create the dataset with embeddings")
        sys.exit(1)
    
    # Run ingestion
    await ingest_radiology_data(
        dataset_path=args.dataset_path,
        neo4j_uri=neo4j_uri,
        neo4j_username=neo4j_username,
        neo4j_password=neo4j_password,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    asyncio.run(main())

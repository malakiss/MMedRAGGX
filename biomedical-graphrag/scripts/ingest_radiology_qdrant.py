"""Ingest radiology images into Qdrant vector store.

This script loads the radiology dataset with embeddings and ingests it into Qdrant.
Images are stored as separate points with report embeddings as dense vectors.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from biomedical_graphrag.config import settings
from biomedical_graphrag.infrastructure.qdrant_db.qdrant_vectorstore import AsyncQdrantVectorStore
from biomedical_graphrag.utils.logger_util import setup_logging

logger = setup_logging()


async def ingest_radiology_to_qdrant(
    radiology_json_path: str,
    collection_name: str | None = None,
    batch_size: int = 50
) -> None:
    """Ingest radiology images into Qdrant.
    
    Args:
        radiology_json_path: Path to radiology JSON with embeddings
        collection_name: Qdrant collection name (default: from config)
        batch_size: Batch size for ingestion
    """
    # Load radiology data
    logger.info(f"Loading radiology data from {radiology_json_path}")
    with open(radiology_json_path, "r") as f:
        radiology_data = json.load(f)
    
    logger.info(f"Loaded {len(radiology_data) if isinstance(radiology_data, list) else len(radiology_data.get('images', []))} images")
    
    # Initialize Qdrant client
    vectorstore = AsyncQdrantVectorStore()
    if collection_name:
        vectorstore.collection_name = collection_name
    
    try:
        # Check if collection exists, create if not
        collections = await vectorstore.client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        if vectorstore.collection_name not in collection_names:
            logger.info(f"Collection {vectorstore.collection_name} does not exist, creating...")
            await vectorstore.create_collection()
        else:
            logger.info(f"Using existing collection: {vectorstore.collection_name}")
        
        # Ingest radiology images
        await vectorstore.upsert_radiology_images(radiology_data, batch_size=batch_size)
        
        logger.info("✅ Radiology ingestion to Qdrant complete!")
        
    finally:
        await vectorstore.close()


async def ingest_combined_data(
    pubmed_json_path: str,
    radiology_json_path: str,
    collection_name: str | None = None,
    batch_size: int = 50
) -> None:
    """Ingest both PubMed papers and radiology images into Qdrant.
    
    Args:
        pubmed_json_path: Path to PubMed dataset JSON
        radiology_json_path: Path to radiology dataset with embeddings
        collection_name: Qdrant collection name (default: from config)
        batch_size: Batch size for ingestion
    """
    # Load both datasets
    logger.info(f"Loading PubMed data from {pubmed_json_path}")
    with open(pubmed_json_path, "r") as f:
        pubmed_data = json.load(f)
    
    logger.info(f"Loading radiology data from {radiology_json_path}")
    with open(radiology_json_path, "r") as f:
        radiology_data = json.load(f)
    
    # Initialize Qdrant client
    vectorstore = AsyncQdrantVectorStore()
    if collection_name:
        vectorstore.collection_name = collection_name
    
    try:
        # Check if collection exists, create if not
        collections = await vectorstore.client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        if vectorstore.collection_name not in collection_names:
            logger.info(f"Collection {vectorstore.collection_name} does not exist, creating...")
            await vectorstore.create_collection()
        else:
            logger.info(f"Using existing collection: {vectorstore.collection_name}")
        
        # First, ingest papers with radiology images attached
        logger.info("📚 Ingesting PubMed papers with linked radiology images...")
        await vectorstore.upsert_points(pubmed_data, iuxray_data=radiology_data, batch_size=batch_size)
        
        # Then, ingest radiology images as separate points
        logger.info("🩻 Ingesting radiology images as separate points...")
        await vectorstore.upsert_radiology_images(radiology_data, batch_size=batch_size)
        
        logger.info("✅ Combined ingestion complete!")
        
        # Print statistics
        info = await vectorstore.client.get_collection(vectorstore.collection_name)
        logger.info(f"\n{'='*60}")
        logger.info(f"QDRANT COLLECTION STATISTICS")
        logger.info(f"{'='*60}")
        logger.info(f"Collection: {vectorstore.collection_name}")
        logger.info(f"Total points: {info.points_count}")
        logger.info(f"Vector dimension: {info.config.params.vectors['Dense'].size}")
        logger.info(f"{'='*60}\n")
        
    finally:
        await vectorstore.close()


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest radiology data into Qdrant")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["radiology_only", "combined"],
        default="radiology_only",
        help="Ingestion mode: radiology_only or combined (with PubMed)"
    )
    parser.add_argument(
        "--radiology_json",
        type=str,
        default="/home/m.ismail/MMed-RAG/biomedical-graphrag/data/radiology_with_embeddings.json",
        help="Path to radiology JSON with embeddings"
    )
    parser.add_argument(
        "--pubmed_json",
        type=str,
        default="data/pubmed_dataset.json",
        help="Path to PubMed dataset JSON (for combined mode)"
    )
    parser.add_argument(
        "--collection_name",
        type=str,
        default=None,
        help="Qdrant collection name (default: from config)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
        help="Batch size for ingestion"
    )
    
    args = parser.parse_args()
    
    # Validate radiology file exists
    if not Path(args.radiology_json).exists():
        logger.error(f"Radiology JSON not found: {args.radiology_json}")
        logger.error("Please run extract_embeddings.py first to create the dataset with embeddings")
        sys.exit(1)
    
    if args.mode == "radiology_only":
        await ingest_radiology_to_qdrant(
            radiology_json_path=args.radiology_json,
            collection_name=args.collection_name,
            batch_size=args.batch_size
        )
    elif args.mode == "combined":
        if not Path(args.pubmed_json).exists():
            logger.error(f"PubMed JSON not found: {args.pubmed_json}")
            sys.exit(1)
        
        await ingest_combined_data(
            pubmed_json_path=args.pubmed_json,
            radiology_json_path=args.radiology_json,
            collection_name=args.collection_name,
            batch_size=args.batch_size
        )


if __name__ == "__main__":
    asyncio.run(main())

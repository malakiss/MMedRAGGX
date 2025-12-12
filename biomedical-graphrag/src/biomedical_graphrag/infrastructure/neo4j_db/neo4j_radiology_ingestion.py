"""Neo4j graph ingestion for radiology images and reports."""

import asyncio
from typing import Any

from biomedical_graphrag.domain.radiology import RadiologyDataset
from biomedical_graphrag.infrastructure.neo4j_db.neo4j_client import AsyncNeo4jClient
from biomedical_graphrag.utils.logger_util import setup_logging

logger = setup_logging()


class Neo4jRadiologyIngestion:
    """Async ingestion of radiology images and reports into Neo4j."""

    def __init__(
        self, client: AsyncNeo4jClient, concurrency_limit: int = 25, batch_size: int = 100
    ) -> None:
        self.client = client
        self.semaphore = asyncio.Semaphore(concurrency_limit)
        self.batch_size = batch_size

    async def create_constraints(self) -> None:
        """Create unique constraints for radiology nodes."""
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (img:RadiologyImage) REQUIRE img.image_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (r:Report) REQUIRE r.report_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (f:Finding) REQUIRE f.name IS UNIQUE",
        ]
        for c in constraints:
            await self.client.create_graph(c)
        logger.info("✅ Radiology constraints created.")

    async def create_indexes(self) -> None:
        """Create indexes for efficient querying."""
        indexes = [
            "CREATE INDEX IF NOT EXISTS FOR (img:RadiologyImage) ON (img.modality)",
            "CREATE INDEX IF NOT EXISTS FOR (f:Finding) ON (f.name)",
        ]
        for idx in indexes:
            await self.client.create_graph(idx)
        logger.info("✅ Radiology indexes created.")

    async def ingest_radiology_dataset(self, dataset: RadiologyDataset) -> None:
        """Ingest radiology images and create relationships."""
        await self.create_constraints()
        await self.create_indexes()

        logger.info(f"🩻 Ingesting {len(dataset.images)} radiology images...")

        # Batch image nodes
        for i in range(0, len(dataset.images), self.batch_size):
            batch = dataset.images[i : i + self.batch_size]
            await self._create_image_batch(batch)
            logger.info(f"  → Inserted {i + len(batch)} / {len(dataset.images)} images")

        # Create relationships concurrently
        tasks = [self._safe_ingest_image_relationships(img) for img in dataset.images]
        await asyncio.gather(*tasks)
        logger.info("✅ Radiology ingestion complete.")

    async def _create_image_batch(self, images: list[Any]) -> None:
        """Insert image nodes in batches using UNWIND."""
        query = """
        UNWIND $batch AS row
        MERGE (img:RadiologyImage {image_id: row.image_id})
        SET img.image_path = row.image_path,
            img.image_root = row.image_root,
            img.report = row.report,
            img.modality = row.modality,
            img.image_embedding = row.image_embedding,
            img.report_embedding = row.report_embedding
        """
        params = {
            "batch": [
                {
                    "image_id": img.image_id,
                    "image_path": img.image_path,
                    "image_root": img.image_root,
                    "report": img.report,
                    "modality": img.modality,
                    "image_embedding": img.image_embedding or [],
                    "report_embedding": img.report_embedding or [],
                }
                for img in images
            ]
        }
        await self.client.create_graph(query, params)

    async def _safe_ingest_image_relationships(self, image: Any) -> None:
        """Create relationships for findings."""
        async with self.semaphore:
            try:
                # Extract findings
                if not image.findings:
                    image.extract_findings()

                # Link image to findings
                for finding in image.findings:
                    await self._create_finding_relationship(image.image_id, finding)
            except Exception as e:
                logger.warning(f"⚠️ Failed to ingest relationships for image {image.image_id}: {e}")

    async def _create_finding_relationship(self, image_id: str, finding: str) -> None:
        """Link image to finding."""
        query = """
        MATCH (img:RadiologyImage {image_id: $image_id})
        MERGE (f:Finding {name: $finding})
        MERGE (img)-[:HAS_FINDING]->(f)
        """
        await self.client.create_graph(query, {"image_id": image_id, "finding": finding})

    async def link_images_to_papers(self, image_paper_mapping: dict[str, str]) -> None:
        """Link radiology images to PubMed papers.
        
        Args:
            image_paper_mapping: Dict mapping image_id -> pmid
        """
        logger.info(f"🔗 Linking {len(image_paper_mapping)} images to papers...")
        
        edges = [{"image_id": img_id, "pmid": pmid} for img_id, pmid in image_paper_mapping.items()]
        
        for i in range(0, len(edges), self.batch_size):
            batch = edges[i : i + self.batch_size]
            query = """
            UNWIND $batch AS edge
            MATCH (img:RadiologyImage {image_id: edge.image_id})
            MATCH (p:Paper {pmid: edge.pmid})
            MERGE (img)-[:ILLUSTRATED_IN]->(p)
            """
            await self.client.create_graph(query, {"batch": batch})
        
        logger.info(f"✅ Linked {len(edges)} images to papers.")

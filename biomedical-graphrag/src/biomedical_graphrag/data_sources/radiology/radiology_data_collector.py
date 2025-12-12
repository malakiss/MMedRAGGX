"""Data collector for IU X-Ray dataset."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from biomedical_graphrag.data_sources.base import BaseDataSource
from biomedical_graphrag.domain.radiology import (
    RadiologyDataset,
    RadiologyDatasetMetadata,
    RadiologyImage,
)
from biomedical_graphrag.utils.logger_util import setup_logging

logger = setup_logging()


class RadiologyDataCollector(BaseDataSource):
    """Data collector for IU X-Ray radiology images and reports."""

    def __init__(self, json_path: str) -> None:
        """Initialize the radiology data collector.

        Args:
            json_path: Path to the IU X-Ray JSON dataset file.
        """
        super().__init__()
        self.json_path = json_path

    async def search(self, query: str, max_results: int) -> list[str]:
        """Search is not applicable for static radiology dataset.

        Args:
            query: Not used.
            max_results: Not used.

        Returns:
            Empty list (not applicable for this dataset).
        """
        logger.warning("Search not applicable for radiology dataset")
        return []

    async def fetch_images(self, image_ids: list[str]) -> list[RadiologyImage]:
        """Fetch specific radiology images by ID.

        Args:
            image_ids: List of image IDs to fetch.

        Returns:
            List of RadiologyImage objects.
        """
        logger.info(f"Fetching {len(image_ids)} radiology images")
        all_images = await self._load_dataset()
        
        # Filter by requested IDs
        image_dict = {img.image_id: img for img in all_images}
        fetched = [image_dict[img_id] for img_id in image_ids if img_id in image_dict]
        
        logger.info(f"Fetched {len(fetched)} radiology images")
        return fetched

    async def fetch_entities(self, entity_ids: list[str]) -> list[Any]:
        """Fetch radiology images (unified interface with other collectors).

        Args:
            entity_ids: List of image IDs to fetch.

        Returns:
            List of RadiologyImage objects.
        """
        return await self.fetch_images(entity_ids)

    async def collect_dataset(self) -> RadiologyDataset:
        """Collect the complete IU X-Ray dataset.

        Returns:
            RadiologyDataset with all images and metadata.
        """
        logger.info(f"Collecting radiology dataset from {self.json_path}")
        
        images = await self._load_dataset()
        
        metadata = RadiologyDatasetMetadata(
            collection_date=datetime.now().isoformat(),
            total_images=len(images),
            dataset_name="IU X-Ray",
            modality="chest_xray",
        )
        
        logger.info(f"Collected {len(images)} radiology images")
        return RadiologyDataset(metadata=metadata, images=images)

    async def _load_dataset(self) -> list[RadiologyImage]:
        """Load radiology dataset from JSON file.

        Returns:
            List of RadiologyImage objects.
        """
        logger.info(f"Loading radiology data from {self.json_path}")
        
        with open(self.json_path, "r") as f:
            data = json.load(f)
        
        images = []
        for item in data:
            image = RadiologyImage(
                image_id=item.get("id", ""),
                image_path=item.get("image", ""),
                image_root=item.get("image_root", ""),
                report=item.get("report", ""),
                modality="chest_xray",
                findings=item.get("findings", []),
            )
            images.append(image)
        
        logger.info(f"Loaded {len(images)} radiology images")
        return images


if __name__ == "__main__":
    import asyncio
    from biomedical_graphrag.config import settings

    async def main() -> None:
        """Test radiology data collector."""
        # Update path to your IU X-ray JSON file
        json_path = "/home/m.ismail/MMed-RAG/data/training/retriever/radiology/rad_iu.json"
        
        collector = RadiologyDataCollector(json_path=json_path)
        dataset = await collector.collect_dataset()
        
        # Save to configured path
        output_path = "data/radiology_dataset.json"
        with open(output_path, "w") as f:
            # Convert to dict for JSON serialization
            data = {
                "metadata": {
                    "collection_date": dataset.metadata.collection_date,
                    "total_images": dataset.metadata.total_images,
                    "dataset_name": dataset.metadata.dataset_name,
                    "modality": dataset.metadata.modality,
                },
                "images": [
                    {
                        "image_id": img.image_id,
                        "image_path": img.image_path,
                        "image_root": img.image_root,
                        "report": img.report,
                        "modality": img.modality,
                        "findings": img.findings or [],
                    }
                    for img in dataset.images
                ],
            }
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved radiology dataset to {output_path}")
        print(f"✓ Collected {dataset.metadata.total_images} radiology images")

    asyncio.run(main())

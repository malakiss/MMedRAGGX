from typing import Any

# from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Batch, models
from google import genai
from google.generativeai import types
# Remove: from openai import AsyncOpenAI
from biomedical_graphrag.config import settings
from biomedical_graphrag.domain.citation import CitationNetwork
from biomedical_graphrag.domain.gene import GeneRecord
from biomedical_graphrag.domain.paper import Paper
from biomedical_graphrag.utils.logger_util import setup_logging

logger = setup_logging()


class AsyncQdrantVectorStore:
    """
    Async Qdrant client for managing collections and points.
    """

    def __init__(self) -> None:
        """
        Initialize the async Qdrant client with connection parameters.
        """
        self.url = settings.qdrant.url
        self.api_key = settings.qdrant.api_key
        self.collection_name = settings.qdrant.collection_name
        self.embedding_dimension = settings.qdrant.embedding_dimension

        # self.openai_client = AsyncOpenAI(api_key=settings.openai.api_key.get_secret_value())

        # self.client = AsyncQdrantClient(
        #     url=self.url, api_key=self.api_key.get_secret_value() if self.api_key else None
        # )
        
            
        # Replace OpenAI with Gemini
        self.gemini_client = genai.Client(api_key=settings.gemini.api_key.get_secret_value())
    
        self.client = AsyncQdrantClient(
            url=self.url, api_key=self.api_key.get_secret_value() if self.api_key else None
        )

    async def close(self) -> None:
        """Close the async Qdrant client."""
        await self.client.close()

    async def create_collection(self) -> None:
        """
        Create a new collection in Qdrant (async).
        Args:
                collection_name (str): Name of the collection.
                kwargs: Additional parameters for collection creation.
        """
        logger.info(f"🔧 Creating Qdrant collection: {self.collection_name}")
        await self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "Dense": models.VectorParams(
                    size=self.embedding_dimension, distance=models.Distance.COSINE
                )
            },
        )
        logger.info(f"✅ Collection '{self.collection_name}' created successfully")

    async def delete_collection(self) -> None:
        """
        Delete a collection from Qdrant (async).
        Args:
                collection_name (str): Name of the collection.
        """
        logger.info(f"🗑️ Deleting Qdrant collection: {self.collection_name}")
        await self.client.delete_collection(collection_name=self.collection_name)
        logger.info(f"✅ Collection '{self.collection_name}' deleted successfully")

    # async def _dense_vectors(self, text: str) -> list[float]:
    #     """
    #     Get the embedding vector for the given text (async).
    #     Args:
    #             text (str): Input text to embed.
    #     Returns:
    #             list[float]: The embedding vector.
    #     """
    #     try:
    #         embedding = await self.openai_client.embeddings.create(
    #             model=settings.qdrant.embedding_model, input=text
    #         )
    #         return embedding.data[0].embedding
    #     except Exception as e:
    #         logger.error(f"❌ Failed to create embedding: {e}")
    #         raise

    async def _dense_vectors(self, text: str) -> list[float]:
        """Get the embedding vector for the given text using Gemini."""
        try:
            embedding = self.gemini_client.models.embed_content(
                model=settings.gemini.embedding_model,
                contents=text,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
            )
            return embedding.embeddings[0].values
        except Exception as e:
            logger.error(f"❌ Failed to create embedding: {e}")
            raise
        
    async def _multimodal_vectors(self, text: str, image_path: str) -> list[float]:
        """Get embedding for text + image using multimodal model."""
        try:
            from pathlib import Path
            image_bytes = Path(image_path).read_bytes()
            
            embedding = self.gemini_client.models.embed_content(
                model="multimodalembedding",
                contents=[text, image_bytes],
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
            )
            return embedding.embeddings[0].values
        except Exception as e:
            logger.error(f"❌ Failed to create multimodal embedding: {e}")
            # Fallback to text-only embedding
            return await self._dense_vectors(text)
        
    async def upsert_points(
        self, pubmed_data: dict[str, Any], iuxray_data: dict[str, Any] | None = None, batch_size: int = 50
    ) -> None:
        
        """
        Upsert points into a collection from pubmed_dataset.json and IU X-ray dataset,
        attaching related radiology images to papers in the payload (async).
        Args:
            pubmed_data (dict): Parsed JSON data from pubmed_dataset.json.
            iuxray_data (dict | None): Parsed JSON data from radiology dataset with embeddings.
            batch_size (int): Number of points to process in each batch.
        """
        papers = pubmed_data.get("papers", [])
        citation_network_dict = pubmed_data.get("citation_network", {})

        logger.info(f"📚 Starting ingestion of {len(papers)} papers with batch size {batch_size}")

        # Build PMID -> [RadiologyImage] index
        pmid_to_images: dict[str, list[dict]] = {}
        if iuxray_data is not None:
            images = iuxray_data if isinstance(iuxray_data, list) else iuxray_data.get("images", [])
            logger.info(f"🩻 Processing {len(images)} radiology images for paper-image relationships")
            
            # Build index mapping papers to their images
            # Assuming images have a 'pmid' or 'paper_id' field, or we infer from report
            for img in images:
                image_record = {
                    "image_id": img.get("image_id"),
                    "image_path": img.get("image_path"),
                    "report": img.get("report"),
                    "modality": img.get("modality", "chest_xray"),
                    "findings": img.get("findings", []),
                    "image_embedding": img.get("image_embedding"),
                    "report_embedding": img.get("report_embedding"),
                }
                
                # If image has associated PMID, link it to paper
                linked_pmid = img.get("pmid") or img.get("paper_id")
                if linked_pmid:
                    pmid_to_images.setdefault(linked_pmid, []).append(image_record)
            
            logger.info(f"🔗 Built radiology image index for {len(pmid_to_images)} papers")

        total_processed = 0
        total_skipped = 0

        # Process papers in batches
        for i in range(0, len(papers), batch_size):
            batch_papers = papers[i : i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(papers) + batch_size - 1) // batch_size

            logger.info(f"📦 Processing batch {batch_num}/{total_batches} ({len(batch_papers)} papers)")

            batch_ids = []
            batch_payloads = []
            batch_vectors = []
            batch_skipped = 0

            for paper in batch_papers:
                pmid = paper.get("pmid")
                title = paper.get("title")
                abstract = paper.get("abstract")
                publication_date = paper.get("publication_date")
                journal = paper.get("journal")
                doi = paper.get("doi")
                authors = paper.get("authors", [])
                mesh_terms = paper.get("mesh_terms", [])

                if not title or not abstract or not pmid:
                    batch_skipped += 1
                    continue  # skip incomplete papers

                try:
                    vector = await self._dense_vectors(abstract)

                    # Get citation network for this paper if available
                    citation_info = citation_network_dict.get(pmid, {})
                    citation_network = CitationNetwork(**citation_info) if citation_info else None

                    paper_model = Paper(
                        pmid=pmid,
                        title=title,
                        abstract=abstract,
                        authors=authors,
                        mesh_terms=mesh_terms,
                        publication_date=publication_date,
                        journal=journal,
                        doi=doi,
                    )

                    payload = {
                        "paper": paper_model.model_dump(),
                        "citation_network": citation_network.model_dump() if citation_network else None,
                        "radiology_images": pmid_to_images.get(pmid, []),
                    }

                    batch_ids.append(int(pmid))
                    batch_payloads.append(payload)
                    batch_vectors.append(vector)

                except Exception as e:
                    logger.error(f"❌ Failed to process paper {pmid}: {e}")
                    batch_skipped += 1
                    continue

            # Upsert batch if we have any valid papers
            if batch_ids:
                try:
                    await self.client.upsert(
                        collection_name=self.collection_name,
                        points=Batch(
                            ids=[str(i) for i in batch_ids],
                            payloads=batch_payloads,
                            vectors={"Dense": [list(v) for v in batch_vectors]},
                        ),
                    )
                    total_processed += len(batch_ids)
                    total_skipped += batch_skipped
                    logger.info(
                        f"✅ Batch {batch_num} completed: {len(batch_ids)} papers upserted, \
                            {batch_skipped} skipped"
                    )
                except Exception as e:
                    logger.error(f"❌ Failed to upsert batch {batch_num}: {e}")
                    raise
            else:
                logger.warning(f"⚠️ Batch {batch_num} had no valid papers to process")

        logger.info(
            f"🎉 Ingestion complete! Total: {total_processed} papers processed, {total_skipped} skipped"
        )
    
    async def upsert_radiology_images(
        self, iuxray_data: dict[str, Any] | list[dict], batch_size: int = 50
    ) -> None:
        """
        Upsert radiology images as separate points in Qdrant.
        Uses report embeddings for dense vectors and stores image embeddings in payload.
        
        Args:
            iuxray_data: Radiology dataset (list of images or dict with 'images' key)
            batch_size: Number of images to process in each batch
        """
        images = iuxray_data if isinstance(iuxray_data, list) else iuxray_data.get("images", [])
        logger.info(f"🩻 Starting ingestion of {len(images)} radiology images with batch size {batch_size}")
        
        total_processed = 0
        total_skipped = 0
        
        # Process images in batches
        for i in range(0, len(images), batch_size):
            batch_images = images[i : i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(images) + batch_size - 1) // batch_size
            
            logger.info(f"📦 Processing batch {batch_num}/{total_batches} ({len(batch_images)} images)")
            
            batch_ids = []
            batch_payloads = []
            batch_vectors = []
            batch_skipped = 0
            
            for img in batch_images:
                image_id = img.get("image_id")
                report = img.get("report", "")
                image_embedding = img.get("image_embedding", [])
                report_embedding = img.get("report_embedding", [])
                
                if not image_id:
                    batch_skipped += 1
                    continue
                
                # Skip if no embeddings available
                if not report_embedding and not image_embedding:
                    logger.warning(f"⚠️ Image {image_id} has no embeddings, skipping")
                    batch_skipped += 1
                    continue
                
                try:
                    # Use report embedding as the primary dense vector
                    # If not available, fall back to generating from report text or using image embedding
                    if report_embedding:
                        vector = report_embedding
                    elif report:
                        vector = await self._dense_vectors(report)
                    elif image_embedding:
                        # Use image embedding as fallback (may need dimension adjustment)
                        vector = image_embedding
                    else:
                        batch_skipped += 1
                        continue
                    
                    payload = {
                        "type": "radiology_image",
                        "image_id": image_id,
                        "image_path": img.get("image_path"),
                        "image_root": img.get("image_root"),
                        "report": report,
                        "modality": img.get("modality", "chest_xray"),
                        "findings": img.get("findings", []),
                        "image_embedding": image_embedding,  # Store visual embedding in payload
                        "pmid": img.get("pmid") or img.get("paper_id"),  # Link to paper if available
                    }
                    
                    # Use image_id as the point ID (convert to hash if needed for int)
                    point_id = abs(hash(image_id)) % (10 ** 10)
                    
                    batch_ids.append(point_id)
                    batch_payloads.append(payload)
                    batch_vectors.append(vector)
                    
                except Exception as e:
                    logger.error(f"❌ Failed to process image {image_id}: {e}")
                    batch_skipped += 1
                    continue
            
            # Upsert batch if we have any valid images
            if batch_ids:
                try:
                    await self.client.upsert(
                        collection_name=self.collection_name,
                        points=Batch(
                            ids=[str(i) for i in batch_ids],
                            payloads=batch_payloads,
                            vectors={"Dense": [list(v) for v in batch_vectors]},
                        ),
                    )
                    total_processed += len(batch_ids)
                    total_skipped += batch_skipped
                    logger.info(
                        f"✅ Batch {batch_num} completed: {len(batch_ids)} images upserted, {batch_skipped} skipped"
                    )
                except Exception as e:
                    logger.error(f"❌ Failed to upsert batch {batch_num}: {e}")
                    raise
            else:
                logger.warning(f"⚠️ Batch {batch_num} had no valid images to process")
        
        logger.info(
            f"🎉 Radiology ingestion complete! Total: {total_processed} images processed, {total_skipped} skipped"
        )

"""Extract image and text embeddings for radiology dataset.

This script:
1. Loads OpenCLIP model to extract visual embeddings from X-ray images
2. Uses Gemini API to extract text embeddings from radiology reports
3. Updates the radiology dataset with both embedding types
"""

import asyncio
import json
import sys
from pathlib import Path

import google.generativeai as genai
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from biomedical_graphrag.config import settings
from biomedical_graphrag.domain.radiology import RadiologyImage
from biomedical_graphrag.utils.logger_util import setup_logging

logger = setup_logging()


class EmbeddingExtractor:
    """Extract image and text embeddings for radiology data."""

    def __init__(self, openclip_checkpoint: str, gemini_api_key: str):
        """Initialize embedding models.
        
        Args:
            openclip_checkpoint: Path to fine-tuned OpenCLIP checkpoint
            gemini_api_key: Gemini API key for text embeddings
        """
        self.openclip_checkpoint = openclip_checkpoint
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Configure Gemini
        genai.configure(api_key=gemini_api_key)
        
        # Image preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.clip_model = None
        logger.info(f"Initialized EmbeddingExtractor with device: {self.device}")

    def load_openclip_model(self):
        """Load fine-tuned OpenCLIP model."""
        logger.info(f"Loading OpenCLIP from {self.openclip_checkpoint}")
        
        try:
            import open_clip
            
            # Load model architecture
            model, _, _ = open_clip.create_model_and_transforms(
                'RN50',
                pretrained=None,
                device=self.device
            )
            
            # Load fine-tuned weights
            checkpoint = torch.load(self.openclip_checkpoint, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            
            # Remove 'module.' prefix if present (from DataParallel)
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            
            self.clip_model = model
            logger.info("✅ OpenCLIP model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load OpenCLIP: {e}")
            raise

    def extract_image_embedding(self, image_path: str) -> list[float]:
        """Extract visual embedding from X-ray image using OpenCLIP.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Image embedding as list of floats
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Extract embedding
            with torch.no_grad():
                embedding = self.clip_model.encode_image(image_tensor)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)  # Normalize
            
            return embedding.squeeze().cpu().numpy().tolist()
        
        except Exception as e:
            logger.warning(f"Failed to extract image embedding for {image_path}: {e}")
            return []

    def extract_report_embedding(self, report_text: str) -> list[float]:
        """Extract text embedding from radiology report using Gemini.
        
        Args:
            report_text: Radiology report text
            
        Returns:
            Text embedding as list of floats
        """
        try:
            if not report_text or len(report_text.strip()) == 0:
                return []
            
            # Use Gemini embedding model
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=report_text,
                task_type="retrieval_document"
            )
            
            return result['embedding']
        
        except Exception as e:
            logger.warning(f"Failed to extract report embedding: {e}")
            return []

    async def extract_embeddings_for_dataset(
        self, 
        json_path: str, 
        output_path: str,
        batch_size: int = 10
    ) -> None:
        """Extract embeddings for entire dataset and save.
        
        Args:
            json_path: Path to IU X-ray JSON file
            output_path: Path to save dataset with embeddings
            batch_size: Number of reports to process in parallel
        """
        logger.info(f"Loading dataset from {json_path}")
        
        with open(json_path, "r") as f:
            data = json.load(f)
        
        logger.info(f"Processing {len(data)} images...")
        
        # Load OpenCLIP model
        self.load_openclip_model()
        
        # Process images
        images_with_embeddings = []
        
        for i in tqdm(range(0, len(data), batch_size), desc="Extracting embeddings"):
            batch = data[i:i + batch_size]
            
            for item in batch:
                # Create RadiologyImage object
                image = RadiologyImage(
                    image_id=item.get("id", ""),
                    image_path=item.get("image", ""),
                    image_root=item.get("image_root", ""),
                    report=item.get("report", ""),
                    modality="chest_xray"
                )
                
                # Extract image embedding
                full_path = image.get_full_path()
                if Path(full_path).exists():
                    image.image_embedding = self.extract_image_embedding(full_path)
                else:
                    logger.warning(f"Image not found: {full_path}")
                    image.image_embedding = []
                
                # Extract report embedding
                image.report_embedding = self.extract_report_embedding(image.report)
                
                # Extract findings
                image.extract_findings()
                
                images_with_embeddings.append({
                    "image_id": image.image_id,
                    "image_path": image.image_path,
                    "image_root": image.image_root,
                    "report": image.report,
                    "modality": image.modality,
                    "findings": image.findings,
                    "image_embedding": image.image_embedding,
                    "report_embedding": image.report_embedding,
                })
            
            # Save progress periodically
            if (i + batch_size) % 100 == 0:
                logger.info(f"Processed {i + batch_size}/{len(data)} images")
        
        # Save results
        logger.info(f"Saving embeddings to {output_path}")
        with open(output_path, "w") as f:
            json.dump(images_with_embeddings, f, indent=2)
        
        logger.info(f"✅ Saved {len(images_with_embeddings)} images with embeddings")
        
        # Print statistics
        with_image_emb = sum(1 for img in images_with_embeddings if img["image_embedding"])
        with_report_emb = sum(1 for img in images_with_embeddings if img["report_embedding"])
        logger.info(f"  - Images with visual embeddings: {with_image_emb}/{len(images_with_embeddings)}")
        logger.info(f"  - Images with report embeddings: {with_report_emb}/{len(images_with_embeddings)}")


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract embeddings for radiology dataset")
    parser.add_argument(
        "--json_path",
        type=str,
        default="/home/m.ismail/MMed-RAG/data/training/retriever/radiology/rad_iu.json",
        help="Path to IU X-ray JSON file"
    )
    parser.add_argument(
        "--openclip_checkpoint",
        type=str,
        default="/home/m.ismail/MMed-RAG/train/open_clip/src/logs/exp_name/checkpoints/epoch_360.pt",
        help="Path to fine-tuned OpenCLIP checkpoint"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/home/m.ismail/MMed-RAG/biomedical-graphrag/data/radiology_with_embeddings.json",
        help="Path to save dataset with embeddings"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Batch size for processing"
    )
    
    args = parser.parse_args()
    
    # Get Gemini API key from settings
    gemini_api_key = settings.gemini.api_key.get_secret_value()
    
    if not gemini_api_key:
        logger.error("Gemini API key not found. Please set GEMINI__API_KEY in .env file")
        sys.exit(1)
    
    # Create extractor
    extractor = EmbeddingExtractor(
        openclip_checkpoint=args.openclip_checkpoint,
        gemini_api_key=gemini_api_key
    )
    
    # Extract embeddings
    await extractor.extract_embeddings_for_dataset(
        json_path=args.json_path,
        output_path=args.output_path,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    asyncio.run(main())

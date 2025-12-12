# Radiology Data Integration for GraphRAG

This guide explains how to integrate IU X-ray radiology images with embeddings into the Neo4j knowledge graph.

## Overview

The integration extracts two types of embeddings:
1. **Image embeddings**: Visual features from OpenCLIP (fine-tuned on radiology data)
2. **Report embeddings**: Text embeddings from Gemini API for radiology reports

Both embeddings are stored as node properties in Neo4j for hybrid retrieval.

## Prerequisites

1. **Neo4j Database**: Running at `bolt://localhost:7687`
2. **Gemini API Key**: Set in `.env` file
3. **OpenCLIP Checkpoint**: Fine-tuned model from training
4. **IU X-ray Dataset**: JSON file with images and reports

## Step 1: Configure Environment

Update your `.env` file:

```bash
# Gemini API
GEMINI__API_KEY=your_gemini_api_key_here

# Neo4j
NEO4J__URI=bolt://localhost:7687
NEO4J__USERNAME=neo4j
NEO4J__PASSWORD=your_neo4j_password

# Radiology Data
JSON_DATA__RADIOLOGY_JSON_PATH=/home/m.ismail/MMed-RAG/data/training/retriever/radiology/rad_iu.json
```

## Step 2: Extract Embeddings

Extract both image and text embeddings from the dataset:

```bash
cd /home/m.ismail/MMed-RAG/biomedical-graphrag

python scripts/extract_embeddings.py \
    --json_path /home/m.ismail/MMed-RAG/data/training/retriever/radiology/rad_iu.json \
    --openclip_checkpoint /home/m.ismail/MMed-RAG/train/open_clip/src/logs/exp_name/checkpoints/epoch_360.pt \
    --output_path data/radiology_with_embeddings.json \
    --batch_size 10
```

### What this does:
- Loads fine-tuned OpenCLIP model
- Extracts 512-dim visual embeddings for each X-ray image
- Uses Gemini `text-embedding-004` to extract text embeddings from reports
- Extracts radiological findings using keyword matching
- Saves enriched dataset to `data/radiology_with_embeddings.json`

**Expected output:**
```
Loading dataset from rad_iu.json
Processing 7470 images...
✅ OpenCLIP model loaded successfully
Extracting embeddings: 100%|████████| 747/747 [25:30<00:00]
✅ Saved 7470 images with embeddings
  - Images with visual embeddings: 7470/7470
  - Images with report embeddings: 7470/7470
```

## Step 3: Ingest into Neo4j

Load the dataset with embeddings into Neo4j:

```bash
python scripts/ingest_radiology.py \
    --dataset_path data/radiology_with_embeddings.json \
    --batch_size 100
```

### What this does:
- Creates Neo4j constraints and indexes
- Inserts `RadiologyImage` nodes with both embedding types
- Creates `Finding` nodes and `HAS_FINDING` relationships
- Uses batched async operations for fast ingestion

**Expected output:**
```
Connecting to Neo4j...
✅ Radiology constraints created.
✅ Radiology indexes created.
🩻 Ingesting 7470 radiology images...
  → Inserted 100 / 7470 images
  → Inserted 200 / 7470 images
  ...
✅ Radiology ingestion complete!

============================================================
INGESTION STATISTICS
============================================================
Total images ingested: 7470
Images with visual embeddings: 7470/7470
Images with report embeddings: 7470/7470
Images with findings: 5234/7470
============================================================
```

## Neo4j Graph Schema

After ingestion, your graph will have:

### Nodes

**RadiologyImage**
- `image_id` (unique): Image identifier
- `image_path`: Relative path to image file
- `image_root`: Root directory for images
- `report`: Full radiology report text
- `modality`: Imaging modality (e.g., "chest_xray")
- `image_embedding`: 512-dim float array (OpenCLIP)
- `report_embedding`: 768-dim float array (Gemini)

**Finding**
- `name` (unique): Finding name (e.g., "pneumothorax", "cardiomegaly")

### Relationships

- `(RadiologyImage)-[:HAS_FINDING]->(Finding)`
- `(RadiologyImage)-[:ILLUSTRATED_IN]->(Paper)` (if linked to PubMed)

## Querying the Graph

### Example Cypher Queries

**Find images with specific finding:**
```cypher
MATCH (img:RadiologyImage)-[:HAS_FINDING]->(f:Finding {name: "pneumothorax"})
RETURN img.image_id, img.report
LIMIT 10
```

**Find similar images by visual embedding (cosine similarity):**
```cypher
MATCH (img1:RadiologyImage {image_id: "CXR1_1_IM-0001"})
MATCH (img2:RadiologyImage)
WHERE img1 <> img2
WITH img1, img2, 
     reduce(dot = 0.0, i IN range(0, size(img1.image_embedding)-1) | 
            dot + img1.image_embedding[i] * img2.image_embedding[i]) AS similarity
RETURN img2.image_id, similarity
ORDER BY similarity DESC
LIMIT 10
```

**Get images by modality:**
```cypher
MATCH (img:RadiologyImage {modality: "chest_xray"})
OPTIONAL MATCH (img)-[:HAS_FINDING]->(f:Finding)
RETURN img.image_id, img.report, collect(f.name) AS findings
LIMIT 20
```

## Integration with GraphRAG Query Service

The radiology query tools are already integrated:

1. **`get_similar_images_by_finding(finding, limit)`**: Find images with specific findings
2. **`get_image_report(image_id)`**: Get full report for an image
3. **`get_images_by_modality(modality, limit)`**: Filter by imaging modality

These tools are available in Gemini function calling for hybrid retrieval.

## Verification

Verify the ingestion:

```bash
# Count total images
docker exec -it neo4j cypher-shell -u neo4j -p your_password \
    "MATCH (img:RadiologyImage) RETURN count(img) AS total_images"

# Count findings
docker exec -it neo4j cypher-shell -u neo4j -p your_password \
    "MATCH (f:Finding) RETURN count(f) AS total_findings"

# Check embedding dimensions
docker exec -it neo4j cypher-shell -u neo4j -p your_password \
    "MATCH (img:RadiologyImage) RETURN size(img.image_embedding) AS img_dim, size(img.report_embedding) AS report_dim LIMIT 1"
```

## Troubleshooting

### Missing OpenCLIP checkpoint
```bash
# Find your checkpoint
find /home/m.ismail/MMed-RAG/train/open_clip -name "*.pt"
```

### Gemini API rate limits
The script includes automatic retry with exponential backoff. If you hit rate limits:
- Reduce `--batch_size`
- Add delays between requests

### Image files not found
Ensure `image_root` in JSON matches actual image locations:
```python
# Check paths in JSON
import json
with open('rad_iu.json') as f:
    data = json.load(f)
print(data[0]['image_root'], data[0]['image'])
```

## Performance

**Embedding extraction** (7470 images):
- OpenCLIP: ~2-3 images/sec on GPU
- Gemini API: ~10 reports/sec
- **Total time: ~25-30 minutes**

**Neo4j ingestion** (7470 images):
- Batch size 100
- Async concurrency 25
- **Total time: ~2-3 minutes**

## Next Steps

1. **Vector similarity search**: Use Neo4j vector indexes for efficient embedding search
2. **Link to PubMed**: Connect images to related papers using `ILLUSTRATED_IN` relationships
3. **Multi-modal retrieval**: Combine image and text embeddings for hybrid queries
4. **Clinical validation**: Fine-tune finding extraction with medical NER models

## Files Created

```
biomedical-graphrag/
├── scripts/
│   ├── extract_embeddings.py       # Extract OpenCLIP + Gemini embeddings
│   └── ingest_radiology.py         # Ingest into Neo4j
├── src/biomedical_graphrag/
│   ├── domain/radiology.py         # Domain models
│   ├── data_sources/radiology/     # Data collectors
│   └── infrastructure/neo4j_db/
│       └── neo4j_radiology_ingestion.py  # Neo4j ingestion
└── data/
    └── radiology_with_embeddings.json    # Output with embeddings
```

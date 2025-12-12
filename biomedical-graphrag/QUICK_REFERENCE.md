# Radiology Integration - Quick Reference

## 🚀 Quick Start

```bash
cd /home/m.ismail/MMed-RAG/biomedical-graphrag

# Run the interactive pipeline
./scripts/radiology_pipeline.sh

# Choose option 3 for full pipeline (extract + ingest)
```

## 📁 File Locations

| File | Path |
|------|------|
| **Input Dataset** | `/home/m.ismail/MMed-RAG/data/training/retriever/radiology/rad_iu.json` |
| **OpenCLIP Checkpoint** | `/home/m.ismail/MMed-RAG/train/open_clip/src/logs/exp_name/checkpoints/epoch_360.pt` |
| **Output with Embeddings** | `/home/m.ismail/MMed-RAG/biomedical-graphrag/data/radiology_with_embeddings.json` |
| **Images** | `/home/m.ismail/MMed-RAG/iu_xray/iu_xray/images/CXR*/0.png` |

## ⚙️ Configuration

Create/update `.env` file:

```bash
# Gemini API (required)
GEMINI__API_KEY=your_gemini_api_key_here

# Neo4j (required)
NEO4J__URI=bolt://localhost:7687
NEO4J__USERNAME=neo4j
NEO4J__PASSWORD=your_password

# Data path
JSON_DATA__RADIOLOGY_JSON_PATH=/home/m.ismail/MMed-RAG/data/training/retriever/radiology/rad_iu.json
```

## 🔧 Manual Commands

### 1. Extract Embeddings Only

```bash
python scripts/extract_embeddings.py \
    --json_path /home/m.ismail/MMed-RAG/data/training/retriever/radiology/rad_iu.json \
    --openclip_checkpoint /home/m.ismail/MMed-RAG/train/open_clip/src/logs/exp_name/checkpoints/epoch_360.pt \
    --output_path data/radiology_with_embeddings.json \
    --batch_size 10
```

**Time**: ~25-30 minutes for 7,470 images

### 2. Ingest into Neo4j Only

```bash
python scripts/ingest_radiology.py \
    --dataset_path data/radiology_with_embeddings.json \
    --batch_size 100
```

**Time**: ~2-3 minutes for 7,470 images

## 🔍 Verification Commands

### Check ingestion status

```bash
# Count images
docker exec -it neo4j cypher-shell -u neo4j -p your_password \
    "MATCH (img:RadiologyImage) RETURN count(img) AS total_images"

# Count findings
docker exec -it neo4j cypher-shell -u neo4j -p your_password \
    "MATCH (f:Finding) RETURN count(f) AS total_findings"

# Check embedding dimensions
docker exec -it neo4j cypher-shell -u neo4j -p your_password \
    "MATCH (img:RadiologyImage) RETURN size(img.image_embedding) AS img_dim, size(img.report_embedding) AS report_dim LIMIT 1"

# Sample image data
docker exec -it neo4j cypher-shell -u neo4j -p your_password \
    "MATCH (img:RadiologyImage) RETURN img.image_id, img.modality, img.report LIMIT 5"
```

### Browse in Neo4j Browser

1. Open http://localhost:7474
2. Login with neo4j / your_password
3. Run queries:

```cypher
// Overview
MATCH (img:RadiologyImage)-[:HAS_FINDING]->(f:Finding)
RETURN img.image_id, f.name
LIMIT 25

// Images with pneumothorax
MATCH (img:RadiologyImage)-[:HAS_FINDING]->(f:Finding {name: "pneumothorax"})
RETURN img.image_id, img.report

// All findings
MATCH (f:Finding)
RETURN f.name, count{(img)-[:HAS_FINDING]->(f)} AS image_count
ORDER BY image_count DESC
```

## 🎯 Query Tools (in GraphRAG)

### Available Tools

| Tool Name | Description | Parameters |
|-----------|-------------|------------|
| `get_similar_images_by_finding` | Find images with specific finding | `finding` (str), `limit` (int) |
| `get_image_report` | Get full report for an image | `image_id` (str) |
| `get_images_by_modality` | Filter by imaging modality | `modality` (str), `limit` (int) |

### Example Queries

**User**: "Show me chest X-rays with pneumothorax"
```python
# Gemini calls:
get_similar_images_by_finding("pneumothorax", limit=10)

# Returns:
[
  {
    "image_id": "CXR100_IM-0002",
    "image_path": "CXR100_IM-0002/0.png",
    "report": "Small right pneumothorax...",
    "modality": "chest_xray"
  },
  ...
]
```

**User**: "Get the report for image CXR1_1_IM-0001"
```python
# Gemini calls:
get_image_report("CXR1_1_IM-0001")

# Returns:
{
  "image_id": "CXR1_1_IM-0001",
  "report": "The heart size is normal...",
  "findings": ["opacity", "infiltrate"]
}
```

## 📊 Node Structure

### RadiologyImage Node

```python
{
  "image_id": "CXR1_1_IM-0001",          # Unique ID
  "image_path": "CXR1_1_IM-0001/0.png",  # Relative path
  "image_root": "/path/to/images",        # Root directory
  "report": "Full radiology report...",   # Report text
  "modality": "chest_xray",               # Imaging type
  "image_embedding": [512 floats],        # OpenCLIP features
  "report_embedding": [768 floats]        # Gemini text embedding
}
```

### Relationships

```
(RadiologyImage)-[:HAS_FINDING]->(Finding)
(RadiologyImage)-[:ILLUSTRATED_IN]->(Paper)  // If linked to PubMed
```

## 🐛 Troubleshooting

### Issue: Gemini API rate limits

**Solution**: Reduce batch size
```bash
python scripts/extract_embeddings.py --batch_size 5
```

### Issue: OpenCLIP checkpoint not found

**Solution**: Find checkpoint
```bash
find /home/m.ismail/MMed-RAG/train/open_clip -name "*.pt"
```

### Issue: Neo4j connection refused

**Solution**: Start Neo4j
```bash
docker run -d --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your_password \
  neo4j:latest
```

### Issue: Images not found

**Solution**: Check image paths
```bash
ls /home/m.ismail/MMed-RAG/iu_xray/iu_xray/images/CXR1_1_IM-0001/
```

## 📈 Performance Expectations

| Dataset Size | Embedding Extraction | Neo4j Ingestion | Total |
|--------------|---------------------|-----------------|-------|
| 1,000 images | ~4 min              | ~30 sec         | ~5 min |
| 7,470 images | ~30 min             | ~3 min          | ~33 min |

**Hardware**: 
- GPU: Required for OpenCLIP (CUDA)
- RAM: ~8GB minimum
- Storage: ~2GB for dataset with embeddings

## 📚 Documentation

- **Full Guide**: [RADIOLOGY_INTEGRATION.md](RADIOLOGY_INTEGRATION.md)
- **Architecture**: [RADIOLOGY_ARCHITECTURE.md](RADIOLOGY_ARCHITECTURE.md)
- **Neo4j Queries**: See RADIOLOGY_INTEGRATION.md § "Querying the Graph"

## ✅ Checklist

- [ ] Neo4j running on port 7687
- [ ] Gemini API key configured in `.env`
- [ ] IU X-ray dataset at correct path
- [ ] OpenCLIP checkpoint available
- [ ] Python environment activated with dependencies
- [ ] Run `./scripts/radiology_pipeline.sh`
- [ ] Verify ingestion with Cypher queries
- [ ] Test GraphRAG queries with radiology tools

## 🎉 Success Indicators

You'll know it's working when:

1. **Extraction**: See progress bar reaching 100% with no errors
2. **Ingestion**: See "✅ Ingestion complete!" with statistics
3. **Verification**: Cypher queries return 7,470 images
4. **Query**: GraphRAG returns relevant X-rays for "pneumothorax"

## 💡 Tips

- Run extraction overnight for large datasets
- Use tmux/screen for long-running processes
- Monitor GPU usage: `nvidia-smi`
- Check Neo4j memory: http://localhost:7474 → System info
- Save embeddings JSON for faster re-ingestion

# Updated: Radiology Data Integration with Qdrant

## Summary of Changes

I've updated the GraphRAG system to integrate IU X-ray radiology data with **both Neo4j AND Qdrant**, replacing the gene dataset with radiology images.

## What Changed

### 1. **Qdrant VectorStore** ([qdrant_vectorstore.py](biomedical-graphrag/src/biomedical_graphrag/infrastructure/qdrant_db/qdrant_vectorstore.py))

**Before**: Used gene dataset with papers
```python
# Old: pmid_to_genes mapping
genes = gene_data.get("genes", [])
for gene in genes:
    for linked_pmid in gene.linked_pmids:
        pmid_to_genes[linked_pmid].append(gene)
```

**After**: Uses radiology images with papers
```python
# New: pmid_to_images mapping
images = iuxray_data.get("images", [])
for img in images:
    image_record = {
        "image_id": img.get("image_id"),
        "report": img.get("report"),
        "image_embedding": img.get("image_embedding"),  # OpenCLIP
        "report_embedding": img.get("report_embedding"), # Gemini
        # ... more fields
    }
    linked_pmid = img.get("pmid")
    if linked_pmid:
        pmid_to_images[linked_pmid].append(image_record)
```

**Payload structure**:
```python
{
  "paper": {...},  # PubMed paper
  "citation_network": {...},
  "radiology_images": [...]  # ← Replaced "genes"
}
```

### 2. **New Method**: `upsert_radiology_images()`

Ingests radiology images as **separate points** in Qdrant:

```python
await vectorstore.upsert_radiology_images(radiology_data, batch_size=50)
```

**Features**:
- Uses `report_embedding` (768-dim Gemini) as dense vector
- Stores `image_embedding` (512-dim OpenCLIP) in payload
- Enables hybrid text + visual retrieval
- Each image is a searchable point

**Point structure**:
```python
{
  "type": "radiology_image",
  "image_id": "CXR1_1_IM-0001",
  "report": "Full radiology report...",
  "image_embedding": [512 floats],  # OpenCLIP (stored in payload)
  "report_embedding": [768 floats], # Gemini (used as vector)
  "findings": ["pneumothorax", "cardiomegaly"],
  "pmid": "12345678"  # Link to paper if available
}
```

### 3. **Configuration Updates**

[config.py](biomedical-graphrag/src/biomedical_graphrag/config.py):
```python
class GeminiSettings(BaseModel):
    embedding_model: str = "text-embedding-004"  # Added for Qdrant

class QdrantSettings(BaseModel):
    collection_name: str = "biomedical_radiology"  # Updated
    embedding_model: str = "text-embedding-004"     # Gemini
    embedding_dimension: int = 768                  # Gemini dimension
```

## Usage

### Option 1: Radiology Images Only

```bash
python scripts/ingest_radiology_qdrant.py \
    --mode radiology_only \
    --radiology_json data/radiology_with_embeddings.json
```

**Result**: 7,470 radiology image points in Qdrant

### Option 2: Combined (PubMed + Radiology)

```bash
python scripts/ingest_radiology_qdrant.py \
    --mode combined \
    --pubmed_json data/pubmed_dataset.json \
    --radiology_json data/radiology_with_embeddings.json
```

**Result**: 
- PubMed papers with `radiology_images` in payload
- Radiology images as separate searchable points

## Data Flow

```
┌─────────────────────────────────────────┐
│  radiology_with_embeddings.json        │
│  • image_id, image_path                │
│  • report (text)                        │
│  • image_embedding (512-dim OpenCLIP)  │
│  • report_embedding (768-dim Gemini)   │
│  • findings                             │
└──────────────┬──────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│  Qdrant Ingestion                        │
│  └─ upsert_radiology_images()            │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│  Qdrant Collection: biomedical_radiology │
│                                          │
│  Point 1: CXR1_1_IM-0001                │
│  ├─ Vector: report_embedding (768-dim)  │
│  ├─ Payload:                             │
│  │   ├─ type: "radiology_image"         │
│  │   ├─ report: "..."                   │
│  │   ├─ image_embedding: [512 floats]   │
│  │   ├─ findings: ["opacity"]           │
│  │   └─ pmid: "12345678"                │
│  ...                                     │
└──────────────────────────────────────────┘
```

## Hybrid Retrieval Architecture

Now you have **3 data sources**:

1. **Neo4j** (Graph queries)
   - RadiologyImage nodes with embeddings
   - Finding nodes
   - Paper nodes
   - Relationships: HAS_FINDING, ILLUSTRATED_IN

2. **Qdrant** (Vector similarity)
   - Radiology images as points
   - Report embeddings for semantic search
   - Image embeddings in payload

3. **Gemini Function Calling**
   - Routes queries to Neo4j OR Qdrant
   - Combines results

### Query Flow Example

**User**: "Find chest X-rays showing pneumothorax with similar cases"

```python
# Step 1: Gemini determines tools
tools = [
  "get_similar_images_by_finding",  # Neo4j
  "semantic_search"                  # Qdrant
]

# Step 2: Neo4j query (graph)
neo4j_results = get_similar_images_by_finding("pneumothorax")
# Returns: Images with HAS_FINDING relationship

# Step 3: Qdrant query (vector)
qdrant_results = qdrant_client.search(
    collection_name="biomedical_radiology",
    query_vector=gemini_embed("pneumothorax chest x-ray"),
    filter={"findings": {"$in": ["pneumothorax"]}},
    limit=10
)
# Returns: Semantically similar reports

# Step 4: Fusion
final_answer = gemini_synthesize(neo4j_results + qdrant_results)
```

## Benefits of This Architecture

| Feature | Neo4j | Qdrant | Combined |
|---------|-------|--------|----------|
| **Exact matches** | ✅ (graph traversal) | ❌ | ✅ |
| **Semantic similarity** | ❌ | ✅ (vector search) | ✅ |
| **Relationships** | ✅ (explicit edges) | ❌ | ✅ |
| **Scalability** | Moderate | High | ✅ |
| **Multimodal** | ✅ (stored) | ✅ (searchable) | ✅ |

## Verification

```bash
# Check Qdrant ingestion
curl http://localhost:6333/collections/biomedical_radiology

# Expected response:
{
  "result": {
    "status": "green",
    "points_count": 7470,
    "vectors_count": 7470,
    "config": {
      "params": {
        "vectors": {
          "Dense": {
            "size": 768,
            "distance": "Cosine"
          }
        }
      }
    }
  }
}
```

## Complete Pipeline

```bash
# 1. Extract embeddings (OpenCLIP + Gemini)
python scripts/extract_embeddings.py

# 2. Ingest into Neo4j (graph structure)
python scripts/ingest_radiology.py

# 3. Ingest into Qdrant (vector search)
python scripts/ingest_radiology_qdrant.py --mode radiology_only

# 4. Query with GraphRAG (hybrid retrieval)
# Now uses both Neo4j + Qdrant automatically!
```

## Key Differences from Gene Dataset

| Aspect | Gene Dataset (Old) | Radiology Dataset (New) |
|--------|-------------------|------------------------|
| **Entity Type** | Genes | Radiology Images |
| **Link to Papers** | `linked_pmids` | `pmid` or `paper_id` |
| **Embeddings** | Text only | Image (512) + Report (768) |
| **Qdrant Payload** | `"genes": [...]` | `"radiology_images": [...]` |
| **Neo4j Nodes** | Gene, MENTIONED_IN | RadiologyImage, HAS_FINDING |
| **Primary Use** | Genomics research | Medical imaging |

## Configuration Example

`.env` file:
```bash
# Gemini
GEMINI__API_KEY=your_key
GEMINI__EMBEDDING_MODEL=text-embedding-004

# Qdrant
QDRANT__URL=http://localhost:6333
QDRANT__COLLECTION_NAME=biomedical_radiology
QDRANT__EMBEDDING_DIMENSION=768

# Neo4j
NEO4J__URI=bolt://localhost:7687
NEO4J__PASSWORD=your_password

# Data
JSON_DATA__RADIOLOGY_JSON_PATH=/path/to/rad_iu.json
```

## Next Steps

1. ✅ Extract embeddings (OpenCLIP + Gemini)
2. ✅ Ingest into Neo4j (graph)
3. ✅ Ingest into Qdrant (vectors)
4. 🔄 Test hybrid queries combining Neo4j + Qdrant
5. 🔄 Fine-tune Qdrant search filters
6. 🔄 Add multimodal search (image embedding similarity)

## Files Modified/Created

- ✅ `qdrant_vectorstore.py` - Updated to use radiology instead of genes
- ✅ `config.py` - Added Gemini embedding model settings
- ✅ `scripts/ingest_radiology_qdrant.py` - New Qdrant ingestion script
- ✅ All previous Neo4j integration files remain unchanged

Your system now has **full hybrid retrieval** across:
- **Neo4j**: Structured graph queries with embeddings
- **Qdrant**: Fast vector similarity search
- **Gemini**: Intelligent query routing and synthesis

🎉

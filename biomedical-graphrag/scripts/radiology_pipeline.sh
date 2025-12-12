#!/bin/bash
# Quick start script for radiology data integration

set -e

echo "=================================================="
echo "Radiology Data Integration - Quick Start"
echo "=================================================="
echo ""

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Paths
IU_XRAY_JSON="/home/m.ismail/MMed-RAG/data/training/retriever/radiology/rad_iu.json"
OPENCLIP_CHECKPOINT="/home/m.ismail/MMed-RAG/train/open_clip/src/logs/exp_name/checkpoints/epoch_360.pt"
OUTPUT_WITH_EMBEDDINGS="$PROJECT_ROOT/data/radiology_with_embeddings.json"

# Check prerequisites
echo "Checking prerequisites..."
echo ""

# Check if .env exists
if [ ! -f "$PROJECT_ROOT/.env" ]; then
    echo "❌ .env file not found!"
    echo "   Please copy .env.example to .env and configure it:"
    echo "   cp .env.example .env"
    echo "   nano .env  # Set your GEMINI__API_KEY and NEO4J__PASSWORD"
    exit 1
fi

# Check if Neo4j is running
echo -n "Checking Neo4j connection... "
if command -v docker &> /dev/null; then
    if docker ps | grep -q neo4j; then
        echo "✅ Running"
    else
        echo "❌ Not running"
        echo "   Please start Neo4j:"
        echo "   docker run -d --name neo4j -p 7474:7474 -p 7687:7687 \\"
        echo "     -e NEO4J_AUTH=neo4j/your_password neo4j:latest"
        exit 1
    fi
else
    echo "⚠️  Cannot check (docker not found)"
fi

# Check if dataset exists
if [ ! -f "$IU_XRAY_JSON" ]; then
    echo "❌ IU X-ray dataset not found at: $IU_XRAY_JSON"
    exit 1
fi
echo "✅ Dataset found: $IU_XRAY_JSON"

# Check if OpenCLIP checkpoint exists
if [ ! -f "$OPENCLIP_CHECKPOINT" ]; then
    echo "⚠️  OpenCLIP checkpoint not found at: $OPENCLIP_CHECKPOINT"
    echo "   Searching for checkpoint..."
    FOUND_CHECKPOINT=$(find /home/m.ismail/MMed-RAG/train/open_clip -name "*.pt" -o -name "*.pth" | head -1)
    if [ -n "$FOUND_CHECKPOINT" ]; then
        echo "   Found: $FOUND_CHECKPOINT"
        OPENCLIP_CHECKPOINT="$FOUND_CHECKPOINT"
    else
        echo "❌ No OpenCLIP checkpoint found!"
        exit 1
    fi
fi
echo "✅ OpenCLIP checkpoint: $OPENCLIP_CHECKPOINT"
echo ""

# Menu
echo "What would you like to do?"
echo "1) Extract embeddings (OpenCLIP + Gemini)"
echo "2) Ingest into Neo4j (requires embeddings from step 1)"
echo "3) Full pipeline (extract + ingest)"
echo "4) Verify ingestion"
echo ""
read -p "Enter choice [1-4]: " choice

case $choice in
    1)
        echo ""
        echo "=================================================="
        echo "Step 1: Extracting Embeddings"
        echo "=================================================="
        echo "This will:"
        echo "  - Load fine-tuned OpenCLIP model"
        echo "  - Extract visual embeddings from X-ray images"
        echo "  - Extract text embeddings from reports (Gemini API)"
        echo "  - Save to: $OUTPUT_WITH_EMBEDDINGS"
        echo ""
        echo "Estimated time: ~25-30 minutes for 7470 images"
        echo ""
        read -p "Continue? [y/N] " confirm
        if [[ $confirm != [yY] ]]; then
            echo "Cancelled."
            exit 0
        fi
        
        python scripts/extract_embeddings.py \
            --json_path "$IU_XRAY_JSON" \
            --openclip_checkpoint "$OPENCLIP_CHECKPOINT" \
            --output_path "$OUTPUT_WITH_EMBEDDINGS" \
            --batch_size 10
        
        echo ""
        echo "✅ Embedding extraction complete!"
        echo "   Output: $OUTPUT_WITH_EMBEDDINGS"
        ;;
    
    2)
        echo ""
        echo "=================================================="
        echo "Step 2: Ingesting into Neo4j"
        echo "=================================================="
        
        if [ ! -f "$OUTPUT_WITH_EMBEDDINGS" ]; then
            echo "❌ Embeddings file not found: $OUTPUT_WITH_EMBEDDINGS"
            echo "   Please run option 1 first to extract embeddings."
            exit 1
        fi
        
        echo "This will:"
        echo "  - Create Neo4j constraints and indexes"
        echo "  - Insert RadiologyImage nodes with embeddings"
        echo "  - Create Finding nodes and relationships"
        echo ""
        echo "Estimated time: ~2-3 minutes for 7470 images"
        echo ""
        read -p "Continue? [y/N] " confirm
        if [[ $confirm != [yY] ]]; then
            echo "Cancelled."
            exit 0
        fi
        
        python scripts/ingest_radiology.py \
            --dataset_path "$OUTPUT_WITH_EMBEDDINGS" \
            --batch_size 100
        
        echo ""
        echo "✅ Ingestion complete!"
        ;;
    
    3)
        echo ""
        echo "=================================================="
        echo "Full Pipeline: Extract + Ingest"
        echo "=================================================="
        echo "Estimated time: ~30-35 minutes total"
        echo ""
        read -p "Continue? [y/N] " confirm
        if [[ $confirm != [yY] ]]; then
            echo "Cancelled."
            exit 0
        fi
        
        # Step 1: Extract embeddings
        echo ""
        echo "Step 1/2: Extracting embeddings..."
        python scripts/extract_embeddings.py \
            --json_path "$IU_XRAY_JSON" \
            --openclip_checkpoint "$OPENCLIP_CHECKPOINT" \
            --output_path "$OUTPUT_WITH_EMBEDDINGS" \
            --batch_size 10
        
        # Step 2: Ingest
        echo ""
        echo "Step 2/2: Ingesting into Neo4j..."
        python scripts/ingest_radiology.py \
            --dataset_path "$OUTPUT_WITH_EMBEDDINGS" \
            --batch_size 100
        
        echo ""
        echo "✅ Full pipeline complete!"
        ;;
    
    4)
        echo ""
        echo "=================================================="
        echo "Verifying Ingestion"
        echo "=================================================="
        
        # Read Neo4j credentials from .env
        source <(grep -E '^NEO4J__' .env | sed 's/__/=/g')
        
        echo "Checking Neo4j database..."
        echo ""
        
        docker exec -it neo4j cypher-shell -u "$NEO4J_USERNAME" -p "$NEO4J_PASSWORD" \
            "MATCH (img:RadiologyImage) RETURN count(img) AS total_images" || true
        
        echo ""
        docker exec -it neo4j cypher-shell -u "$NEO4J_USERNAME" -p "$NEO4J_PASSWORD" \
            "MATCH (f:Finding) RETURN count(f) AS total_findings" || true
        
        echo ""
        docker exec -it neo4j cypher-shell -u "$NEO4J_USERNAME" -p "$NEO4J_PASSWORD" \
            "MATCH (img:RadiologyImage) RETURN size(img.image_embedding) AS img_emb_dim, size(img.report_embedding) AS report_emb_dim LIMIT 1" || true
        
        echo ""
        echo "✅ Verification complete!"
        ;;
    
    *)
        echo "Invalid choice. Please run again and select 1-4."
        exit 1
        ;;
esac

echo ""
echo "=================================================="
echo "Done!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "  - Query the graph using Neo4j Browser: http://localhost:7474"
echo "  - Use GraphRAG query service with radiology tools"
echo "  - See RADIOLOGY_INTEGRATION.md for example queries"
echo ""

#!/bin/bash
# EXO GGUF Distributed Deployment Script
# 
# This script helps deploy distributed GGUF inference across multiple nodes.
# Each node should have:
#   1. Ollama installed with the same model
#   2. EXO cloned from the same repository
#   3. Network connectivity between nodes
#
# Usage:
#   ./deploy_distributed.sh <model_name> <node_list>
#
# Example:
#   ./deploy_distributed.sh qwen3-coder:30b "192.168.1.10,192.168.1.11"

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║       EXO GGUF Distributed Deployment                       ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check arguments
if [ "$#" -lt 2 ]; then
    echo -e "${RED}Usage: $0 <model_name> <node_list>${NC}"
    echo "  model_name: Ollama model name (e.g., qwen3-coder:30b)"
    echo "  node_list:  Comma-separated list of node IPs"
    echo ""
    echo "Example:"
    echo "  $0 qwen3-coder:30b \"192.168.1.10,192.168.1.11\""
    exit 1
fi

MODEL_NAME="$1"
NODE_LIST="$2"
BASE_PORT=50100

# Parse nodes
IFS=',' read -ra NODES <<< "$NODE_LIST"
WORLD_SIZE=${#NODES[@]}

echo -e "${YELLOW}Model:${NC} $MODEL_NAME"
echo -e "${YELLOW}Nodes:${NC} ${NODES[*]}"
echo -e "${YELLOW}World Size:${NC} $WORLD_SIZE"
echo ""

# Build node addresses with ports
NODE_ADDRESSES=""
for i in "${!NODES[@]}"; do
    PORT=$((BASE_PORT + i))
    if [ -n "$NODE_ADDRESSES" ]; then
        NODE_ADDRESSES="$NODE_ADDRESSES,"
    fi
    NODE_ADDRESSES="$NODE_ADDRESSES${NODES[$i]}:$PORT"
done

echo -e "${GREEN}Node Configuration:${NC}"
for i in "${!NODES[@]}"; do
    PORT=$((BASE_PORT + i))
    ROLE="middle"
    [ $i -eq 0 ] && ROLE="head"
    [ $i -eq $((WORLD_SIZE - 1)) ] && ROLE="tail"
    echo "  Node $i (${NODES[$i]}:$PORT) - $ROLE"
done
echo ""

# Generate start commands for each node
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Commands to run on each node:${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo ""

for i in "${!NODES[@]}"; do
    NODE="${NODES[$i]}"
    PORT=$((BASE_PORT + i))
    
    ROLE="MIDDLE"
    [ $i -eq 0 ] && ROLE="HEAD"
    [ $i -eq $((WORLD_SIZE - 1)) ] && ROLE="TAIL"
    
    echo -e "${YELLOW}[$ROLE - Node $i (${NODE})]${NC}"
    echo "ssh ${NODE} 'cd /path/to/exo && \\"
    echo "  uv run python -m exo.worker.engines.gguf.launcher \\"
    echo "    --model $MODEL_NAME \\"
    echo "    --rank $i \\"
    echo "    --world-size $WORLD_SIZE \\"
    echo "    --nodes \"$NODE_ADDRESSES\" \\"
    
    if [ $i -eq 0 ]; then
        echo "    --interactive'"
    else
        echo "'"
    fi
    echo ""
done

echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Quick Local Test (single node):${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo ""
echo "uv run python -m exo.worker.engines.gguf.launcher \\"
echo "  --model $MODEL_NAME \\"
echo "  --rank 0 \\"
echo "  --world-size 1 \\"
echo "  --nodes \"localhost:$BASE_PORT\" \\"
echo "  --interactive"
echo ""

# Check if running locally
THIS_HOST=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "localhost")

echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Pre-flight Checks:${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo ""

# Check if Ollama is installed
if command -v ollama &> /dev/null; then
    echo -e "${GREEN}✓${NC} Ollama is installed"
    
    # Check if model exists
    if ollama list 2>/dev/null | grep -q "$MODEL_NAME"; then
        echo -e "${GREEN}✓${NC} Model '$MODEL_NAME' is available locally"
    else
        echo -e "${RED}✗${NC} Model '$MODEL_NAME' not found locally"
        echo "  Run: ollama pull $MODEL_NAME"
    fi
else
    echo -e "${RED}✗${NC} Ollama is not installed"
    echo "  Install from: https://ollama.ai"
fi

# Check if uv is installed
if command -v uv &> /dev/null; then
    echo -e "${GREEN}✓${NC} uv is installed"
else
    echo -e "${RED}✗${NC} uv is not installed"
    echo "  Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
fi

echo ""
echo -e "${GREEN}Done! Copy the commands above to each respective node.${NC}"

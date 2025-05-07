#!/bin/bash

# Retrieval the results 

MODEL_DIR="Model"

for i in {21..25}; do
    MODEL_FILE="${MODEL_DIR}/model-run1-${i}.pth"
    RESULT_FILE="First_stage_result/result_run1_${i}.txt"
    if [ -f "$MODEL_FILE" ]; then
        echo "Running generate_vector.py with model: $MODEL_FILE"
        python generate_vector.py --model_path "$MODEL_FILE"
        python retrieval_result.py --output_file  "$RESULT_FILE"
    else
        echo "Model file $MODEL_FILE not found, skipping..."
    fi
done

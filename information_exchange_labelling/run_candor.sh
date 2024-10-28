#!/bin/bash

# Define variables
#MODEL_ID="gpt2-large"
MODEL_ID="unsloth/Meta-Llama-3.1-8B-bnb-4bit"
DEVICE="cuda:3"
DATA_PATH="/u/sebono/conversational_dominance/data/processed/CANDOR/conversations.csv"
INDEX_PATH="/u/sebono/conversational_dominance/data/processed/CANDOR/group_7.csv"
PERPL_FUNC="p1"
OUTPUT_PATH="/u/sebono/conversational_dominance/notebooks/information_exchange_labelling/dataset_perplexity_results/CANDOR_${PERPL_FUNC}_${MODEL_ID}/"

python perplexity_labelling.py --model_id "$MODEL_ID" --device "$DEVICE" --data_path "$DATA_PATH" --index_path "$INDEX_PATH" --output_path "$OUTPUT_PATH" --perplexity_func "$PERPL_FUNC"
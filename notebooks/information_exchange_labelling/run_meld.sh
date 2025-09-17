#!/bin/bash

# Define variables
MODEL_ID="gpt2-large"
DEVICE="cuda:5"
DATA_PATH="/u/sebono/conversational_dominance/data/processed/MELD/conversations.csv"
INDEX_PATH="/u/sebono/conversational_dominance/data/processed/MELD/group_4.csv"
PERPL_FUNC="p1"
OUTPUT_PATH="/u/sebono/conversational_dominance/notebooks/information_exchange_labelling/dataset_perplexity_results/MELD_$PERPL_FUNC/"

python perplexity_labelling.py --model_id "$MODEL_ID" --device "$DEVICE" --data_path "$DATA_PATH" --index_path "$INDEX_PATH" --output_path "$OUTPUT_PATH" --perplexity_func "$PERPL_FUNC"
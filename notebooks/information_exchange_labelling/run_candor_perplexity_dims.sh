#!/bin/bash

# Define variables
MODEL_ID="gpt2-large"
#MODEL_ID="unsloth/Meta-Llama-3.1-8B-bnb-4bit"
DEVICE="cuda:1"
DATA_PATH="/u/sebono/conversational_dominance/data/processed/CANDOR_${MODEL_ID}/conversations.csv"
PERPL_FUNC="p3"
PPL_PATH="/u/sebono/conversational_dominance/information_exchange_labelling/dataset_perplexity_results/CANDOR_p1_${MODEL_ID}/"
OUTPUT_PATH="/u/sebono/conversational_dominance/data/processed/CANDOR_${MODEL_ID}/"
TIME_FIELD='{"START": "start", "STOP": "stop"}'
LABELS="/u/sebono/conversational_dominance/data/external/CANDOR_${MODEL_ID}/transcript_audiophile/"
BIN_WIDTH=1
python compute_perplexity_dims.py --model_id "$MODEL_ID" --device "$DEVICE" --data_path "$DATA_PATH" --ppl_path "$PPL_PATH" --output_path "$OUTPUT_PATH" --perplexity_func "$PERPL_FUNC" --time_field "$TIME_FIELD" --labels "$LABELS" --bin_width "$BIN_WIDTH"
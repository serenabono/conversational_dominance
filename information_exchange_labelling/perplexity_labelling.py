
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from torch.nn.functional import pad

from transformers import LlamaForCausalLM, LlamaTokenizerFast

def get_token_nll(input_ids, model, device, trg_len=-1, tokenizer=None, debug=False):
    input_ids = input_ids.to(device)
    target_ids = input_ids.clone()
    target_ids[:, :trg_len] = -100  # mask all tokens except the last one

    masked = [tokenizer.decode(token, skip_special_tokens=True) for token in input_ids[:, :trg_len]]
    computed = [tokenizer.decode(token, skip_special_tokens=True) for token in target_ids[:, trg_len:]]
    
    if debug:
        print(f"{masked} | {computed}")

    with torch.no_grad():
        output = model(input_ids, labels=target_ids)
        loss = output.loss.item()

    return loss


def get_token_spans(dialog, tokenizer):
    """Returns list of token spans for each turn."""
    spans = []
    total_tokens = []
    for turn in dialog:
        tokens = tokenizer(turn, return_tensors="pt").input_ids[0]
        spans.append((len(total_tokens), len(total_tokens) + len(tokens)))
        total_tokens.extend(tokens.tolist())
    return spans, torch.tensor(total_tokens)


def compute_p1(encodings, token_list, tokenizer, model, device, max_length = 500, start_of_sentence=" ", pattern = r'<(?:SPK[0-9]|MOD)>', debug=False):
    stride = 1
    
    pad_token_id = 0
    seq_len = encodings.input_ids.size(1)
    padding_len = max_length -1 
    padded_input_ids = pad(torch.tensor([], dtype=torch.long), (0, padding_len), value=pad_token_id).unsqueeze(dim=0)
    padded_input_ids = torch.cat([padded_input_ids, encodings.input_ids], dim=1)
    seq_len = padded_input_ids.size(1)
    
    nlls = []
    prev_end_loc = padding_len
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from max_length on the last loop 
        begin_loc = max(padding_len, begin_loc)
        input_ids = padded_input_ids[:, begin_loc:end_loc].to(device)
    
        loss = get_token_nll(
            input_ids=input_ids,
            model=model,
            device=device,
            trg_len=-trg_len,
            tokenizer=tokenizer,
            debug=debug,
        )
        
        nlls.append(loss)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    return nlls

def compute_p2(encodings, token_list, tokenizer, model, device, max_length = 500, start_of_sentence=" ", pattern = r'<(?:SPK[0-9]|MOD)>', debug=False):
    stride = 1

    pad_token_id = 0
    assert len([t for token in token_list for t in token]) == encodings.input_ids.size(1)

    turn_len = [len(token) for token in token_list]
    turn_mask = offset_mask = np.concatenate([
        np.arange(length) for length in turn_len
    ])
    assert len(turn_mask) == sum(turn_len)

    seq_len = encodings.input_ids.size(1)
    padding_len = max_length -1 
    padded_input_ids = pad(torch.tensor([], dtype=torch.long), (0, padding_len), value=pad_token_id).unsqueeze(dim=0)
    padded_input_ids = torch.cat([padded_input_ids, encodings.input_ids], dim=1)
    seq_len = padded_input_ids.size(1)

    nlls = []
    prev_end_loc = padding_len
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        offset = turn_mask[begin_loc]
        trg_len = end_loc - prev_end_loc + offset  # may be different from max_length on the last loop 
        begin_loc = max(padding_len, begin_loc)
        input_ids = padded_input_ids[:, begin_loc:end_loc].to(device)

        loss = get_token_nll(
            input_ids=input_ids,
            model=model,
            device=device,
            trg_len=-trg_len,
            tokenizer=tokenizer,
            debug=debug
        )

        nlls.append(loss)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    return nlls


def compute_p3(encodings, token_list, tokenizer, model, device, max_length = 500, start_of_sentence=" ", pattern = r'<(?:SPK[0-9]|MOD)>', debug=False):
    
    stride = 1
    pad_token_id = 0
    assert len([t for token in token_list for t in token]) == encodings.input_ids.size(1)

    turn_len = [len(token) for token in token_list]
    turn_mask = np.repeat(range(len(turn_len)), turn_len)
    max_val = turn_mask.max()
    turn_mask = torch.tensor([(turn_mask + 1) % (max_val + 1)])

    assert len(turn_mask[0]) == sum(turn_len)

    seq_len = encodings.input_ids.size(1)
    padding_len = max_length -1 
    padded_input_ids = pad(torch.tensor([], dtype=torch.long), (0, padding_len), value=pad_token_id).unsqueeze(dim=0)
    turn_mask = torch.cat([padded_input_ids, turn_mask], dim=1)
    padded_input_ids = torch.cat([padded_input_ids, encodings.input_ids], dim=1)
    seq_len = padded_input_ids.size(1)

    nlls = []
    prev_end_loc = padding_len
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        begin_loc = max(padding_len, begin_loc)

        sentence = tokenizer.decode(token_list[turn_mask[0, prev_end_loc]],skip_special_tokens=True)
        match = re.findall(pattern, sentence)
        len_match = tokenizer(match, return_tensors="pt").input_ids.size(1)
        offset = token_list[turn_mask[0, prev_end_loc]][:len_match].unsqueeze(0)
        input_ids_aug = torch.cat([padded_input_ids[:, begin_loc:end_loc], offset], dim=1).to(device)
        loss = get_token_nll(
            input_ids=input_ids_aug,
            model=model,
            device=device,
            trg_len=-len_match,
            tokenizer=tokenizer,
            debug=debug
        )
        nlls.append(loss)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break


    return nlls

from transformers import GPT2LMHeadModel, GPT2TokenizerFast

import argparse
import pickle
import os

if __name__ == "__main__":
    # Define argparse arguments
    parser = argparse.ArgumentParser(description="Compute perplexity and generate graphs for dialogue transcripts")
    parser.add_argument("--model_id", type=str, default="gpt2-large", help="ID of the GPT-2 model to use")
    parser.add_argument("--device", type=str, default="cuda:1", help="Device to run the model on (e.g., 'cuda:0' for GPU or 'cpu' for CPU)")
    parser.add_argument("--data_path", type=str, default="/u/sebono/conversational_dominance/data/processed/topical/conversations.csv", help="Path to the CSV file containing the dialogue transcripts")
    parser.add_argument("--index_path", type=str, default="/u/sebono/conversational_dominance/data/processed/topical/group_1.csv", help="Path to the CSV file containing the dialogue transcripts")
    parser.add_argument("--output_path", type=str, default="/u/sebono/conversational_dominance/notebooks/dominance_scores_topical.pkl", help="Path to save the output pickle file")
    parser.add_argument("--perplexity_func", type=str, default="default", choices=["p1", "p2","p3"], 
                        help="Function to use for calculating perplexity ('default' for perplexity_of_fixedlength_models or 'per_user' for perplexity_of_fixedlength_models_per_user)")

    args = parser.parse_args()

    # Use the provided arguments
    model_id = args.model_id
    device = args.device
    data_path = args.data_path
    index_path = args.index_path
    output_path = args.output_path
    perplexity_func = args.perplexity_func


    # Load the model and tokenizer with device map
    if "Llama" in model_id:
        model = LlamaForCausalLM.from_pretrained(model_id, device_map="auto")
        tokenizer = LlamaTokenizerFast.from_pretrained(model_id)
        start_of_sentence='<|begin_of_text|>'
    if "gpt2" in model_id:
        model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
        tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
        start_of_sentence =" "

    print(f"Loaded model: {model_id}")
    # Load dataset from the specified data_path
    dataset = pd.read_csv(data_path)
    indices = pd.read_csv(index_path)["conversation_id"]

    filtered_d = dataset[dataset["file_name"].isin(indices)]

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    tot_n = len(filtered_d['file_content'])
    pattern='<(SPK[0-9]|MOD)>'
    for idx, (dialog, path) in enumerate(zip(filtered_d["file_content"], filtered_d["file_name"])):
        ppl = {}
        if os.path.exists(f"{output_path}/dominance_scores_{path}.pkl"):
            print(f"skipping {path} ...")
            continue
        print(f"{idx}/{tot_n}")
        print(f"processing file {path}")
        pattern = r'<(?:SPK[0-9]|MOD)>'
        dialog_lines = re.sub(r"[\[\(].*?[\]\)]", "", dialog).replace("<", "\n<").split("\n")[1:]
        matches = [f"{start_of_sentence}{match} " for match in re.findall(pattern, f"{start_of_sentence}".join(dialog_lines))]
        token_list = [tokenizer(token, return_tensors="pt").input_ids[0] for token in dialog_lines]
        encodings = tokenizer(f"{start_of_sentence}".join(dialog_lines), return_tensors="pt")
        
        if perplexity_func == "p1":
            perpl = compute_p1(encodings, token_list, tokenizer, model, device, start_of_sentence=start_of_sentence, max_length = 100, pattern=pattern, debug=False)
        elif perplexity_func == "p2":  # Assuming 'per_user' is the only other option
            perpl = compute_p2(encodings, token_list, tokenizer, model, device, start_of_sentence=start_of_sentence, max_length = 100, pattern=pattern, debug=False)
        elif perplexity_func == "p3":
            perpl = compute_p3(encodings, token_list, tokenizer, model, device, start_of_sentence=start_of_sentence, max_length = 100, pattern=pattern, debug=False)
        else:
            print(f"Error: {perplexity_func}, not a known perplexity type") 
        ppl[path] = perpl
        #assert len(ppl[path]) == tokenizer(f"{start_of_sentence}".join(dialog_lines), return_tensors="pt", return_offsets_mapping=True).input_ids.size(1)
        #assert len(matches) == len(dialog_lines)
        with open(f"{output_path}/dominance_scores_{path}.pkl", 'wb') as file:
            pickle.dump(ppl, file)
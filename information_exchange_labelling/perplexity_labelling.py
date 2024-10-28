
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from torch.nn.functional import pad

from transformers import LlamaForCausalLM, LlamaTokenizerFast

def p1(dialog, start_of_sentence=" "):
    
    #max_length = model.config.n_positions
    #max_length =  model.config.max_position_embeddings
    max_length = 500
    stride = 1
    
    pad_token_id = 0
    encodings = tokenizer(f"{start_of_sentence}".join(dialog), return_tensors="pt")
    seq_len = encodings.input_ids.size(1)
    padding_len = max_length -1 
    padded_input_ids = pad(torch.tensor([], dtype=torch.long), (0, padding_len), value=pad_token_id).unsqueeze(dim=0)
    encodings.input_ids = torch.cat([padded_input_ids, encodings.input_ids], dim=1)
    seq_len = encodings.input_ids.size(1)
    
    nlls = []
    prev_end_loc = padding_len
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from max_length on the last loop 
        begin_loc = max(padding_len, begin_loc)
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood.item())

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    return nlls

def p2(dialog, start_of_sentence=" "):
    
    #max_length = model.config.n_positions
    max_length =  model.config.max_position_embeddings
    stride = 1
    #encodings = tokenizer(f"{start_of_sentence}".join(dialog), return_tensors="pt")
    tokens = [tokenizer(token, return_tensors="pt", return_offsets_mapping=True).input_ids[0] for token in dialog]
    encodings = tokenizer(f"{start_of_sentence}".join(dialog), return_tensors="pt")
    
    tokens_ids_per_sentence = np.cumsum([t.size(0) for t in tokens])
    assert tokens_ids_per_sentence[-1] == encodings.input_ids.size(1)
    pad_token_id = 0    
    padding_len = max_length -1 
    
    tokens_ids_per_sentence+=padding_len
    padded_input_ids = pad(torch.tensor([], dtype=torch.long), (0, padding_len), value=pad_token_id).unsqueeze(dim=0)
    encodings.input_ids = torch.cat([padded_input_ids, encodings.input_ids], dim=1)
    seq_len = encodings.input_ids.size(1)
    #seq_len_tter = encodings.input_ids.size(1)
    seq_len_tter = tokens_ids_per_sentence 
    nlls = []
    
    begin_loc_win=padding_len
    i=0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        begin_loc = max(padding_len, begin_loc)
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-begin_loc_win] = -100
        #print([tokenizer.decode(token, skip_special_tokens=True) for token in encodings.input_ids[:, begin_loc_win:end_loc]])
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood.item())
        if seq_len_tter[i] == end_loc:
            begin_loc_win = seq_len_tter[i]
            i+=1
        if end_loc == seq_len:
            break
    
    return nlls

def p3(dialog, start_of_sentence=" "):
    
    #max_length = model.config.n_positions
    max_length =  model.config.max_position_embeddings
    stride = 1
    
    pad_token_id = 0
    encodings = tokenizer(f"{start_of_sentence}".join(dialog), return_tensors="pt")
    encoding_nxt_line = tokenizer("\n", return_tensors="pt").input_ids.to(device)
    seq_len = encodings.input_ids.size(1)
    padding_len = max_length -1 
    padded_input_ids = pad(torch.tensor([], dtype=torch.long), (0, padding_len), value=pad_token_id).unsqueeze(dim=0)
    encodings.input_ids = torch.cat([padded_input_ids, encodings.input_ids], dim=1)
    seq_len = encodings.input_ids.size(1)
    
    nlls = []
    prev_end_loc = padding_len
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc # may be different from max_length on the last loop 
        begin_loc = max(padding_len, begin_loc)
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        input_ids_xt = torch.cat([input_ids, encoding_nxt_line], dim=1)
        target_ids = input_ids_xt.clone()
        target_ids[:, :-trg_len] = -100
        with torch.no_grad():
            outputs = model(input_ids_xt, labels=target_ids)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood.item())

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    return nlls

import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import torch

nltk.download('punkt')
nltk.download('stopwords')
STOP_WORDS = set(stopwords.words('english'))

# Assuming 'matches', 'dialog', 'offset', and 'perpl' are defined earlier in your code
def perplexity_to_info(dialog, tokens, perpl, answers, pattern = '<(SPK[1-9]|MOD)>'):

    matches = re.findall(pattern, "".join(dialog))
    unique_matches = np.unique(matches)
    #encodings = tokenizer(f"{start_of_sentence}".join(dialog), return_tensors="pt")
    encodings = torch.cat(tokens)
    tokens_ids_per_sentence = np.cumsum([t.size(0) for t in tokens])
    assert tokens_ids_per_sentence[-1] == len(encodings)
    assert len(encodings) == len(perpl)
    
    ppl_to_info = []
    prev_idx_pp = 0
    for idx, (match,answer) in enumerate(zip(matches,answers)):
        idx_pp = tokens_ids_per_sentence[idx]
        patt = matches[idx]
        label = re.sub(r'\[([^\]]+)\]: ', '', dialog[idx])
        tokens = encodings[prev_idx_pp:idx_pp]
        decoded = [tokenizer.decode([token], skip_special_tokens=True) for token in tokens]
        perpl_per_sent = perpl[prev_idx_pp:idx_pp]
        mean_value=np.nanmean(np.asarray(perpl_per_sent))
        prev_idx_pp=idx_pp
        ppl_to_info.append({"label":answer, "perpl": np.asarray(perpl_per_sent)})
    
    return ppl_to_info

def filter_out_common_words(words, perpl):
    # Remove stop words and corresponding perplexity values
    filtered_words = [word for word in words if word.lower().strip() not in STOP_WORDS]
    filtered_word_indices = [i for i, word in enumerate(words) if word.lower().strip() not in STOP_WORDS]
    assert len(list(np.asarray(perpl)[filtered_word_indices])) == len(filtered_words)
    return list(np.asarray(perpl)[filtered_word_indices]), filtered_word_indices

def perplexity_to_info_filtering_out_common_words(dialog, tokens, perpl, answers, matches, pattern = '<(SPK[1-9]|MOD)>'):
    
    encodings = torch.cat(tokens)
    unique_matches = np.unique(matches)
    tokens_ids_per_sentence = np.cumsum([t.size(0) for t in tokens])
    assert tokens_ids_per_sentence[-1] == len(encodings)
    assert len(encodings) == len(perpl)
    
    
    ppl_to_info = []
    prev_idx_pp = 0
    dialog_filtered = []
    for idx, (match,answer) in enumerate(zip(matches,answers)):
        idx_pp = tokens_ids_per_sentence[idx]
        patt = matches[idx]
        label = re.sub(pattern, '', dialog[idx])
        tokens = encodings[prev_idx_pp:idx_pp]
        decoded = [tokenizer.decode([token], skip_special_tokens=True) for token in tokens]
        assert len(decoded) == len(perpl[prev_idx_pp:idx_pp])
        perpl_per_sent, filtered_word_indices = filter_out_common_words(decoded, perpl[prev_idx_pp:idx_pp])
        assert len(perpl_per_sent) == len(tokens[filtered_word_indices])
        dialog_filtered.append(tokens[filtered_word_indices])
        mean_value=np.nanmean(np.asarray(perpl_per_sent))
        prev_idx_pp=idx_pp
        ppl_to_info.append({"label":answer, "perpl": np.asarray(perpl_per_sent)})
        
    return dialog_filtered, ppl_to_info

def compute_per_user_mean_perplexity_filtering_out_common_words(dialog, tokens, perpl, matches, pattern = '<(SPK[1-9]|MOD)>'):

    encodings = torch.cat(tokens)
    unique_matches = np.unique(matches)
    tokens_ids_per_sentence = np.cumsum([t.size(0) for t in tokens])
    assert tokens_ids_per_sentence[-1] == len(encodings)
    assert len(encodings) == len(perpl)
    
    prev_idx_pp = 0
    user_to_ppl = {}
    decoded_utterances = []
    for idx, match in enumerate(matches):
        idx_pp = tokens_ids_per_sentence[idx]
        patt = matches[idx]
        label = re.sub(pattern, '', dialog[idx])
        tokens = encodings[prev_idx_pp:idx_pp]
        decoded = [tokenizer.decode([token], skip_special_tokens=True) for token in tokens]
        assert len(decoded) == len(perpl[prev_idx_pp:idx_pp])
        perpl_per_sent, filtered_word_indices = filter_out_common_words(decoded, perpl[prev_idx_pp:idx_pp])
        decoded_utterances.append([tokenizer.decode([token], skip_special_tokens=True) for token in tokens[filtered_word_indices]])
        assert len(perpl_per_sent) == len(tokens[filtered_word_indices])
        mean_value=np.nanmean(np.asarray(perpl_per_sent))
        prev_idx_pp=idx_pp
        if patt not in user_to_ppl:
            user_to_ppl[patt] = []
        user_to_ppl[patt].append(mean_value)
        
    return user_to_ppl, decoded_utterances

def compute_per_user_mean_perplexity(dialog, tokens, perpl, matches, pattern = '<(SPK[1-9]|MOD)>'):
    
    encodings = torch.cat(tokens)
    #matches = re.findall(pattern, "".join(dialog))
    unique_matches = np.unique(matches)
    tokens_ids_per_sentence = np.cumsum([t.size(0) for t in tokens])
    assert tokens_ids_per_sentence[-1] == len(encodings)
    assert len(encodings) == len(perpl)


    prev_idx_pp = 0
    user_to_ppl = {}
    for idx in range(len(matches)):
        idx_pp = tokens_ids_per_sentence[idx]
        patt = matches[idx]
        tokens = encodings[prev_idx_pp:idx_pp]
        decoded = [tokenizer.decode([token], skip_special_tokens=True) for token in tokens]
        perpl_per_sent = perpl[prev_idx_pp:idx_pp]
        mean_value=np.nanmean(np.asarray(perpl_per_sent))
        if patt not in user_to_ppl:
            user_to_ppl[patt] = []
        user_to_ppl[patt].append(mean_value)
        
    return  user_to_ppl

def create_mask(arr):
    transformed_arr = []
    for num in arr:
        if num == 0:
            transformed_arr.append(0)
        else:
            transformed_arr.extend([1] * num)
    return np.asarray(transformed_arr, dtype=np.int64).cumsum() - 1

import warnings

def compute_per_utterance_mean_perplexity(dialog, tokens, perpl, matches, pattern = '<(SPK[1-9]|MOD)>'):
    
    masked_speakers = []
    encodings = torch.cat(tokens)    
    tokens_ids_per_sentence = np.cumsum([t.size(0) for t in tokens])
    assert tokens_ids_per_sentence[-1] == len(encodings)
    assert len(encodings) == len(perpl)
    
    warnings.filterwarnings("ignore", message="Mean of empty slice")

    pos_nxt_speaker_token = np.asarray([0] + [len(tokenizer(d, return_tensors="pt").input_ids[0]) for d in dialog])
    pos_nxt_speaker_token_cumsum = pos_nxt_speaker_token.cumsum()
    bin_edges = np.concatenate([[0], tokens_ids_per_sentence])
    idx_bin = np.digitize(pos_nxt_speaker_token_cumsum, bin_edges) - 1

    assert len(idx_bin) - 1 == len(dialog)
    assert len(pos_nxt_speaker_token_cumsum) - 1 == len(dialog)

    prev_idx_pp = 0
    utterance_to_ppl = []
    current_sum = 0
    nxs_spk = []
    substring_ppl_ls = []
    utterance_to_ppl_sent = []
    
    for idx in range(len(tokens)):
        idxs_nxt_speaker_token = pos_nxt_speaker_token_cumsum[np.argwhere(np.isin(idx_bin, idx)).flatten()] - prev_idx_pp
        idx_pp = tokens_ids_per_sentence[idx]
        encodings_ = encodings[prev_idx_pp:idx_pp]
        perpl_per_sent = perpl[prev_idx_pp:idx_pp]
        substring_ppl = [np.nanmean(perpl_per_sent[start:end]) for start, end in zip([0] + list(idxs_nxt_speaker_token), list(idxs_nxt_speaker_token) + [None])]
        substrings_detoken = [encodings_[start:end] for start, end in zip([0] + list(idxs_nxt_speaker_token), list(idxs_nxt_speaker_token) + [None])] 
        substring_ppl_ls = [substring_ppl[num] for num, s in enumerate(substrings_detoken) if s.numel() != 0]
        utterance_to_ppl.append(substring_ppl_ls)     
        utterance_to_ppl_sent.append([tokenizer.decode(token, skip_special_tokens=True) for token in substrings_detoken])
        prev_idx_pp = idx_pp
        
        if len(idxs_nxt_speaker_token)>0 and len(substring_ppl_ls) != len(idxs_nxt_speaker_token):
            nxs_spk.append(0)
        nxs_spk.append(len(idxs_nxt_speaker_token))

    return  utterance_to_ppl, utterance_to_ppl_sent, create_mask(nxs_spk)
    
def compute_graph_perplexity(tokens, p1, p2, matches, pattern = '<(SPK[1-9]|MOD)>', answers=None):
    
    dialog = [tokenizer.decode(token, skip_special_tokens=True) for token in tokens]
    unique_matches = np.unique(matches)
    
    rows = int(np.ceil(np.sqrt(len(dialog))))
    # Create an 8x8 grid of subplots
    fig, axes = plt.subplots(rows, rows, figsize=(30, 30))
    num_plots = len(dialog)
    # Set smaller font size
    plt.rcParams.update({'font.size': 8})
    
    assert num_plots == len(matches)
    encodings = torch.cat(tokens)
    tokens_ids_per_sentence = np.cumsum([t.size(0) for t in tokens])
    assert tokens_ids_per_sentence[-1] == len(p1)
    
    prev_idx_pp = 0
    for idx, ax in enumerate(axes.flatten()):
        if idx < num_plots:
            idx_pp = tokens_ids_per_sentence[idx]
            patt = matches[idx]
            tokens = encodings[prev_idx_pp:idx_pp]
            decoded = [tokenizer.decode([token], skip_special_tokens=True) for token in tokens]
            p1_per_sent = p1[prev_idx_pp:idx_pp]
            p2_per_sent = p2[prev_idx_pp:idx_pp]
            ax.plot(np.asarray(p1_per_sent), label=f'{patt} p1')
            ax.plot(np.asarray(p2_per_sent), label=f'{patt} p2', color='r')
            #mean_value=np.nanmean(np.asarray(perpl_per_sent))
            ax.axhline(p1_per_sent[0], color='g', label=f'{patt} p3')  # Fixed the color argument
            ax.set_xticks(np.arange(len(decoded)))
            ax.set_xticklabels(decoded, rotation=90)
            if answers is not None:
                ax.set_title(f'{answers[idx]}')
            ax.legend()
            prev_idx_pp=idx_pp

    # Hide any remaining empty subplots
    for ax in axes.flatten()[num_plots:]:
        ax.axis('off')
        
    plt.subplots_adjust(hspace=0.5, top=0.95)  
    plt.suptitle("Per-Word Perplexity across the Dataset", fontsize=30)
    plt.legend()
    plt.show()

from tqdm import tqdm

def create_extended_list(lst, arr):
    result = []
    prev_count = 0
    for i, count in enumerate(arr):
        result.extend([lst[i]] * (count - prev_count))
        prev_count = count
    return result

def compute_p3(candor_df, perplexity_scores_p1):
    perplexity_scores_p3 = {}
    for content, name in tqdm(zip(candor_df["file_content"], candor_df["file_name"]), total=len(candor_df)):
        try:
            pattern = '<(SPK[0-9]|MOD)>'
            matches = re.findall(pattern, "".join(content))
            content = re.sub(r'\<', r'\n<', content).split("\n")[1:]
            content = [re.sub(pattern,">", d) for d in content]
            assert len(perplexity_scores_p1[name]) == tokenizer(f"{start_of_sentence}".join(content), return_tensors="pt", return_offsets_mapping=True).input_ids.size(1)
            tokens = [tokenizer(token, return_tensors="pt", return_offsets_mapping=True).input_ids[0] for token in content]
            assert len(torch.cat(tokens)) == len(perplexity_scores_p1[name])
            tokens_ids_per_sentence = np.cumsum([t.size(0) for t in tokens])
            value = np.asarray([0] + list(np.asarray(perplexity_scores_p1[name])[tokens_ids_per_sentence[:-1]]))
            #tokens_value = np.asarray(torch.cat(tokens))[tokens_ids_per_sentence[:-1]]
            perplexity_scores_p3[name] = create_extended_list(value, tokens_ids_per_sentence)
            assert len(perplexity_scores_p3[name]) == tokenizer(f"{start_of_sentence}".join(content), return_tensors="pt", return_offsets_mapping=True).input_ids.size(1)
        except:
            print(f'missing {name}')

    return perplexity_scores_p3


from transformers import GPT2LMHeadModel, GPT2TokenizerFast

import argparse
import pickle
import os

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
for idx, (el, path) in enumerate(zip(filtered_d["file_content"], filtered_d["file_name"])):
    ppl = {}
    if os.path.exists(f"{output_path}/dominance_scores_{path}.pkl"):
        print(f"skipping {path} ...")
        continue
    print(f"{idx}/{tot_n}")
    print(f"processing file {path}")
    dialog = re.sub(r'\<', r'\n<', el).split("\n")[1:]
    matches = re.findall(pattern, "".join(dialog))
    dialog_no_s = [re.sub(pattern,">", d) for d in dialog]
    if perplexity_func == "p1":
        perpl = p1(dialog_no_s, start_of_sentence=start_of_sentence)
    elif perplexity_func == "p2":  # Assuming 'per_user' is the only other option
        perpl = p2(dialog_no_s, start_of_sentence=start_of_sentence)
    elif perplexity_func == "p3":
        perpl = p3(dialog_no_s, start_of_sentence=start_of_sentence)
    else:
        print(f"Error: {perplexity_func}, not a known perplexity type") 
    ppl[path] = perpl
    assert len(ppl[path]) == tokenizer(f"{start_of_sentence}".join(dialog_no_s), return_tensors="pt", return_offsets_mapping=True).input_ids.size(1)
    assert len(matches) == len(dialog_no_s)
    with open(f"{output_path}/dominance_scores_{path}.pkl", 'wb') as file:
        pickle.dump(ppl, file)
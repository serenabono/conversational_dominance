
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from torch.nn.functional import pad

def p1(dialog, start_of_sentence=" "):
    
    max_length = model.config.n_positions
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
    
    max_length = model.config.n_positions
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
    
    max_length = model.config.n_positions
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

def compute_per_utterance_mean_perplexity(dialog, tokens, idx_bin, perpl, matches, pattern = '<(SPK[1-9]|MOD)>'):
     
    encodings = torch.cat(tokens)    
    tokens_ids_per_sentence = np.cumsum([t.size(0) for t in tokens])
    assert tokens_ids_per_sentence[-1] == len(encodings)
    assert len(encodings) == len(perpl)

    pos_nxt_speaker_token = np.asarray([0] + [len(tokenizer(d, return_tensors="pt").input_ids[0]) for d in dialog])
    pos_nxt_speaker_token_cumsum = pos_nxt_speaker_token.cumsum()

    per_utterance_ppl = [perpl[pos_nxt_speaker_token_cumsum[i-1]:pos_nxt_speaker_token_cumsum[i]] if i < len(pos_nxt_speaker_token_cumsum) else perplexity_scores_p1[name][pos_nxt_speaker_token_cumsum[i]:] for i in range(1,len(pos_nxt_speaker_token_cumsum))]

    #checking everything is correct
    per_utterance_ppl_len = np.asarray([len(d) for d in per_utterance_ppl])
    per_utterance_len = pos_nxt_speaker_token[1:]
    assert all(per_utterance_ppl_len) == all(per_utterance_len)

    #creating per-utterance bin
    per_utterance_bin = [idx_bin[pos_nxt_speaker_token_cumsum[i-1]:pos_nxt_speaker_token_cumsum[i]] if i < len(pos_nxt_speaker_token_cumsum) else idx_bin[pos_nxt_speaker_token_cumsum[i]:] for i in range(1,len(pos_nxt_speaker_token_cumsum))]

    return  per_utterance_ppl, per_utterance_bin
    
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

def compute_perplexity_scores_p3(dataset, perplexity_scores_p1):
    perplexity_scores_p3 = {}
    for content, name in tqdm(zip(dataset["file_content"], dataset["file_name"]), total=len(dataset)):
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

# load perplexity
import os
import pickle

def compute_perplexity_scores_p1(path_base):
    perplexity_scores_list = []
    print("Computing perplexity scores p1")
    for root, dirs, files in os.walk(path_base):
        for file in files:
            if file.endswith('.pkl'):
                file_path = os.path.join(root, file)
                with open(file_path, 'rb') as f:
                    loaded = pickle.load(f)
                    perplexity_scores_list.append(loaded)

    perplexity_scores_p1 = {k:v for element in perplexity_scores_list for k,v in element.items()}
    return perplexity_scores_p1

def create_extended_list(lst, arr):
    result = []
    prev_count = 0
    for i, count in enumerate(arr):
        result.extend([lst[i]] * (count - prev_count))
        prev_count = count
    return result


def compute_bins(zero_pos, tokenizer, content, t_u, bin_width):
    len_per_token = [len(tokenizer(dialogue, return_tensors="pt", add_special_tokens=False).input_ids[0].detach().numpy()) for dialogue in content]
    tokens = [tokenizer(dialogue, return_tensors="pt").input_ids[0].detach().numpy() for dialogue in content]
    
    time_per_token = []

    for idx in range(len(content)):   
        if len(tokens[idx]) == 0:
            time_per_token.append(0)
        else:
            time_per_token.append(t_u[idx] / len(tokens[idx]))
    all_tokens = []
    for dialogue in content:
        all_tokens += list(tokenizer(dialogue, return_tensors="pt", add_special_tokens=False).input_ids[0].detach().numpy())
    
    assert len(all_tokens) == tokenizer(f" ".join(content), return_tensors="pt", add_special_tokens=False).input_ids.size(1)

    all_tokens = np.asarray(all_tokens)
    assert len(all_tokens) == np.asarray(len_per_token).sum()
    assert len(len_per_token) == len(time_per_token)

    elements = [(count, value) for count, value in zip(len_per_token, time_per_token)]
    arr = sum([[val] * num for num, val in elements], [])
    data = np.asarray(arr).cumsum()

    bin_edges = np.arange(0, np.ceil(data.max()) + bin_width, bin_width)
    idx_bin = np.digitize(data, bin_edges) + int(zero_pos // bin_width)
    assert len(all_tokens) == len(idx_bin)
    return idx_bin 


import glob 

def compute_dominance_ppl(tokenizer, dataset,labels_path, perplexity_scores, time_field, bin_width, start_of_sentence=" "):
    
    print("Computing dominance perplexity")
    dominance_p = {}
    dominance_p_decoded_sent = {}
    list_skps_per_utterance = {}

    for content, name in tqdm(zip(dataset["file_content"], dataset["file_name"]), total=len(dataset)):
        if name not in perplexity_scores:
            continue
      
        filepath = glob.os.path.join(labels_path, name) + ".csv"
        d = pd.read_csv(filepath)
        d_to_t = [d[time_field["START"]][i] - d[time_field["START"]][i-1] for i in range(1, len(d[time_field["START"]]))]+[np.asarray(d[time_field["STOP"]])[-1] - np.asarray(d[time_field["START"]])[-1]]
        pattern = '<(SPK[0-9]|MOD)>'
        matches = np.asarray(re.findall(pattern, "".join(content)))
        content = re.sub(r'\<', r'\n<', content).split("\n")[1:]
        content = [re.sub(pattern,">", d) for d in content]
        assert len(content) == len(matches)
        try:
            assert len(content) == len(d_to_t)
        except:
            continue
        assert len(matches) == len(content)
        idx_bin = compute_bins(d[time_field["START"]][0], tokenizer, content, d_to_t, bin_width)
        assert len(perplexity_scores[name]) == tokenizer(f"{start_of_sentence}".join(content), return_tensors="pt").input_ids.size(1)
        assert len(idx_bin) == tokenizer(f"{start_of_sentence}".join(content), return_tensors="pt").input_ids.size(1)

        tokens = [tokenizer(token, return_tensors="pt").input_ids[0] for token in content]
        
        per_utterance_ppl, per_utterance_bin = compute_per_utterance_mean_perplexity(content,tokens,idx_bin, perplexity_scores[name],matches)
        dominance_p[name] = {}
        list_skps_per_utterance[name] = {}
        dominance_p_decoded_sent[name] = {}
        for utterance_id in range(len(per_utterance_bin)):
            spk = matches[utterance_id]
            for token in range(len(per_utterance_bin[utterance_id])):
                if per_utterance_bin[utterance_id][token] not in dominance_p[name]:
                    list_skps_per_utterance[name][per_utterance_bin[utterance_id][token]] = []
                    dominance_p[name][per_utterance_bin[utterance_id][token]] = {}
                    dominance_p_decoded_sent[name][per_utterance_bin[utterance_id][token]] = {}
                if spk not in dominance_p[name][per_utterance_bin[utterance_id][token]]:
                    dominance_p[name][per_utterance_bin[utterance_id][token]][spk] = []
                    dominance_p_decoded_sent[name][per_utterance_bin[utterance_id][token]][spk] = []
                dominance_p[name][per_utterance_bin[utterance_id][token]][spk].append(per_utterance_ppl[utterance_id][token])
                dominance_p_decoded_sent[name][per_utterance_bin[utterance_id][token]][spk].append(tokens[utterance_id][token])
                list_skps_per_utterance[name][per_utterance_bin[utterance_id][token]].append(spk)
        for utterance_id in range(len(per_utterance_bin)):
            for token in range(len(per_utterance_bin[utterance_id])):
                for spk in dominance_p[name][per_utterance_bin[utterance_id][token]]:
                    dominance_p[name][per_utterance_bin[utterance_id][token]][spk] = np.mean(dominance_p[name][per_utterance_bin[utterance_id][token]][spk])
        
        df = pd.DataFrame.from_dict(dominance_p[name], orient='index')
        complete_index = range(0, df.index.max() + 1)
        complete_df = df.reindex(complete_index, fill_value=None)
        dominance_p[name] = complete_df.to_dict(orient='index')
        
        df_dominance_p_decoded_sent = pd.DataFrame.from_dict(dominance_p_decoded_sent[name], orient='index')
        complete_index = range(0, df_dominance_p_decoded_sent.index.max() + 1)
        complete_df_dominance_p_decoded_sent = df_dominance_p_decoded_sent.reindex(complete_index, fill_value=[torch.tensor(220)])
        dominance_p_decoded_sent[name] = complete_df_dominance_p_decoded_sent.to_dict(orient='index')
        
        assert len(list(dominance_p[name].keys())) == len(dominance_p_decoded_sent[name].keys())
        assert len(list(dominance_p[name].keys())) == per_utterance_bin[-1][-1] + 1 
        assert len(list(dominance_p[name].keys())) in range(int(np.ceil(d["stop"].tolist()[-1])) + 1 - 1, int(np.ceil(d["stop"].tolist()[-1])) + 1 + 1)

        
    return dominance_p,dominance_p_decoded_sent,list_skps_per_utterance

from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers import LlamaForCausalLM, LlamaTokenizerFast

import argparse
import pickle
import os
import json

# Define argparse arguments
parser = argparse.ArgumentParser(description="Compute perplexity and generate graphs for dialogue transcripts")
parser.add_argument("--model_id", type=str, default="gpt2-large", help="ID of the GPT-2 model to use")
parser.add_argument("--device", type=str, default="cuda:1", help="Device to run the model on (e.g., 'cuda:0' for GPU or 'cpu' for CPU)")
parser.add_argument("--data_path", type=str, default="/u/sebono/conversational_dominance/data/processed/topical/conversations.csv", help="Path to the CSV file containing the dialogue transcripts")
parser.add_argument("--output_path", type=str, default="/u/sebono/conversational_dominance/notebooks/dominance_scores_topical.pkl", help="Path to save the output pickle file")
parser.add_argument("--ppl_path", type=str, default="u/sebono/conversational_dominance/notebooks/information_exchange_labelling/dataset_perplexity_results/CANDOR_p1", help="Path to load perplexity of dataset")
parser.add_argument("--perplexity_func", type=str, default="default", choices=["p1", "p2","p3"], 
                    help="Function to use for calculating perplexity ('default' for perplexity_of_fixedlength_models or 'per_user' for perplexity_of_fixedlength_models_per_user)")
parser.add_argument("--time_field", type=str, default="start", help="the name of the time elapsed in the dataset dataframe")
parser.add_argument("--labels_path", type=str, default="start", help="affective labels path")
parser.add_argument("--bin_width", type=str, default="start", help="affective labels frequency(s)")
args = parser.parse_args()

# Use the provided arguments
model_id = args.model_id
device = args.device
data_path = args.data_path
ppl_path = args.ppl_path
output_path = args.output_path
perplexity_func = args.perplexity_func
time_field = json.loads(args.time_field)
labels_path= args.labels_path
bin_width= int(args.bin_width)

# Load the model and tokenizer with device map
if "Llama" in model_id:
    model = LlamaForCausalLM.from_pretrained(model_id, device_map="auto")
    tokenizer = LlamaTokenizerFast.from_pretrained(model_id)
    start_of_sentence='<|begin_of_text|>'
if "gpt2" in model_id:
    model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
    start_of_sentence=" "

print(f"Loaded model: {model_id}")
dataset = pd.read_csv(data_path)
if not os.path.exists(output_path):
    os.makedirs(output_path)  

    
if perplexity_func == "p1":
    p1_file_path = os.path.join(output_path, 'perplexity_scores_p1.pkl')
    # Compute Perplexity
    if os.path.exists(p1_file_path):
        with open(p1_file_path, 'rb') as file:
            perplexity_scores_p1 = pickle.load(file)
    else:
        perplexity_scores_p1 = compute_perplexity_scores_p1(ppl_path)
    with open(p1_file_path, 'wb') as file:
        pickle.dump(perplexity_scores_p1, file)
    
    # Compute Dominance
    if not os.path.exists(f'{output_path}/dominance_p1.pkl') or not os.path.exists(f'{output_path}/list_skps_per_utterance_p1.pkl'):
        dominance_p1,dominance_p_decoded_sent_p1,list_skps_per_utterance = compute_dominance_ppl(tokenizer, dataset,labels_path, perplexity_scores_p1, time_field, bin_width, start_of_sentence=start_of_sentence)
    
    with open(f'{output_path}/dominance_p1.pkl', 'wb') as f:
        pickle.dump(dominance_p1, f)
    with open(f'{output_path}/list_skps_per_utterance_p1.pkl', 'wb') as f:
        pickle.dump(list_skps_per_utterance, f)

if perplexity_func == "p3":
    # Compute Perplexity
    p3_file_path = os.path.join(output_path, 'perplexity_scores_p3.pkl')
    if os.path.exists(p3_file_path):
        with open(p3_file_path, 'rb') as file:
            perplexity_scores_p3 = pickle.load(file)
    else:
        perplexity_scores_p3 = compute_perplexity_scores_p3(ppl_path)
    with open(p3_file_path, 'wb') as file:
        pickle.dump(perplexity_scores_p3, file)
    
    # Compute Dominance
    dominance_p3,dominance_p_decoded_sent_p3,list_skps_per_utterance = compute_dominance_ppl(tokenizer, dataset,labels_path, perplexity_scores_p3, time_field, bin_width)
    with open(f'{output_path}/dominance_p3.pkl', 'wb') as f:
        pickle.dump(dominance_p3, f)
    with open(f'{output_path}/list_skps_per_utterance_p3.pkl', 'wb') as f:
        pickle.dump(list_skps_per_utterance, f)

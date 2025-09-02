from transformers import PreTrainedTokenizer, BatchEncoding
from typing import List, Dict, Union
import numpy as np
import torch

def check_token_mapping(shuffled_dialog):
    orig_enc = shuffled_dialog["original_tokens"]
    shuf_enc = shuffled_dialog["shuffled_tokens"]
    token_map = shuffled_dialog["original_to_shuffled_token_map"]

    orig_ids = orig_enc["input_ids"][0]
    shuf_ids = shuf_enc["input_ids"][0]

    # 1. Lengths must match
    assert orig_ids.size(0) == shuf_ids.size(0), "Token length mismatch"

    # 2. All indices should be mapped exactly once
    assert set(token_map.keys()) == set(range(len(orig_ids))), "Not all original tokens mapped"
    assert set(token_map.values()) == set(range(len(shuf_ids))), "Not all shuffled tokens reached"

    # 3. Check if remapping reproduces the shuffled sequence
    remapped = torch.tensor([shuf_ids[token_map[i]] for i in range(len(orig_ids))])
    assert torch.equal(remapped, orig_ids), "Remapped shuffled tokens do not match original sequence"
    print("✅ Token mapping check passed.")

    
def get_shuffled_dialog_data(
    dialog_lines: List[str],
    tokenizer: PreTrainedTokenizer,
    pattern: str = r"\b[B|E]\b",
    start_of_sentence: str = " ",
    by_turn: bool = True,
    seed: int = 42,
) -> Dict[str, Union[List, Dict, BatchEncoding]]:
    """
    Shuffle a dialog by turn or token while keeping a flat input_ids list aligned with the full encoding.
    Returns both per-turn and flat representations for original and shuffled dialogs, and a mapping from original to shuffled token indices.
    """
    rng = np.random.default_rng(seed)
    pad_token_id = tokenizer.pad_token_id or 0

    # Tokenize each utterance into a list of tensors
    tokenized_turns = [tokenizer(line, return_tensors="pt")["input_ids"][0] for line in dialog_lines]
    turn_lens = [len(t) for t in tokenized_turns]
    total_tokens = sum(turn_lens)

    # Flatten original input_ids
    original_input_ids = torch.cat(tokenized_turns, dim=0)
    original_attention_mask = torch.ones_like(original_input_ids)

    # Full dialog encoding for validation
    full_encoding = tokenizer(start_of_sentence.join(dialog_lines), return_tensors="pt")
    assert full_encoding.input_ids.size(1) == total_tokens, "Mismatch with tokenizer full encoding"

    # Offsets: original turn → token indices
    if by_turn:
        shuffled_indices = rng.permutation(len(dialog_lines))
        shuffled_turns = [tokenized_turns[i] for i in shuffled_indices]
        shuffled_dialog = [dialog_lines[i] for i in shuffled_indices]
        
        original_offsets = [0]
        for t in tokenized_turns:
            original_offsets.append(original_offsets[-1] + len(t))

        # Build token-level mapping
        mapping = {}
        flat_idx = 0
        for shuffled_pos, orig_turn_idx in enumerate(shuffled_indices):
            start = original_offsets[orig_turn_idx]
            end = original_offsets[orig_turn_idx + 1]
            for orig_tok_idx in range(start, end):
                mapping[orig_tok_idx] = flat_idx
                flat_idx += 1
    else:
        # Shuffle tokens globally
        shuffled_indices = rng.permutation(total_tokens)
        shuffled_input_ids = original_input_ids[shuffled_indices]

        # Reconstruct shuffled turns by original lengths
        shuffled_turns = []
        cursor = 0
        for length in turn_lens:
            chunk = shuffled_input_ids[cursor:cursor + length]
            shuffled_turns.append(chunk)
            cursor += length
        shuffled_dialog = [tokenizer.decode(t, skip_special_tokens=True) for t in shuffled_turns]

        # Mapping: original token index → new shuffled index
        mapping = {int(orig): int(new) for new, orig in enumerate(shuffled_indices)}

    # Flatten shuffled turns
    flat_shuffled_input_ids = torch.cat(shuffled_turns, dim=0)
    shuffled_attention_mask = torch.ones_like(flat_shuffled_input_ids)

    # Final consistency checks
    assert flat_shuffled_input_ids.numel() == total_tokens
    assert original_input_ids.numel() == total_tokens

    result = {
        "original_dialog": dialog_lines,
        "shuffled_dialog": shuffled_dialog,
        "original_tokens": BatchEncoding({
            "input_ids": original_input_ids.unsqueeze(0),
            "attention_mask": original_attention_mask.unsqueeze(0),
        }),
        "shuffled_tokens": BatchEncoding({
            "input_ids": flat_shuffled_input_ids.unsqueeze(0),
            "attention_mask": shuffled_attention_mask.unsqueeze(0),
        }),
        "original_per_turn_tokens": tokenized_turns,
        "shuffled_per_turn_tokens": shuffled_turns,
        "original_to_shuffled_token_map": mapping,
    }
    check_token_mapping(result)
    return result


import matplotlib.pyplot as plt

def plot_comparison(
    p_original,
    p_shuffled,
    tokenizer,
    original_token_list,
    shuffled_token_list,
    original_to_shuffled_token_map: dict,
    title="Impact of Token Shuffling on P1"
):
    """
    Plots token-level comparisons between original and shuffled scores using per-token mapping,
    and returns both the list of figures and reordered shuffled scores.
    
    Args:
        p_original (List[float]): Original token-level scores.
        p_shuffled (List[float]): Shuffled token-level scores.
        tokenizer: HF tokenizer used.
        original_token_list (List[Tensor]): List of token tensors per turn (original).
        shuffled_token_list (List[Tensor]): List of token tensors per turn (shuffled).
        original_to_shuffled_token_map (dict): Maps each original token idx → new shuffled idx.
        title (str): Title prefix.

    Returns:
        figures (List[matplotlib.figure.Figure])
        reordered_shuffled (List[float])
    """

    def compute_token_offsets(token_list):
        offsets = [0]
        for tokens in token_list:
            offsets.append(offsets[-1] + len(tokens))
        return offsets

    original_offsets = compute_token_offsets(original_token_list)
    figures = []
    reordered_shuffled = []

    flat_token_list = [tok for utt in original_token_list for tok in utt]

    for i, turn_tokens in enumerate(original_token_list):
        # Get P1s for the original turn
        start_o, end_o = original_offsets[i], original_offsets[i + 1]
        y_orig = p_original[start_o:end_o]

        # Retrieve shuffled P1s by mapping token indices
        y_shuff = [p_shuffled[original_to_shuffled_token_map[idx]] for idx in range(start_o, end_o)]

        # Decode tokens
        decoded_tokens = tokenizer.convert_ids_to_tokens(turn_tokens.tolist())
        decoded_tokens = [t.replace("Ġ", "").replace("▁", "") for t in decoded_tokens]
        assert len(decoded_tokens) == len(y_orig)
        assert len(y_shuff) == len(y_orig)

        reordered_shuffled.extend(y_shuff)

        # Plot
        x = list(range(len(decoded_tokens)))
        fig, ax = plt.subplots(figsize=(max(8, len(decoded_tokens) * 0.5), 3))
        ax.plot(x, y_orig, label="Original", color="blue", linewidth=2, marker='o')
        ax.plot(x, y_shuff, label="Shuffled", color="orange", linestyle="--", linewidth=2, marker='x')

        ax.set_xticks(x)
        ax.set_xticklabels(decoded_tokens, rotation=45, ha='right')
        ax.set_title(f"{title} — Turn {i}")
        ax.set_xlabel("Tokens")
        ax.set_ylabel("Perplexity")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        fig.tight_layout()
        plt.subplots_adjust(bottom=0.3)
        figures.append(fig)
        plt.close

    return figures, reordered_shuffled

def reorder_tokens(
    p_original,
    p_shuffled,
    tokenizer,
    original_token_list,
    shuffled_token_list,
    original_to_shuffled_token_map: dict,
    title="Impact of Token Shuffling on P1"
):
    """
    Plots token-level comparisons between original and shuffled scores using per-token mapping,
    and returns both the list of figures and reordered shuffled scores.
    
    Args:
        p_original (List[float]): Original token-level scores.
        p_shuffled (List[float]): Shuffled token-level scores.
        tokenizer: HF tokenizer used.
        original_token_list (List[Tensor]): List of token tensors per turn (original).
        shuffled_token_list (List[Tensor]): List of token tensors per turn (shuffled).
        original_to_shuffled_token_map (dict): Maps each original token idx → new shuffled idx.
        title (str): Title prefix.

    Returns:
        figures (List[matplotlib.figure.Figure])
        reordered_shuffled (List[float])
    """

    def compute_token_offsets(token_list):
        offsets = [0]
        for tokens in token_list:
            offsets.append(offsets[-1] + len(tokens))
        return offsets

    original_offsets = compute_token_offsets(original_token_list)
    figures = []
    reordered_shuffled = []

    flat_token_list = [tok for utt in original_token_list for tok in utt]

    for i, turn_tokens in enumerate(original_token_list):
        # Get P1s for the original turn
        start_o, end_o = original_offsets[i], original_offsets[i + 1]
        y_orig = p_original[start_o:end_o]

        # Retrieve shuffled P1s by mapping token indices
        y_shuff = [p_shuffled[original_to_shuffled_token_map[idx]] for idx in range(start_o, end_o)]

        # Decode tokens
        decoded_tokens = tokenizer.convert_ids_to_tokens(turn_tokens.tolist())
        decoded_tokens = [t.replace("Ġ", "").replace("▁", "") for t in decoded_tokens]
        assert len(decoded_tokens) == len(y_orig)
        assert len(y_shuff) == len(y_orig)

        reordered_shuffled.extend(y_shuff)

    return reordered_shuffled

import re
import random
from typing import List, Dict, Union
from transformers import PreTrainedTokenizer, BatchEncoding
from torch.nn.utils.rnn import pad_sequence
import torch

def swap_speaker_tokens_with_metadata(
    dialog_lines: List[str],
    tokenizer: PreTrainedTokenizer,
    pattern: str = r"[A-Z]+",
    invert: bool = True,
    swap_percent: float = 0.10
) -> Dict[str, Union[List, BatchEncoding]]:
    """
    Swaps speaker tokens in a dialog and returns both per-turn and flattened tokenizations.
    """

    swapped_dialog = dialog_lines.copy()
    num_lines = len(dialog_lines)

    if invert:
        indices_to_swap = list(range(0, num_lines - 1, 2))
    else:
        max_pairs = (num_lines - 1) // 2
        num_turns_to_swap = max(1, int(num_lines * swap_percent))
        num_pairs_to_swap = min(max_pairs, num_turns_to_swap // 2)

        candidates = list(range(0, num_lines - 1))
        random.shuffle(candidates)

        used = set()
        indices_to_swap = []
        for idx in candidates:
            if idx in used or idx + 1 in used:
                continue
            indices_to_swap.append(idx)
            used.update({idx, idx + 1})
            if len(indices_to_swap) >= num_pairs_to_swap:
                break

    for i in indices_to_swap:
        m1 = re.match(pattern, dialog_lines[i])
        m2 = re.match(pattern, dialog_lines[i + 1])
        if m1 and m2:
            s1, s2 = m1.group(0), m2.group(0)
            b1 = re.sub(pattern, "", dialog_lines[i], count=1)
            b2 = re.sub(pattern, "", dialog_lines[i + 1], count=1)
            swapped_dialog[i] = f"{s2}{b1}"
            swapped_dialog[i + 1] = f"{s1}{b2}"

    # Tokenize each turn and store encoding per-turn (dicts)
    original_tokens_per_turn = [
        tokenizer(line, return_tensors="pt", padding=False, truncation=False)
        for line in dialog_lines
    ]
    swapped_tokens_per_turn = [
        tokenizer(line, return_tensors="pt", padding=False, truncation=False)
        for line in swapped_dialog
    ]

    # Flatten all input_ids and masks
    def flatten(enc_list: List[BatchEncoding]) -> BatchEncoding:
        input_ids = torch.cat([enc["input_ids"][0] for enc in enc_list], dim=0).unsqueeze(0)
        attention_mask = torch.cat([enc["attention_mask"][0] for enc in enc_list], dim=0).unsqueeze(0)
        return BatchEncoding({"input_ids": input_ids, "attention_mask": attention_mask})

    return {
        "original_dialog": dialog_lines,
        "swapped_dialog": swapped_dialog,
        "original_per_turn_tokens": [t.input_ids[0] for t in original_tokens_per_turn],
        "swapped_per_turn_tokens": [t.input_ids[0] for t in swapped_tokens_per_turn],
        "original_tokens": flatten(original_tokens_per_turn),
        "swapped_tokens": flatten(swapped_tokens_per_turn),
    }


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

def compute_per_utterance_mean_perplexity(tokenizer, dialog, tokens, idx_bin, perpl, matches, pattern = '<(SPK[1-9]|MOD)>', ):
     
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
    
def compute_graph_perplexity(tokenizer, tokens_list, p1, p2, matches, names, answers=None):
    dialog = [tokenizer.decode(token, skip_special_tokens=True) for token in tokens_list]
    
    # rows = int(np.ceil(np.sqrt(len(dialog))))
    rows = int(np.ceil(len(dialog)/6.0))
    # Create an 8x8 grid of subplots
    fig, axes = plt.subplots(rows, 6, figsize=(30, 15))
    num_plots = len(dialog)
    # Set smaller font size
    plt.rcParams.update({'font.size': 8})
    
    assert num_plots == len(matches)
    encodings = torch.cat(tokens_list)
    tokens_ids_per_sentence = np.cumsum([t.size(0) for t in tokens_list])
    assert tokens_ids_per_sentence[-1] == len(p1)
    
    # Compute global y-limits across all p1 and p2 values
    all_values = []
    prev_idx_pp = 0
    for idx in range(len(dialog)):
        idx_pp = tokens_ids_per_sentence[idx]
        p1_per_sent = p1[prev_idx_pp:idx_pp]
        p2_per_sent = p2[prev_idx_pp:idx_pp]
        all_values.extend(p1_per_sent)
        all_values.extend(p2_per_sent)
        prev_idx_pp = idx_pp

    # Filter out NaNs and compute global limits
    all_values = np.array(all_values)
    global_min = np.nanmin(all_values)
    global_max = np.nanmax(all_values)

    prev_idx_pp = 0
    for idx, ax in enumerate(axes.flatten()):
        if idx < num_plots:
            idx_pp = tokens_ids_per_sentence[idx]
            patt = matches[idx]
            sub_tokens = encodings[prev_idx_pp:idx_pp]
            decoded = [tokenizer.decode([token], skip_special_tokens=True) for token in sub_tokens]
            p1_per_sent = p1[prev_idx_pp:idx_pp]
            p2_per_sent = p2[prev_idx_pp:idx_pp]
            p1_name = names['p1']
            ax.plot(np.asarray(p1_per_sent), label=f'{patt} {p1_name}')
            #ax.plot(np.asarray(p2_per_sent), label=f'{patt} p2', color='r')
            #mean_value=np.nanmean(np.asarray(perpl_per_sent))
            # ax.axhline(p1_per_sent[0], color='g', label=f'{patt} p3')  # Fixed the color argument
            p2_name = names['p2']
            ax.plot(np.asarray(p2_per_sent), label=f'{patt} {p2_name}', color='g')
            ax.set_xticks(np.arange(len(decoded)))
            ax.set_xticklabels(decoded, rotation=90)
            if answers is not None:
                ax.set_title(f'{answers[idx]}')
            ax.legend()
            prev_idx_pp=idx_pp

            ax.set_ylim(global_min, global_max)

    # Hide any remaining empty subplots
    for ax in axes.flatten()[num_plots:]:
        ax.axis('off')
        
    plt.subplots_adjust(hspace=0.5, top=0.95)  
    plt.suptitle("Per-Word Perplexity across the Dataset", fontsize=30)
    plt.legend()
    plt.show()

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import plotly
import plotly.io as pio
pio.renderers.default = 'iframe'
import plotly.express as px
plotly.offline.init_notebook_mode(connected=True)
import seaborn as sns

cmp = 'algae'
def correlation_heatmap(y_cols, x_cols, full_data):
    '''
    Uses scipy.stats.spearmanr function
    Params:
    y_cols, x_cols: sets of column titles (strings)
    full_data: pandas dataframe that includes all columns listed in y_cols, x_cols
    Returns:
    corr: Spearman correlation coefficient matrix (y_cols = rows, x_cols = cols of matrix)
    fig_corr: annotated plotly heatmap of coefficients
    p: Spearman p-value matrix
    fig_p: annotated plotly heatmap of p-values
    '''
    cols = y_cols+x_cols
    all_correlations = scipy.stats.spearmanr(full_data[cols], nan_policy='omit')
    corr = all_correlations.correlation[:len(y_cols), -len(x_cols):]
    corr = pd.DataFrame(corr)
    corr.columns = x_cols
    corr.index = y_cols

    p = all_correlations.pvalue[:len(y_cols), -len(x_cols):]
    p = pd.DataFrame(p)
    p.columns = x_cols
    p.index = y_cols
    
    fig_corr = px.imshow(corr, text_auto=True, aspect='auto', color_continuous_scale='agsunset')
    fig_r2 = px.imshow(corr**2, text_auto=True, aspect='auto', color_continuous_scale='agsunset')
    fig_p = px.imshow(p, text_auto=True, aspect='auto', color_continuous_scale='gray_r')

    return corr, fig_corr, p, fig_p, fig_r2

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde


import matplotlib.pyplot as plt
from IPython.display import display, HTML
import matplotlib.colors as mcolors

def display_colored_sentences(tokens_list, p1, tokenizer, bound = None):
    encodings = torch.cat(tokens_list)
    decoded_tokens = [tokenizer.decode([token], skip_special_tokens=False) for token in encodings]

    tokens_ids_per_sentence = np.cumsum([t.size(0) for t in tokens_list])
    assert tokens_ids_per_sentence[-1] == len(p1)

    global_min = np.nanmin(p1)
    global_max = np.nanmax(p1)
    if bound != None:
        global_min, global_max = bound

    # Create a white-to-red colormap
    cmap = plt.cm.Reds
    cmap = cmap(np.linspace(0, 1, 256))
    # cmap[:50, :] = 1  # make the first 50 entries white (RGBA=1,1,1,1)
    white_to_red = mcolors.ListedColormap(cmap)

    html_output = ""
    prev_idx_pp = 0

    for idx in range(len(tokens_list)):
        idx_pp = tokens_ids_per_sentence[idx]
        sentence_tokens = decoded_tokens[prev_idx_pp:idx_pp]
        sentence_p1 = p1[prev_idx_pp:idx_pp]

        colored_sentence = ""
        for token, score in zip(sentence_tokens, sentence_p1):
            token = token.strip()
            token = token.replace("<", "&lt;").replace(">", "&gt;")

            if global_max == global_min:
                norm_score = 0.0
            else:
                norm_score = (score - global_min) / (global_max - global_min)
            r, g, b, _ = white_to_red(norm_score)
            color = mcolors.to_hex((r, g, b))
            colored_sentence += f'<span style="color:{color}">{str(token)}</span> '
        html_output += f"<span style='line-height:1.8em; font-size:130%; font-weight: bolder;'>{colored_sentence}</span><br>"
        prev_idx_pp = idx_pp

    # Display the HTML sentences
    display(HTML(html_output))

    # Create the color bar
    fig, ax = plt.subplots(figsize=(6, 1))
    norm = plt.Normalize(vmin=global_min, vmax=global_max)
    fig.subplots_adjust(bottom=0.5)

    cb1 = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=white_to_red),
        cax=ax, orientation='horizontal'
    )
    cb1.set_label('Perplexity')
    plt.show()

import re

def preprocess(text, pattern=r'<(SPK[1-9]|MOD)>'):
    matches = re.findall(pattern, text)
    dialog = text.replace('\n', '')
    dialog = re.sub(pattern, lambda m: '\n' + m.group(0), dialog)
    dialog_lines = [
        line.strip() + ' '
        for line in dialog.split('\n')
        if line.strip()
    ]

    return dialog_lines, matches


def rolling_kde_heatmap_with_turns(
    ppls,
    token_list,
    window_size=50,
    step=10,
    bandwidth=0.5,
    vmin=None,
    vmax=None,
    title="[SPK]",
    return_densities=True
):
    """
    Plots a rolling KDE heatmap over token-level perplexities and marks turn boundaries.
    X-axis is labeled by turn indices.

    Args:
        ppls (List[float]): Flattened token-level perplexities.
        token_list (List[Tensor]): List of per-turn token tensors.
        window_size (int): Window size for KDE.
        step (int): Step size for moving window.
        bandwidth (float): Bandwidth for Gaussian KDE.
        vmin, vmax: Color limits for the heatmap.
        title (str): Title string.
        return_densities (bool): If True, return the densities matrix.

    Returns:
        densities (ndarray): KDE matrix of shape (n_windows x n_bins)
    """
    ppls = np.array(ppls)
    xs = np.linspace(0, np.nanmax(ppls), 100)
    densities = []

    # Compute KDE over rolling windows
    for i in range(0, len(ppls) - window_size, step):
        window = ppls[i:i + window_size]
        if np.isnan(window).any():
            densities.append(np.zeros_like(xs))  # pad with zeros
        else:
            kde = gaussian_kde(window, bw_method=bandwidth)
            densities.append(kde(xs))

    densities = np.array(densities)

    # Compute token indices for turn boundaries
    turn_token_ends = np.cumsum([len(tok) for tok in token_list])
    turn_boundaries_token_idx = turn_token_ends[:-1]

    # Map each window midpoint to a turn index
    window_starts = np.arange(0, len(ppls) - window_size, step)
    window_midpoints = window_starts + window_size // 2
    window_turn_idxs = [np.searchsorted(turn_token_ends, mid) for mid in window_midpoints]

    # Mark turn boundaries on the x-axis
    turn_boundaries_bins = [np.searchsorted(window_starts, tb) for tb in turn_boundaries_token_idx]
    unique_turn_bins = sorted(set(turn_boundaries_bins))

    # Plot heatmap
    plt.figure(figsize=(12, 5))
    ax = sns.heatmap(
        densities.T,
        xticklabels=False,
        yticklabels=False,
        cmap="viridis",
        cbar=True,
        vmin=vmin,
        vmax=vmax
    )

    for xb in unique_turn_bins:
        ax.axvline(x=xb, color='white', linestyle='--', linewidth=0.5, alpha=0.7)

    # Set sparse x-ticks using turn indices
    xtick_pos = np.linspace(0, len(window_turn_idxs) - 1, num=min(10, len(window_turn_idxs)), dtype=int)
    xtick_labels = [f"Turn {window_turn_idxs[i]}" for i in xtick_pos]
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_labels, rotation=45)

    plt.title(f"Rolling KDE Heatmap of PPL for {title}")
    plt.xlabel("Turn Index")
    plt.ylabel("Perplexity Bins")
    plt.tight_layout()
    plt.show()

    if return_densities:
        return densities

from matplotlib import cm
from matplotlib.colors import Normalize
from IPython.display import HTML

from matplotlib import cm
from matplotlib.colors import Normalize
from IPython.display import HTML

from matplotlib import cm
from matplotlib.colors import Normalize
from IPython.display import HTML

def display_turns_colored_by_kde(
    token_list,
    ppls,
    densities,
    window_size,
    step,
    tokenizer,
    matches,
    vmin=None,
    vmax=None,
    cmap_name="viridis"
):
    """
    Colors each *turn* using the average of token-level rolling KDE values.
    
    Args:
        token_list: List of token tensors per turn.
        ppls: Flattened list of token-level perplexities.
        densities: 2D array of KDE values (n_windows x n_bins).
        window_size: Size of rolling window used for KDE.
        step: Step size used in rolling KDE.
        tokenizer: HF tokenizer.
        matches: List of speaker tags per turn (e.g., ["<SPK1>", "<SPK2>", ...]).
        vmin, vmax: KDE value scale for coloring.
        cmap_name: Name of matplotlib colormap.

    Returns:
        IPython HTML display of color-coded turns grouped by speaker.
    """
    num_tokens = len(ppls)
    num_windows = densities.shape[0]
    token_kde_scores = np.zeros(num_tokens)
    token_counts = np.zeros(num_tokens)

    # Aggregate KDE scores per token
    for win_idx in range(num_windows):
        start = win_idx * step
        end = min(start + window_size, num_tokens)
        window_density = np.max(densities[win_idx])
        for i in range(start, end):
            token_kde_scores[i] += window_density
            token_counts[i] += 1

    # Compute per-token average KDE score
    with np.errstate(divide='ignore', invalid='ignore'):
        averaged_scores = np.divide(
            token_kde_scores,
            token_counts,
            out=np.zeros_like(token_kde_scores),
            where=token_counts != 0
        )

    # Normalize color scale
    norm = Normalize(
        vmin=vmin if vmin is not None else np.nanmin(averaged_scores),
        vmax=vmax if vmax is not None else np.nanmax(averaged_scores)
    )
    cmap = cm.get_cmap(cmap_name)

    # Decode tokens
    flat_tokens = [tok.item() for turn in token_list for tok in turn]
    decoded_tokens = tokenizer.convert_ids_to_tokens(flat_tokens)

    # Compute averaged scores per turn
    html = "<div style='font-family: monospace; line-height: 1.8;'>"
    idx = 0
    for turn_idx, turn in enumerate(token_list):
        turn_len = len(turn)
        speaker = matches[turn_idx].strip() if isinstance(matches[turn_idx], str) else str(matches[turn_idx])

        # Average KDE scores over the turn's tokens
        turn_scores = averaged_scores[idx:idx + turn_len]
        avg_score = np.mean(turn_scores) if len(turn_scores) > 0 else 0.0
        color = cm.colors.rgb2hex(cmap(norm(avg_score)))

        # Decode and join tokens
        turn_tokens = decoded_tokens[idx:idx + turn_len]
        decoded_str = " ".join(tok.replace("Ġ", " ").replace("▁", " ").strip() for tok in turn_tokens)

        # Format HTML
        html += f"<div><strong>{speaker}</strong>: <span style='background-color:{color}; padding:2px 6px; border-radius:3px;'>{decoded_str}</span></div>\n"
        idx += turn_len

    html += "</div>"
    return HTML(html)

import numpy as np
from itertools import combinations
from scipy import stats

def compute_significance(
    original_ppls_p_spk,
    spk_tokens,
    *,
    alternative_mw="two-sided",
    equal_var_ttest=False,
    adjust_p="fdr_bh"  # None or "fdr_bh"
):
    """
    Pairwise significance tests (KS, Mann-Whitney U, Welch's t) across multiple speakers.

    Parameters
    ----------
    original_ppls_p_spk : dict[str, array-like]
        Mapping like {"<SPK0>": arr, "<SPK1>": arr, ...}.
    spk_tokens : list[str]
        Speaker tokens to include (order doesn't matter).
    alternative_mw : {"two-sided","less","greater"}
        Alternative for Mann-Whitney U.
    equal_var_ttest : bool
        If False, do Welch's t-test; if True, classic independent t-test.
    adjust_p : {None, "fdr_bh"}
        If set, applies Benjamini–Hochberg FDR across *all* p-values (per test family).

    Returns
    -------
    results_list : list[dict]
        Tidy rows with keys: test, spk_i, spk_j, n_i, n_j, statistic, p_value, p_value_adj (if requested).
    results_dict : dict
        Nested dict results_dict[test][(spk_i, spk_j)] = {statistic, p_value, ...}
    """
    # Clean arrays
    arrays = {}
    for s in spk_tokens:
        arr = np.asarray(original_ppls_p_spk[s], dtype=float)
        arrays[s] = arr[~np.isnan(arr)]

    pairs = list(combinations(spk_tokens, 2))
    results = {"KS": {}, "Mann-Whitney U": {}, "T-test": {}}
    rows_ks, rows_mw, rows_tt = [], [], []

    for a, b in pairs:
        x, y = arrays[a], arrays[b]
        n_x, n_y = len(x), len(y)

        # If any side empty, skip with NaNs
        if n_x == 0 or n_y == 0:
            ks_stat = mw_stat = tt_stat = np.nan
            ks_p = mw_p = tt_p = np.nan
        else:
            ks_stat, ks_p = stats.ks_2samp(x, y, alternative="two-sided", mode="auto")
            mw_stat, mw_p = stats.mannwhitneyu(x, y, alternative=alternative_mw, method="auto")
            tt_stat, tt_p = stats.ttest_ind(x, y, equal_var=equal_var_ttest)

        rows_ks.append({"test":"KS","spk_i":a,"spk_j":b,"n_i":n_x,"n_j":n_y,"statistic":ks_stat,"p_value":ks_p})
        rows_mw.append({"test":"Mann-Whitney U","spk_i":a,"spk_j":b,"n_i":n_x,"n_j":n_y,"statistic":mw_stat,"p_value":mw_p})
        rows_tt.append({"test":"T-test","spk_i":a,"spk_j":b,"n_i":n_x,"n_j":n_y,"statistic":tt_stat,"p_value":tt_p})

        results["KS"][(a,b)] = {"statistic": ks_stat, "p_value": ks_p, "n_i": n_x, "n_j": n_y}
        results["Mann-Whitney U"][(a,b)] = {"statistic": mw_stat, "p_value": mw_p, "n_i": n_x, "n_j": n_y}
        results["T-test"][(a,b)] = {"statistic": tt_stat, "p_value": tt_p, "n_i": n_x, "n_j": n_y}

    # O`ptional Benjamini–Hochberg adjustment per test family
    def fdr_bh(rows):
        # ignore NaNs in correction
        pvals = np.array([r["p_value"] for r in rows], dtype=float)
        idx = np.where(~np.isnan(pvals))[0]
        m = len(idx)
        if m == 0: 
            return rows
        order = idx[np.argsort(pvals[idx])]
        ranks = np.empty_like(order, dtype=float)
        ranks[np.arange(m)] = np.arange(1, m+1)

        # compute adjusted p for the ordered set, then enforce monotonicity
        adj = np.empty(m, dtype=float)
        sorted_p = pvals[order]
        adj_vals = (sorted_p * m) / ranks
        adj_vals = np.minimum.accumulate(adj_vals[::-1])[::-1]
        # place back
        p_adj = np.full_like(pvals, np.nan, dtype=float)
        p_adj[order] = np.clip(adj_vals, 0.0, 1.0)

        # write into rows
        for i, r in enumerate(rows):
            r["p_value_adj"] = p_adj[i]
        return rows

    if adjust_p == "fdr_bh":
        rows_ks = fdr_bh(rows_ks)
        rows_mw = fdr_bh(rows_mw)
        rows_tt = fdr_bh(rows_tt)

        # also add to dict
        for row in rows_ks:
            a, b = row["spk_i"], row["spk_j"]
            results["KS"][(a,b)]["p_value_adj"] = row["p_value_adj"]
        for row in rows_mw:
            a, b = row["spk_i"], row["spk_j"]
            results["Mann-Whitney U"][(a,b)]["p_value_adj"] = row["p_value_adj"]
        for row in rows_tt:
            a, b = row["spk_i"], row["spk_j"]
            results["T-test"][(a,b)]["p_value_adj"] = row["p_value_adj"]

    results_list = rows_ks + rows_mw + rows_tt
    return results_list, results


from transformers import AutoTokenizer
import numpy as np

def compute_dominance_per_spk(perplexity, token_list, matches, tokenizer):
    prev_idx_pp = 0
    tokens_ids_per_sentence = np.cumsum([t.size(0) for t in token_list])
    dialog = [tokenizer.decode(token, skip_special_tokens=True) for token in token_list]
    all_values = []
    ppls_p_spk={}
    for idx, match in enumerate(matches):
        if match not in ppls_p_spk:
            ppls_p_spk[match] = []
        for px in range(len(token_list[idx])):
            fin_idx = px + np.sum([len(l) for l in token_list[:idx]], dtype=int)
            ppls_p_spk[match].append(perplexity[fin_idx])
    return ppls_p_spk

from transformers import AutoTokenizer
import numpy as np

def remove_match_prefix_ppl(token_list, ppl, matches, tokenizer, min_len=3):
    """
    Removes matching prefixes from each line and filters out entire turns if remaining token length is below threshold.

    Args:
        token_list (List[List[int]]): Tokenized dialog lines.
        ppl (List[float]): Flat list of token-level PPL values.
        matches (List[str]): List of string prefixes to remove from each line.
        tokenizer: HuggingFace tokenizer used.
        min_len (int): Minimum length (after prefix removal) to keep the turn.

    Returns:
        Tuple[
            List[List[int]],  # filtered_tokens
            List[float],      # filtered_ppl
            List[int],        # flat_token_ids
            List[str]         # filtered_matches
        ]
    """
    assert len(token_list) == len(matches), "Mismatch between token list and matches"

    line_offsets = np.cumsum([0] + [len(t) for t in token_list[:-1]])  # start index in flat PPL array

    filtered_tokens = []
    filtered_ppl = []
    flat_token_ids = []
    filtered_matches = []

    for i, (token, match) in enumerate(zip(token_list, matches)):
        tok_match = tokenizer(match, return_tensors="pt")
        match_tok_len = len(tok_match.input_ids[0])

        # Slice off the match prefix
        token_filtered = token[match_tok_len:]

        if len(token_filtered) < min_len:
            continue  # skip turn entirely if below threshold

        # Keep this turn
        filtered_tokens.append(token_filtered)
        filtered_matches.append(match)

        # Extract aligned PPL values
        start = line_offsets[i] + match_tok_len
        end = line_offsets[i] + len(token)
        turn_ppl = ppl[start:end]

        assert len(turn_ppl) == len(token_filtered), f"PPL/token mismatch at turn {i}"
        
        filtered_ppl.extend(turn_ppl)
        flat_token_ids.extend(token_filtered)

    return filtered_tokens, filtered_ppl, flat_token_ids, filtered_matches

def extract_tokens_and_ppls_by_turn_indices(turn_indices, token_list, ppls):
    """
    Extracts a filtered list of token lists and aligned PPLs from a flat PPL array.

    Args:
        turn_indices (List[int]): Indices of turns to keep.
        token_list (List[List[int]]): Full list of token lists.
        ppls (List[float]): Flattened list of per-token PPLs.

    Returns:
        Tuple:
            filtered_token_list (List[List[int]]): Subset of token_list
            filtered_ppl (List[float]): Subset of PPL values aligned with token IDs
    """
    token_lengths = [len(t) for t in token_list]
    cum_starts = np.cumsum([0] + token_lengths)

    filtered_token_list = []
    filtered_ppl = []

    for idx in turn_indices:
        start = cum_starts[idx]
        end = cum_starts[idx + 1]
        filtered_token_list.append(token_list[idx])
        filtered_ppl.extend(ppls[start:end])

    return filtered_token_list, filtered_ppl

from transformers import AutoTokenizer
import numpy as np

def remove_match_prefix_ppl(token_list, ppl, matches, tokenizer, min_len=1):
    """
    Removes matching prefixes from each line and filters out entire turns if remaining token length is below threshold.

    Args:
        token_list (List[List[int]]): Tokenized dialog lines.
        ppl (List[float]): Flat list of token-level PPL values.
        matches (List[str]): List of string prefixes to remove from each line.
        tokenizer: HuggingFace tokenizer used.
        min_len (int): Minimum length (after prefix removal) to keep the turn.

    Returns:
        Tuple[
            List[List[int]],  # filtered_tokens
            List[float],      # filtered_ppl
            List[int],        # flat_token_ids
            List[str]         # filtered_matches
        ]
    """
    assert len(token_list) == len(matches), "Mismatch between token list and matches"

    line_offsets = np.cumsum([0] + [len(t) for t in token_list[:-1]])  # start index in flat PPL array

    filtered_tokens = []
    filtered_ppl = []
    flat_token_ids = []
    filtered_matches = []

    for i, (token, match) in enumerate(zip(token_list, matches)):
        tok_match = tokenizer(match, return_tensors="pt")
        match_tok_len = len(tok_match.input_ids[0])

        # Slice off the match prefix
        token_filtered = token[match_tok_len:]

        if len(token_filtered) < min_len:
            continue  # skip turn entirely if below threshold

        # Keep this turn
        filtered_tokens.append(token_filtered)
        filtered_matches.append(match)

        # Extract aligned PPL values
        start = line_offsets[i] + match_tok_len
        end = line_offsets[i] + len(token)
        turn_ppl = ppl[start:end]

        assert len(turn_ppl) == len(token_filtered), f"PPL/token mismatch at turn {i}"
        
        filtered_ppl.extend(turn_ppl)
        flat_token_ids.extend(token_filtered)

    return filtered_tokens, filtered_ppl, flat_token_ids, filtered_matches

import pandas as pd

def average_ppl_per_turn(df, ppl_dict, idx_col='ppl', annotation_cols=None):
    """
    For each row in df, average values from ppl_dict at indices listed in df[idx_col].

    Args:
        df (pd.DataFrame): DataFrame with a column of token indices (e.g., 'ppl').
        ppl_dict (dict): Dictionary with name -> list of PPL values.
        idx_col (str): Column in df with list of indices per turn.
        annotation_cols (List[str]): Optional columns to carry forward.

    Returns:
        pd.DataFrame: Averaged PPL values per turn, optionally with annotations.
    """
    import numpy as np
    import pandas as pd

    records = []
    for _, row in df.iterrows():
        token_ids = row[idx_col]
        if not isinstance(token_ids, list) or len(token_ids) == 0:
            continue

        record = {}
        for name, ppl_values in ppl_dict.items():
            selected = [ppl_values[i] for i in token_ids if i < len(ppl_values)]
            record[f'{name}_avg'] = np.mean(selected) if selected else None
            record['tokens'] = token_ids
            record['tok_len'] = len(token_ids)

        if annotation_cols:
            for col in annotation_cols:
                record[col] = row.get(col, None)

        records.append(record)

    return pd.DataFrame(records)

def expand_multiple_ppl_by_token(bin_df, ppl_dict, annotation_cols=None):
    """
    Expands a bin-level DataFrame into token-level rows, adding multiple PPL values per token.

    Args:
        bin_df (pd.DataFrame): must contain 'ppl' as a list of token indices.
        ppl_dict (dict): dictionary of {name: List[float]}, e.g. {'ppl1': [...], 'ppl2': [...]}
        annotation_cols (List[str]): optional list of annotation columns to repeat per token.

    Returns:
        pd.DataFrame: one row per token with multiple ppl values and repeated annotations.
    """
    # Validate that all ppl lists have the same length
    lengths = [len(v) for v in ppl_dict.values()]
    if not all(l == lengths[0] for l in lengths):
        raise ValueError("All PPL arrays must have the same length.")

    records = []

    for _, row in bin_df.iterrows():
        token_ids = row["ppl"]
        if not isinstance(token_ids, list) or len(token_ids) == 0:
            continue

        for tok_id in token_ids:
            if tok_id >= lengths[0]:
                continue  # skip out-of-bounds

            record = {"token_id": tok_id}
            for name, ppl_values in ppl_dict.items():
                record[name] = ppl_values[tok_id]

            if annotation_cols:
                for col in annotation_cols:
                    record[col] = row.get(col, None)

            records.append(record)

    return pd.DataFrame(records)


import numpy as np
import pandas as pd
from collections import defaultdict
import torch  # make sure torch is imported

def assign_words_to_bins(df, tokenizer, bin_size=1.0, per_speaker=True):
    """
    Assigns words to time bins based on uniform spread over the utterance duration.
    Returns: dict of pd.DataFrames, one per speaker
    """
    # Fix nested defaultdict
    bin_tok = defaultdict(lambda: defaultdict(list))
    bin_ppl = defaultdict(lambda: defaultdict(list))
    tok_len = 0

    for _, row in df.iterrows():
        spk = row['speaker']
        start = row["start"]
        stop = row["stop"]
        duration = stop - start
            
        if duration <= 0 or pd.isna(start) or pd.isna(stop):
            continue

        tokens = row["tokens"]
        n_tokens = len(tokens)
        if n_tokens == 0:
            continue

        # Uniformly spread tokens over time
        tok_times = np.linspace(start, stop, n_tokens + 1)
        for i, tok in enumerate(tokens):
            tok_start = tok_times[i]
            tok_end = tok_times[i + 1]

            bin_start_idx = int(np.floor(tok_start / bin_size))
            bin_end_idx = int(np.floor(tok_end / bin_size))

            for b in range(bin_start_idx, bin_end_idx + 1):
                bin_tok[spk][b].append(tok)
                bin_ppl[spk][b].append(tok_len + i)

        tok_len += n_tokens

    # Determine full range of bins
    max_bin = int(np.ceil(df["stop"].max() / bin_size))
    all_bins = list(range(max_bin + 1))
    all_speakers = df["speaker"].unique()

    # Build a DataFrame per speaker
    bins_df = {
        spk: pd.DataFrame([
            {
                "time_bin": b,
                "start_time": b * bin_size,
                "end_time": (b + 1) * bin_size,
                "words": tokenizer.decode(torch.tensor(bin_tok[spk][b]), skip_special_tokens=True) if bin_tok[spk][b] else None,
                "ppl": bin_ppl[spk][b],
                "n_tok": len(bin_tok[spk][b])
            }
            for b in all_bins
        ])
        for spk in all_speakers
    }

    return bins_df

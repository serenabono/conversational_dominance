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
    
def compute_graph_perplexity(tokenizer, tokens, p1, p2, matches, pattern = '<(SPK[1-9]|MOD)>', answers=None):
    
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
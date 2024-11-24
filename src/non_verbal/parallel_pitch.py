from concurrent.futures import ProcessPoolExecutor
import librosa
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import re
from scipy import stats
import pickle

# Read the Dominance-assesment.xlsx file
xlsx_file_path = 'data/external/multisimo/Dominance-assesment-mapped.xlsx'
dominance_df = pd.read_excel(xlsx_file_path)

# Define the function to process a single session
def process_session(session_number):
    try:
        # Load audio file
        wav_file_path = f'data/external/multisimo/STEREO_to_MONO/S{session_number}_STE-MONO.wav'
        y, sr = librosa.load(wav_file_path, sr=None)
        y = y / np.max(np.abs(y))

        # Load the annotation file
        annotation_file = f'data/external/multisimo/annotations/speech transcription_Elan/S{session_number}.eaf'
        tree = ET.parse(annotation_file)
        root = tree.getroot()

        # Extract time slots
        time_order = root.find('.//TIME_ORDER')
        time_slots = {ts.attrib['TIME_SLOT_ID']: int(ts.attrib['TIME_VALUE']) for ts in time_order.findall('TIME_SLOT')}

        # Extract P1 and P2
        annotation_values = []
        for tier in root.findall('.//TIER'):
            if tier.attrib.get('LINGUISTIC_TYPE_REF') == 'TurnType' and tier.attrib.get('TIER_ID') == 'Turns':
                for annotation in tier.findall('.//ANNOTATION/ALIGNABLE_ANNOTATION'):
                    value = annotation.find('ANNOTATION_VALUE').text
                    if re.match(r'^P\d{3}$', value) and value not in annotation_values:
                        annotation_values.append(value)
        
        P1_number, P2_number = sorted(annotation_values[:2])

        # Extract sections for P1 and P2
        sections_P1, sections_P2 = [], []
        for tier in root.findall('.//TIER'):
            if tier.attrib.get('LINGUISTIC_TYPE_REF') == 'TurnType' and tier.attrib.get('TIER_ID') == 'Turns':
                for annotation in tier.findall('.//ANNOTATION/ALIGNABLE_ANNOTATION'):
                    value = annotation.find('ANNOTATION_VALUE').text
                    if value in annotation_values:
                        start_time = time_slots[annotation.attrib['TIME_SLOT_REF1']] / 1000.0
                        end_time = time_slots[annotation.attrib['TIME_SLOT_REF2']] / 1000.0
                        if value == P1_number:
                            sections_P1.append((start_time, end_time))
                        elif value == P2_number:
                            sections_P2.append((start_time, end_time))

        # Concatenate audio sections
        def concatenate_sections(sections):
            concatenated = np.array([])
            for start, end in sections:
                start_sample = int(start * sr)
                end_sample = int(end * sr)
                concatenated = np.concatenate((concatenated, y[start_sample:end_sample]))
            return concatenated

        concatenated_P1 = concatenate_sections(sections_P1)
        concatenated_P2 = concatenate_sections(sections_P2)

        # Compute f0 using pyin
        def compute_f0(audio):
            f0, voiced_flag, _ = librosa.pyin(audio[:len(audio) // 10],
                                              fmin=librosa.note_to_hz('C2'),
                                              fmax=librosa.note_to_hz('C7'),
                                              )
            return np.mean(f0[voiced_flag]) if len(f0[voiced_flag]) > 0 else None

        f0_P1 = compute_f0(concatenated_P1)
        f0_P2 = compute_f0(concatenated_P2)

        # Read dominance scores
        row = dominance_df[dominance_df.iloc[:, 0] == f'S{session_number}']
        dominance_P1 = row.iloc[:, 1:6].mean(axis=1).values[0]
        dominance_P2 = row.iloc[:, 7:12].mean(axis=1).values[0]

        # Hypothesis testing
        ret = {"session": session_number, "f0_P1": f0_P1, "f0_P2": f0_P2, "dominance_P1": dominance_P1, "dominance_P2": dominance_P2}
        if f0_P1 < f0_P2:
            if dominance_P1 > dominance_P2:
                ret["hypo"] = "true"
            elif dominance_P1 == dominance_P2:
                ret["hypo"] = "maybe"
            else:
                ret["hypo"] = "false"
        else:
            if dominance_P1 < dominance_P2:
                ret["hypo"] = "true"
            elif dominance_P1 == dominance_P2:
                ret["hypo"] = "maybe"
            else:
                ret["hypo"] = "false"
        
        return ret

    except Exception as e:
        print(f"Error processing session {session_number}: {e}")
        return None

# Parallel execution
if __name__ == "__main__":
    session_numbers = ["02", "04", "05", "07", "08", "09", "10", "11", "13", "14", "17", "18", "19", "20", "21", "22", "23"]

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_session, session_numbers))

    # Organize the results
    list_hypo_true = [r["session"] for r in results if r and r["hypo"] == "true"]
    list_hypo_maybe = [r["session"] for r in results if r and r["hypo"] == "maybe"]
    list_hypo_false = [r["session"] for r in results if r and r["hypo"] == "false"]

    print("true: ", list_hypo_true)
    print("maybe: ", list_hypo_maybe)
    print("false: ", list_hypo_false)

    with open("data/results/non_verbal_multi/pitch_results.pkl", "wb") as f:
        pickle.dump(pd.DataFrame(results), f)
    
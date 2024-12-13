from concurrent.futures import ProcessPoolExecutor
import librosa
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import re
from scipy import stats
import pickle

# Define the function to process a single session
def process_session(session_number):
    try:
        # Load audio file
        wav_file_path = f'./STEREO_to_MONO/S{session_number}_STE-MONO.wav'
        y, sr = librosa.load(wav_file_path, sr=None)
        y = y / np.max(np.abs(y))

        # Parse the .eaf file
        annotation_file = f'./annotations/speech transcription_Elan/S{session_number}.eaf'
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
        sections = []
        for tier in root.findall('.//TIER'):
            if tier.attrib.get('LINGUISTIC_TYPE_REF') == 'TurnType' and tier.attrib.get('TIER_ID') == 'Turns':
                for annotation in tier.findall('.//ANNOTATION/ALIGNABLE_ANNOTATION'):
                    speaker = annotation.find('ANNOTATION_VALUE').text
                    if speaker in annotation_values:
                        start_time = time_slots[annotation.attrib['TIME_SLOT_REF1']] / 1000.0
                        end_time = time_slots[annotation.attrib['TIME_SLOT_REF2']] / 1000.0
                        sections.append({
                            "speaker": speaker,
                            "section": (start_time, end_time)
                        })

        # get audio of sections
        def get_section_audio(section):
            start_sample = int(section["section"][0] * sr)
            end_sample = int(section["section"][1] * sr)
            return y[start_sample:end_sample]

        # Compute f0 using pyin
        def compute_f0(audio):
            f0, voiced_flag, _ = librosa.pyin(audio,
                                              fmin=librosa.note_to_hz('C2'),
                                              fmax=librosa.note_to_hz('C7'),
                                              )
            return np.mean(f0[voiced_flag]) if len(f0[voiced_flag]) > 0 else None

        for i, section in enumerate(sections):
            sections[i]["f0"] = compute_f0(get_section_audio(section))
        
        return {"session" : session_number, "pitch_series": sections}

    except Exception as e:
        print(f"Error processing session {session_number}: {e}")
        return None

# Parallel execution
if __name__ == "__main__":
    session_numbers = ["02", "04", "05", "07", "08", "09", "10", "11", "13", "14", "17", "18", "19", "20", "21", "22", "23"]

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_session, session_numbers))


    with open("pitch_series.pkl", "wb") as f:
        pickle.dump(results, f)
    
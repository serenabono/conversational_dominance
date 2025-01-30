from concurrent.futures import ProcessPoolExecutor
import librosa
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import re
from scipy import stats
import pickle
import os

def get_audio(session_number, data_path):
    # Load audio file
    wav_file_path = os.path.join(data_path, f'STEREO_to_MONO/S{session_number}_STE-MONO.wav')
    y, sr = librosa.load(wav_file_path, sr=None)
    y = y / np.max(np.abs(y))

    # Parse the .eaf file
    annotation_file = os.path.join(data_path, f'annotations/speech transcription_Elan/S{session_number}.eaf')
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

    speaker_audio = {
        "sr": sr,
        P1_number: np.array([]),
        P2_number: np.array([])
    }

    for tier in root.findall('.//TIER'):
        if tier.attrib.get('LINGUISTIC_TYPE_REF') == 'TurnType' and tier.attrib.get('TIER_ID') == 'Turns':
            for annotation in tier.findall('.//ANNOTATION/ALIGNABLE_ANNOTATION'):
                speaker = annotation.find('ANNOTATION_VALUE').text
                if speaker in annotation_values:
                    start_time = time_slots[annotation.attrib['TIME_SLOT_REF1']] / 1000.0
                    end_time = time_slots[annotation.attrib['TIME_SLOT_REF2']] / 1000.0
                    start_sample = int(start_time * sr)
                    end_sample = int(end_time * sr)
                    if speaker == P1_number:
                        speaker_audio[speaker] = np.concatenate([speaker_audio[speaker], y[start_sample:end_sample]])

    speaker_audio[P1_number] = speaker_audio[P1_number].reshape(1, -1).astype("float32")
    speaker_audio[P2_number] = speaker_audio[P2_number].reshape(1, -1).astype("float32")

    return speaker_audio    

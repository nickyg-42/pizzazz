import librosa
import numpy as np
from music21 import *
import os

def process_audio_to_sheet_music(audio_path):
    # Load the audio file
    y, sr = librosa.load(audio_path)
    
    # Perform onset detection
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    
    # Pitch detection
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    
    # Create a music21 score
    score = stream.Score()
    part = stream.Part()
    
    # Convert detected pitches to notes
    for i in range(len(onset_times)):
        start_time = onset_times[i]
        if i < len(onset_times) - 1:
            duration = onset_times[i + 1] - start_time
        else:
            duration = 0.5  # Default duration for last note
            
        # Find the strongest pitch at onset
        frame_idx = librosa.time_to_frames(start_time, sr=sr)
        pitch_idx = magnitudes[:, frame_idx].argmax()
        pitch_hz = pitches[pitch_idx, frame_idx]
        
        if pitch_hz > 0:  # If a pitch was detected
            # Convert frequency to midi note
            midi_note = librosa.hz_to_midi(pitch_hz)
            note = note.Note(midi_note)
            note.duration.quarterLength = duration
            part.append(note)
    
    score.append(part)
    
    # Export as PDF
    output_path = os.path.join('static', 'output.pdf')
    score.write('pdf', fp=output_path)
    
    return output_path
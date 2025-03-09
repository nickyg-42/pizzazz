import librosa
import numpy as np
from music21 import stream, note, meter
import os

def get_tempo_and_beats(y, sr):
    """
    Extract tempo and beat information from audio
    """
    # Use onset detection instead of beat tracking
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    pulse = librosa.beat.plp(onset_envelope=onset_env, sr=sr)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
    return tempo, 60.0 / tempo  # Return tempo and beat length

def quantize_duration(duration, base_note_length=0.25):
    """
    Quantize a duration to the nearest standard note length
    base_note_length: 0.25 represents a quarter note
    """
    # Standard note lengths (in quarter notes)
    std_lengths = [4.0, 2.0, 1.5, 1.0, 0.75, 0.5, 0.25, 0.125]
    
    # Convert duration to quarter note units
    duration_in_quarters = duration / base_note_length
    
    # Find closest standard duration
    closest_duration = min(std_lengths, key=lambda x: abs(x - duration_in_quarters))
    return closest_duration * base_note_length

def process_audio_to_sheet_music(audio_path):
    # Load the audio file
    y, sr = librosa.load(audio_path)
    
    # Get tempo and beat information
    tempo, beat_length = get_tempo_and_beats(y, sr)
    
    # Perform onset detection with more sensitive parameters
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        units='frames',
        hop_length=512,
        backtrack=True
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    
    # Pitch detection with improved parameters
    pitches, magnitudes = librosa.piptrack(
        y=y, 
        sr=sr,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7')
    )
    
    # Create a music21 score
    score = stream.Score()
    part = stream.Part()
    
    # Add time signature
    part.append(meter.TimeSignature('4/4'))
    
    # Convert detected pitches to notes
    for i in range(len(onset_times)):
        start_time = onset_times[i]
        if i < len(onset_times) - 1:
            duration = onset_times[i + 1] - start_time
        else:
            duration = beat_length  # Default to one beat for last note
        
        # Quantize the duration to standard note lengths
        quantized_duration = quantize_duration(duration, beat_length)
        
        # Find the strongest pitch at onset
        frame_idx = librosa.time_to_frames(start_time, sr=sr)
        if frame_idx < magnitudes.shape[1]:  # Check if frame index is valid
            pitch_idx = magnitudes[:, frame_idx].argmax()
            pitch_hz = pitches[pitch_idx, frame_idx]
            
            if pitch_hz > 0:  # If a pitch was detected
                # Convert frequency to midi note
                midi_note = librosa.hz_to_midi(pitch_hz)
                n = note.Note(int(round(midi_note)))  # Round to nearest integer
                n.duration.quarterLength = max(0.25, quantized_duration / beat_length)  # Ensure minimum duration
                part.append(n)
    
    score.append(part)
    
    # Create static directory if it doesn't exist
    os.makedirs('static', exist_ok=True)
    
    # Export as MusicXML
    output_xml = os.path.join('static', 'output.musicxml')
    score.write('musicxml', fp=output_xml)
    
    return output_xml
import librosa
import numpy as np
from music21 import stream, note, meter, clef, chord
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

def split_notes_by_clef(midi_note, magnitude):
    """
    Improved hand separation logic for more natural piano playing
    """
    MIDDLE_C = 60
    TRANSITION_RANGE = 7  # Wider transition range
    
    if midi_note >= MIDDLE_C + TRANSITION_RANGE:
        return ('treble', midi_note, magnitude)
    elif midi_note < MIDDLE_C - TRANSITION_RANGE:
        return ('bass', midi_note, magnitude)
    else:
        # In the transition range, prefer keeping single notes in treble
        if magnitude > 0.4:  # Lower threshold for melody detection
            return ('treble', midi_note, magnitude)
        return ('bass', midi_note, magnitude)

def find_simultaneous_notes(pitches, magnitudes, frame_idx, threshold=0.02):  # Even lower threshold
    """
    Optimized for single note detection with much higher sensitivity
    """
    frame_magnitudes = magnitudes[:, frame_idx]
    max_magnitude = frame_magnitudes.max()
    if max_magnitude == 0:
        return []
    
    normalized_magnitudes = frame_magnitudes / max_magnitude
    
    # Find the single strongest peak
    strongest_peak = None
    strongest_mag = 0
    
    for i, mag in enumerate(normalized_magnitudes):
        if mag > threshold and pitches[i, frame_idx] > 0:
            # Very simple peak detection - just check immediate neighbor
            is_peak = True
            if i > 0 and normalized_magnitudes[i-1] > mag * 1.2:  # More forgiving comparison
                is_peak = False
            if i < len(normalized_magnitudes)-1 and normalized_magnitudes[i+1] > mag * 1.2:
                is_peak = False
            
            if is_peak and mag > strongest_mag:
                midi_note = librosa.hz_to_midi(pitches[i, frame_idx])
                midi_note = round(midi_note)
                strongest_mag = mag
                strongest_peak = (midi_note, mag)
    
    return [strongest_peak] if strongest_peak else []

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
    # Load the audio file with higher sample rate and normalize
    y, sr = librosa.load(audio_path, sr=44100)
    y = librosa.util.normalize(y)
    
    # Enhanced pre-processing with less aggressive pre-emphasis
    y = librosa.effects.preemphasis(y, coef=0.85)  # Reduced from 0.97
    
    # Add harmonic extraction before pitch detection
    y_harmonic = librosa.effects.harmonic(y, margin=4.0)
    
    # Get tempo and beat information first
    tempo, beat_length = get_tempo_and_beats(y, sr)
    
    # Balanced onset detection
    onset_env = librosa.onset.onset_strength(
        y=y, 
        sr=sr,
        hop_length=32,
        aggregate=np.median,
        fmax=4000,  # Reduced to focus on main note frequencies
        n_mels=128  # Reduced for less noise
    )
    
    # More balanced onset detection parameters
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        units='frames',
        hop_length=32,
        backtrack=True,
        pre_max=7,  # Slightly increased
        post_max=7,
        pre_avg=15,  # Increased for better peak detection
        post_avg=15,
        delta=0.015,  # Balanced sensitivity
        wait=2  # Slight wait to avoid double triggers
    )
    
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=32)
    
    # More precise pitch detection with updated parameters
    pitches, magnitudes = librosa.piptrack(
        y=y_harmonic,  # Use harmonic signal
        sr=sr,
        n_fft=4096,    # Increased for better frequency resolution
        hop_length=32,
        fmin=librosa.note_to_hz('A0'),  # Full piano range
        fmax=librosa.note_to_hz('C8'),
        threshold=0.012,  # Slightly lower for better sensitivity
        win_length=4096,  # Match n_fft for better accuracy
        center=True
    )
    
    # Add pitch correction
    def adjust_pitch(midi_note):
        """
        Enhanced pitch correction with octave adjustment
        """
        # First, round to nearest semitone
        semitone = round(midi_note)
        cents_diff = (midi_note - semitone) * 100
        
        # More aggressive pitch correction
        if abs(cents_diff) < 40:  # Increased from 30 to catch more slight deviations
            # Check if note needs octave adjustment
            note_in_octave = semitone % 12
            if note_in_octave == 11 and cents_diff > 20:  # If very sharp B
                return semitone + 1  # Adjust to C of next octave
            elif note_in_octave == 0 and cents_diff < -20:  # If very flat C
                return semitone - 1  # Adjust to B of previous octave
            return semitone
        
        # For larger deviations, try to find nearest common piano note
        piano_notes = [21, 23, 24, 26, 28, 29, 31, 33, 35, 36, 38, 40, 41, 43, 45, 
                      47, 48, 50, 52, 53, 55, 57, 59, 60, 62, 64, 65, 67, 69, 71, 
                      72, 74, 76, 77, 79, 81, 83, 84, 86, 88, 89, 91, 93, 95, 96, 108]
        return min(piano_notes, key=lambda x: abs(x - midi_note))

    # Create score and parts
    score = stream.Score()
    treble_part = stream.Part()
    bass_part = stream.Part()
    
    # Add clefs and time signature
    treble_part.append(clef.TrebleClef())
    bass_part.append(clef.BassClef())
    treble_part.append(meter.TimeSignature('4/4'))
    bass_part.append(meter.TimeSignature('4/4'))
    
    # Modified note detection loop
    for i in range(len(onset_times)):
        start_time = onset_times[i]
        if i < len(onset_times) - 1:
            duration = onset_times[i + 1] - start_time
        else:
            duration = beat_length
        
        quantized_duration = quantize_duration(duration, beat_length)
        quarter_length = max(0.125, quantized_duration / beat_length)
        
        frame_idx = librosa.time_to_frames(start_time, sr=sr, hop_length=32)
        if frame_idx >= magnitudes.shape[1]:
            continue
        
        # More selective frame checking
        notes_set = set()
        for offset in [0, -1]:  # Reduced to just current and previous frame
            check_idx = frame_idx + offset
            if 0 <= check_idx < magnitudes.shape[1]:
                notes = find_simultaneous_notes(pitches, magnitudes, check_idx, threshold=0.015)
                if notes and notes[0]:
                    notes_set.add(notes[0][0])
        
        # Process detected notes with minimum gap
        prev_midi = None
        for midi_note in sorted(notes_set):
            # Skip notes too close to previous note
            if prev_midi and abs(midi_note - prev_midi) < 2:
                continue
                
            # Round to nearest integer and adjust octave if needed
            adjusted_note = int(round(midi_note))
            if adjusted_note >= 60:
                n = note.Note(adjusted_note, quarterLength=quarter_length)
                treble_part.append(n)
            else:
                n = note.Note(adjusted_note, quarterLength=quarter_length)
                bass_part.append(n)
            prev_midi = adjusted_note
    
    score.append(treble_part)
    score.append(bass_part)
    
    # Create static directory if it doesn't exist
    os.makedirs('static', exist_ok=True)
    
    # Export as MusicXML
    output_xml = os.path.join('static', 'output.musicxml')
    score.write('musicxml', fp=output_xml)
    
    return output_xml
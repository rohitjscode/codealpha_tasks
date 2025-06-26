import pickle
import numpy as np
from keras.models import load_model
from music21 import note, chord, stream
import os

# Load notes
with open("data/notes.pkl", "rb") as f:
    notes = pickle.load(f)

n_vocab = len(set(notes))
pitchnames = sorted(set(notes))
note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

# Prepare network input
sequence_length = 100
network_input = []

for i in range(0, len(notes) - sequence_length):
    seq_in = notes[i:i + sequence_length]
    network_input.append([note_to_int[char] for char in seq_in])

start = np.random.randint(0, len(network_input) - 1)
pattern = network_input[start]

# Load trained model
model = load_model("music_model.h5")

# Generate new music
prediction_output = []

for _ in range(500):  # generate 500 notes
    input_seq = np.reshape(pattern, (1, len(pattern), 1)) / float(n_vocab)
    prediction = model.predict(input_seq, verbose=0)
    index = np.argmax(prediction)
    result = int_to_note[index]
    prediction_output.append(result)
    pattern.append(index)
    pattern = pattern[1:]

# Convert predictions to music21 notes/chords
output_notes = []

for pattern in prediction_output:
    if '.' in pattern or pattern.isdigit():
        chord_notes = [note.Note(int(n)) for n in pattern.split('.')]
        new_chord = chord.Chord(chord_notes)
        output_notes.append(new_chord)
    else:
        output_notes.append(note.Note(pattern))

# Write to MIDI
os.makedirs("output", exist_ok=True)
midi_stream = stream.Stream(output_notes)
midi_stream.write('midi', fp='output/generated_music.mid')

print("✅ Music generated and saved to output/generated_music.mid")

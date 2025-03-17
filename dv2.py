import numpy as np
from pydub import AudioSegment
import noisereduce as nr
from scipy.io.wavfile import write

def damp_audio(input_file, output_file):
    # Convert m4a to wav using pydub
    audio = AudioSegment.from_file(input_file, format="m4a")
    audio = audio.set_channels(1)  # Convert to mono (if needed)
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)

    # Normalize to range -1 to 1
    samples = samples / np.max(np.abs(samples))

    # Reduce noise using noisereduce
    reduced_noise = nr.reduce_noise(y=samples, sr=audio.frame_rate)

    # Convert back to int16 format
    reduced_noise = (reduced_noise * 32767).astype(np.int16)

    # Save to wav file using scipy
    write(output_file, audio.frame_rate, reduced_noise)

    print(f"Damped audio saved to: {output_file}")

# Example usage
input_file = "noise_city.m4a"
output_file = "output.wav"
damp_audio(input_file, output_file)

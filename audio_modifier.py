import pyaudio
import wave
import numpy as np
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
import time
import os

def record_audio(filename, duration=5, sample_rate=44100, chunk_size=1024):
    """
    Record audio from microphone and save it to a file
    """
    p = pyaudio.PyAudio()

    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=chunk_size)

    print("* Recording audio...")

    frames = []
    for i in range(0, int(sample_rate / chunk_size * duration)):
        data = stream.read(chunk_size)
        frames.append(np.frombuffer(data, dtype=np.float32))

    print("* Done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    audio_data = np.concatenate(frames)

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(4)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())

    return audio_data, sample_rate

def generate_anti_wave(audio_data):
    """
    Generate inverse of the audio wave
    """
    return -1 * audio_data

def create_damped_audio(audio_data, anti_wave):
    """
    Apply exponential damping to the audio
    """
    # time = np.linspace(0, 1, len(audio_data))
    # damping = np.exp(-damping_factor * time)
    return audio_data + anti_wave

def save_wave_file(filename, audio_data, sample_rate):
    """
    Save audio data as WAV file
    """
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(4)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())

def create_spectrogram(audio_data, sample_rate, title, output_file):
    """
    Create and save a spectrogram using short-time Fourier transform
    """
    plt.figure(figsize=(10, 6))

    # Calculate spectrogram using STFT
    frequencies, times, Sxx = signal.stft(audio_data,
                                        fs=sample_rate,
                                        window='hamming',
                                        nperseg=1024,
                                        noverlap=512)

    # Convert to power spectrum in dB
    Sxx_db = 10 * np.log10(np.abs(Sxx)**2)

    # Plot spectrogram
    plt.pcolormesh(times, frequencies, Sxx_db, shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title(f'Spectrogram - {title}')
    plt.colorbar(label='Power Spectral Density [dB]')

    # Adjust frequency axis to show up to Nyquist frequency
    plt.ylim([0, sample_rate/2])

    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def plot_waveform(audio_data, sample_rate, title, output_file):
    """
    Create and save a waveform plot
    """
    plt.figure(figsize=(10, 4))
    time = np.linspace(0, len(audio_data)/sample_rate, len(audio_data))
    plt.plot(time, audio_data)
    plt.xlabel('Time [sec]')
    plt.ylabel('Amplitude')
    plt.title(f'Waveform - {title}')
    plt.grid(True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def analyze_frequency_content(audio_data, sample_rate, title, output_file):
    """
    Create and save a frequency spectrum plot
    """
    plt.figure(figsize=(10, 4))

    # Calculate frequency spectrum
    frequencies = np.fft.fftfreq(len(audio_data), 1/sample_rate)
    spectrum = np.fft.fft(audio_data)

    # Plot only positive frequencies up to Nyquist frequency
    positive_freq_mask = frequencies >= 0
    plt.plot(frequencies[positive_freq_mask],
             np.abs(spectrum)[positive_freq_mask])

    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    plt.title(f'Frequency Spectrum - {title}')
    plt.grid(True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Create output directories
    output_dir = "audio_output"
    plot_dir = "plot_output"
    for directory in [output_dir, plot_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # File paths
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    input_file = os.path.join(output_dir, f"input_{timestamp}.wav")
    anti_wave_file = os.path.join(output_dir, f"anti_wave_{timestamp}.wav")
    damped_file = os.path.join(output_dir, f"damped_{timestamp}.wav")

    try:
        # Record audio
        print("Starting audio recording...")
        audio_data, sample_rate = record_audio(input_file)
        print(f"Input audio saved to: {input_file}")

        # Generate anti-wave
        anti_wave = generate_anti_wave(audio_data)
        save_wave_file(anti_wave_file, anti_wave, sample_rate)
        print(f"Anti-wave saved to: {anti_wave_file}")

        # Create damped audio
        damped_audio = create_damped_audio(audio_data, anti_wave)
        save_wave_file(damped_file, damped_audio, sample_rate)
        print(f"Damped audio saved to: {damped_file}")

        # Create analysis plots
        print("\nGenerating analysis plots...")

        # For original audio
        create_spectrogram(audio_data, sample_rate, "Original Audio",
                          os.path.join(plot_dir, f"spectrogram_original_{timestamp}.png"))
        plot_waveform(audio_data, sample_rate, "Original Audio",
                     os.path.join(plot_dir, f"waveform_original_{timestamp}.png"))
        analyze_frequency_content(audio_data, sample_rate, "Original Audio",
                                os.path.join(plot_dir, f"spectrum_original_{timestamp}.png"))

        # For anti-wave
        create_spectrogram(anti_wave, sample_rate, "Anti-wave",
                          os.path.join(plot_dir, f"spectrogram_antiwave_{timestamp}.png"))
        plot_waveform(anti_wave, sample_rate, "Anti-wave",
                     os.path.join(plot_dir, f"waveform_antiwave_{timestamp}.png"))
        analyze_frequency_content(anti_wave, sample_rate, "Anti-wave",
                                os.path.join(plot_dir, f"spectrum_antiwave_{timestamp}.png"))

        # For damped audio
        create_spectrogram(damped_audio, sample_rate, "Damped Audio",
                          os.path.join(plot_dir, f"spectrogram_damped_{timestamp}.png"))
        plot_waveform(damped_audio, sample_rate, "Damped Audio",
                     os.path.join(plot_dir, f"waveform_damped_{timestamp}.png"))
        analyze_frequency_content(damped_audio, sample_rate, "Damped Audio",
                                os.path.join(plot_dir, f"spectrum_damped_{timestamp}.png"))

        # List created files
        print("\nCreated audio files:")
        for file in [input_file, anti_wave_file, damped_file]:
            print(f"- {file}")

        print("\nCreated plot files:")
        for file in os.listdir(plot_dir):
            if file.endswith(f"{timestamp}.png"):
                print(f"- {os.path.join(plot_dir, file)}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
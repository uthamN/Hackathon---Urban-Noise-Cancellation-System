import pyaudio
import numpy as np
import threading
from queue import Queue
import time
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for better compatibility
import matplotlib.pyplot as plt

class AudioProcessor:
    def __init__(self):
        self.CHUNK = 512
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 44100
        try:
            self.p = pyaudio.PyAudio()
            # Print available audio devices
            print("\nAvailable Audio Devices:")
            for i in range(self.p.get_device_count()):
                dev_info = self.p.get_device_info_by_index(i)
                print(f"Device {i}: {dev_info['name']}")
        except Exception as e:
            print(f"Error initializing PyAudio: {e}")
            raise
        self.input_queue = Queue()
        self.output_queue = Queue()
        self.is_running = False

    def start_processing(self):
        """Start audio processing"""
        self.is_running = True

        try:
            # Open input stream
            self.input_stream = self.p.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK,
                stream_callback=self.input_callback
            )
        except Exception as e:
            print(f"Error opening input stream: {e}")
            self.stop_processing()
            return

        print("Analyzing incoming audio...")
        analysis_time = 3  # Number of seconds to analyze
        analysis_samples = []

        # Temporarily store and analyze incoming audio
        analysis_start = time.time()
        while time.time() - analysis_start < analysis_time:
            if not self.input_queue.empty():
                audio_data = self.input_queue.get()
                analysis_samples.append(audio_data)
            time.sleep(0.001)  # Small sleep to prevent CPU hogging

        print(f"Number of samples collected: {len(analysis_samples)}")
        if len(analysis_samples) == 0:
            print("No audio samples collected during analysis period!")
            self.stop_processing()
            return

        # Analyze the collected samples
        if analysis_samples:
            try:
                combined_samples = np.concatenate(analysis_samples)

                # Calculate basic audio statistics
                max_amplitude = np.max(np.abs(combined_samples))
                avg_amplitude = np.mean(np.abs(combined_samples))

                # Perform FFT to get frequency components
                fft_data = np.fft.fft(combined_samples)
                freqs = np.fft.fftfreq(len(combined_samples), 1/self.RATE)
                dominant_freq = abs(freqs[np.argmax(np.abs(fft_data))])

                print(f"\nAudio Analysis Results:")
                print(f"Max Amplitude: {max_amplitude:.4f}")
                print(f"Average Amplitude: {avg_amplitude:.4f}")
                print(f"Dominant Frequency: {dominant_freq:.1f} Hz")

                # Generate anti-wave for visualization
                anti_wave = -0.7 * combined_samples

                # Create visualization
                plt.figure(figsize=(15, 8))

                # Plot original wave
                plt.subplot(2, 1, 1)
                plt.plot(np.arange(len(combined_samples))/self.RATE,
                        combined_samples, 'b-', linewidth=1, label='Input Wave')
                plt.title('Input Audio Waveform')
                plt.xlabel('Time (seconds)')
                plt.ylabel('Amplitude')
                plt.grid(True)
                plt.legend()
                plt.ylim(-1, 1)

                # Plot anti-wave
                plt.subplot(2, 1, 2)
                plt.plot(np.arange(len(anti_wave))/self.RATE,
                        anti_wave, 'r-', linewidth=1, label='Anti-Wave')
                plt.title('Generated Anti-Wave (0.7x amplitude)')
                plt.xlabel('Time (seconds)')
                plt.ylabel('Amplitude')
                plt.grid(True)
                plt.legend()
                plt.ylim(-1, 1)

                plt.tight_layout()
                plt.show(block=False)
                plt.pause(3)
                plt.close()

            except Exception as e:
                print(f"Error during analysis and visualization: {e}")
                self.stop_processing()
                return

        try:
            # Open output stream
            self.output_stream = self.p.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                output=True,
                frames_per_buffer=self.CHUNK,
                stream_callback=self.output_callback
            )
        except Exception as e:
            print(f"Error opening output stream: {e}")
            self.stop_processing()
            return

        # Start processing thread
        self.process_thread = threading.Thread(target=self.process_audio)
        self.process_thread.start()

        print("\nReal-time audio processing started...")
        print("Press Ctrl+C to stop")

    def input_callback(self, in_data, frame_count, time_info, status):
        """Handle input audio"""
        if status:
            print(f"Input callback status: {status}")
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        self.input_queue.put(audio_data)
        return (None, pyaudio.paContinue)

    def output_callback(self, in_data, frame_count, time_info, status):
        """Handle output audio"""
        if status:
            print(f"Output callback status: {status}")
        if not self.output_queue.empty():
            data = self.output_queue.get()
            return (data.tobytes(), pyaudio.paContinue)
        return (np.zeros(self.CHUNK, dtype=np.float32).tobytes(), pyaudio.paContinue)

    def process_audio(self):
        """Process audio in real-time"""
        while self.is_running:
            if not self.input_queue.empty():
                audio_data = self.input_queue.get()
                anti_wave = -0.7 * audio_data
                self.output_queue.put(anti_wave)
            else:
                time.sleep(0.001)  # Prevent CPU hogging

    def stop_processing(self):
        """Stop audio processing"""
        self.is_running = False
        if hasattr(self, 'process_thread'):
            self.process_thread.join()

        if hasattr(self, 'input_stream'):
            self.input_stream.stop_stream()
            self.input_stream.close()

        if hasattr(self, 'output_stream'):
            self.output_stream.stop_stream()
            self.output_stream.close()

        self.p.terminate()
        print("\nAudio processing stopped")

def main():
    try:
        audio_proc = AudioProcessor()
        audio_proc.start_processing()

        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping audio processing...")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        audio_proc.stop_processing()

if __name__ == "__main__":
    main()
import pyaudio
import numpy as np
import threading
from queue import Queue
import time
from scipy.signal import find_peaks
from scipy.fftpack import fft
import matplotlib.pyplot as plt
from datetime import datetime

class AudioProcessor:
    def __init__(self):
        self.CHUNK = 256
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 44100
        self.p = pyaudio.PyAudio()

        # Analysis buffers
        self.analysis_duration = 1  # seconds for analysis
        self.analysis_buffer_size = self.RATE * self.analysis_duration
        self.analysis_buffer = np.zeros(self.analysis_buffer_size)

        # Wave characteristics
        self.frequency = None
        self.amplitude = None
        self.phase = None

        # State control
        self.is_analyzing = True
        self.is_running = False
        self.calibration_complete = False

        # Phase alignment parameters
        self.target_phase_alignment = None
        self.phase_aligned = False
        self.sync_buffer_size = 1024
        self.phase_tolerance = 0.1  # radians

        # Queues
        self.input_queue = Queue()
        self.output_queue = Queue()

        # Create timestamp
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"Session timestamp: {self.timestamp}")

    def input_callback(self, in_data, frame_count, time_info, status):
        """Callback for input stream"""
        try:
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            self.input_queue.put(audio_data)
            return (None, pyaudio.paContinue)
        except Exception as e:
            print(f"Error in input callback: {e}")
            return (None, pyaudio.paAbort)

    def output_callback(self, in_data, frame_count, time_info, status):
        """Callback for output stream"""
        try:
            if not self.output_queue.empty():
                data = self.output_queue.get()
                return (data.tobytes(), pyaudio.paContinue)
            return (np.zeros(self.CHUNK, dtype=np.float32).tobytes(), pyaudio.paContinue)
        except Exception as e:
            print(f"Error in output callback: {e}")
            return (None, pyaudio.paAbort)

    def calculate_current_phase(self, data):
        """Calculate the current phase of the signal"""
        if self.frequency is None:
            return None

        zero_crossings = np.where(np.diff(np.signbit(data)))[0]
        if len(zero_crossings) >= 2:
            period_samples = int(self.RATE / self.frequency)
            current_position = zero_crossings[0]
            return 2 * np.pi * (current_position % period_samples) / period_samples
        return None

    def analyze_wave(self, data):
        """Analyze incoming wave to determine frequency, amplitude, and phase"""
        self.analysis_buffer = np.roll(self.analysis_buffer, -len(data))
        self.analysis_buffer[-len(data):] = data

        if np.sum(self.analysis_buffer) == 0:
            return False

        # Perform FFT
        fft_data = fft(self.analysis_buffer)
        frequencies = np.fft.fftfreq(len(self.analysis_buffer), 1/self.RATE)
        magnitude_spectrum = np.abs(fft_data)

        # Find dominant frequency
        peak_idx = np.argmax(magnitude_spectrum[:len(magnitude_spectrum)//2])
        self.frequency = abs(frequencies[peak_idx])

        # Calculate amplitude
        self.amplitude = np.max(np.abs(self.analysis_buffer))

        # Calculate phase
        zero_crossings = np.where(np.diff(np.signbit(self.analysis_buffer)))[0]
        if len(zero_crossings) >= 2:
            period_samples = zero_crossings[1] - zero_crossings[0]
            self.phase = 2 * np.pi * (zero_crossings[0] / period_samples)

        print(f"Analysis Results:")
        print(f"Frequency: {self.frequency:.2f} Hz")
        print(f"Amplitude: {self.amplitude:.4f}")
        print(f"Phase: {self.phase:.4f} radians")

        # Plot waves after analysis
        self.plot_waves()
        return True

    def plot_waves(self):
        """Create and save wave diagrams"""
        if not all([self.frequency, self.amplitude, self.phase is not None]):
            print("Wave characteristics not yet analyzed")
            return

        # Time array for three complete cycles
        cycles_to_show = 3
        samples_per_cycle = int(self.RATE / self.frequency)
        t = np.arange(samples_per_cycle * cycles_to_show) / self.RATE

        # Generate waves
        input_wave = self.amplitude * np.sin(2 * np.pi * self.frequency * t + self.phase)
        anti_wave = -self.amplitude * np.sin(2 * np.pi * self.frequency * t + self.phase)
        resultant = input_wave + anti_wave

        # Create plot
        plt.figure(figsize=(15, 10))

        # Input Wave
        plt.subplot(3, 1, 1)
        plt.plot(t * 1000, input_wave, 'b-', label='Input Wave')
        plt.title(f'Input Wave (Frequency: {self.frequency:.1f} Hz)')
        plt.xlabel('Time (ms)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.legend()

        # Anti-Wave
        plt.subplot(3, 1, 2)
        plt.plot(t * 1000, anti_wave, 'r-', label='Anti-Wave')
        plt.title('Anti-Wave')
        plt.xlabel('Time (ms)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.legend()

        # Resultant Wave
        plt.subplot(3, 1, 3)
        plt.plot(t * 1000, resultant, 'g-', label='Resultant')
        plt.title('Resultant Wave (Expected Cancellation)')
        plt.xlabel('Time (ms)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.legend()

        # Save the plot
        filename = f'wave_analysis_{self.timestamp}.png'
        plt.savefig(filename)
        plt.close()
        print(f"Wave diagram saved as: {filename}")

    def wait_for_phase_alignment(self):
        """Wait for the input signal to reach the desired phase"""
        print("Waiting for phase alignment...")

        self.target_phase_alignment = (self.phase + np.pi) % (2 * np.pi)
        alignment_buffer = np.zeros(self.sync_buffer_size)
        samples_needed = int(self.RATE / self.frequency * 2)

        while not self.phase_aligned and self.is_running:
            if not self.input_queue.empty():
                data = self.input_queue.get()
                current_phase = self.calculate_current_phase(data)

                if current_phase is not None:
                    phase_difference = abs(current_phase - self.target_phase_alignment)
                    phase_difference = min(phase_difference, 2*np.pi - phase_difference)

                    if phase_difference < self.phase_tolerance:
                        print(f"Phase aligned! Difference: {phase_difference:.3f} radians")
                        self.phase_aligned = True
                        break

                    if np.random.random() < 0.1:
                        print(f"Current phase: {current_phase:.2f}, "
                              f"Target: {self.target_phase_alignment:.2f}, "
                              f"Difference: {phase_difference:.2f}")

            time.sleep(0.001)

    def generate_anti_wave(self, num_samples):
        """Generate anti-wave with precise phase alignment"""
        if not all([self.frequency, self.amplitude, self.phase is not None]):
            return np.zeros(num_samples)

        t = np.arange(num_samples) / self.RATE
        anti_wave = -self.amplitude * np.sin(2 * np.pi * self.frequency * t +
                                           self.target_phase_alignment)
        return anti_wave.astype(np.float32)

    def start_processing(self):
        """Start audio processing with analysis phase"""
        print("Starting analysis phase...")
        self.is_running = True
        self.is_analyzing = True

        try:
            self.input_stream = self.p.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK,
                stream_callback=self.input_callback
            )

            self.analysis_thread = threading.Thread(target=self.analysis_phase)
            self.analysis_thread.start()

        except Exception as e:
            print(f"Error starting processing: {e}")
            self.stop_processing()
            raise

    def analysis_phase(self):
        """Initial analysis phase"""
        print("Analyzing input wave...")
        samples_collected = 0
        required_samples = self.RATE * self.analysis_duration

        while samples_collected < required_samples and self.is_analyzing:
            if not self.input_queue.empty():
                data = self.input_queue.get()
                samples_collected += len(data)

                if self.analyze_wave(data):
                    print("Wave analysis complete")
                    self.start_cancellation()
                    break

    def start_cancellation(self):
        """Start the active cancellation phase with phase alignment"""
        print("Preparing for active cancellation...")
        self.is_analyzing = False

        try:
            self.wait_for_phase_alignment()

            if not self.phase_aligned:
                print("Failed to achieve phase alignment")
                self.stop_processing()
                return

            print("Starting phase-aligned cancellation...")
            self.calibration_complete = True

            self.output_stream = self.p.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                output=True,
                frames_per_buffer=self.CHUNK,
                stream_callback=self.output_callback
            )

            self.cancellation_thread = threading.Thread(target=self.cancellation_phase)
            self.cancellation_thread.start()

        except Exception as e:
            print(f"Error starting cancellation: {e}")
            self.stop_processing()
            raise

    def cancellation_phase(self):
        """Active cancellation phase with continuous phase tracking"""
        print("Phase-aligned cancellation active")
        samples_processed = 0
        period_samples = int(self.RATE / self.frequency)

        while self.is_running:
            if not self.input_queue.empty():
                input_data = self.input_queue.get()
                anti_wave = self.generate_anti_wave(self.CHUNK)

                samples_processed += self.CHUNK
                if samples_processed >= period_samples:
                    samples_processed = samples_processed % period_samples
                    current_phase = self.calculate_current_phase(input_data)

                    if current_phase is not None:
                        phase_difference = abs(current_phase - self.target_phase_alignment)
                        phase_difference = min(phase_difference, 2*np.pi - phase_difference)

                        if phase_difference > self.phase_tolerance:
                            print(f"Phase drift detected: {phase_difference:.3f} radians")
                            self.target_phase_alignment = (current_phase + np.pi) % (2 * np.pi)

                self.output_queue.put(anti_wave)

    def stop_processing(self):
        """Stop all processing"""
        print("Stopping processing...")
        self.is_running = False
        self.is_analyzing = False

        if hasattr(self, 'input_stream'):
            self.input_stream.stop_stream()
            self.input_stream.close()

        if hasattr(self, 'output_stream'):
            self.output_stream.stop_stream()
            self.output_stream.close()

        self.p.terminate()
        print("Processing stopped")

def main():
    audio_proc = AudioProcessor()

    try:
        audio_proc.start_processing()

        while not audio_proc.calibration_complete and audio_proc.is_running:
            time.sleep(0.1)

        while audio_proc.is_running:
            time.sleep(1)
            if audio_proc.frequency is not None:
                print(f"Active cancellation at {audio_proc.frequency:.1f} Hz "
                      f"(Phase aligned: {audio_proc.phase_aligned})")

    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"Error in main: {e}")
    finally:
        if hasattr(audio_proc, 'stop_processing'):
            audio_proc.stop_processing()

if __name__ == "__main__":
    main()
import pyaudio
import numpy as np
import threading
from queue import Queue
import time
import noisereduce as nr

class AudioProcessor:
    def __init__(self):
        self.CHUNK = 512  # Smaller chunk size for lower latency
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 44100
        self.p = pyaudio.PyAudio()
        self.input_queue = Queue()
        self.output_queue = Queue()
        self.is_running = False

    def start_processing(self):
        """Start audio processing"""
        self.is_running = True

        # Open input stream
        self.input_stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
            stream_callback=self.input_callback
        )

        # Open output stream
        self.output_stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            output=True,
            frames_per_buffer=self.CHUNK,
            stream_callback=self.output_callback
        )

        # Start processing thread
        self.process_thread = threading.Thread(target=self.process_audio)
        self.process_thread.start()

        print("Real-time audio processing started...")
        print("Press Ctrl+C to stop")

    def input_callback(self, in_data, frame_count, time_info, status):
        """Handle input audio"""
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        self.input_queue.put(audio_data)
        return (None, pyaudio.paContinue)

    def output_callback(self, in_data, frame_count, time_info, status):
        """Handle output audio"""
        if not self.output_queue.empty():
            data = self.output_queue.get()
            return (data.tobytes(), pyaudio.paContinue)
        return (np.zeros(self.CHUNK, dtype=np.float32).tobytes(), pyaudio.paContinue)

    def process_audio(self):
        """Process audio in real-time"""
        while self.is_running:
            if not self.input_queue.empty():
                # Get input audio
                audio_data = self.input_queue.get()

                # Generate anti-wave
                anti_wave = -1 * audio_data

                # Create noise-cancelled audio (original + anti-wave)
                nc_audio = audio_data + anti_wave + (0.1*audio_data)
                # nc_audio = anti_wave
                reduced_noise = nr.reduce_noise(y=audio_data, sr=44100)

                # Send to output queue
                self.output_queue.put(reduced_noise)

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
    audio_proc = AudioProcessor()

    try:
        audio_proc.start_processing()
        while True:
            time.sleep(0.1)  # Keep the main thread alive
    except KeyboardInterrupt:
        print("\nStopping audio processing...")
    finally:
        audio_proc.stop_processing()

if __name__ == "__main__":
    main()
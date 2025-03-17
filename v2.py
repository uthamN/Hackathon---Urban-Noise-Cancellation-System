import pyaudio
import numpy as np
import threading
from collections import deque
import time
import psutil
import os
import platform

class LowLatencyAudio:
    def __init__(self):
        # Optimize buffer size and format
        self.CHUNK = 32  # Smaller chunks for lower latency
        self.FORMAT = pyaudio.paFloat32  # Use float32 for better quality
        self.CHANNELS = 1  # Mono for less processing
        self.RATE = 48000  # Higher sample rate for better quality

        # Use ring buffer for smooth streaming
        self.buffer = deque(maxlen=32)
        self.is_running = False

        # Threading events
        self.audio_event = threading.Event()

        # Performance monitoring
        self.latency_measurements = []

        # Initialize PyAudio with optimized settings
        self.init_audio()

    def init_audio(self):
        self.pa = pyaudio.PyAudio()

        # Get default device info
        def_input = self.pa.get_default_input_device_info()
        def_output = self.pa.get_default_output_device_info()

        # Set up optimized streams
        self.input_stream = self.pa.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
            input_device_index=def_input['index'],
            stream_callback=self.input_callback
        )

        self.output_stream = self.pa.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            output=True,
            frames_per_buffer=self.CHUNK,
            output_device_index=def_output['index'],
            stream_callback=self.output_callback
        )

    def optimize_system(self):
        """Optimize system settings for low latency"""
        try:
            # Set process priority
            process = psutil.Process()

            if platform.system() == 'Windows':
                process.nice(psutil.HIGH_PRIORITY_CLASS)
            else:  # Linux/Unix
                os.nice(-20)  # Highest priority

            # Disable CPU throttling (Linux only)
            if platform.system() == 'Linux':
                os.system('echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor')

        except Exception as e:
            print(f"Warning: Could not optimize system settings: {e}")

    def input_callback(self, in_data, frame_count, time_info, status):
        """Optimized input callback"""
        try:
            start_time = time.perf_counter()

            # Convert to numpy array for faster processing
            audio_data = np.frombuffer(in_data, dtype=np.float32)

            # Basic noise gate
            audio_data[abs(audio_data) < 0.01] = 0

            # Add to buffer
            self.buffer.append(audio_data)

            # Measure callback latency
            latency = (time.perf_counter() - start_time) * 1000
            self.latency_measurements.append(latency)

            return (None, pyaudio.paContinue)
        except Exception as e:
            print(f"Input callback error: {e}")
            return (None, pyaudio.paAbort)

    def output_callback(self, in_data, frame_count, time_info, status):
        """Optimized output callback"""
        try:
            if len(self.buffer) > 0:
                data = self.buffer.popleft()
                return (data.tobytes(), pyaudio.paContinue)
            return (np.zeros(frame_count, dtype=np.float32).tobytes(), pyaudio.paContinue)
        except Exception as e:
            print(f"Output callback error: {e}")
            return (np.zeros(frame_count, dtype=np.float32).tobytes(), pyaudio.paAbort)

    def monitor_performance(self):
        """Monitor and report performance metrics"""
        while self.is_running:
            if len(self.latency_measurements) > 100:
                avg_latency = np.mean(self.latency_measurements[-100:])
                max_latency = np.max(self.latency_measurements[-100:])
                print(f"Average latency: {avg_latency:.2f}ms, Max latency: {max_latency:.2f}ms")
                print(f"Buffer usage: {len(self.buffer)}/{self.buffer.maxlen}")
                self.latency_measurements = self.latency_measurements[-100:]
            time.sleep(1)

    def start(self):
        """Start audio processing"""
        self.optimize_system()
        self.is_running = True

        # Start performance monitoring in separate thread
        self.monitor_thread = threading.Thread(target=self.monitor_performance)
        self.monitor_thread.start()

        # Start audio streams
        self.input_stream.start_stream()
        self.output_stream.start_stream()

        print("Audio processing started with optimized settings")
        print(f"Buffer size: {self.CHUNK} samples")
        print(f"Sample rate: {self.RATE} Hz")

    def stop(self):
        """Stop audio processing"""
        self.is_running = False

        # Stop streams
        self.input_stream.stop_stream()
        self.output_stream.stop_stream()
        self.input_stream.close()
        self.output_stream.close()

        # Clean up
        self.pa.terminate()
        print("Audio processing stopped")

def main():
    audio = LowLatencyAudio()
    try:
        audio.start()
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping audio processing...")
        audio.stop()

if __name__ == "__main__":
    main()
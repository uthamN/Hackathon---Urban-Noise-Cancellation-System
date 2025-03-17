import pyaudio
import numpy as np
import threading
import time
from numba import jit, float32
import psutil
import os
from collections import deque

@jit(nopython=True, fastmath=True)
def process_audio_effects(data):
    """Numba-optimized audio effects processing"""
    result = np.empty_like(data)
    for i in range(len(data)):
        # Original distortion effect
        sample = data[i]
        # Combining negative feedback and distortion
        processed = -(0.5 * sample)
        result[i] = processed
    return result

class OptimizedRingBuffer:
    def __init__(self, size: int, dtype=np.float32):
        self.size = size
        self.data = np.zeros(size, dtype=dtype)
        self.write_ptr = 0
        self.read_ptr = 0
        self.lock = threading.Lock()
        self.available = 0

    def write(self, data: np.ndarray) -> bool:
        with self.lock:
            data_len = len(data)
            if data_len > self.size:
                return False

            space_available = self.size - self.available
            if space_available < data_len:
                return False

            write_end = self.write_ptr + data_len
            if write_end <= self.size:
                self.data[self.write_ptr:write_end] = data
            else:
                first_part = self.size - self.write_ptr
                self.data[self.write_ptr:] = data[:first_part]
                self.data[:data_len-first_part] = data[first_part:]

            self.write_ptr = write_end % self.size
            self.available += data_len
            return True

    def read(self, size: int) -> np.ndarray:
        with self.lock:
            if self.available < size:
                return np.zeros(size, dtype=self.data.dtype)

            if self.read_ptr + size <= self.size:
                result = self.data[self.read_ptr:self.read_ptr + size].copy()
            else:
                first_part = self.size - self.read_ptr
                result = np.empty(size, dtype=self.data.dtype)
                result[:first_part] = self.data[self.read_ptr:]
                result[first_part:] = self.data[:size-first_part]

            self.read_ptr = (self.read_ptr + size) % self.size
            self.available -= size
            return result

class AudioProcessor:
    def __init__(self):
        # Audio settings
        self.CHUNK = 64  # Small chunk size for more frequent processing
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 44100
        self.BUFFER_SIZE = 4096

        # Initialize buffers
        self.input_buffer = OptimizedRingBuffer(self.BUFFER_SIZE)
        self.output_buffer = OptimizedRingBuffer(self.BUFFER_SIZE)

        # Pre-allocate arrays
        self.zero_data = np.zeros(self.CHUNK, dtype=np.float32)

        # Performance monitoring
        self.processing_times = deque(maxlen=100)
        self.last_time = time.perf_counter()

        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        self.is_running = False

    def optimize_system(self):
        """Optimize system settings"""
        try:
            process = psutil.Process()
            if os.name == 'posix':
                os.nice(-10)
            else:
                process.nice(psutil.HIGH_PRIORITY_CLASS)

            # Set CPU affinity
            cpu_count = psutil.cpu_count()
            if cpu_count > 2:
                process.cpu_affinity([cpu_count - 1])
        except Exception as e:
            print(f"System optimization failed: {e}")

    def start_processing(self):
        self.optimize_system()
        self.is_running = True

        # Configure stream
        stream_kwargs = {
            'format': self.FORMAT,
            'channels': self.CHANNELS,
            'rate': self.RATE,
            'frames_per_buffer': self.CHUNK,
            'input': True,
            'output': True,
            'stream_callback': None
        }

        # Open audio stream
        self.stream = self.p.open(**stream_kwargs)
        print("Audio destruction processing started...")
        print("Press Ctrl+C to stop")

        # Pre-compile Numba function
        dummy_data = np.zeros(self.CHUNK, dtype=np.float32)
        _ = process_audio_effects(dummy_data)

        # Main processing loop
        while self.is_running:
            try:
                start_time = time.perf_counter()

                # Read input
                input_data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                audio_data = np.frombuffer(input_data, dtype=np.float32)

                # Process audio with distortion effects
                processed_data = process_audio_effects(audio_data)

                # Write output
                self.stream.write(processed_data.tobytes())

                # Performance monitoring
                process_time = time.perf_counter() - start_time
                self.processing_times.append(process_time)

            except Exception as e:
                print(f"Error: {e}")
                time.sleep(0.001)

    def stop_processing(self):
        self.is_running = False

        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()

        self.p.terminate()

        # Print performance statistics
        if self.processing_times:
            avg_time = np.mean(self.processing_times) * 1000
            max_time = np.max(self.processing_times) * 1000
            print(f"\nPerformance Statistics:")
            print(f"Average processing time: {avg_time:.2f}ms")
            print(f"Maximum processing time: {max_time:.2f}ms")
        print("\nAudio processing stopped")

def main():
    audio_proc = AudioProcessor()

    try:
        audio_proc.start_processing()
    except KeyboardInterrupt:
        print("\nStopping audio processing...")
    finally:
        audio_proc.stop_processing()

if __name__ == "__main__":
    main()
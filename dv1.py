import numpy as np
import sounddevice as sd
import threading
import queue
import keyboard
import time
from scipy import signal

class RealtimeWaveDampener:
    def __init__(self):
        # Audio parameters
        self.CHANNELS = 1
        self.RATE = 44100
        self.CHUNK = 1024
        self.DTYPE = np.float32

        # Processing parameters
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.running = False
        self.dampening_active = True
        self.dampening_ratio = 1.0

        # Status monitoring
        self.input_level = 0
        self.output_level = 0

    def audio_callback(self, indata, outdata, frames, time, status):
        """Handle real-time audio I/O"""
        if status:
            print(f"Status: {status}")

        try:
            # Process input
            input_data = np.mean(indata, axis=1) if indata.shape[1] > 1 else indata[:, 0]
            self.input_queue.put(input_data.copy())

            # Get processed output
            if not self.output_queue.empty():
                output_data = self.output_queue.get()
                outdata[:] = output_data.reshape(-1, 1)
            else:
                outdata[:] = indata  # Fallback to direct passthrough

        except Exception as e:
            print(f"Callback error: {e}")
            outdata[:] = indata

    def process_audio(self):
        """Process audio in real-time"""
        while self.running:
            try:
                if not self.input_queue.empty():
                    # Get input data
                    input_data = self.input_queue.get()

                    # Calculate input level
                    self.input_level = np.max(np.abs(input_data))

                    if self.dampening_active:
                        # Generate dampening wave
                        dampening_wave = self.generate_dampening_wave(input_data)

                        # Mix original and dampening
                        output_data = input_data + (dampening_wave * self.dampening_ratio)

                        # Normalize to prevent clipping
                        output_data = np.clip(output_data, -1, 1)
                    else:
                        output_data = input_data

                    # Calculate output level
                    self.output_level = np.max(np.abs(output_data))

                    # Send to output queue
                    self.output_queue.put(output_data)

            except Exception as e:
                print(f"Processing error: {e}")

            time.sleep(0.001)  # Prevent CPU overload

    def generate_dampening_wave(self, input_data):
        """Generate dampening wave based on input"""
        try:
            # Perform FFT
            fft_data = np.fft.rfft(input_data)
            freqs = np.fft.rfftfreq(len(input_data), 1/self.RATE)

            # Generate inverse phase wave
            phase_shift = np.pi  # 180 degree phase shift
            dampening_wave = np.fft.irfft(fft_data * np.exp(1j * phase_shift))

            return dampening_wave

        except Exception as e:
            print(f"Dampening generation error: {e}")
            return np.zeros_like(input_data)

    def display_status(self):
        """Display real-time status"""
        while self.running:
            try:
                print("\033[H\033[J")  # Clear screen
                print("=== Realtime Wave Dampener ===")
                print(f"Dampening: {'ON' if self.dampening_active else 'OFF'}")
                print(f"Dampening Ratio: {self.dampening_ratio:.2f}")
                print(f"Input Level: {'#' * int(self.input_level * 50)}")
                print(f"Output Level: {'#' * int(self.output_level * 50)}")
                print("\nControls:")
                print("SPACE: Toggle dampening")
                print("UP/DOWN: Adjust ratio")
                print("Q: Quit")

                time.sleep(0.1)  # Update rate

            except Exception as e:
                print(f"Display error: {e}")

    def handle_keyboard(self):
        """Handle keyboard controls"""
        while self.running:
            try:
                if keyboard.is_pressed('space'):
                    self.dampening_active = not self.dampening_active
                    time.sleep(0.2)  # Debounce

                elif keyboard.is_pressed('up'):
                    self.dampening_ratio = min(2.0, self.dampening_ratio + 0.1)
                    time.sleep(0.1)

                elif keyboard.is_pressed('down'):
                    self.dampening_ratio = max(0.0, self.dampening_ratio - 0.1)
                    time.sleep(0.1)

                elif keyboard.is_pressed('q'):
                    self.running = False
                    break

            except Exception as e:
                print(f"Keyboard handler error: {e}")

            time.sleep(0.01)

    def run(self):
        """Main run loop"""
        try:
            print("Starting Wave Dampener...")
            self.running = True

            # Start processing thread
            process_thread = threading.Thread(target=self.process_audio)
            process_thread.start()

            # Start display thread
            display_thread = threading.Thread(target=self.display_status)
            display_thread.start()

            # Start keyboard handler
            keyboard_thread = threading.Thread(target=self.handle_keyboard)
            keyboard_thread.start()

            # Start audio stream
            with sd.Stream(channels=self.CHANNELS,
                         samplerate=self.RATE,
                         blocksize=self.CHUNK,
                         dtype=self.DTYPE,
                         callback=self.audio_callback):

                while self.running:
                    time.sleep(0.1)

            # Wait for threads to finish
            process_thread.join()
            display_thread.join()
            keyboard_thread.join()

        except Exception as e:
            print(f"Runtime error: {e}")
        finally:
            self.running = False
            print("\nShutdown complete")

if __name__ == "__main__":
    dampener = RealtimeWaveDampener()
    dampener.run()
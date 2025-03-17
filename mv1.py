import sys
import math
import wave
import struct
import curses
import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Any

class NoiseReducer:
    def __init__(self):
        # 'curses' configuration
        self.stdscr = curses.initscr()
        self.stdscr.nodelay(True)
        curses.noecho()
        curses.cbreak()

        # PyAudio object variable
        self.pa = pyaudio.PyAudio()

        # Constants
        self.CHUNK = 1024  # Increased for better performance
        self.CHANNELS = 2
        self.WIDTH = 2
        self.SAMPLE_RATE = 44100
        self.NTH_ITERATION = 10  # Default value

    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            curses.endwin()
            self.pa.terminate()
        except:
            pass

    def process_args(self) -> None:
        """Process command line arguments"""
        if len(sys.argv) < 2:
            print("Usage: python3 script.py [-f|--file|-l|--live|-p|--playback] [options]")
            sys.exit(1)

        self.MODE = sys.argv[1]
        if self.MODE not in ['-f', '--file', '-l', '--live', '-p', '--playback']:
            print('Invalid mode. Use -f, -l, or -p')
            sys.exit(1)

        if self.MODE != '-p' and self.MODE != '--playback':
            try:
                self.NTH_ITERATION = int(sys.argv[3])
            except (ValueError, IndexError):
                print('The second argument must be a number')
                sys.exit(1)

    def invert(self, data: bytes) -> bytes:
        """Invert audio data"""
        try:
            intwave = np.frombuffer(data, dtype=np.int16)
            inverted = -intwave  # Simple phase inversion
            return inverted.tobytes()
        except Exception as e:
            print(f"Error inverting audio: {e}")
            return data

    def mix_samples(self, sample_1: bytes, sample_2: bytes, ratio: float) -> bytes:
        """Mix two audio samples with given ratio"""
        try:
            ratio_1, ratio_2 = self.get_ratios(ratio)
            s1 = np.frombuffer(sample_1, dtype=np.int16)
            s2 = np.frombuffer(sample_2, dtype=np.int16)
            mixed = (s1 * ratio_1 + s2 * ratio_2).astype(np.int16)
            return mixed.tobytes()
        except Exception as e:
            print(f"Error mixing samples: {e}")
            return sample_1

    def get_ratios(self, ratio: float) -> Tuple[float, float]:
        """Calculate mixing ratios"""
        ratio = max(0.0, min(2.0, float(ratio)))  # Clamp between 0 and 2
        ratio_1 = ratio / 2
        ratio_2 = (2 - ratio) / 2
        return ratio_1, ratio_2

    def calculate_decibel(self, data: bytes) -> float:
        """Calculate decibel level of audio data"""
        try:
            samples = np.frombuffer(data, dtype=np.int16)
            rms = np.sqrt(np.mean(np.square(samples.astype(np.float32) / 32768.0))) + 0.0001
            return 20 * np.log10(rms)
        except Exception as e:
            print(f"Error calculating decibel: {e}")
            return 0.0

    def file_mode(self, filename: str) -> None:
        """Process audio file"""
        try:
            wf = wave.open(filename, 'rb')
            stream = self.pa.open(
                format=self.pa.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True
            )

            self.stdscr.addstr('Processing file...\n')

            decibel_levels = []
            active = True
            ratio = 1.0

            while True:
                data = wf.readframes(self.CHUNK)
                if not data:
                    break

                key = self.stdscr.getch()
                if key == ord('o'):
                    active = not active
                    ratio = 2.0 if not active else 1.0
                elif key == ord('x'):
                    break

                inverted = self.invert(data)
                if active:
                    output = self.mix_samples(data, inverted, ratio)
                    stream.write(output)
                    decibel_levels.append(self.calculate_decibel(output))

            stream.stop_stream()
            stream.close()
            wf.close()

            # Plot results if requested
            if len(decibel_levels) > 0:
                plt.plot(decibel_levels)
                plt.show()

        except Exception as e:
            print(f"Error in file mode: {e}")
        finally:
            self.cleanup()

    def live_mode(self) -> None:
        """Process live audio"""
        try:
            stream = self.pa.open(
                format=pyaudio.paInt16,
                channels=self.CHANNELS,
                rate=self.SAMPLE_RATE,
                input=True,
                output=True,
                frames_per_buffer=self.CHUNK
            )

            self.stdscr.addstr('Processing live audio...\n')

            while True:
                data = stream.read(self.CHUNK)
                inverted = self.invert(data)
                stream.write(inverted, self.CHUNK)

                if self.stdscr.getch() == ord('x'):
                    break

            stream.stop_stream()
            stream.close()

        except Exception as e:
            print(f"Error in live mode: {e}")
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Cleanup resources"""
        curses.endwin()
        self.pa.terminate()

    def run(self) -> None:
        """Main run method"""
        try:
            self.process_args()

            if self.MODE in ['-f', '--file']:
                self.file_mode(sys.argv[4])
            elif self.MODE in ['-l', '--live']:
                self.live_mode()
            elif self.MODE in ['-p', '--playback']:
                self.file_mode(sys.argv[2])

        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.cleanup()

if __name__ == "__main__":
    reducer = NoiseReducer()
    reducer.run()
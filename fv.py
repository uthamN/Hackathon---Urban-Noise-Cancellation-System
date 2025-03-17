import numpy as np
import sounddevice as sd
import queue
import threading
import signal
import sys
import time

class AdaptiveNoiseFilter:
    def __init__(self, filter_length=128, mu=0.05):
        self.filter_length = filter_length
        self.mu = mu
        self.weights = np.zeros(filter_length)
        self.buffer = np.zeros(filter_length)

    def update(self, reference, desired):
        self.buffer = np.roll(self.buffer, 1)
        self.buffer[0] = reference
        output = np.dot(self.weights, self.buffer)
        error = desired - output
        self.weights += self.mu * error * self.buffer
        return output, error

class NoiseReductionSystem:
    def __init__(self, input_device=None, output_device=None,
                 sample_rate=44100, block_size=1024):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.input_device = input_device
        self.output_device = output_device
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.adaptive_filter = AdaptiveNoiseFilter()
        self.running = False
        self.processing_thread = None

    def input_callback(self, indata, frames, time, status):
        if status:
            print(f"Input status: {status}")
        self.input_queue.put(indata.copy())

    def output_callback(self, outdata, frames, time, status):
        if status:
            print(f"Output status: {status}")
        try:
            data = self.output_queue.get_nowait()
            outdata[:] = data
        except queue.Empty:
            outdata.fill(0)

    def process_audio(self):
        print("Processing started. Press Ctrl+C to stop.")
        while self.running:
            try:
                # Get input data
                input_data = self.input_queue.get(timeout=0.1)

                # Process each channel
                # reference = input_data[:, 0]
                # desired = input_data[:, 1]

                reference = input_data[:, 0]
                desired = reference  


                # Process through adaptive filter
                output_buffer = np.zeros(len(reference))
                error_buffer = np.zeros(len(reference))

                for i in range(len(reference)):
                    output_buffer[i], error_buffer[i] = self.adaptive_filter.update(
                        reference[i], desired[i])

                # Prepare anti-noise output
                output_data = -output_buffer.reshape(-1, 1)
                output_data = np.repeat(output_data, 2, axis=1)

                # Send to output queue
                self.output_queue.put(output_data.astype(np.float32))

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")
                continue

    def start(self):
        """Start the noise reduction system"""
        if self.running:
            print("System is already running!")
            return

        self.running = True

        # Initialize streams
        try:
            self.input_stream = sd.InputStream(
                device=self.input_device,
                channels=1,
                callback=self.input_callback,
                samplerate=self.sample_rate,
                blocksize=self.block_size
            )

            self.output_stream = sd.OutputStream(
                device=self.output_device,
                channels=2,
                callback=self.output_callback,
                samplerate=self.sample_rate,
                blocksize=self.block_size
            )

            # Start processing thread
            self.processing_thread = threading.Thread(target=self.process_audio)
            self.processing_thread.daemon = True
            self.processing_thread.start()

            # Start audio streams
            self.input_stream.start()
            self.output_stream.start()

            print("Noise reduction system started successfully!")

        except Exception as e:
            print(f"Error starting system: {e}")
            self.stop()
            raise

    def stop(self):
        """Stop the noise reduction system"""
        print("Stopping noise reduction system...")
        self.running = False

        # Stop and close streams
        if hasattr(self, 'input_stream'):
            try:
                self.input_stream.stop()
                self.input_stream.close()
            except Exception as e:
                print(f"Error closing input stream: {e}")

        if hasattr(self, 'output_stream'):
            try:
                self.output_stream.stop()
                self.output_stream.close()
            except Exception as e:
                print(f"Error closing output stream: {e}")

        # Wait for processing thread to finish
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)

        # Clear queues
        while not self.input_queue.empty():
            try:
                self.input_queue.get_nowait()
            except queue.Empty:
                break

        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except queue.Empty:
                break

        print("Noise reduction system stopped.")

def handle_interrupt(nr_system):
    """Handle keyboard interrupt"""
    def signal_handler(signum, frame):
        print("\nInterrupt received. Shutting down...")
        nr_system.stop()
        sys.exit(0)
    return signal_handler

def main():
    # Print available audio devices
    # print("\nAvailable audio devices:")
    # print(sd.query_devices())

    try:
        # Create noise reduction system
        nr_system = NoiseReductionSystem(
            input_device=None,  # Default input device
            output_device=None, # Default output device
            sample_rate=44100,
            block_size=1024
        )

        # Set up signal handler
        signal.signal(signal.SIGINT, handle_interrupt(nr_system))

        # Start the system
        print("\nStarting noise reduction system...")
        nr_system.start()

        # Keep the main thread alive
        while True:
            time.sleep(1)  # Sleep to prevent CPU hogging

    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Ensure cleanup happens
        if 'nr_system' in locals():
            nr_system.stop()

if __name__ == "__main__":
    main()

# Created/Modified files during execution:
# No files are created or modified during execution
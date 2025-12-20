import csv
import time
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import serial


class LiveUltrasonicLogger:
    """
    Reads streaming sensor data from an ESP32 via serial,
    logs it continuously to a CSV file, and visualizes the
    measurements in real time using Matplotlib.
    """

    def __init__(
        self,
        port: str = "COM8",
        baudrate: int = 115200,
        max_points: int = 800,
    ):
        self.port = port
        self.baudrate = baudrate
        self.max_points = max_points

        self.csv_path = (
            Path(__file__).resolve().parent.joinpath("..", "..", "data", "raw").resolve()
            / f"esp32_scan_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        )
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        self.ser = None

    def run(self) -> None:
        # Open serial connection to the ESP32
        self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
        print(f"\nConnected to {self.port} at {self.baudrate} baud.")
        print(f"Logging to: {self.csv_path}\n")

        # Fixed-size buffers for live plotting (sliding window)
        steps = deque(maxlen=self.max_points)
        distances = deque(maxlen=self.max_points)

        # Enable interactive plotting mode
        plt.ion()
        fig, ax = plt.subplots()

        # Initialize an empty line object for live updates
        (line,) = ax.plot([], [], "-o", markersize=2)
        ax.set_xlabel("step")
        ax.set_ylabel("distance_cm")
        ax.set_title("Live Ultrasonic Measurement")
        ax.grid(True)

        # Open CSV file for continuous logging
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Write CSV header once at the beginning
            writer.writerow(["scan_index", "step", "angle_deg", "distance_cm"])

            try:
                # Main loop: runs until interrupted by the user
                while True:
                    # Read one line from the serial buffer
                    raw = self.ser.readline().decode("utf-8", errors="ignore").strip()
                    if not raw:
                        continue  # Skip empty lines

                    # Handle comment or debug messages from ESP32
                    if raw.startswith("#"):
                        print(raw)
                        continue

                    # Ignore repeated CSV headers sent by the device
                    if raw.lower().startswith("scan_index"):
                        continue

                    # Expect exactly four comma-separated values
                    parts = raw.split(",")
                    if len(parts) != 4:
                        continue

                    scan_index_str, step_str, angle_str, dist_str = parts

                    # Convert incoming strings to numeric values
                    try:
                        scan_index = int(scan_index_str)
                        step = int(step_str)
                        angle = float(angle_str)
                        distance = float(dist_str) if dist_str != "NaN" else float("nan")
                    except ValueError:
                        continue  # Skip malformed data

                    # Write every measurement to the CSV file (always log; including NaN is OK)
                    writer.writerow([scan_index, step, angle, distance])
                    f.flush()  # # Ensure data is written immediately to disk

                    # # Skip NaN values for live plotting
                    if dist_str == "NaN":
                        continue

                    # Update sliding window buffers
                    steps.append(step)
                    distances.append(distance)

                    # Update plot data
                    xs = list(steps)
                    ys = list(distances)
                    line.set_xdata(xs)
                    line.set_ydata(ys)

                    # Dynamically adjust x-axis limits
                    if xs:
                        ax.set_xlim(min(xs), max(xs) + 1)

                    # Dynamically adjust y-axis limits
                    if ys:
                        ymin = min(ys)
                        ymax = max(ys)
                        if ymin == ymax:
                            ymin -= 1
                            ymax += 1
                        ax.set_ylim(ymin - 1, ymax + 1)

                    # Small pause to allow the plot to refresh
                    plt.pause(0.001)

            except KeyboardInterrupt:
                print("Stopped by user.")  # shutdown when user presses Ctrl+C
            finally:
                # Always close serial port and finalize plotting
                if self.ser:
                    self.ser.close()
                plt.ioff()
                plt.show()


if __name__ == "__main__":
    LiveUltrasonicLogger().run()

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

        # CSV writer resources
        self._csv_file = None
        self._csv_writer = None

        # Live plot resources
        self._steps = deque(maxlen=self.max_points)
        self._distances = deque(maxlen=self.max_points)
        self._fig = None
        self._ax = None
        self._line = None

    def _open_serial(self) -> None:
        # Open serial connection to the ESP32
        self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
        print(f"\nConnected to {self.port} at {self.baudrate} baud.")
        print(f"Logging to: {self.csv_path}\n")

    def _setup_csv_writer(self) -> None:
        # Open CSV file for continuous logging
        self._csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_file)

        # Write CSV header once at the beginning
        self._csv_writer.writerow(["scan_index", "step", "angle_deg", "distance_cm"])

    def _setup_live_plot(self) -> None:
        # Enable interactive plotting mode
        plt.ion()
        self._fig, self._ax = plt.subplots()

        # Initialize an empty line object for live updates
        (self._line,) = self._ax.plot([], [], "-o", markersize=2)
        self._ax.set_xlabel("step")
        self._ax.set_ylabel("distance_cm")
        self._ax.set_title("Live Ultrasonic Measurement")
        self._ax.grid(True)

    def _parse_serial_line(self, raw: str):
        # Read one line from the serial buffer
        raw = raw.strip()
        if not raw:
            return None  # Skip empty lines

        # Handle comment or debug messages from ESP32
        if raw.startswith("#"):
            print(raw)
            return None

        # Ignore repeated CSV headers sent by the device
        if raw.lower().startswith("scan_index"):
            return None

        # Expect exactly four comma-separated values
        parts = raw.split(",")
        if len(parts) != 4:
            return None

        scan_index_str, step_str, angle_str, dist_str = parts

        # Convert incoming strings to numeric values
        try:
            scan_index = int(scan_index_str)
            step = int(step_str)
            angle = float(angle_str)
            distance = float(dist_str) if dist_str != "NaN" else float("nan")
        except ValueError:
            return None  # Skip malformed data

        return scan_index, step, angle, distance, dist_str

    def _copy_to_csv(self, scan_index: int, step: int, angle: float, distance: float) -> None:
        # Write every measurement to the CSV file (always log; including NaN is OK)
        self._csv_writer.writerow([scan_index, step, angle, distance])  # type: ignore
        self._csv_file.flush()  # Ensure data is written immediately to disk # type: ignore

    def _update_live_plot(self, step: int, distance: float, dist_str: str) -> None:
        # # Skip NaN values for live plotting
        if dist_str == "NaN":
            return

        # Update sliding window buffers
        self._steps.append(step)
        self._distances.append(distance)

        # Update plot data
        xs = list(self._steps)
        ys = list(self._distances)
        self._line.set_xdata(xs)  # type: ignore
        self._line.set_ydata(ys)  # type: ignore

        # Dynamically adjust x-axis limits
        if xs:
            self._ax.set_xlim(min(xs), max(xs) + 1)  # type: ignore

        # Dynamically adjust y-axis limits
        if ys:
            ymin = min(ys)
            ymax = max(ys)
            if ymin == ymax:
                ymin -= 1
                ymax += 1
            self._ax.set_ylim(ymin - 1, ymax + 1)  # type: ignore

        # Small pause to allow the plot to refresh
        plt.pause(0.001)

    def main_run(self) -> None:
        self._open_serial()
        self._setup_live_plot()
        self._setup_csv_writer()

        try:
            # Main loop: runs until interrupted by the user
            while True:
                raw = self.ser.readline().decode("utf-8", errors="ignore")  # type: ignore
                parsed = self._parse_serial_line(raw)
                if parsed is None:
                    continue

                scan_index, step, angle, distance, dist_str = parsed

                # CSV logging
                self._copy_to_csv(scan_index, step, angle, distance)

                # Live plot update
                self._update_live_plot(step, distance, dist_str)

        except KeyboardInterrupt:
            print("Stopped by user.")  # shutdown when user presses Ctrl+C
        finally:
            # Always close serial port and finalize plotting
            try:
                if self.ser:
                    self.ser.close()
            finally:
                if self._csv_file:
                    self._csv_file.close()
                plt.ioff()
                plt.show()


if __name__ == "__main__":
    LiveUltrasonicLogger().main_run()

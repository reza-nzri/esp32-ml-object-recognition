import serial
import matplotlib.pyplot as plt
from collections import deque

BAUDRATE = 115200

def main():
    ser = serial.Serial(PORT, BAUDRATE, timeout=1)
    print(f"Connected to {PORT} at {BAUDRATE} baud.")

    steps = deque(maxlen=MAX_POINTS)
    distances = deque(maxlen=MAX_POINTS)

    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [], "-o", markersize=2)
    ax.set_xlabel("step")
    ax.set_ylabel("distance_cm")
    ax.set_title("Live Ultrasonic Measurement")
    ax.grid(True)

    try:
        while True:
            raw = ser.readline().decode("utf-8", errors="ignore").strip()

            if not raw:
                continue

            if raw.startswith("#"):
                print(raw)
                continue

            parts = raw.split(",")
            if len(parts) != 4:
                print("Skip:", raw)
                continue

            scan_id_str, step_str, angle_str, dist_str = parts

            if dist_str == "NaN":
                print("No echo:", raw)
                continue

            try:
                step = int(step_str)
                distance = float(dist_str)
            except ValueError:
                print("Parse error:", raw)
                continue

            steps.append(step)
            distances.append(distance)

            line.set_xdata(list(steps))
            line.set_ydata(list(distances))

            if steps:
                ax.set_xlim(min(steps), max(steps) + 1)
                ax.set_ylim(min(distances) - 1, max(distances) + 1)

            plt.pause(0.001)

    except KeyboardInterrupt:
        print("Stopped by user.")

    finally:
        ser.close()
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
    
# 🤖 ESP32 ML Object Recognition Scanner

> **ESP32-based rotating ultrasonic scanner**  
> Data acquisition on embedded hardware + ML-ready dataset generation on PC.

## 📌 Overview

This project builds a **rotating distance scanner** using an **ESP32**, a **28BYJ-48 stepper motor**, and an **HC-SR04 ultrasonic sensor**.  
The ESP32 performs **angle-aware distance measurements**, streams them over **Serial**, and the PC logs + visualizes the data for **machine learning workflows**.

Typical use cases:
- 2D environment scanning
- Obstacle mapping
- Dataset generation for ML / DL
- Embedded + Python integration practice

<details>
<summary>🧠 System Architecture</summary>

### Hardware
- ESP32 Dev Board
- 28BYJ-48 5V Stepper Motor
- ULN2003 Driver Board
- HC-SR04 Ultrasonic Sensor
- External 5V power (motor)

### Software
- **MicroPython** on ESP32
- **Python 3** on PC
- CSV logging + live plotting
- ML-ready data pipeline

### Data Flow

```
Stepper rotation → distance measurement → Serial stream
→ PC logger → CSV → visualization → ML training
```

</details>

<details>
<summary>🔩 Hardware Components</summary>

| Component | Purpose |
|---------|--------|
| ESP32 | Main controller & Serial communication |
| 28BYJ-48 | Rotational scanning |
| ULN2003 | Motor driver (Darlington array) |
| HC-SR04 | Distance measurement |
| Resistor + Diode | Level protection (5V → 3.3V) |

⚠️ **Important:**  
HC-SR04 Echo pin outputs **5V** → must be protected before ESP32 GPIO.</details>

<details>
<summary>⚙️ How It Works</summary>

1. Stepper motor rotates in **fixed step intervals**
2. Every *N steps* → ultrasonic measurement
3. ESP32 prints:
```
scan_index, step, angle_deg, distance_cm
````

4. PC script:
- Reads Serial
- Saves CSV
- Updates live plot
5. Result → clean dataset for ML

✔️ Angle calculation is deterministic  
✔️ Step count stays consistent  
✔️ One measurement per defined step window  
</details>

<details>
<summary>📊 Data Format</summary>

Example CSV row:
```csv
1,120,10.55,43.2
````

| Column      | Description         |
| ----------- | ------------------- |
| scan_index     | Unique scan session |
| step        | Motor step count    |
| angle_deg   | Calculated angle    |
| distance_cm | Measured distance   |

ML-friendly. No post-cleanup needed.
</details>

<details>
<summary>🚀 Running the Project</summary>

### ESP32

* Flash **MicroPython**
* Upload `device_esp32/` files
* Power motor separately
* Open Serial @ correct baudrate

### PC

```bash
uv sync
uv run python training_pc/live_plot.py
```

Live plot starts automatically. CSV is saved to `training_pc/data/raw/`.
</details>

<details>
<summary>🧪 Machine Learning Usage</summary>

You can use the generated data for:

* Clustering
* Obstacle classification
* Shape detection
* 2D occupancy grids
* Supervised / unsupervised learning

Notebooks are provided for:

* Exploration
* Training
* Evaluation
</details>

<details>
<summary>🧼 Git & Data Hygiene</summary>

Ignored by default:

```gitignore
training_pc/data/raw/esp32_scan_*.csv
```

Reason:

* Large files
* Regenerated data
* Clean commits
</details>

## 👤 Author

Built as part of an **embedded + ML university project**
Focus: **clean engineering, correctness, scalability**

⭐ If this helped you, star the repo.

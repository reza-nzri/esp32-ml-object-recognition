import math
import time

# CONFIG
STEPS_PER_REV = 200
SCAN_DELAY_MS = 20

scan_index = 0
step = 0

print("\n\n# Starting fake scan simulator...\n")
print("scan_index,step,angle_deg,distance_cm")  # CSV header

while True:
    angle_deg = (step % STEPS_PER_REV) * (360 / STEPS_PER_REV)

    # Fake distance signal (sinusoidal like a real object scan)
    distance_cm = 80 + 20 * math.sin(math.radians(angle_deg))

    # This line feeds BOTH terminal + Thonny plotter
    print(f"{scan_index},{step},{angle_deg:.2f},{distance_cm:.2f}")

    step += 1

    if step % STEPS_PER_REV == 0:
        scan_index += 1
        print(f"\n# new scan {scan_index} \n")

    time.sleep(SCAN_DELAY_MS / 1000)

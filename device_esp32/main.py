import time
from utils import PinAssignment, StepperMotor, UltrasonicSensor


def scan_one_revolution(
    motor,
    sensor,
    scan_index: int = 0,
    measure_every: int = 10,
) -> None:
    """Perform one full 360° scan using a stepper motor and ultrasonic sensor.

    The motor performs a full revolution using its native step resolution.
    Distance measurements are taken only every N motor steps to reduce
    data redundancy and noise for machine learning applications.

    Only steps where a measurement is taken are logged (printed),
    ensuring correct semantic alignment between angle and distance.

    Args:
        motor (StepperMotor): The initialized stepper motor instance.
        sensor (UltrasonicSensor): The initialized ultrasonic sensor instance.
        scan_index (int): Index of the current scan iteration (used to group samples belonging to the same full rotation).
        measure_every (int): Take one measurement every N motor steps.
    """
    steps_per_rev = motor.steps_per_rev  # 4096 for 28BYJ-48 (half-step)

    for step in range(steps_per_rev):
        # Move motor by one physical/logical step
        step_count, angle_deg = motor.rotate(run=True, direction=1)

        # Only measure every Nth step
        if step_count % measure_every == 0:
            # Measure distance with ultrasonic sensor
            distance_cm = sensor.measure_distance()

            # Print result to serial console (CSV-style line)
            if distance_cm is None:
                print(f"{scan_index},{step_count},{angle_deg:.2f},NaN")
            else:
                print(f"{scan_index},{step_count},{angle_deg:.2f},{distance_cm:.2f}")

            # Small pause to avoid flooding the serial output (motor speed is set in StepperMotor)
            time.sleep_ms(5)

    # turn off coils after one full revolution
    motor.rotate(run=False, direction=1)
    print(f"\n# Finished scan {scan_index}\n")


def main():
    """Main loop of the ESP32 program."""
    pins = PinAssignment()
    motor = StepperMotor(
        in1=pins.IN1, in2=pins.IN2, in3=pins.IN3, in4=pins.IN4, delay_ms=3
    )
    sensor = UltrasonicSensor(trigger_pin=pins.TRIGGER_PIN, echo_pin=pins.ECHO_PIN)

    scan_index = 0

    print("\n")
    print("# Starting continuous scans (one full revolution per scan)...\n")
    print("scan_index,step,angle_deg,distance_cm")  # CSV header

    # scan object only 5 times
    for _ in range(5):
        scan_one_revolution(motor, sensor, scan_index, 10)
        scan_index += 1
        time.sleep(2)  # pause between scans


if __name__ == "__main__":
    main()

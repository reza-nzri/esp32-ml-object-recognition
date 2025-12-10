import time
from utils import PinAssignment, StepperMotor, UltrasonicSensor, Data


def scan_one_revolution(motor, sensor, data):
    """Rotate one full revolution and measure distance at each step."""
    steps_per_rev = motor.steps_per_rev  # 4096 for 28BYJ-48 (half-step)

    for _ in range(steps_per_rev):
        # Move motor by one logical step
        step_count, angle_deg = motor.rotate(run=True, direction=1)

        # Measure distance with ultrasonic sensor
        distance = sensor.measure_distance()

        # Reads measurements into Dataframe
        data.reading(distance=distance, step=step_count)

        # Print result to serial console
        if distance is None:
            print(f"Step: {step_count}, Angle: {angle_deg:.2f}°, Distance: No echo")
        else:
            print(f"Step: {step_count}, Angle: {angle_deg:.2f}°, Distance: {distance:.2f} cm")
        # Small pause to avoid flooding the serial output (motor speed is set in StepperMotor)
        time.sleep_ms(5)

    # After one revolution: stop motor (turn off coils)
    motor.rotate(run=False, direction=1)


def main():
    pins = PinAssignment()
    motor = StepperMotor(in1=pins.IN1, in2=pins.IN2, in3=pins.IN3, in4=pins.IN4, delay_ms=3)
    sensor = UltrasonicSensor(trigger_pin=pins.TRIGGER_PIN, echo_pin=pins.ECHO_PIN)
    data = Data()

    print("Starting continuous scans (one full revolution per scan)...\n")

    while True:
        print("\t=== New scan: One full revolution ===")
        scan_one_revolution(motor, sensor, data)
        print("\t\t=== Scan finished. Waiting 2 seconds... ===\n")
        time.sleep(2)


if __name__ == "__main__":
    main()

import time
from utils import PinAssignment, StepperMotor, UltrasonicSensor


def scan_one_revolution(motor, sensor, scan_id: int = 0) -> None:
    """Rotate one full revolution and measure distance at each step.
    
    Args:
        motor (StepperMotor): The initialized stepper motor instance.
        sensor (UltrasonicSensor): The initialized ultrasonic sensor instance.
        scan_id (int): Identifier for this scan (for logging and later ML use).
    """
    steps_per_rev = motor.steps_per_rev  # 4096 for 28BYJ-48 (half-step)

    for _ in range(steps_per_rev):
        # Move motor by one logical step
        step_count, angle_deg = motor.rotate(run=True, direction=1)

        # Measure distance with ultrasonic sensor
        distance = sensor.measure_distance()

        # Print result to serial console (CSV-style line)
        if distance is None:
            print(f"{scan_id},{step_count},{angle_deg:.2},NaN")
            # print(f"Step: {step_count}, Angle: {angle_deg:.2f}°, Distance: No echo")
        else:
            print(f"{scan_id},{step_count},{angle_deg:.2},{distance:.2f}")
            # print(f"Step: {step_count}, Angle: {angle_deg:.2f}°, Distance: {distance:.2f} cm")
        # Small pause to avoid flooding the serial output (motor speed is set in StepperMotor)
        time.sleep_ms(5)

    # turn off coils after one full revolution
    motor.rotate(run=False, direction=1)
    print(f"# Finished scan {scan_id}\n")


def main():
    """Main loop of the ESP32 program."""
    pins = PinAssignment()
    motor = StepperMotor(in1=pins.IN1, in2=pins.IN2, in3=pins.IN3, in4=pins.IN4, delay_ms=3)
    sensor = UltrasonicSensor(trigger_pin=pins.TRIGGER_PIN, echo_pin=pins.ECHO_PIN)

    scan_id = 0
    print("# Starting continuous scans (one full revolution per scan)...\n")
    print("scan_id,step,angle_deg,distance_cm")  # CSV header
    
    while True:
        scan_one_revolution(motor, sensor)
        scan_id += 1
        time.sleep(2)  # pause between scans


if __name__ == "__main__":
    main()

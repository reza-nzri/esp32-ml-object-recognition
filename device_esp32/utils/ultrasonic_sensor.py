from machine import time_pulse_us
import time


class UltrasonicSensor:
    """HC-SR04 ultrasonic distance sensor driver.

    Attributes:
        trigger (Pin): Output pin used for sending the trigger pulse.
        echo (Pin): Input pin used to receive the echo pulse.
    """

    def __init__(self, trigger_pin, echo_pin) -> None:
        self.trigger = trigger_pin
        self.echo = echo_pin

        # Make sure trigger starts LOW
        self.trigger.value(0)
        time.sleep_ms(2)

    def measure_distance(self):
        """Sends a 10 µs trigger pulse, measures echo duration,
        and converts it into distance in centimeters.

        Returns:
            distance_cm: Distance in centimeters, or None if no echo detected.
        """
        # Send a short HIGH pulse
        self.trigger.value(1)
        time.sleep_us(10)  # microseconds
        self.trigger.value(0)

        # Measure echo pulse duration
        duration = time_pulse_us(self.echo, 1)

        # Negative duration means timeout/no echo
        if duration < 0:
            return None

        # Convert ms to cm
        distance_cm: float = (
            duration * 0.0343
        ) / 2  # 343 m/s: (Datasheet) Speed ​​of sound in air in 20°
        return distance_cm


if __name__ == "__main__":
    from pin_assignment import PinAssignment

    pins = PinAssignment()
    sensor = UltrasonicSensor(trigger_pin=pins.TRIGGER_PIN, echo_pin=pins.ECHO_PIN)

    while True:
        dist = sensor.measure_distance()

        if dist is None:
            print("No echo detected")
        else:
            print(f"Distance: {dist} cm")

        time.sleep(0.2)

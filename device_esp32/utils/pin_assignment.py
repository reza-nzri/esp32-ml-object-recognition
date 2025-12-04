from machine import Pin

class PinAssignment:
    """
    A class to define and manage pin assignments.
    
    Attributes:
        TRIGGER_PIN: For the ultrasonic sensor.
        ECHO_PIN: For the ultrasonic sensor.
    """
    
    def __init__(self):
        """
        Initializes the PinAssignment class by setting up the GPIO pins.
        """
        # Ultrasonic sensor pins
        self.ECHO_PIN = Pin(35, Pin.IN)
        self.TRIGGER_PIN = Pin(32, Pin.OUT)

        # Stepper motor pins
        self.IN1 = Pin(2, Pin.OUT)
        self.IN2 = Pin(4, Pin.OUT)
        self.IN3 = Pin(5, Pin.OUT)
        self.IN4 = Pin(18, Pin.OUT)

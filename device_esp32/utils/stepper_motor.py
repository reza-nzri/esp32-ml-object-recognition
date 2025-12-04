from machine import Pin
from time import sleep_ms


class StepperMotor:
    """ULN2003 stepper motor driver for 28BYJ-48 using half-step sequence."""

    # Half-step sequence (8 steps)
    HALF_STEP_SEQUENCE = [
        (1, 0, 0, 0),
        (1, 1, 0, 0),
        (0, 1, 0, 0),
        (0, 1, 1, 0),
        (0, 0, 1, 0),
        (0, 0, 1, 1),
        (0, 0, 0, 1),
        (1, 0, 0, 1),
    ]

    # Type Hinting (type annotation "PEP 484"), z.B.:
        # -->  `in1: Pin` = “Parameter in1 should be an object of class Pin.”
        # --> `mode: str = "half"` = a function parameter named mode, expected to be a string, with "half" as its default value.
    def __init__(self, in1: Pin, in2: Pin, in3: Pin, in4: Pin, delay_ms: int = 3):
        """Initialize the stepper motor with GPIO pins.

        Args:
            in1 (Pin): IN1 control pin.
            in2 (Pin): IN2 control pin.
            in3 (Pin): IN3 control pin.
            in4 (Pin): IN4 control pin.
            delay_ms (int): Delay between steps (speed control).
        """
        self.in1 = in1
        self.in2 = in2
        self.in3 = in3
        self.in4 = in4
        self.delay_ms = delay_ms
        
        self.sequence = StepperMotor.HALF_STEP_SEQUENCE
        self.steps_per_rev = 4096

    # `_`: internal/private methode in class | not intended for external use
    def _set_step(self, a: int, b: int, c: int, d: int):
        """Set the output values of the 4 control pins.  
        
        Set the coil activation state for all four control pins. Each argument
        represents the desired on/off state for a specific coil.
        
        Args:
            a (int): Logic level for IN1 (0 or 1).
            b (int): Logic level for IN2 (0 or 1).
            c (int): Logic level for IN3 (0 or 1).
            d (int): Logic level for IN4 (0 or 1).
        """
        self.in1.value(a)
        self.in2.value(b)
        self.in3.value(c)
        self.in4.value(d)

    def rotate(self, run: bool = True, direction: int = 1) -> tuple[int, float]:
        """Rotate the motor in a simple continuous way while tracking steps and angle.

        Args:
            run (bool): If True → motor rotates. If False → motor stops.
            direction (int): +1 for forward, -1 for backward.
            
        Returns:
            tuple[int, float]: (step_count, angle_degrees)
        """
        if direction not in (1, -1):
            raise ValueError("direction must be +1 or -1")

        # initializations
        if not hasattr(self, "_step_index"):
            ## Goal: keep the movement smooth acroos multiple rotate calls 
            ## instead of restarting from zero every time
            self._step_index = 0     # `_` private attribute
        
        if not hasattr(self, "_step_count"):
            self._step_count = 0     # total logical steps taken

        if not run:
            # stop motor: turn off coils
            self._set_step(0, 0, 0, 0)
        else:    
            seq_len = len(self.sequence)

            # move one step
            self._step_index += direction

            # wrap forward to zero 
            if self._step_index == seq_len: 
                self._step_index = 0 
                
            # wrap backward 
            if self._step_index == -1: 
                self._step_index = seq_len - 1

            # apply coil pattern
            a, b, c, d = self.sequence[self._step_index]
            self._set_step(a, b, c, d)

            # update counters
            self._step_count += direction

            sleep_ms(self.delay_ms)
        
        # compute angle
        angle = (self._step_count % self.steps_per_rev) * (360 / self.steps_per_rev)
        
        return self._step_count, angle


if __name__ == "__main__":
    print("StepperMotor standalone test...")
    
    from pin_assignment import PinAssignment

    pins = PinAssignment()

    # Create motor object
    motor = StepperMotor(in1=pins.IN1, in2=pins.IN2, in3=pins.IN3, in4=pins.IN4, delay_ms=3)

    print("Motor is rotating continuously...\n")

    while True:
        # one step forward per loop
        step_count, angle = motor.rotate(run=True, direction=1)

        # print current state
        print(f"Step: {step_count}, \t Angle: {angle}°")

        # slow down printing (not motor)
        sleep_ms(10)

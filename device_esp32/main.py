"""
Entry point for the ESP32 application.

Starts the app, connects hardware modules, and runs the main control loop for ML inference and sensor interaction.
"""

from machine import Pin, Timer

led = Pin(1, Pin.OUT)  # internal LED

def blink(timer):
    led.value(not led.value())

timer = Timer(0)
timer.init(freq=2, mode=Timer.PERIODIC, callback=blink)


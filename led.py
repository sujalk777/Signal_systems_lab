import RPi.GPIO as GPIO
import time
# Setup GPIO Pins
LED1_PIN = 23  # Adjust the GPIO pins as per your setup
LED2_PIN = 24

GPIO.setmode(GPIO.BCM)
GPIO.setup(LED1_PIN, GPIO.OUT)
GPIO.setup(LED2_PIN, GPIO.OUT)

print("Hello World")
avg = 0.08328627581158153
avg_f = [
    0.00758391, 0.00634574, 0.01027966, 0.01008122, 0.05813246, 0.09862552,
    0.09265026, 0.16097827, 0.29374871, 0.25202988, 0.04867215, 0.08282148,
    0.06490334, 0.05293078, 0.06121145, 0.04871144, 0.00891782, 0.03528598,
    0.04265941, 0.22915604
]

for i in avg_f:
    if i > avg:  # Compare each element with avg, not the list
        GPIO.output(LED1_PIN, GPIO.HIGH)
        GPIO.output(LED2_PIN, GPIO.LOW)
        print(1)
    else:
        GPIO.output(LED1_PIN, GPIO.LOW)
        GPIO.output(LED2_PIN, GPIO.HIGH)
        print(0)

    time.sleep(1)  # Adjust delay as needed

GPIO.cleanup()

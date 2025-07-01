import os, time

while True:
    os.system("echo 1 | sudo tee /sys/class/leds/ACT/brightness")
    time.sleep(1)
    os.system("echo 0 | sudo tee /sys/class/leds/ACT/brightness")
    time.sleep(1)
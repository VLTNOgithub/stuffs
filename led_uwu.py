import os

while True:
    ledstatus = input("z is on x is off ")

    if ledstatus == "z":
        os.system("echo 1 | sudo tee /sys/class/leds/ACT/brightness")
    elif ledstatus == "x":
        os.system("echo 0 | sudo tee /sys/class/leds/ACT/brightness")


   
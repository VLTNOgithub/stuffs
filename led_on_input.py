import os

asd = input("Enter on or off here: ")

if asd == "on":
    os.system("echo 1 | sudo tee /sys/class/leds/ACT/brightness")
elif asd == "off":
    os.system("echo 0 | sudo tee /sys/class/leds/ACT/brightness")
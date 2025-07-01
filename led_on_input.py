import os
while True:
    turnon = input("Enter on or off here: ")

    if turnon.lower() == "z":
        os.system("echo 1 | sudo tee /sys/class/leds/ACT/brightness")
    elif turnon.lower() == "x":
        os.system("echo 0 | sudo tee /sys/class/leds/ACT/brightness")
   
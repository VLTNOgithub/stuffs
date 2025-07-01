import os
os.execl("/bin/sh", "sh", "-c", "echo 1 > /sys/class/leds/ACT/brightness")
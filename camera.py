from picamzero import Camera
from time import sleep

cam = Camera()
cam.start_preview()
cam.record_video("hellp.h264", 999999)
sleep(100)


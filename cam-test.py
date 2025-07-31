from time import sleep
from picamera2 import Picamera2

picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={"size": (1920, 1080)}
)

picam2.start(config, show_preview=True)
sleep(100)
picam2.stop()
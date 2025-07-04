import random
import time
import math
from sense_hat import SenseHat

sense = SenseHat()

sense.show_message("Ask a question", 0.045)

time.sleep(1.5)

replies = ["Yes", "No", "Maybe", "Of course!", "That would be dumb", "Fate has decided", "Another time", "WTF bro??", "WTF bro??"]

while True:
    x, y, z = sense.get_accelerometer_raw().values()
    x = abs(x)
    y = abs(y)
    z = abs(z)
 
    if x > 2 or y > 2 or z > 2:
        sense.show_message(random.choice(replies), 0.045)
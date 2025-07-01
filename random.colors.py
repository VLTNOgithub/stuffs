from sense_hat import *
from random import choice 

sense = SenseHat()

red = [255, 0, 0]
green = [0, 255, 0]
blue = [0, 0, 255]

colours = [red, green, blue]

while True:
    grid = []

    for event in sense.stick.get_events():
        if event.action == "pressed":
            if event.direction == "middle":
                for x in range(64):
                    colour = choice(colours)
                    grid.append(colour)

                sense.set_pixels(grid)
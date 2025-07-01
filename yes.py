from sense_hat import SenseHat
sense = SenseHat()

yellow = [255,255,0]
blue = [0,0,255]
red = [255,0,0]
black = [0, 0, 0]

smiley_face = [
[], [], [], [], [], [], [], [],
[], [], red, [], [], red, [], [],
[], [], [], [], [], [], [], [],
[], [], red, [], [], [], red, [],
[], [], [], red, [], [], red, [],
[], [], [], [], red, red, [], [],
[], [], [], [], [], [], [], [],
[], [], [], [], [], [], [], []
]

sense.clear()
sense.set_pixels(smiley_face)
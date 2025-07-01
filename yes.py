from sense_hat import SenseHat
from utils import parse_matrix
sense = SenseHat()

yellow = [255,255,0]
blue = [0,0,255]
red = [255,0,0]
black = [0, 0, 0]

smiley_face = [
[], [], [], [], [], [], [], [],
[], [], red, red, [], red, red, [],
[], [], red, red, [], red, red, [],
[], red, [], [], [], [], [], red,
[], [], red, [], [], [], red, [],
[], [], [], red, red, red, [], [],
[], [], [], [], [], [], [], [],
[], [], [], [], [], [], [], []
]

sense.clear()

smiley_face = parse_matrix(smiley_face)

sense.set_pixels(smiley_face)
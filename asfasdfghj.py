import libcamera._libcamera
from sense_hat import SenseHat
import time
import math
import cv2
from picamera2 import Picamera2
from libcamera import Transform

sense = SenseHat()

def show_message(input, background=[0, 0, 0]):
    print(f"Showing: '{input}'.")
    sense.show_message(input, 0.05, back_colour=background)


# # message
# message = "... --- ... ..- / ..-- .- --."

# # show on leds
# while True:


#     sense.show_message(message,00.05, text_colour = [255,255,255], back_colour= [255,0,0])

# # print lol

# print(message)

# sense.clear()

# num1 = float(input("whats ur first number lol:" ))
# num2 = float(input("whats ur second number lol:" ))

# result = num1 ** num2

# print(f"you have {result} lol")

# sense.show_message(f"you have {result}")


# first_name = input("First name here: ")
# second_name = input("Second name here: ")

# print(f"Hello, {first_name} {second_name}!")

# show_message(sense, f"Hello, {first_name} {second_name}!")

# names = ["Jonathan", "Reinan", "Valentino"]
# names.append("""
# SUPA POOPOO
# """)

# for our_names_lol in names:
#     print(our_names_lol)

# number = int(input("enter a number: "))

# Pos_or_neg = ""
# if number > 0:
#     Pos_or_neg = "POSITIVE"
# elif number < 0:
#     Pos_or_neg = "NEGATIVE"
# else:
#     Pos_or_neg = "ZERO"

# show_message(f"{number} is {Pos_or_neg}")

# import time
# from sense_hat import SenseHat
# sense = SenseHat()
# sense.clear()

# def display_counter(count):
#     print(count)
#     sense.show_message(f"Count: {count}", scroll_speed=0.025)

# def update_counter(count, direction):
#     if direction == "up":
#         count += 1
#     elif direction == "down":
#         count -= 1
#     return count

# def check_press_and_update():
#     count = 0
#     while True:
#         for event in sense.stick.get_events():
#             if event.action == "pressed":
#                 count = update_counter(count, event.direction)
#                 display_counter(count)
#         time.sleep(0.1)

# check_press_and_update()

# sense.show_letter("V")

# angles = [0, 90, 180, 270, 0, 90, 180, 270]

# def wrap(degrees):
#     if degrees >= 360:
#         return degrees - 360
#     elif degrees < 0:
#         return degrees + 360
#     else:
#         return degrees

# while True:
#     rotation = sense.get_orientation_degrees()["pitch"]
#     if ((rotation >= 335 and rotation < 0) or (rotation <= 30 and rotation > 0)):
#         sense.set_rotation(0)
#     elif (rotation >= 60 and rotation <= 70):
#         sense.set_rotation(90)port libcamera
#     print(rotation)

# while True:
#     temperature = round(sense.get_temperature(), 1)
#     pressure = round(sense.get_pressure(), 1)
#     humidity = round(sense.get_humidity(), 1)

#     if temperature > 18.3 and temperature < 26.7:
#         background = [0, 100, 0]
#     else:
#         background = [100, 0, 0]
    
#     message = "Temp %s, Pres = %s, Hum = %s" % (temperature, pressure, humidity)

#     show_message(message, background=background)

# while True:
#     pitch, roll, yaw = sense.get_orientation().values()
#     print("pitch=%s, roll=%s, yaw=%s" % (pitch, roll, yaw))

# while True:
#     x, y, z = sense.get_accelerometer_raw().values()
#     x = round(x, 0)
#     y = round(y, 0)
#     z = round(z, 0)

#     print("x = %s, y = %s, z = %s" % (x, y, z))
#     time.sleep(0.4)

# sense.show_letter("V")

# while True:
#     x, y, z = sense.get_accelerometer_raw().values()

#     x = round(x, 0)
#     y = round(y, 0)

#     if x == -1:
#         sense.set_rotation(90)
#     elif x == 1:
#         sense.set_rotation(270)
#     elif y == -1:
#         sense.set_rotation(180)
#     elif y == 1:
#         sense.set_rotation(0)

# picam = Picamera2()

# picam.preview_configuration.size = (1920, 1080)
# picam.configure()

# picam.start(show_preview=True)
# time.sleep(100)
# picam.stop()

picam = Picamera2()

preview_config = picam.create_preview_configuration(
    transform=Transform(vflip=True)
)

picam.configure(preview_config)
picam.start(show_preview=True)
time.sleep(100)
picam.stop()
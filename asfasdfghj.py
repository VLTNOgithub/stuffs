from sense_hat import SenseHat
import time

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
#         sense.set_rotation(90)
#     print(rotation)

while True:
    temperature = round(sense.get_temperature(), 1)
    pressure = round(sense.get_pressure(), 1)
    humidity = round(sense.get_humidity(), 1)

    if temperature > 18.3 and temperature < 26.7:
        background = [0, 100, 0]
    else:
        background = [100, 0, 0]
    
    message = "Temp %s, Pres = %s, Hum = %s" % (temperature, pressure, humidity)

    show_message(message, background=background)
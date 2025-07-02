from sense_hat import SenseHat

sense = SenseHat()

def show_message(input):
    print(input)
    sense.show_message(input, 0.05)




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

number = int(input("enter a number: "))

Pos_or_neg = ""
if number > 0:
    Pos_or_neg = "POSITIVE"
elif number < 0:
    Pos_or_neg = "NEGATIVE"
else:
    Pos_or_neg = "ZERO"

show_message(f"{number} is {Pos_or_neg}")

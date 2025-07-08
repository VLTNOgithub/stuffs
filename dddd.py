import math
import time

#  # strings

# first_name = "reinan"
# food =  "pizza"
# email = "hiballs"

# print (f"hello {first_name}")
# print(f"you like {food}")
# print(f"your email is {email}")

# # integers
# age = 15 
# quanity = 3
# numstudents = 30


# print(f"you are {age}")
# print(f"you are buying {quanity} items" )
# print(f"your class has {numstudents} students")

# # float
# price = 10.99
# gpa = 3.2
# distance = 5.5

# print (f"the price is £{price}")
# print (f"your gpa is {gpa}")
# print(f"you ran {distance}km")

# # booleans

# is_student = False
# sale = True
# online=True

# print(f"are you a student?: {is_student}")

# if is_student:
#     print("you are a student")
# else:
#     print("you are not a student")

# if sale:
#     print("that is on sale")
# else:
#     print("not available")

#typecasting

# name = ""
# age = 15
# gpa = 3.2
# student = True

# name = bool (name)
# print(name)

# # input()

# name = input("what is your name?:")
# age = int(input("how old are you?:"))

# age = age + 1

# print(f"hello {name}!")
# print("happy bday dude")
# print(f"you are {age} years old")

# # rect calc

# length = float(input("enter length:"))
# width = float(input("enter width:"))
# area = length * width

# print(f"the area is: {area}cm^2")

# #shopping cart

# item = input("what item would you like to buy: ")
# price = float(input("what is the price: "))
# quantity = int(input("how many would you like: "))
# total = price * quantity

# print(f"you have bought {quantity} x {item}(s)")
# print(f"your total is: ${total}")

# #madlibs

# adjective1 = input("enter and adjective(description): ")
# noun1 = input("enter a noun(person,place,thing): ")
# adjective2 = input("enter and adjective(description): ")
# verb1 = input("enter a verb(doing word): ")
# adjective3 = input("enter and adjective(description): ")

# print(f"today i went to a {adjective1} zoo.")
# print(f"in an exhibit, i saw a {noun1}")
# print(f"{noun1} was {adjective2} and {verb1}")
# print(f"i was {adjective3}!")

#arithmetic

# friends = 10

# friends = friends + 1
# friends += 1

# friends = friends - 2
# friends -= 2

# friends = friends * 3
# friends *= 3

# friends = friends / 2
# friends /= 2

# friends = friends ** 2
# friends **= 2
# remainder = friends % 3


# print(remainder)

# x = 3.14
# y = 4
# z = 5

# result = round(x)
# result = abs(y)
# result = pow(4,3)
# result = max(x, y, z)
# result = min(x, y, z)

# print(round(result, 3))

#x = 9.9 

# print(math.pi)
# print(math.e)
# result = math.sqrt(x)
# result = math.ceil(x)
# result = math.floor(x)

# print(round(result, 3))


# radius = float(input("enter the radius of a circle: "))
# circumference = 2 * math.pi * radius
# print(f"the circumference is: {round(circumference)}cm")

# radius = float(input("enter the radius of a circle: "))
# area= math.pi * pow(radius, 2)
# print(f"the area of the circle is: {round(area, 2)}cm^2")

# a = float(input("enter side a:"))
# b = float(input("enter side b:"))
# c =math.sqrt(pow(a, 2) +(pow (b, 2)))

# print(f"side c ={c}")

#if statements

# age = int(input("enter your age:"))

# if age >= 100:
#     print("too old")
# elif age >= 18:
#     print("your are now signed up")
# elif age < 0:
#     print("you havent been born yet")
# else:
#     print("you must be 18+ to sign up")

# response = input("would you like food? (Y/N): ")

# if response == "y":
#     print("have some food!")
# else:
#     print("no food for you lol")

# name = input("enter your name: ")

# if name == "":
#     print("you didnt type your name")
# else:
#     print(f"hello {name}")

# online = False

# if online:
#     print("this user is online")
# else:
#     print("this user is not online")

# # python calc

# operator = input("enter an operator (+ - * /): ")
# num1 = float(input("enter the first number: "))
# num2 = float(input("enter the second number: "))

# if operator == "+":
#     result = num1 + num2
#     print(round(result, 3))
# elif operator == "-":
#     result = num1 - num2
#     print(round(result, 3))
# elif operator == "*":
#     result = num1 * num2
#     print(round(result, 3))
# elif operator == "/":
#     result = num1 / num2
#     print(round(result, 3))
# else:
#     print(f"{operator} is not a valid operator")

#python weight converter

# weight = float(input("enter your weight: "))
# unit = input("kilograms or pounds? (K or L): ")

# if unit == "K":
#     weight = weight * 2.205
#     unit = "Lbs"
#     print(f"your weight is: {round(weight, 1)} {unit}")
# elif unit == "L":
#     weight = weight / 2.205
#     unit = "Kgs"
#     print(f"your weight is: {round(weight, 1)} {unit}")
# else:
#     print(f"{unit} was not valid")

# #temp conversion

# unit = input("is this temperature in celsius or fahrenheit (C/F): ")
# temp = float(input("enter the temperature: "))

# if unit == "C":
#     temp = round((9 * temp) / 5 + 326, 1)
#     print(f"the temperature in fahrenheit is: {temp}°F")
# elif unit == "F":
#     temp = round((temp - 32) * 5 / 9, 1)
#     print(f"the temperature in celsius is {temp}C")
# else:
#     print(f"{unit} is an invalid unit of measurement")

# #logical operators ; or, and, not

# temp = 21

# is_raining = False
 
# if temp > 35 or temp < 0 or is_raining:
#     print("the outdoor event is cancelled")
# else:
#     print("the outdoor event is still scheduled")
    
# is_sunny =True

# if temp>= 28 and is_sunny:
#     print("it is hot outside")
#     print("it is sunny")
# elif temp<= 0 and is_sunny:
#     print("it is cold outside")
#     print("it is sunny")
# elif temp > 28 and temp > 0 and is_sunny:
#     print ("it is warm outside")
#     print("it is sunny")
# elif temp>= 28 and not is_sunny:
#     print("it is hot outside")
#     print("it is cloudy")
# elif temp<= 0 and not is_sunny:
#     print("it is cold outside")
#     print("it is cloudy")
# elif temp > 28 and not temp > 0 and is_sunny:
#     print ("it is warm outside")
#     print("it is cloudy")

#conditional expression

# num = 5
# a=6
# b= 7
# age = 13
# temp = 30
# user_role = "admin"

# print("postitive" if num > 0 else "negative")
# result = "even" if num % 2 == 0 else "odd"
# max_num = a if a > b else b
# min_num = a if a < b else b
# status = "adult" if age >= 18 else "child"
# weather = "hot" if temperature = 20 else "cold"
# access_level = "full access" if user_role == "admin" else "limited access"

# print(access_level)

#string methods

# name = input("enter your full name: ")
# phone_number = input("enter your phone number #:")

# result = len(name)
# result = name.find(" ")
# result = name.rfind("o")
# name = name.capitalize
# name = name.upper()
# name=name.lower
# result = name.isdigit()
# result = name.isalpha()
# result = phone_number.count("-")
# result = phone_number.replace("-"," ")

# print(result)

# print(help(str))

#username thing

# done = False

# while not done:

#     username = input("enter username: ")

#     if len(username) == 0:
#         print("please type a username")
#     elif len(username) > 12:
#         print("username must be below 12 characters")
#     elif username.find(" ") > 0:
#         print("username must not contain spaces")
#     elif not username.isalpha():
#         print("username must not contain numbers")
#     else:
#         print(f"hello {username}")
#         done = True
#         break

#indexing

# credit_number = "1234-5678-9012-3456"

# print(credit_number[4])
# print(credit_number[:5])
# print(credit_number[5:9])
# print(credit_number[0:])
# print(credit_number[-6])
# print(credit_number[: :3])

# last_digits = credit_number[-4:]
# print(f"XXXX-XXXX-XXXX-{last_digits}")

# credit_number = credit_number[::-1]
# print(credit_number)
   
# format specifiers

# price1 = 3000.14159
# price2 = -9807.65
# price3 = 1200.34

# print(f"Price 1 is £{price1:+,.2f}")
# print(f"Price 2 is £{price2:+,.2f}")
# print(f"Price 3 is £{price3:+,.2f}")

#while loops
# name = input("enter your name:")

# while name == "":
#     print("you did not enter your name")
#     name = input("enter your name: ")
# print(f"hello {name}")

# age = int(input("enter your age: "))

# while age < 0:
#     print("age cant be negative")
#     age = int(input("enter your age: "))
# print(f"You are {age} years old")

# food = input("enter a food you like: (q to quit): ")

# while not food == "q":
#     print(f"you like {food}")
#     food = input("enter another food you like (q to quit: )")
# print("bye")

# num = int(input("enter a number between 1-10: "))

# while num < 1 or num > 10:
#     print(f"{num} is not valid")
#     num = int(input("enter a number between 1-10: "))
# print(f"your number is {num}")
 
#compound intrest calculator
  
# done = False
# while not done:

#     principle = float(input("please enter your principle: "))
#     rate = float(input("please enter your interest rate: "))
#     time = float(input("please enter the time (in years): "))

#     if principle <= 0:
#         time("invalid principle")
#     elif time <= 0:
#         print("invalid time")
#     elif rate <= 0:
#         print("invalid interest rate")
#     else:
#         interest = principle * pow((1 + rate), time)
#         print(f"your compound interest is: {interest:.2f}")
#         done = True
    
#for loops

# credit_card = "1234-5678-9012-3456"

# for x in credit_card:
#     print(x)
    
# for x in reversed(range(1, 21)):
#     if x == 13:
#         continue #can be continue or break, both do different things
#     else:
#         print(x)

#countdown program

# my_time = int(input("enter the time in seconds: "))

# for x in range(my_time, 0, -1):
#     seconds = x % 60
#     minutes = int(x / 60) % 60
#     hours = int(x / 3600)
#     print(f"{hours:02}:{minutes:02}:{seconds:02}")
#     time.sleep(1)

# print("times up")

#nested loops

rows = int(input("enter the amount of rows you want: "))
columns = int(input("enter the amount of columns you want: "))

for x in range(rows):
    for y in range(columns):
        print("-" , end="")
    print()






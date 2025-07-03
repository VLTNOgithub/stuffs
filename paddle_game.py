from sense_hat import SenseHat
import time
import random

sense = SenseHat()

# Game settings
paddle_position = 3 #paddle starts in the middle of the bottom row
falling_object = [random.randint(0,7),0]
game_over = False
total_points = 0
points = 0
lives = 2

#update display with falling object and paddle
def update_display():
    sense.clear()
    sense.set_pixel(falling_object[0], falling_object[1], (255, 0,0))#red object falling
    for i in range(3): #DRAW PADDle (3 pixels wide)
        sense.set_pixel(paddle_position + i, 7, (0,255,0))#green paddle

#move paddle based on joystick input
def move_paddle():
    global paddle_position
    for event in sense.stick.get_events():
        if event.action == "pressed": 
            if event.direction == "left" and paddle_position > 0:
                paddle_position -= 1 #most paddle left
            elif event.direction == "right" and paddle_position < 5:
                paddle_position += 1 #move paddle right

def move_falling_object():
    global falling_object
    falling_object[1] += 1 #move the object down by 1
    if falling_object[1] > 7: #if the object is at the bottom
        falling_object[1] = 0 #object caught, reset to top
        falling_object[0] = random.randint(0,7) # random x position

def check_collision():
    global paddle_position, falling_object, points
    if paddle_position <= falling_object[0] < paddle_position + 3:
        falling_object[1] = 0 #object caught, reset to top
        falling_object[0] = random.randint(0,7) # random x position
        points += 1
        print(points)
        
        return True #caught the object
    return False #failure u idiot

def main_game_loop():
    global game_over, lives, points, total_points
    while not game_over:
        move_paddle()#move the paddle based on joystick input
        move_falling_object()#move the falling object
        
        update_display() #update the LED matrix

        if falling_object[1] == 7 and not check_collision():
            game_over = True
        
        time.sleep(0.3) # speed of falling object
    message = f"{lives} LIVES LEFT"
    if lives == 1:
        message = f"{lives} LIFE LEFT"
    sense.show_message(message, 0.05)
    
    if lives >= 1:
        game_over = False
        lives -= 1
        total_points += points
        points = 0
        main_game_loop()
    else:
        sense.show_message(f"GAME OVER. {total_points} POINTS", 0.05)

main_game_loop()
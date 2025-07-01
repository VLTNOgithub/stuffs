import cv2
import numpy as np
import picamera
import picamera.array
from pyzbar import pyzbar
from sense_hat import SenseHat
from random import choice, randint
from time import sleep, time

sense = SenseHat()
sense.clear()

MINIGAMES = [
    "simon_says",
    "snake",
    "reaction_test"
]

def display_message(msg, color):
    sense.show_message(msg, text_colour=color, scroll_speed=0.05)

def run_minigame(minigame):
    if minigame == "simon_says":
        simon_says()
    elif minigame == "snake":
        snake()
    elif minigame == "reaction_test":
        reaction_test()
    else:
        display_message("Invalid minigame!", [255,0,0])

def simon_says():
    directions = ["U", "D", "L", "R"]
    dir_to_arrow = {"U": [0,255,0], "D": [255,0,0], "L": [0,0,255], "R": [255,255,0]}
    sequence = []
    round_num = 1
    display_message("Simon says!", [0,255,0])
    while True:
        sequence.append(choice(directions))
        # Show sequence
        display_message(sequence)
        # Get user input
        display_message("Your turn!", [255,255,255])
        for i, d in enumerate(sequence):
            event = None
            while event is None:
                events = sense.stick.get_events()
                for e in events:
                    if e.action == "pressed":
                        event = e
                        break
            user_dir = None
            if event.direction == "up":
                user_dir = "U"
            elif event.direction == "down":
                user_dir = "D"
            elif event.direction == "left":
                user_dir = "L"
            elif event.direction == "right":
                user_dir = "R"
            if user_dir != d:
                display_message("Wrong! Score: {}".format(round_num-1), [255,0,0])
                return
        display_message("Good!", [0,255,0])
        round_num += 1

def snake():
    # Minimal snake game for Sense HAT
    display_message("Snake!", [0,255,0])
    sense.clear()
    width, height = 8, 8
    snake = [(4,4)]
    direction = (0,1) # Start moving right
    food = (randint(0,7), randint(0,7))
    while food in snake:
        food = (randint(0,7), randint(0,7))
    score = 0
    game_over = False
    while not game_over:
        # Draw snake and food
        sense.clear()
        for x, y in snake:
            sense.set_pixel(x, y, 0,255,0)
        sense.set_pixel(food[0], food[1], 255,0,0)
        sleep(0.3)
        # Handle input
        events = sense.stick.get_events()
        for event in events:
            if event.action == "pressed":
                if event.direction == "up" and direction != (0,1):
                    direction = (0,-1)
                elif event.direction == "down" and direction != (0,-1):
                    direction = (0,1)
                elif event.direction == "left" and direction != (1,0):
                    direction = (-1,0)
                elif event.direction == "right" and direction != (-1,0):
                    direction = (1,0)
        # Move snake
        new_head = ((snake[0][0] + direction[0]) % width, (snake[0][1] + direction[1]) % height)
        if new_head in snake:
            game_over = True
            break
        snake = [new_head] + snake
        if new_head == food:
            score += 1
            while True:
                food = (randint(0,7), randint(0,7))
                if food not in snake:
                    break
        else:
            snake.pop()
    display_message("Game Over! Score: {}".format(score), [255,0,0])

def reaction_test():
    display_message("Reaction!", [0,255,255])
    sense.clear()
    sleep(randint(2,5))
    sense.clear(0,255,0)
    start = time()
    pressed = False
    while not pressed:
        events = sense.stick.get_events()
        for event in events:
            if event.action == "pressed":
                pressed = True
                break
    reaction_time = time() - start
    sense.clear()
    display_message("Time: {:.2f}s".format(reaction_time), [0,255,255])

def detect_qr(frame):
    decoded_objs = pyzbar.decode(frame)
    for obj in decoded_objs:
        data = obj.data.decode("utf-8")
        if data in MINIGAMES:
            return data
    return None


def main():
    with picamera.PiCamera() as camera:
        camera.resolution = (320, 240)
        sleep(2)
        with picamera.array.PiRGBArray(camera) as output:
            display_message("Start Hunt!", [255,255,0])
            for frame in camera.capture_continuous(output, format="bgr", use_video_port=True):
                img = frame.array
                marker = detect_qr(img)
                if marker:
                    run_minigame(marker)
                    break
                output.truncate(0)

                # Press joystick down to quit
                for event in sense.stick.get_events():
                    if event.action == "pressed" and event.direction == "middle":
                        display_message("Bye!", [255,0,0])
                        return

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sense.clear()
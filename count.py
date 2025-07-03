import time
from sense_hat import SenseHat
sense = SenseHat()
sense.clear()

def display_counter(count):
    print(count)
    sense.show_message(f"Count: {count}", scroll_speed=0.025)

def update_counter(count, direction):
    if direction == "up":
        count += 1
    elif direction == "down":
        count -= 1
    return count

def check_press_and_update():
    count = 0
    while True:
        for event in sense.stick.get_events():
            if event.action == "pressed":
                count = update_counter(count, event.direction)
                display_counter(count)
        time.sleep(0.1)

check_press_and_update()
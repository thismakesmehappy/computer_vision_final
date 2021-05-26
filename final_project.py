import tkinter #Tk, Canvas, Frame, BOTH, ARC
import random
import cv2
import numpy as np
from colorsys import rgb_to_hsv, hsv_to_rgb
from colorutils import hsv_to_hex, rgb_to_hex
import inspect
import sys
import time
#TODO: Look at motion_test.py to figure out how to do the loop properly

def mirror(canvas, canvas_width, canvas_height, columns, rows, spacing):
    shape_width = (canvas_width // columns) - spacing
    shape_height = (canvas_height // rows) - spacing
    resize_height, resize_width = calculate_resize_parameters(rows, columns)
    max_flow_tolerance = 1.5
    take_picture = True
    total_wait = 1.5
    motion_wait = .1
    bg_colors = ['#FF9999', '#FFFF99', '#99FF99']
    
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    while True:
        if take_picture:
            canvas.delete("all")
            ret, frame = cap.read()
            take_picture = False
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small_frame = frame = cv2.resize(frame, (resize_width, resize_height), interpolation=cv2.INTER_AREA)
        small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
        shapes = np.empty(shape=(rows, columns, 2))

        draw_grid_of_shapes(canvas_width, canvas_height, columns, rows, spacing, shape_width, shape_height, canvas, small_frame, shapes, h_multiplier=1, s_multiplier=1, v_multiplier=1)
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (480, 320), interpolation = cv2.INTER_AREA)
        avg_flow = 0
        while True:
            canvas.update()
            #TODO: Solve _tkinter.TclError: can't invoke "update" command: application has been destroyed
            if avg_flow > max_flow_tolerance:
                take_picture = True
                for color in bg_colors:
                    canvas.configure(background=color)
                    canvas.update()
                    individual_wait = total_wait / len(bg_colors)
                    time.sleep(individual_wait)
                canvas.configure(background='#FFFFFF')
                break
            ret, frame = cap.read()
            prev_gray = gray
            # Our operations on the frame come here
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (480, 320), interpolation = cv2.INTER_AREA)
            flow = np.array(cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0))
            avg_flow = np.average(flow)
            time.sleep(motion_wait)
 
    

def draw_grid_of_shapes(canvas_width, canvas_height, columns, rows, spacing, shape_width, shape_height, canvas, small_frame, shapes, h_multiplier=1, s_multiplier=1, v_multiplier=1):
    for row in range(rows):
        for column in range(columns):
            fill_hsv = small_frame[row][column]
            fill_h = (int(fill_hsv[0] * 2) * abs(h_multiplier)) % 360
            fill_s = min((fill_hsv[1] / 255) * abs(s_multiplier), 1)
            fill_v = min((fill_hsv[2] / 255) * abs(v_multiplier), 1)
            stroke_h = (fill_h * abs(h_multiplier)) % 360
            stroke_s = min(fill_s * .75 * abs(s_multiplier), 1)
            stroke_v = min(fill_v * .25 * abs(v_multiplier), 1)
            fill_color = hsv_to_hex((fill_h, fill_s, fill_v))
            stroke_color = hsv_to_hex((stroke_h, stroke_s, stroke_v))
            origin_y, origin_x, middle_x, middle_y = calculate_image_size_and_place(shape_width, shape_height, row, column, rows, columns, canvas_height, canvas_width, spacing)
            shapes[row][column][0], shapes[row][column][1] = draw_random_shape(canvas, shape_width, shape_height, origin_y, origin_x, middle_x, middle_y, fill_color=fill_color, stroke_color=stroke_color)


def calculate_resize_parameters(rows, columns):
    ''' Given the size of the captured image, determine the width and the height of the resized image, and how to crop it, to mantain proportions with the rows and columns of the pixel art '''
    cap = cv2.VideoCapture(0)
    capture_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    capture_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width_resized = (rows * capture_width) // capture_height
    if width_resized > columns:
        resize_height = rows
        resize_width = width_resized
    else:
        resize_height = (columns * capture_height) // capture_width
        resize_width = columns
    return resize_height, resize_width

def draw_random_shape(canvas, width, height, origin_y, origin_x, middle_x, middle_y, fill_color='#fff', stroke_color='#000'):
    random_shape = random.randint(0, 6)
    border_width = 1
    shrink_factor = 3
    small_width = width // shrink_factor
    small_height = height // shrink_factor
    small_origin_x = middle_x - small_width // 2
    small_origin_y = middle_y - small_height // 2
    # TODO: Correct the math so when it rounds pixels it's evenly centered

    # TODO: Shift triangles so the look more centered
    if random_shape == 0: # Diamond
        shape_big = canvas.create_polygon([middle_x, origin_y, origin_x + width, middle_y, middle_x, origin_y + height, origin_x, middle_y], fill=fill_color, outline=stroke_color, width=border_width)
        shape_small = canvas.create_polygon([middle_x, small_origin_y, small_origin_x + small_width, middle_y, middle_x, small_origin_y + small_height, small_origin_x, middle_y], fill=stroke_color, width=0)
    elif random_shape == 1: # Triangle up
        shape_big = canvas.create_polygon([middle_x, origin_y, origin_x + width, origin_y + height, origin_x, origin_y + height], fill=fill_color, outline=stroke_color, width=border_width)
        shape_small = canvas.create_polygon([middle_x, small_origin_y, small_origin_x + small_width, small_origin_y + small_height, small_origin_x, small_origin_y + small_height], fill=stroke_color, width=0)
    elif random_shape == 2: # Triangle down
        shape_big = canvas.create_polygon([middle_x, origin_y + height, origin_x + width, origin_y, origin_x, origin_y], fill=fill_color, outline=stroke_color, width=border_width)
        shape_small = canvas.create_polygon([middle_x, small_origin_y + small_height, small_origin_x + small_width, small_origin_y, small_origin_x, small_origin_y], fill=stroke_color, width=0)
    elif random_shape == 3: # Triangle right
        shape_big = canvas.create_polygon([origin_x, origin_y, origin_x + width, middle_y, origin_x, origin_y + height], fill=fill_color, outline=stroke_color, width=border_width)
        shape_small = canvas.create_polygon([small_origin_x, small_origin_y, small_origin_x + small_width, middle_y, small_origin_x, small_origin_y + small_height], fill=stroke_color, width=0)
    elif random_shape == 4: # Triangle left
        shape_big = canvas.create_polygon([origin_x + width, origin_y, origin_x, middle_y, origin_x + width, origin_y + height], fill=fill_color, outline=stroke_color, width=border_width)
        shape_small = canvas.create_polygon([small_origin_x + small_width, small_origin_y, small_origin_x, middle_y, small_origin_x + small_width, small_origin_y + small_height], fill=stroke_color, width=0)
    elif random_shape == 5: # Rectangle
        shape_big = canvas.create_rectangle(origin_x, origin_y, origin_x + width, origin_y + height, fill=fill_color, outline=stroke_color, width=border_width)
        shape_small = canvas.create_rectangle(small_origin_x, small_origin_y, small_origin_x + small_width, small_origin_y + small_height, fill=stroke_color, width=0)
    else: # Oval
        shape_big = canvas.create_oval(origin_x, origin_y, origin_x + width, origin_y + height, fill=fill_color, outline=stroke_color, width=border_width)
        shape_small = canvas.create_oval(small_origin_x, small_origin_y, small_origin_x + small_width, small_origin_y + small_height, fill=stroke_color, width=0)
    return shape_big, shape_small


def calculate_image_size_and_place(width, height, row, column, rows, columns, canvas_height, canvas_width, spacing):
    # Calculate initial position for each shape
    origin_y = row * (canvas_height / rows) + spacing // 2
    origin_x = column * (canvas_width / columns) + spacing // 2
    # Calculate middle points in each direction
    middle_x = origin_x + (width // 2)
    middle_y = origin_y + (height // 2)
    return origin_y, origin_x, middle_x, middle_y

# The main window of the animation
def create_window(canvas_width, canvas_height, canvas_origin_x, canvas_origin_y, title):
    geo = f'{canvas_width}x{canvas_height}+{canvas_origin_x}+{canvas_origin_y}' 
    window = tkinter.Tk()
    window.title(title)
    # Uses python 3.6+ string interpolation
    window.geometry(geo)
    return window

# Create a canvas for animation and add it to main window
def create_canvas(window, canvas_width, canvas_height):
    canvas = tkinter.Canvas(window, width=canvas_width, height=canvas_width)
    canvas.configure(bg="white")
    canvas.pack(fill="both", expand=True)
    return canvas

def main():

    # TODO: Account for wdith of window frame
    canvas_width = 800
    canvas_height = 600
    canvas_origin_x = 00
    canvas_origin_y = 00
    title = 'Geomirror'
    columns = 88
    rows = 41
    spacing = 2

    window = create_window(canvas_width, canvas_height, canvas_origin_x, canvas_origin_y, title)
    canvas = create_canvas(window, canvas_width, canvas_height)

        # while True:
    #     if avg_flow > 1.5:
    #         canvas.destroy()
    #         break
    #     print('looping')
    #     ret, frame = cap.read()
    #     prev_gray = gray
    #     # Our operations on the frame come here
    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     gray = cv2.resize(gray, (640, 360), interpolation = cv2.INTER_AREA)
    #     flow = np.array(cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0))
    #     avg_flow = np.average(flow)
    mirror(canvas, canvas_width, canvas_height, columns, rows, spacing)
    window.mainloop()



if __name__ == '__main__':
    main()
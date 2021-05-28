import tkinter #Tk, Canvas, Frame, BOTH, ARC
import random
import cv2
import numpy as np
from colorsys import rgb_to_hsv, hsv_to_rgb
from colorutils import hsv_to_hex
import inspect
import sys
import time


# https://www.geeksforgeeks.org/python-smile-detection-using-opencv/


def mirror(canvas, canvas_width, canvas_height, columns, rows, spacing, window_padding):
    resize_height, resize_width = calculate_resize_parameters(rows, columns)
    max_flow_tolerance = 1.5
    take_picture = True
    smile_detected = False
    total_wait = 1.5
    motion_wait = .1
    bg_colors = ['#FF9999', '#FFFF99', '#99FF99']
    h_multiplier = s_multiplier = v_multiplier = 1
    
    cap = cv2.VideoCapture(0)
    # The first photo tends to be darker, so we're taking a couple of photos 
    #   before entering the loop to "warm up" the camera
    _, frame = cap.read()
    #time.sleep(.05)
    #_, frame = cap.read()
    # Adaptive histogram equalization to equalize the color / contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    while True:
        if take_picture:
            canvas.delete("all")
            _, frame = cap.read()
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            lab = cv2.split(frame)
            lab[0] = clahe.apply(lab[0])
            lab[1] = clahe.apply(lab[1])
            lab[2] = clahe.apply(lab[2])
            frame = cv2.merge(lab)
            take_picture = False
        if smile_detected:
            canvas.delete("all")
            s_multiplier = 1.5
            v_multiplier = 1.5
            smile_detected = False
        small_frame = frame = cv2.resize(frame, (resize_width, resize_height), interpolation=cv2.INTER_AREA)
        small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
        #shapes = np.empty(shape=(rows, columns, 2))

        draw_grid_of_shapes(canvas_width, canvas_height, columns, rows, spacing, window_padding, canvas, small_frame, h_multiplier, s_multiplier, v_multiplier)
        h_multiplier = s_multiplier = v_multiplier = 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = clahe.apply(gray)
        gray = cv2.resize(gray, (480, 320), interpolation = cv2.INTER_AREA)
        avg_flow = 0
        while True:
            canvas.update()
            #TODO: Solve _tkinter.TclError: can't invoke "update" command: application has been destroyed
            key = cv2.waitKey(1)
            if key == 27 or key == ord('q') or key == ('Q'): # exit on ESC
                sys.exit()
            if avg_flow > max_flow_tolerance:
                take_picture = True
                for color in bg_colors:
                    canvas.configure(background=color)
                    canvas.update()
                    individual_wait = total_wait / len(bg_colors)
                    time.sleep(individual_wait)
                canvas.configure(background='#FFFFFF')
                break
            if detect_smile(gray):
                smile_detected = True
                break
            _, frame_looping = cap.read()
            prev_gray = gray
            # Our operations on the frame come here
            gray = cv2.cvtColor(frame_looping, cv2.COLOR_BGR2GRAY)
            gray = clahe.apply(gray)
            gray = cv2.resize(gray, (480, 320), interpolation = cv2.INTER_AREA)
            flow = np.array(cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0))
            avg_flow = np.average(flow)
            time.sleep(motion_wait)
 
    

def draw_grid_of_shapes(canvas_width, canvas_height, columns, rows, spacing, window_padding, canvas, small_frame, h_multiplier=1, s_multiplier=1, v_multiplier=1):
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
            origin_y, origin_x, middle_x, middle_y, shape_width, shape_height = calculate_image_size_and_place(row, column, rows, columns, canvas_height, canvas_width, spacing, window_padding)
            draw_random_shape(canvas, shape_width, shape_height, origin_y, origin_x, middle_x, middle_y, fill_color=fill_color, stroke_color=stroke_color)


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
    small_origin_x = middle_x - small_width / 2
    small_origin_y = middle_y - small_height / 2
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

def calculate_image_size_and_place(row, column, rows, columns, canvas_height, canvas_width, spacing, window_padding):

    shape_width = ((canvas_width - 2 * window_padding) / columns) - spacing
    shape_height = ((canvas_height - 2 * window_padding) / rows) - spacing
    # Calculate initial position for each shape
    origin_y = row * ((canvas_height - 2 * window_padding) / rows) + spacing / 2 + window_padding
    origin_x = column * ((canvas_width - 2 * window_padding) / columns) + spacing / 2 + window_padding
    # Calculate middle points in each direction
    middle_x = origin_x + (shape_width / 2)
    middle_y = origin_y + (shape_height / 2)


    return origin_y, origin_x, middle_x, middle_y, shape_width, shape_height

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
    canvas = tkinter.Canvas(window, width=canvas_width, height=canvas_height)
    canvas.configure(bg="white")
    canvas.pack(fill="both", expand=True)
    return canvas

def detect_smile(gray):
    # Adapted from https://dev.to/hammertoe/smile-detector-using-opencv-to-detect-smiling-faces-in-a-video-4l80
    # detect faces within the greyscale version of the frame
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_smile.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 3)
    num_smiles = 0

    # For each face we find...
    for (x, y, w, h) in faces:

        # Calculate the "region of interest", ie the are of the frame
        # containing the face
        roi_gray = gray[y:y+h, x:x+w]
        # roi_color = frame[y:y+h, x:x+w]

        # Within the grayscale ROI, look for smiles 
        smiles = smile_cascade.detectMultiScale(roi_gray, 3, 8)

        # If we find smiles then increment our counter
        if len(smiles):
            num_smiles += 1

    return num_smiles > 0

def main():

    CANVAS_WIDTH = 800
    CANVAS_HEIGHT = 600
    CANVAS_ORIGIN_X = 00
    CANVAS_ORIGIN_Y = 00
    
    TITLE = 'Geomirror'
    COLUMNS = 88
    ROWS = 41
    SPACING = 2
    WINDOW_PADDING = 5

    # Smile Detection https://www.geeksforgeeks.org/python-smile-detection-using-opencv/
    # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
    # eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_eye.xml')
    # smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_smile.xml')
    # faces = face_cascade.detectMultiScale(gray, 1.3, 5)


    window = create_window(CANVAS_WIDTH, CANVAS_HEIGHT, CANVAS_ORIGIN_X, CANVAS_ORIGIN_Y, TITLE)
    canvas = create_canvas(window, CANVAS_WIDTH, CANVAS_HEIGHT)

    mirror(canvas, CANVAS_WIDTH, CANVAS_HEIGHT, COLUMNS, ROWS, SPACING, WINDOW_PADDING)


if __name__ == '__main__':
    main()
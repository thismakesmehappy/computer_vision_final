import tkinter #Tk, Canvas, Frame, BOTH, ARC
import random
import cv2
import numpy as np
from colorsys import rgb_to_hsv, hsv_to_rgb
from colorutils import hsv_to_hex
import time
import argparse
import dlib
import math
import traceback
BLINK_RATIO_THRESHOLD = 9

def calculate_image_size_and_place(row, column, rows, columns, canvas_height, canvas_width, spacing, window_padding):
    '''
    Determines the size and location of a shape
    Args:
        row, column (ints): the row and column for the pixel
        rows, columns (ints): total rows and columns in the image
        canvas_height, canvas_width (ints): dcanvas dimentions
        spacing (int): spacing between rows and columns
        window_padding (int): spacing between the shapes and the window
    Returns:
        the locaition and dimentions of the shape
    '''

    shape_width = ((canvas_width - 2 * window_padding) / columns) - spacing
    shape_height = ((canvas_height - 2 * window_padding) / rows) - spacing
    # Calculate initial position for each shape
    origin_y = row * ((canvas_height - 2 * window_padding) / rows) + spacing / 2 + window_padding
    origin_x = column * ((canvas_width - 2 * window_padding) / columns) + spacing / 2 + window_padding
   
    return origin_y, origin_x, shape_width, shape_height

def calculate_resize_parameters(rows, columns):
    ''' 
    Given the size of the captured image, determine the width and the height 
    the resized image to mantain proportions with the rows and columns of the 
    pixel art. We will later crop the image to center it in the pizel art
    Args. Since we are capturing from camera, we use the camera's resolution
    to determine the source's dimensions
    Args:
        rows, columns (ints): Rows and columns for the pixel art
    Returns:
        ints with the width and height to resize the image proportionally
    '''
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

def create_canvas(window, canvas_width, canvas_height):
    '''
    Initialies a Tkinter canvas
    Args:
        window: the parent window object
        canvas_width, canvas_height (ints): canvas dimetnions
        window_origin_x, window_origin_y (ints): placement for the canvas
    Returns:
        a canvas object

    '''
    canvas = tkinter.Canvas(window, width=canvas_width, height=canvas_height)
    canvas.configure(bg="white")
    canvas.pack(fill="both", expand=True)
    return canvas

def create_window(window_width, window_height, window_origin_x, window_origin_y, title):
    '''
    Initialies a Tkinter window
    Args:
        window_width, window_height (ints): window dimetnions
        window_origin_x, window_origin_y (ints): placement for the canvas
        title (str): the title of the window
    Returns:
        a window object

    '''
    geo = f'{window_width}x{window_height}+{window_origin_x}+{window_origin_y}' 
    window = tkinter.Tk()
    window.title(title)
    # Uses python 3.6+ string interpolation
    window.geometry(geo)
    return window

def detect_blink(gray, detector, predictor):
    '''
    Determine if the user blinked
    # Adapted from https://medium.com/algoasylum/blink-detection-using-python-737a88893825
    Args:
        gray: grayscale image capture
        detector: detector object (we inject it to avoid loading it multiple times)
        predictor: trained data to detect blinkcs (we inkect it to avoit loading
            it multiple times)
    Returns:
        True if a blink was detected, False if not
    '''
    left_eye_landmarks = [36, 37, 38, 39, 40, 41]
    right_eye_landmarks = [42, 43, 44, 45, 46, 47]
    # Blink detection
    faces,_,_ = detector.run(image = gray, upsample_num_times = 0, adjust_threshold = 0.0)
    for face in faces:
        landmarks = predictor(gray, face)
        left_eye_ratio  = get_blink_ratio(left_eye_landmarks, landmarks)
        right_eye_ratio = get_blink_ratio(right_eye_landmarks, landmarks)
        blink_ratio     = (left_eye_ratio + right_eye_ratio) / 2
        if blink_ratio > BLINK_RATIO_THRESHOLD:
                        #Blink detected! Do Something!
            return True
    return False

def detect_smile(gray, face_cascade, smile_cascade):
    '''
    Detect if the user smiled
    # Adapted from https://dev.to/hammertoe/smile-detector-using-opencv-to-detect-smiling-faces-in-a-video-4l80
    Args:
        gray: graysfale image
        face_cascade, smile_cascade: trained data for cascade detectors
    Returns: 
        True if a smile was detected, False otherwise
    '''
    # detect faces within the greyscale version of the frame

    faces = face_cascade.detectMultiScale(gray, 1.1, 3)

    # For each face we find...
    for (x, y, w, h) in faces:

        # Calculate the "region of interest", ie the are of the frame
        # containing the face
        roi_gray = gray[y:y+h, x:x+w]
        # roi_color = frame[y:y+h, x:x+w]

        # Within the grayscale ROI, look for smiles 
        smiles = smile_cascade.detectMultiScale(roi_gray, 4, 9)

        # If we find smiles then increment our counter
        if len(smiles):
            return True

    return False

def draw_grid_of_shapes(canvas_width, canvas_height, rows, columns, spacing, window_padding, canvas, small_frame, border, h_multiplier=1, s_multiplier=1, v_multiplier=1):
    '''
    Produces a grid of shapes that corresponds to the pixels in the resized 
    image to produce the pixel art. This function will determine the color of
    each pixel, process it if necessary, and will put a shape in each 
    position corresponding to the pixel
    Args:
        canvas_width, canvas_height (ints): Canvas dimentions
        rows, columns (ints): Number of rows and columns in the pixel art
        spacing (int): number of pixels between each row or column
        window_padding (int): spacing between the window borders and the shapes
        canvas (Tkinter canvas): canvas where we'll draw the shapes
        small_frame (image): resized image to translate into pixel art
        border (int): border width
        h_miltiplier, s_miltiplier, v_multiplier (ints): multipliers to 
            modify the HSV values if needed
    Returns:
        draws a grid of shapes in the canvas
    '''

    # Since the image was resized to preserve proportions, the number of rows
    #   and columns will not always match. We will calculate how many rows or
    #   columns to shift to center the image in our pixel art
    image_height = small_frame.shape[0]
    image_width = small_frame.shape[1]
    start_row = (image_height - rows) // 2
    start_column = (image_width - columns) // 2

    # For every pixel in the resized image, take the pixel's color and draw the
    #   corresponding shape with its color
    for row in range(rows):
        for column in range(columns):
            fill_hsv = small_frame[row + start_row][column + start_column]
            fill_h = (int(fill_hsv[0] * 2) * abs(h_multiplier)) % 360
            fill_s = min((fill_hsv[1] / 255) * abs(s_multiplier), 1)
            fill_v = min((fill_hsv[2] / 255) * abs(v_multiplier), 1)
            stroke_h = (fill_h * abs(h_multiplier)) % 360
            stroke_s = min(fill_s * .75 * abs(s_multiplier), 1)
            stroke_v = min(fill_v * .25 * abs(v_multiplier), 1)
            fill_color = hsv_to_hex((fill_h, fill_s, fill_v))
            stroke_color = hsv_to_hex((stroke_h, stroke_s, stroke_v))
            origin_y, origin_x, shape_width, shape_height = calculate_image_size_and_place(row, column, rows, columns, canvas_height, canvas_width, spacing, window_padding)
            draw_random_shape(canvas, shape_width, shape_height, origin_y, origin_x, border, fill_color=fill_color, stroke_color=stroke_color)

def draw_random_shape(canvas, width, height, origin_y, origin_x, border,fill_color='#fff', stroke_color='#000'):
    '''
    Given the pixel color and the location and size for the current shape,
    draw a random shaape (elipse, rectangle, diamond, or triangle pointing up,
    down, left or right). Each shape will have a border, slightly darker than
    the fill, and a smalle version of the same shape in the middle. 
    Args:
        canvas (Tkinter canvas): Canvas to draw the shapes in
        width, height (ints): width and height for the shape
        origin_y, origin_x (ints): the shape's distance from the top and 
            left side of the window
        fill_color, stroke_color (hex): Hexadecimal values for the fill and
            stroke colors
    Returns:
        Draws a shape in the canvas
    '''
    # Calculate secondary measurements as needed
    middle_x = origin_x + (width / 2)
    middle_y = origin_y + (height / 2)
    border
    shrink_factor = 3
    small_width = width / shrink_factor
    small_height = height / shrink_factor
    small_origin_x = middle_x - small_width / 2
    small_origin_y = middle_y - small_height / 2

    # Draw a random number to determine which shape will be drawn
    random_shape = random.randint(0, 6)
    if random_shape == 0: # Diamond
        shape_big = canvas.create_polygon([middle_x, origin_y, origin_x + width, middle_y, middle_x, origin_y + height, origin_x, middle_y], fill=fill_color, outline=stroke_color, width=border)
        shape_small = canvas.create_polygon([middle_x, small_origin_y, small_origin_x + small_width, middle_y, middle_x, small_origin_y + small_height, small_origin_x, middle_y], fill=stroke_color, width=0)
    elif random_shape == 1: # Triangle up
        shape_big = canvas.create_polygon([middle_x, origin_y, origin_x + width, origin_y + height, origin_x, origin_y + height], fill=fill_color, outline=stroke_color, width=border)
        small_origin_y += small_height // 2
        shape_small = canvas.create_polygon([middle_x, small_origin_y, small_origin_x + small_width, small_origin_y + small_height, small_origin_x, small_origin_y + small_height], fill=stroke_color, width=0)
    elif random_shape == 2: # Triangle down
        shape_big = canvas.create_polygon([middle_x, origin_y + height, origin_x + width, origin_y, origin_x, origin_y], fill=fill_color, outline=stroke_color, width=border)
        small_origin_y -= small_height // 2
        shape_small = canvas.create_polygon([middle_x, small_origin_y + small_height, small_origin_x + small_width, small_origin_y, small_origin_x, small_origin_y], fill=stroke_color, width=0)
    elif random_shape == 3: # Triangle right
        shape_big = canvas.create_polygon([origin_x, origin_y, origin_x + width, middle_y, origin_x, origin_y + height], fill=fill_color, outline=stroke_color, width=border)
        small_origin_x -= small_width // 3
        shape_small = canvas.create_polygon([small_origin_x, small_origin_y, small_origin_x + small_width, middle_y, small_origin_x, small_origin_y + small_height], fill=stroke_color, width=0)
    elif random_shape == 4: # Triangle left
        shape_big = canvas.create_polygon([origin_x + width, origin_y, origin_x, middle_y, origin_x + width, origin_y + height], fill=fill_color, outline=stroke_color, width=border)
        small_origin_x += small_width // 3
        shape_small = canvas.create_polygon([small_origin_x + small_width, small_origin_y, small_origin_x, middle_y, small_origin_x + small_width, small_origin_y + small_height], fill=stroke_color, width=0)
    elif random_shape == 5: # Rectangle
        shape_big = canvas.create_rectangle(origin_x, origin_y, origin_x + width, origin_y + height, fill=fill_color, outline=stroke_color, width=border)
        shape_small = canvas.create_rectangle(small_origin_x, small_origin_y, small_origin_x + small_width, small_origin_y + small_height, fill=stroke_color, width=0)
    else: # Oval
        shape_big = canvas.create_oval(origin_x, origin_y, origin_x + width, origin_y + height, fill=fill_color, outline=stroke_color, width=border)
        shape_small = canvas.create_oval(small_origin_x, small_origin_y, small_origin_x + small_width, small_origin_y + small_height, fill=stroke_color, width=0)
    return shape_big, shape_small

def euclidean_distance(point1 , point2):
    '''
    Helper function for blink detector. Calculates the eucledian distance
    between two points
    Args:
        point1, point2 (tuples of ints: Two points
    Returns:
        Tuple of ints
    '''
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def equalize_image(frame):
    '''
    Equalize color image using adaptive histogram equalization 
    Args:
        frame: image in BGR color space
    Returns:
        same image, but with equalizes histogram
    '''
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lab = cv2.split(frame)
    lab[0] = clahe.apply(lab[0])
    lab[1] = clahe.apply(lab[1])
    lab[2] = clahe.apply(lab[2])
    frame = cv2.merge(lab)
    return frame

def get_blink_ratio(eye_points, facial_landmarks):
    '''
    Helper function for blink detector. Calculates the blink ration for each eye
    Args:
        eye_points (list of ints): indexes for the features around each eye
        facial_landmarks: Trained data for the feature predictor
    Returns:
        Int
    '''
    
    #loading all the required points
    corner_left  = (facial_landmarks.part(eye_points[0]).x, 
                    facial_landmarks.part(eye_points[0]).y)
    corner_right = (facial_landmarks.part(eye_points[3]).x, 
                    facial_landmarks.part(eye_points[3]).y)
    
    center_top    = midpoint(facial_landmarks.part(eye_points[1]), 
                             facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), 
                             facial_landmarks.part(eye_points[4]))

    #calculating distance
    horizontal_length = euclidean_distance(corner_left,corner_right)
    vertical_length = euclidean_distance(center_top,center_bottom)

    ratio = horizontal_length / vertical_length

    return ratio

def midpoint(point1 ,point2):
    '''
    Helper function for blink detector. Determines the mid point beteween 
    two points
    Args:
        point1, point2 (tuples of ints: Two points
    Returns:
        Tuple of ints
    '''
    return (point1.x + point2.x)/2,(point1.y + point2.y)/2

def mirror(window, canvas, canvas_width, canvas_height, columns, rows, spacing, window_padding, border):
    '''
    This is the main function for this program. We first load the necessary data for facial feature recognition and initialized our image capture. We have two nexted loops The larger one is an infinite loop that takes the photo, downsamples it to obtain the right pixels for the pixel art, and then translats the image into a grid of shapes with the corresponding colors. In this larger loop we also determine the HSV modifiers for the output graphic depending on whether the person smiled or blinked. The inner loop is also an infinite loop that determines if there is enough flow, if the person smiled, or blinked. If any of these actions happen, we go back to the outer loop and take the appropriate action (take a new picture if there was enough flow, brighten the image if there was a smile, or darken the image if there was a wink) and repeat the process.
    Args:
        window (Tkinter window): The parent container for the pixel art
        canvas (Tkinter Canvas): Canvas to hold the shapes we'll draw
        canvas_width, canvas height (int): The width and height of the canvas
        columns, rows (int): the number of columns and rows in the output pixel art
        spacing (int): number of pixels between each row or column
        window_padding (int): blank space between the shapes and the borders of the window
    Returns:
        Renders the pixel art in the window
    '''
    # Determine the dimensions to resize the image based on the rows and columnts
    resize_height, resize_width = calculate_resize_parameters(rows, columns)

    max_flow_tolerance = 1.5
    take_picture = True
    smile_detected = False
    blink_detected = False
    total_wait = 1.5
    motion_wait = .1
    bg_colors = ['#FF9999', '#FFFF99', '#99FF99']
    h_multiplier = s_multiplier = v_multiplier = 1
    # Load data for the face classifier to recognize a smile
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_smile.xml')
    # Initialize the predictor to detect a blink and initialize the data
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('data/shape_predictor_68_face_landmarks.dat')


    # Initialize the image capture through camera. Since the first photo tends 
    #   to be darker, we're taking a couple of photos before entering the loop 
    #   to "warm up" the camera
    cap = cv2.VideoCapture(0)
    _, frame = cap.read()
    time.sleep(.05)
    _, frame = cap.read()

    # This is the outer loop that draws the grid of images
    try:
        while True:
            if take_picture:
                canvas.delete("all")
                _, frame = cap.read()
                frame = equalize_image(frame)
                take_picture = False
            if smile_detected:
                canvas.delete("all")
                h_multiplier = 1
                s_multiplier = 1.5
                v_multiplier = 1.75
                smile_detected = False
            if blink_detected:
                canvas.delete("all")
                h_multiplier = 1.3
                s_multiplier = .5
                v_multiplier = 0.8
                blink_detected = False

            # Resize the image to have the right number of pixels for the 
            #   pixel art, then draw the shapes
            small_frame = cv2.resize(frame, (resize_width, resize_height), interpolation=cv2.INTER_AREA)
            small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)

            draw_grid_of_shapes(canvas_width, canvas_height, rows, columns, spacing, window_padding, canvas, small_frame, border, h_multiplier, s_multiplier, v_multiplier)
            h_multiplier = s_multiplier = v_multiplier = 1
            
            # Convert the image to grayscale to measure the flow. Within the 
            #   inner loop, We will lways keep track of the current and previous 
            #   grayscale frame to calculate the flow. We resize the image to
            #   reduce the computational cost of calculating flow
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (480, 320), interpolation = cv2.INTER_AREA)

            # We initiallize the flow to 0 because we currently only have one  
            #   image, but once we complete the following loop we will have two
            #   images to compare and calculate flow
            avg_flow = 0
            while True:
                window.update()
                # If we have enough flow, give a visual signal that we will 
                #   exit the loop to take another photo. Change the flag to
                #   take another picture and exit the inner loop
                if avg_flow > max_flow_tolerance:
                    take_picture = True
                    for color in bg_colors:
                        canvas.configure(background=color)
                        window.update()
                        individual_wait = total_wait / len(bg_colors)
                        time.sleep(individual_wait)
                    canvas.configure(background='#FFFFFF')
                    break

                # Detect a smile or blink, change the flag for either event,
                #   and exit the inner loop
                if detect_smile(gray, face_cascade, smile_cascade):
                    smile_detected = True
                    break
                
                if detect_blink(gray, detector, predictor):
                    blink_detected = True
                    break
                
                # Store the current picture as the previous one and take a new 
                #   picture, Then calculate flow
                _, frame_looping = cap.read()
                frame_looping = equalize_image(frame_looping)
                prev_gray = gray
                # Our operations on the frame come here
                gray = cv2.cvtColor(frame_looping, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (480, 320), interpolation = cv2.INTER_AREA)
                flow = np.array(cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0))
                avg_flow = np.average(flow)

                # We're waiting a fraction of a second to reduce the computational cost
                time.sleep(motion_wait)
    except:
    #    pass
        traceback.print_exc()

def main():
    '''
    This is the main function that kicks off the program
    '''

    # Gather the argument vector
    parser = argparse.ArgumentParser(description='Creates pixel art from image capture. Detects motion to determine when to refresh the iamge.')
    parser.add_argument('--width', type=str, help='Canvas width', default='800')
    parser.add_argument('--height', type=str, help='Canvas height', default='600')
    parser.add_argument('--originx', type=str, help='X-position for the canvas', default='0')
    parser.add_argument('--originy', type=str, help='Y-position for the canvas', default='0')
    parser.add_argument('--columns', type=str, help='Number of columns in pixel art', default='81')
    parser.add_argument('--rows', type=str, help='Number of rows in pixel art', default='41')
    parser.add_argument('--spacing', type=str, help='Space between pixel shapes', default='2')
    parser.add_argument('--border', type=str, help='Boder width', default='1')
    parser.add_argument('--padding', type=str, help='Padding around the window', default='5')
    args = parser.parse_args()
    
    CANVAS_WIDTH = int(args.width)
    CANVAS_HEIGHT = int(args.height)
    CANVAS_ORIGIN_X = int(args.originx)
    CANVAS_ORIGIN_Y = int(args.originy)
    COLUMNS = int(args.columns)
    ROWS = int(args.rows)
    SPACING = int(args.spacing)
    WINDOW_PADDING = int(args.padding)
    BORDER_WIDTH = int(args.border)
    TITLE = 'Geomirror'

    # Initialize window and canvas
    window = create_window(CANVAS_WIDTH, CANVAS_HEIGHT, CANVAS_ORIGIN_X, CANVAS_ORIGIN_Y, TITLE)
    canvas = create_canvas(window, CANVAS_WIDTH, CANVAS_HEIGHT)
    # Detect if the escape key was pressed, if so exit
    window.bind_all('<Escape>', lambda x: window.destroy())
    # Run main loop
    mirror(window, canvas, CANVAS_WIDTH, CANVAS_HEIGHT, COLUMNS, ROWS, SPACING, WINDOW_PADDING, BORDER_WIDTH)


if __name__ == '__main__':
    main()
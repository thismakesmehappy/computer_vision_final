import tkinter as tk #Tk, Canvas, Frame, BOTH, ARC
import random
import cv2
import numpy as np
from colorsys import rgb_to_hsv, hsv_to_rgb
from colorutils import hsv_to_hex, rgb_to_hex
import inspect

class Example(tk.Frame):

    def __init__(self,canvas_height, canvas_width, columns, rows, spacing):
        super().__init__()
        self.canvas_height = canvas_height
        self.canvas_width = canvas_width
        self.columns = columns
        self.rows = rows
        self.spacing = spacing
        self.shape_width = (self.canvas_width // self.columns) - self.spacing
        self.shape_height = (self.canvas_height // self.rows) - self.spacing
        self.current_image = np.ndarray(shape=(rows, columns))
        self.resize_height = 0
        self.resize_width = 0
        cap = cv2.VideoCapture(0)
        self.capture_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.capture_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.calculate_resize_parameters()


        self.initUI()

    def initUI(self):

        self.master.title("Lines")
        self.pack(fill=tk.BOTH, expand=1)

        canvas = tk.Canvas(self)


        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()

        
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #gray = cv2.resize(gray, (640, 360), interpolation = cv2.INTER_AREA)
        #frame[0, 0, 0] = 255
        #frame[0,0, 1]=0
        #frame[0,0, 2]=0
        #color = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        #color = cv2.resize(color, (columns, rows), interpolation = cv2.INTER_AREA)
        #color = cv2.cvtColor(color, cv2.COLOR_HSV2BGR)
        frame = cv2.resize(frame, (self.columns, self.rows), interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        while True:
            
            for row in range (self.rows):
                for column in range(self.columns):
                    color = rgb_to_hex((frame[row, column, :]))
                    origin_y, origin_x, middle_x, middle_y = self.calculate_image_size_and_place(self.shape_width, self.shape_height, row, column)
                    self.draw_random_shape(canvas, self.shape_width, self.shape_height, origin_y, origin_x, middle_x, middle_y, fill_color=color)
            break

        canvas.pack(fill=tk.BOTH, expand=1)

    def calculate_resize_parameters(self):
        ''' Given the size of the captured image, determine the width and the height of the resized image, and how to crop it, to mantain proportions with the rows and columns of the pixel art '''
        width_resized = (self.rows * self.capture_width) // self.capture_height
        if width_resized > self.columns:
            self.resize_height = self.rows
            self.resize_width = width_resized
        else:
            self.resize_height = (self.columns * self.capture_height) // self.capture_width
            self.resize_width = self.columns

    def calculate_image_size_and_place(self, width, height, row, column):
        # Calculate initial position for each shape
        origin_y = row * (self.canvas_height / self.rows) + self.spacing // 2
        origin_x = column * (self.canvas_width / self.columns) + self.spacing // 2
        # Calculate middle points in each direction
        middle_x = origin_x + (width // 2)
        middle_y = origin_y + (height // 2)
        return origin_y,origin_x,middle_x,middle_y

    def draw_random_shape(self, canvas, width, height, origin_y, origin_x, middle_x, middle_y, fill_color='#fff', outline_color='#000'):
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
            canvas.create_polygon([middle_x, origin_y, origin_x + width, middle_y, middle_x, origin_y + height, origin_x, middle_y], fill=fill_color, outline=outline_color, width=border_width)
            canvas.create_polygon([middle_x, small_origin_y, small_origin_x + small_width, middle_y, middle_x, small_origin_y + small_height, small_origin_x, middle_y], fill=outline_color, width=0)
        elif random_shape == 1: # Triangle up
            canvas.create_polygon([middle_x, origin_y, origin_x + width, origin_y + height, origin_x, origin_y + height], fill=fill_color, outline=outline_color, width=border_width)
            canvas.create_polygon([middle_x, small_origin_y, small_origin_x + small_width, small_origin_y + small_height, small_origin_x, small_origin_y + small_height], fill=outline_color, width=0)
        elif random_shape == 2: # Triangle down
            canvas.create_polygon([middle_x, origin_y + height, origin_x + width, origin_y, origin_x, origin_y], fill=fill_color, outline=outline_color, width=border_width)
            canvas.create_polygon([middle_x, small_origin_y + small_height, small_origin_x + small_width, small_origin_y, small_origin_x, small_origin_y], fill=outline_color, width=0)
        elif random_shape == 3: # Triangle right
            canvas.create_polygon([origin_x, origin_y, origin_x + width, middle_y, origin_x, origin_y + height], fill=fill_color, outline=outline_color, width=border_width)
            canvas.create_polygon([small_origin_x, small_origin_y, small_origin_x + small_width, middle_y, small_origin_x, small_origin_y + small_height], fill=outline_color, width=0)
        elif random_shape == 4: # Triangle left
            canvas.create_polygon([origin_x + width, origin_y, origin_x, middle_y, origin_x + width, origin_y + height], fill=fill_color, outline=outline_color, width=border_width)
            canvas.create_polygon([small_origin_x + small_width, small_origin_y, small_origin_x, middle_y, small_origin_x + small_width, small_origin_y + small_height], fill=outline_color, width=0)
        elif random_shape == 5: # Rectangle
            canvas.create_rectangle(origin_x, origin_y, origin_x + width, origin_y + height, fill=fill_color, outline=outline_color, width=border_width)
            canvas.create_rectangle(small_origin_x, small_origin_y, small_origin_x + small_width, small_origin_y + small_height, fill=outline_color, width=0)
        else: # Oval
            canvas.create_oval(origin_x, origin_y, origin_x + width, origin_y + height, fill=fill_color, outline=outline_color, width=border_width)
            canvas.create_oval(small_origin_x, small_origin_y, small_origin_x + small_width, small_origin_y + small_height, fill=outline_color, width=0)


def main():

    # TODO: Account for wdith of window frame
    canvas_width = 800
    canvas_height = 600
    canvas_origin_x = 00
    canvas_origin_y = 00
    columns = 88
    rows = 41
    spacing = 2
    root = tk.Tk()
    ex = Example(canvas_height, canvas_width, columns, rows, spacing)
    geo = f'{canvas_width}x{canvas_height}+{canvas_origin_x}+{canvas_origin_y}'
    root.geometry(geo)

    root.mainloop()



if __name__ == '__main__':
    main()
import tkinter as tk #Tk, Canvas, Frame, BOTH, ARC
import random
import cv2
import numpy as np
from colorsys import rgb_to_hsv, hsv_to_rgb

class Example(tk.Frame):

    def __init__(self,canvas_height, canvas_width, columns, rows, spacing):
        super().__init__()
        self.canvas_height = canvas_height
        self.canvas_width = canvas_width
        self.columns = columns
        self.rows = rows
        self.spacing = spacing

        self.initUI()

    def initUI(self):

        self.master.title("Lines")
        self.pack(fill=tk.BOTH, expand=1)

        canvas = tk.Canvas(self)
        width = (self.canvas_width // self.columns) - self.spacing
        height = (self.canvas_height // self.rows) - self.spacing
        for row in range (self.rows):
            for column in range(self.columns):
                # Calculate initial position for each shape
                origin_y = row * (self.canvas_height / self.rows) + self.spacing // 2
                origin_x = column * (self.canvas_width / self.columns) + self.spacing // 2
                # Calculate middle points in each direction
                middle_x = origin_x + (width // 2)
                middle_y = origin_y + (height // 2)

                self.draw_random_shape(canvas, width, height, origin_y, origin_x, middle_x, middle_y)


        canvas.pack(fill=tk.BOTH, expand=1)

    def draw_random_shape(self, canvas, width, height, origin_y, origin_x, middle_x, middle_y, fill_color='#fff', outline_color='#000'):
        random_shape = random.randint(0, 6)
        border_width = 2
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
    canvas_width = 600
    canvas_height = 400
    canvas_origin_x = 00
    canvas_origin_y = 00
    columns = 21
    rows = 15
    spacing = 8
    root = tk.Tk()
    ex = Example(canvas_height, canvas_width, columns, rows, spacing)
    geo = f'{canvas_width}x{canvas_height}+{canvas_origin_x}+{canvas_origin_y}'
    root.geometry(geo)
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (640, 360), interpolation = cv2.INTER_AREA)
    dst = cv2.edgePreservingFilter(frame, flags=1, sigma_s=60, sigma_r=0.4)
    color = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    print('got here')
    color = cv2.resize(color, (10, 10), interpolation = cv2.INTER_AREA)
    print(color)
    cv2.imshow('frame',color)
    print(color[0,0,0], color[0, 0, 1], color[0, 0, 2])
    root.mainloop()



if __name__ == '__main__':
    main()
import tkinter as tk #Tk, Canvas, Frame, BOTH, ARC
import random
import cv2

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

                random_shape = random.randint(0, 6)
                
                if random_shape == 0:
                    canvas.create_polygon([middle_x, origin_y, origin_x + width, middle_y, middle_x, origin_y + height, origin_x, middle_y])
                elif random_shape == 1:
                    canvas.create_polygon([middle_x, origin_y, origin_x + width, origin_y + height, origin_x, origin_y + height])
                elif random_shape == 2:
                    canvas.create_polygon([middle_x, origin_y + height, origin_x + width, origin_y, origin_x, origin_y])
                elif random_shape == 3:
                    canvas.create_polygon([origin_x, origin_y, origin_x + width, middle_y, origin_x, origin_y + height])
                elif random_shape == 4:
                    canvas.create_polygon([origin_x + width, origin_y, origin_x, middle_y, origin_x + width, origin_y + height])
                elif random_shape == 5:
                    canvas.create_rectangle(origin_x, origin_y, origin_x + width, origin_y + height)
                else:
                    canvas.create_oval(origin_x, origin_y, origin_x + width, origin_y + height)


        canvas.pack(fill=tk.BOTH, expand=1)


def main():

    
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
    root.mainloop()


if __name__ == '__main__':
    main()
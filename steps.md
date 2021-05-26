- create window
- calculate sizes for columns and rows
-capture initial image
-determine resize parameters
    -determine proportions of columns to row
    -determine proportions of image width to image height
    -determine the direction to resize over
    -determine how the image will need to be cropped
loop
    -capture camera image
    -convert to hsv
    -do any necessary image manipulations
    -draw the grid
        -extract each color from the image
        -convert the color to hex
        -draw the corresponding shape with the color picked
    -loop camera capture
        -downsample (smaller image, and grayscale)
        -calculate the dense flow
        -if the dense flow is above the threshold
           -signal 3 seconds to give the prson time
                -lighten the shapes to 75%, then 50%, then 25%, one second each
                (change: make the background red, yellow, green)
            -exit loop

take parameters for different variables
recognize wink or smile and change colors around that

    
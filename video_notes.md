# Pitch - 1-2 sentences on why the problem you worked on is important
- Coming from a creative background, I'm interested in finding ways to combine computing with design. As we saw with some papers, there are many creative applications for computer vision, and in this case I wanted to come from the creative side and use computing to generate artwork. I made a pixel art generator to showcase how we can use code to create images and to show how we can use computer vision to find innovative ways to interact with art.
# Explanation of Contribution - What did you do?
- For this project, I created a script that captures an image from a computer camera and transforms it to pixel art. The main functionality relies on two loops; for the external loop we capture an image, resize it, and then produce a grid of shapes that reflect the colors in each pixel of the resized image. The inner loop will continue capturing images from the camera until it detects enough movement, a smile, or a blink, and when it does it will go back to the outer loop to generate a new grid. While the premise behind the pixel generator is simple, I think the interesting part is how we can interact with the piece through gestures that don't involve physical contact with the computer.
# Example results - have fun here!
- Default settings
python final_project.py 
- A higher numbner of pixels. The rows and columns are not proportional, so we crop the center of the image. The sam would happen inte other direction.
python final_project.py --width 1500 --height 900 --rows 100 --columns 50 --spacing 5 --border 2
- We can get interesting results with a lower number of rows and columns
python final_project.py --width 1500 --height 900 --rows 10 --columns 30 --spacing 20 --border 5
- Higher flow, adjusting the smile sensibility and reducing blink
python final_project.py --width 1500 --height 900 --rows 5 --columns 8 --spacing 20 --border 10 --flow 3 --smile 4,9 --blink 20 --flow 0.5
- Higher flow, adjusting the smile sensibility and reducing blink
python final_project.py --width 1500 --height 900 --rows 5 --columns 8 --spacing 20 --border 10 --flow 3 --smile 4,9 --blink 10 --flow 2
# Future Work - What would you do next if you had time?
- One of the issues I'm experiencing is that the flow detector, smile detector, and blink detector react differently under different light condition, with different parameters, and depending on the subject's distance to the camera. I added parameters for these three factors and applied a historgram equalizer to improve the image, but with more time I would like to find other ways to normalize the reaction under different light conditions. I would also like to test it with other people to see how the application recognizes different faces, and make adjustments for that.
- I also simplified the math to calculate the size and location for the different shapes. Particularly, I relied on floatin point division, which causes rounding errors when translating to whole pixels. I would like to go back and adjust the math to make the shapes more precise.
- Since this is an interactive pixel art generator, I would like to build it as a stand-alone kiosk. I'll port the script to a Raspberry Pi with a screen and camera. I would also like to explore other interactions, like saving the image, choosing a single shape, or more color manipulations with different gestures.


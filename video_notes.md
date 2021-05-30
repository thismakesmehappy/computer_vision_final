# Pitch - 1-2 sentences on why the problem you worked on is important
- Coming from a creative background, I'm interested in finding ways to combine computing with design. As we saw with some papers, there are many creative applications for computer vision. With this project I wanted to create an interactive pixel art generator to showcase how we can use code to generate creative images.
# Explanation of Contribution - What did you do?
- For this project, I created a script that captures an image from a computer camera and transforms it to pixel art. The main functionality relies on two infinite loop; for the external loop we capture an image, resize it, and then produce a grid of shapes that reflect the colors in each pixel of the resized image. The inner loop will continue capturing images from the camera until it detects enough movement, a smile, or a blink.
# Example results - have fun here!
- Default settings
python final_project.py 
- A higher numbner of pixels. The rows and columns are not proportional, so we crop the center of the image. The sam would happen inte other direction.
python final_project.py --width 1500 --height 1000 --rows 100 --columns 50 --spacing 5 --border 2
- We can get interesting results with a lower number of rows and columns
python final_project.py --width 1500 --height 1000 --rows 10 --columns 30 --spacing 20 --border 5
- We can adjust the flow sensibility to change the image more often
python final_project.py --width 1500 --height 1000 --rows 5 --columns 8 --spacing 20 --border 10 --flow 0.5
- Higher flow, adjusting the smile sensibility
python final_project.py --width 1500 --height 1000 --rows 5 --columns 8 --spacing 20 --border 10 --flow 3 --smile 2,12
- More column than rows (resetting the flow)
python final_project.py --width 1500 --height 1000 --rows 5 --columns 25 --spacing 20 --border 10 --smile 3,12
- More rows than columns
 python final_project.py --width 1500 --height 1000 --rows 30 --columns 5 --spacing 10 --border 5 --smile 3,12
# Future Work - What would you do next if you had time?
- One of the issues I'm experiencing is that the flow detector, smile detector, and blink detector react differently under different light conditions. The different parameters also affect the sensibility to each feature. I added parameters for these three factors and applied a historgram equalizer to improve the image, but with more time I would like to find other ways to manipulate the input to normalize the reaction under different light conditions.
- I also simplified the math to calculate the size and location for the different shapes. Particularly, I relied on floatin point division, which causes rounding errors when translating to whole pixels. I would like to go back and adjust the math to make the shapes more precise.
- Since this is an interactive pixel art generator, I would like to build it as a stand-alone kiosk. I'll port the script to a Raspberry Pi with a screen and camera. I would also like to explore other interactions, like saving the image or more color manipulation with different gestures.


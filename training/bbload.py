import cv2 as cv
import json
import os
import math
print(os.getcwd())

image_dir_path = "../dataset/images/aze_passport"
annotations_file_path = "../custom_dataset/annotations/aze_passport.json"

#load image
target_img = "96.jpg"
img_path = image_dir_path + "/" + target_img
print(f"Loading Image : {img_path}")

#file size
file_size = os.path.getsize(img_path)


image = cv.imread(img_path)

#scale_image
[scale_x, scale_y] = [0.6,0.6]
image_width, image_height = image.shape[:2]

new_dimensions = (math.floor(image_height*scale_y), math.floor(image_width * scale_x))
image = cv.resize(image,new_dimensions , interpolation=cv.INTER_AREA)
print(image.shape[:2], new_dimensions)

#load_annotation
def annotate():
    with open(annotations_file_path, "r") as file:
        annotations_data = json.load(file)
        target_file_name = target_img
        annotations = annotations_data[target_file_name]

        for i in annotations:
            [[xi, yi],[xf, yf]] = annotations[i]
            xi = math.floor(xi * scale_x)
            xf = math.floor(xf * scale_x)
            yi = math.floor(yi * scale_y)
            yf = math.floor(yf * scale_y)
            #draw each rect
            cv.rectangle(image, (xi,yi), (xf, yf), (0,0,255), 2)



annotate()
cv.imshow('Window Title', image)
cv.waitKey(0)
cv.destroyAllWindows()




###
import cv2
import numpy as np
import math

img = cv2.imread('coin.jpg')

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (7,7),1.2)
canny = cv2.Canny(blur, 50, 255)

# Detect circles using HoughCircles
# circles = cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)

# if circles is not None:
#     circles = np.uint16(np.around(circles))
#     for i in circles[0, :]:
#         # Draw the outer circle
#         cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)

circles = cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)

if circles is not None:
    circles = np.uint16(np.around(circles))
    
    # Find the circle with the highest confidence (strongest edge)
    best_circle = None
    best_param2 = 0
    
    for i in circles[0, :]:
        x, y, radius = i
        circle_canny = canny[y - radius:y + radius, x - radius:x + radius]
        
        # Calculate the confidence score for the circle based on edge intensity
        confidence = np.sum(circle_canny) / (circle_canny.shape[0] * circle_canny.shape[1])
        
        if confidence > best_param2:
            best_circle = i
            best_param2 = confidence
    
    # Draw the best circle
    if best_circle is not None:
        x, y, radius = best_circle
        diameter = 2 * radius
        print(f"Circle diameter: {diameter}")
        cv2.circle(img, (x, y), radius, (0, 255, 0), 2)
        cv2.circle(img, (x, y), 2, (0, 0, 255), 3)

    # Size of the image
    height, width, channels = img.shape
    print(f"Image size: {width}x{height}")

    #ratio of circle area to img area
    circle_area = math.pi * radius**2
    image_area = img.shape[0] * img.shape[1]
    ratio = circle_area / image_area
    print(f"Ratio of circle area to image area: {ratio:.6f}")

cv2.imshow('image', img)
# cv2.imshow('gray', gray)
# cv2.imshow('blur', blur)
cv2.imshow('canny', canny)
cv2.waitKey(0)
cv2.destroyAllWindows()
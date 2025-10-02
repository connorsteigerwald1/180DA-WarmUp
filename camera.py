"""
Object Tracker with HSV Thresholding and Bounding Box

Based on:
  - “Contour Features” tutorial, OpenCV-Python Tutorials — using contours, boundingRect, area filtering
  - “Changing Colorspaces” tutorial, OpenCV-Python Tutorials — using cvtColor, HSV thresholding, inRange

Improvements / modifications in this version:
  1. Added morphological opening/closing (kernel) to clean up noise before contour detection.  
  2. Enforced a minimum area threshold (`MIN_AREA`) to ignore small spurious contours.  
  3. Blurs the frame (Gaussian) before converting to HSV to reduce high-frequency noise.  

"""


import cv2
import numpy as np

cap = cv2.VideoCapture(0)

#black color
#lower_hsv = np.array([0,   0,   0], dtype=np.uint8)
#upper_hsv = np.array([180, 255, 50], dtype=np.uint8)

#green color
lower_hsv = np.array([40, 70, 70], dtype=np.uint8)
upper_hsv = np.array([80, 255, 255], dtype=np.uint8)

MIN_AREA = 1000  # ignore tiny specks
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # Clean noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area > MIN_AREA:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Area: {int(area)}", (x, max(0, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Mask", mask)
    cv2.imshow("Tracked Object", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

"""A script to capture webcam feed, identify the dominant color in a central rectangle using K-Means, and display it."""

import cv2
import numpy as np
from sklearn.cluster import KMeans

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # Define a central rectangle
    rect_w, rect_h = w // 4, h // 4
    x1, y1 = w//2 - rect_w//2, h//2 - rect_h//2
    x2, y2 = x1 + rect_w, y1 + rect_h

    # Extract the region of interest
    roi = frame[y1:y2, x1:x2]

    roi_pixels = roi.reshape((-1, 3))

    # Use KMeans to find dominant color
    kmeans = KMeans(n_clusters=1, n_init=5, random_state=42)
    kmeans.fit(roi_pixels)
    dominant_color = kmeans.cluster_centers_[0].astype(int)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    color_box = np.zeros((100, 100, 3), dtype=np.uint8)
    color_box[:] = dominant_color

    cv2.imshow("Webcam Feed", frame)
    cv2.imshow("Dominant Color", color_box)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
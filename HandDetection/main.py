# STEP 1: Import the necessary modules.
import mediapipe as mp
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import logging

from drawHand import draw_landmarks_on_image

# STEP 2: Create an HandLandmarker object.
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# STEP 3: Load the input image.


image = mp.Image.create_from_file("hand_sum_2.jpeg")

# STEP 4: Detect hand landmarks from the input image.
detection_result = detector.detect(image)

# STEP 5: Process the classification result. In this case, visualize it.
annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
annotated_image_gray = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2GRAY)
ret,thresh1 = cv2.threshold(annotated_image_gray,127,255,cv2.THRESH_BINARY)
im2, contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.imshow('MediaPipe Hands', im2)

cv2.waitKey(0)
cv2.destroyAllWindows()
##cv2.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))




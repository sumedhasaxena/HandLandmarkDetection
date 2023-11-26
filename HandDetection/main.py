# STEP 1: Import the necessary modules.
import mediapipe as mp
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from drawHand import draw_landmarks_on_image, get_and_draw_contours

IMG_FILE = 'hand_prof_1.jpeg'
#IMG_FILE = 'testContour.png'
#IMG_FILE = 'hand_sum_1.jpeg'

# STEP 2: Create an HandLandmarker object.
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# STEP 3: Load the input image.
image = mp.Image.create_from_file(IMG_FILE)

# STEP 4: Detect hand landmarks from the input image.
detection_result = detector.detect(image)

# STEP 5: Process the classification result. In this case, visualize it.
#annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
#annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

annotated_image = get_and_draw_contours(image.numpy_view())

cv2.imshow('Hand Landmarks', annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()




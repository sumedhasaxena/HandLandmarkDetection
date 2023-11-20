# @markdown We implemented some functions to visualize the hand landmark detection results. <br/> Run the following cell to activate the functions.
import math

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green
COORDINATES_TEXT_COLOR = (0, 0, 0)

landmark_coordinate_list = {}
def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(max(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv2.putText(annotated_image, f"{handedness[0].category_name}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

        # draw coordinates on image
        for i in range(len(hand_landmarks)):
            x_coordinate_text = int(hand_landmarks[i].x * width)
            y_coordinate_text = int(hand_landmarks[i].y * height)
            landmark_coordinate_list[i] = {"x" : x_coordinate_text,"y": y_coordinate_text}
            cv2.putText(annotated_image,
                        f"{x_coordinate_text},{y_coordinate_text}",
                        (x_coordinate_text, y_coordinate_text),
                        cv2.FONT_HERSHEY_DUPLEX,
                        .3, COORDINATES_TEXT_COLOR,
                        1, cv2.LINE_AA)

        #print(landmark_coordinate_list)

        distance_0_1 = math.hypot(landmark_coordinate_list[1]['x'] - landmark_coordinate_list[0]['x'],
                                  landmark_coordinate_list[1]['y'] - landmark_coordinate_list[0]['y'])
        print("dd 0-1 :" + str(distance_0_1))
        distance_1_2 = math.hypot(landmark_coordinate_list[2]['x'] - landmark_coordinate_list[1]['x'],
                                  landmark_coordinate_list[2]['y'] - landmark_coordinate_list[1]['y'])
        print("dd 1-2 :" + str(distance_1_2))

    return annotated_image

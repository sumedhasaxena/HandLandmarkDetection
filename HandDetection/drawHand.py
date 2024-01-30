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
DISTANCE_TEXT_COLOR = (255, 255, 255)
COORDINATES_TEXT_COLOR = (0, 0, 0)
font = cv2.FONT_HERSHEY_COMPLEX
landmark_coordinate_list = {}
scaling_factor = 1
A4_WIDTH_CMS = 21
A4_HEIGHT_CMS = 29.7


def get_and_draw_contours(image):

    annotated_image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    annotated_image_gray = cv2.GaussianBlur(annotated_image_gray, (5, 5), 0)
    _, threshold = cv2.threshold(annotated_image_gray, 190, 255, cv2.THRESH_BINARY)
    threshold = cv2.erode(threshold, None, iterations=3)
    threshold = cv2.dilate(threshold, None, iterations=3)

    # Detecting contours in image.
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    maxc = max(contours, key=cv2.contourArea)

    approx = cv2.approxPolyDP(maxc, 0.050 * cv2.arcLength(maxc, True), True)

    top1_x = approx[0][0][0]
    top1_y = approx[0][0][1]
    top2_x = approx[1][0][0]
    top2_y = approx[1][0][1]

    distance_pixel = math.hypot((top1_x - top2_x), (top1_y - top2_y))

    global scaling_factor

    if abs(top1_x - top2_x) > abs(top1_y - top2_y):
        scaling_factor = A4_WIDTH_CMS / distance_pixel
    else:
        scaling_factor = A4_HEIGHT_CMS / distance_pixel

    # find coordinates for top left and right of A4 sheet

    cv2.drawContours(image, [approx], 0, (0, 0, 255), 1)
    n = approx.ravel()
    i = 0
    for j in n:
        if (i % 2 == 0):
            x = n[i]
            y = n[i + 1]
            string = str(x) + " " + str(y)
            cv2.putText(image, string, (x, y), font, .5, (255, 255, 0))
        i = i + 1

    return image


def get_and_print_distance_between_landmarks(annotated_image, landmark1_index, landmark2_index):
    distance_1_2 = math.hypot(
        landmark_coordinate_list[landmark2_index]['x'] - landmark_coordinate_list[landmark1_index]['x'],
        landmark_coordinate_list[landmark2_index]['y'] - landmark_coordinate_list[landmark1_index]['y'])

    distance_in_cm = round(distance_1_2 * scaling_factor, 2)

    distance_1_2_text_x, distance_1_2_text_y = get_dist_text_coordinates(landmark_coordinate_list[landmark1_index]['x'],
                                                                         landmark_coordinate_list[landmark1_index]['y'],
                                                                         landmark_coordinate_list[landmark2_index]['x'],
                                                                         landmark_coordinate_list[landmark2_index]['y'])

    cv2.putText(annotated_image,
                f"{int(distance_1_2)}px, {distance_in_cm}cm",
                (distance_1_2_text_x, distance_1_2_text_y),
                cv2.FONT_HERSHEY_DUPLEX,
                .5, DISTANCE_TEXT_COLOR,
                1, cv2.LINE_AA)


def get_dist_text_coordinates(x1, y1, x2, y2):
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


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

        # https://mediapipe.readthedocs.io/en/latest/solutions/hands.html#multi-handedness : hack to flip handedness if the back of hand is photographed
        # Draw handedness (left or right hand) on the image.
        if handedness[0].category_name == "Right":
            handedness = "Left"
        else:
            handedness = "Right"
        cv2.putText(annotated_image, f"{handedness}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

        # draw coordinates on image
        for i in range(len(hand_landmarks)):
            x_coordinate_text = int(hand_landmarks[i].x * width)
            y_coordinate_text = int(hand_landmarks[i].y * height)
            landmark_coordinate_list[i] = {"x": x_coordinate_text, "y": y_coordinate_text}
            # cv2.putText(annotated_image,
            #           f"  ({x_coordinate_text},{y_coordinate_text})",
            #          (x_coordinate_text, y_coordinate_text),
            #         cv2.FONT_HERSHEY_DUPLEX,
            #        .4, COORDINATES_TEXT_COLOR,
            #       1, cv2.LINE_AA)

        print(landmark_coordinate_list)

        # mark the pixel distance between landmarks on image
        get_and_print_distance_between_landmarks(annotated_image, 0, 1)
        get_and_print_distance_between_landmarks(annotated_image, 0, 5)
        get_and_print_distance_between_landmarks(annotated_image, 0, 17)
        get_and_print_distance_between_landmarks(annotated_image, 1, 2)
        get_and_print_distance_between_landmarks(annotated_image, 2, 3)
        get_and_print_distance_between_landmarks(annotated_image, 3, 4)
        get_and_print_distance_between_landmarks(annotated_image, 5, 6)
        get_and_print_distance_between_landmarks(annotated_image, 6, 7)
        get_and_print_distance_between_landmarks(annotated_image, 7, 8)
        get_and_print_distance_between_landmarks(annotated_image, 5, 9)
        get_and_print_distance_between_landmarks(annotated_image, 9, 10)
        get_and_print_distance_between_landmarks(annotated_image, 10, 11)
        get_and_print_distance_between_landmarks(annotated_image, 11, 12)
        get_and_print_distance_between_landmarks(annotated_image, 9, 13)
        get_and_print_distance_between_landmarks(annotated_image, 13, 14)
        get_and_print_distance_between_landmarks(annotated_image, 14, 15)
        get_and_print_distance_between_landmarks(annotated_image, 15, 16)
        get_and_print_distance_between_landmarks(annotated_image, 13, 17)
        get_and_print_distance_between_landmarks(annotated_image, 17, 18)
        get_and_print_distance_between_landmarks(annotated_image, 18, 19)
        get_and_print_distance_between_landmarks(annotated_image, 19, 20)

    return annotated_image

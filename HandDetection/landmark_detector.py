import mediapipe as mp
from PIL import Image

from image_annotator import (draw_handedness_on_image,
                             draw_result_data_on_image,
                             get_and_set_scaling_factor)
from image_helper import resize_with_aspect_ratio, find_contours
from mediapipe_helper import (init_mediapipe_settings,
                              draw_landmarks_on_image)

draw_contour_lines = False
draw_contour_coordinates = False


class LandmarksDetector:

    def detect_and_draw_landmarks(self, image_path):
        # Load the input image.
        image = mp.Image.create_from_file(image_path)

        # Initialize mediapipe
        detector = init_mediapipe_settings()

        # Detect hand landmarks from the input image using mediapipe library.
        detection_result = detector.detect(image)

        # Find/Draw contours of the A4 sheet in image
        approxed_contours, contoured_image = find_contours(image.numpy_view(), draw_contour_lines,
                                                           draw_contour_coordinates)

        # Get/Set the scaling factor from the recognized A4 corner coordinates
        get_and_set_scaling_factor(approxed_contours)

        # visualize the classification result
        annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
        annotated_image = draw_handedness_on_image(annotated_image, detection_result)
        annotated_image = draw_result_data_on_image(annotated_image, detection_result)
        annotated_image = resize_with_aspect_ratio(annotated_image)

        img = Image.fromarray(annotated_image)
        return img

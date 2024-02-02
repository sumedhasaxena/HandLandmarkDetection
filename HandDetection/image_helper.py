import cv2 as cv2

contour_coordinates_font = cv2.FONT_HERSHEY_COMPLEX
contour_coordinates_font_size = 0.5

A4_WIDTH_CMS = 21
A4_HEIGHT_CMS = 29.7


def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(image, dim, interpolation=inter)


def find_contours(image, draw_contour_lines, draw_contour_coordinates):
    annotated_image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    annotated_image_gray = cv2.GaussianBlur(annotated_image_gray, (5, 5), 0)
    _, threshold = cv2.threshold(annotated_image_gray, 190, 255, cv2.THRESH_BINARY)
    threshold = cv2.erode(threshold, None, iterations=3)
    threshold = cv2.dilate(threshold, None, iterations=3)

    # Detecting contours in image.
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    biggest_contour = max(contours, key=cv2.contourArea)

    approxed_contours = cv2.approxPolyDP(biggest_contour, 0.01 * cv2.arcLength(biggest_contour, True), True)

    if draw_contour_lines:
        cv2.drawContours(image, [approxed_contours], 0, (0, 0, 255), 1)

    if draw_contour_coordinates:
        n = approxed_contours.ravel()
        i = 0
        for j in n:
            if (i % 2 == 0):
                x = n[i]
                y = n[i + 1]
                string = str(x) + " " + str(y)
                cv2.putText(image, string, (x, y), contour_coordinates_font, contour_coordinates_font_size, (255, 255, 0))
            i = i + 1
    return approxed_contours, image

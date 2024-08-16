import cv2
import numpy as np
# from tensorflow.keras.models import load_model
from src import const

# sign_detection_model = load_model("best_sign_detection_model.h5")
car_mask = np.load("./car_mask.npy")


def get_masks(img):
    white_color_lower, white_color_upper = [240, 240, 240], [255, 255, 255]
    white_color_mask = cv2.inRange(
        img, np.array(white_color_lower), np.array(white_color_upper)
    ) * (1 - car_mask)

    side_mask = cv2.inRange(img, np.array([130, 0, 108]), np.array([160, 160, 200])) * (
        1 - car_mask
    )

    return {"white_color": white_color_mask, "side": side_mask}


def get_lines(image):
    rho = 1
    angle = np.pi / 180
    min_threshold = 10
    lines = cv2.HoughLinesP(
        image, rho, angle, min_threshold, np.array([]), minLineLength=8, maxLineGap=4
    )

    return lines if lines is not None else np.array([])


def get_next_x_reference(frame, lines):
    # two_lines_mask = np.zeros_like(frame)
    two_lines_mask = frame.copy()
    try:
        left_line_x, left_line_y = [], []
        right_line_x, right_line_y = [], []

        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) > const.VERTICAL_LINE_MIN_SLOPE:
                    if slope <= 0:
                        left_line_x.extend([x1, x2])
                        left_line_y.extend([y1, y2])
                    else:
                        right_line_x.extend([x1, x2])
                        right_line_y.extend([y1, y2])

        min_y = int(frame.shape[0] * (3 / 5))  # below the horizon
        max_y = int(frame.shape[0])  # bottom of the image

        poly_left = np.poly1d(np.polyfit(left_line_y, left_line_x, deg=1))  # left line
        poly_right = np.poly1d(
            np.polyfit(right_line_y, right_line_x, deg=1)
        )  # right line

        right_x_start = int(poly_right(max_y))
        right_x_end = int(poly_right(min_y))
        left_x_start = int(poly_left(max_y))
        left_x_end = int(poly_left(min_y))

        # drawing the 2 lines -> for debuggin
        cv2.line(
            two_lines_mask, (left_x_start, max_y), (left_x_end, min_y), [255, 255, 0], 5
        )
        cv2.line(
            two_lines_mask,
            (right_x_start, max_y),
            (right_x_end, min_y),
            [255, 255, 0],
            5,
        )

        next_x_ref = (left_x_end + right_x_end) / 2
    except:
        # if the algorithm failed -> the next x reference will be middle of the input  image
        next_x_ref = frame.shape[1] / 2
    return two_lines_mask, next_x_ref


def get_vertical_lines_roi(frame):  # ? more on this
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_gauss_gray = cv2.GaussianBlur(img_gray, (3, 3), 3)

    mask1 = np.zeros_like(img_gray, dtype=np.uint8)

    # assuming the input image is 256 by 256
    pts = np.array([[(76, 120), (180, 120), (291, 256), (-35, 256)]], dtype=np.int32)

    cv2.fillPoly(mask1, pts, 255)
    img_roi = cv2.bitwise_and(img_gauss_gray, mask1)

    img_edges = cv2.Canny(img_roi, 100, 230)

    # mask2 is for removing the detected edges around the ROI
    mask2 = np.zeros_like(img_edges, dtype=np.uint8)
    pts = np.array([[(76, 125), (180, 125), (286, 256), (-30, 256)]], dtype=np.int32)

    cv2.fillPoly(mask2, pts, 255)

    img_edges_roi = cv2.bitwise_and(img_edges, mask2)

    return img_edges, img_edges_roi


def is_horiz_lines(frame):
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_gauss_gray = cv2.GaussianBlur(img_gray, (3, 3), 3)
    img_gauss_gray_edges = cv2.Canny(img_gauss_gray, 100, 230)

    roi = img_gauss_gray_edges[155:180, 96:160]
    SMOOTHING = 1e-5  # to avoid division by zero
    try:
        lines = get_lines(roi)
        lines = lines.reshape(-1, 2, 2)
        lines_slopes = (lines[:, 1, 1] - lines[:, 0, 1]) / (
            lines[:, 1, 0] - lines[:, 0, 0] + SMOOTHING
        )

        horiz_lines = lines[np.where(abs(lines_slopes) < const.HORIZ_LINE_MAX_SLOPE)]
        if len(horiz_lines) != 0:
            horiz_line_detected = True
        else:
            horiz_line_detected = False
    except:
        horiz_line_detected = False

    return horiz_line_detected


def get_turn_x(mask):
    roi = mask[100:190, :]
    lines = get_lines(roi)
    lines = lines.reshape(-1, 2, 2)
    lines_slopes = (lines[:, 1, 1] - lines[:, 0, 1]) / (lines[:, 1, 0] - lines[:, 0, 0])

    horiz_lines = lines[np.where(abs(lines_slopes) < const.HORIZ_LINE_MAX_SLOPE)]
    lines_x_mean = np.mean(horiz_lines[:, :, 0])

    return lines_x_mean


def get_side_x(side_mask):
    HEIGHT = side_mask.shape[0]
    side_roi = side_mask[150:HEIGHT, :]

    white_pixels_yx = np.where(side_roi > 0)
    white_pixels_x = white_pixels_yx[1]

    side_x = np.mean(white_pixels_x)
    return side_x


def get_sign_state(frame, hsv_frame):
    signs_classes = ["left", "straight", "right"]
    mask = cv2.inRange(hsv_frame, np.array([100, 160, 90]), np.array([160, 220, 220]))
    mask[:30, :] = 0
    try:
        points, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        sorted_points = sorted(points, key=len)

        if cv2.contourArea(sorted_points[-1]) > 30:
            x, y, w, h = cv2.boundingRect(sorted_points[-1])
            if (x > 5) and (x + w < 251) and (y > 5) and (y + h < 251):  # ?
                sign = frame[y : y + h, x : x + w]
                sign = cv2.resize(sign, (25, 25)) / 255  # resize + normalize

                frame = cv2.rectangle(
                    frame, (x, y), (x + w, y + h), (0, 255, 255), 2
                )  # extra for showing the detected sign on the frame

                preds = sign_detection_model.predict(sign.reshape(1, 25, 25, 3))
                next_sign = signs_classes[np.argmax(preds)]
                return mask, next_sign
            else:
                return mask, "nothing"
        else:
            return mask, "nothing"
    except:
        return mask, "nothing"


def is_stop_sign(frame_hsv):
    red_color_mask = cv2.inRange(
        frame_hsv, np.array([140, 70, 0]), np.array([255, 255, 255])
    ) * (1 - car_mask)

    contours_points, _ = cv2.findContours(
        red_color_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours_points:
        length_sorted_points = sorted(contours_points, key=len)

        red_area = cv2.contourArea(length_sorted_points[-1])
        if red_area > 60:
            return True
        else:
            return False
    else:
        return False

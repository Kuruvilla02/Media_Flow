import csv
import copy
import argparse

import cv2 as cv
import numpy as np
import mediapipe as mp

import pyautogui
from utils import CvFpsCalc
from model import KeyPointClassifier

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args


def main():
    # Argument parsing #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # Camera preparation ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model load #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()

    #Declaring states
    prevhandstate = 0
    curhandstate = 0


    # Read labels ###########################################################
    with open('model/keypoint_classifier/keypoint_classifier_label3.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]


    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    mode = 0

    while True:
        fps = cvFpsCalc.get()

        # Process Key (ESC: end) #################################################
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break

        # Camera capture #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        #  ####################################################################

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                # Bounding box calculation
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Hand sign classification
                hand_sign_id = keypoint_classifier(landmark_list)
                prevhandstate = curhandstate
                curhandstate = hand_sign_id
                #print(hand_sign_id)
                # if(prevhandstate == 5 ) and (curhandstate == 3 or curhandstate == 6 or curhandstate == 8):
                #     print("Played")
                #     pyautogui.press('space')
                # if(curhandstate == 10) and (prevhandstate == 10):
                #     print("Vol Up")
                #     pyautogui.press('up')
                # if (curhandstate == 9) and (prevhandstate == 9):
                #     print("Vol Down")
                #     pyautogui.press('down')
                if (prevhandstate == 1) and (curhandstate == 2):
                    print("Played")
                    # pyautogui.press('up')
                if (prevhandstate == 5) and (curhandstate == 6):
                    print("Right")
                    # pyautogui.press('up')
                if (prevhandstate == 6) and (curhandstate == 5):
                    print("Left")
                if (prevhandstate == 7) and (curhandstate == 8):
                    print("Down")
                if (prevhandstate == 8) and (curhandstate == 7):
                    print("Up")
                debug_image = draw_info_text(
                    debug_image,
                    keypoint_classifier_labels[hand_sign_id],
                )
        else:
            debug_image = draw_info_text(
                debug_image,
                "Nothing",
            )

        debug_image = draw_info(debug_image, fps)

        # Screen reflection #############################################################
        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_point.append(landmark.x)
        landmark_point.append(landmark.y)
        landmark_point.append(landmark.z)
    return landmark_point


def draw_info_text(image, finger_gesture_text):

    if finger_gesture_text != "":
        # cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 500),
        #            cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 5), 4, cv.LINE_AA)
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (16, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 235), 2,
                   cv.LINE_AA)

    return image


def draw_info(image, fps):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    return image


if __name__ == '__main__':
    main()
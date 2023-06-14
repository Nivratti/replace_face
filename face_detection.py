import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from loguru import logger
import math
from typing import Dict, Tuple
from nb_utils.error_handling import trace_error

import json

import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

# init detector
face_detector = FaceAnalysis(
    name="buffalo_s", # default name="buffalo_l",
    allowed_modules=['detection'], providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)
face_detector.prepare(ctx_id=0, det_size=(640, 640))


def detect_faces(img):
    """
    Detect faces in image and returns box and kps i.e 5 points landmarks for each detected face

    Args:
        img (np.ndarray): Np array in RGB color

    Returns:
        list: list of faces
    """
    return face_detector.get(img)

def detect_crop_face(img, faces=[], image_color_space="BGR"):
    """
    Crop highest detected confidence score face, if it's non-blue
    """
    if not faces:
        faces = face_detector.get(img)

    if faces:
        total_faces = len(faces)
        for idx, face in enumerate(faces):
            bbox = face["bbox"]
            key_points = face["kps"]
            det_score = face["det_score"]

            # print(f"face det_score: {det_score}")
            # import pdb;pdb.set_trace()
            
            ## Filter out low confidence faces -- to avoid watermarked faces coming as a main face
            ## To avoid Gandhi's face coming as main person face when user photo not available in PAN ID image
            ## To avoid detection of watermarked blue face as a main face of person in passport, if main photo blurred, or tempered
            if det_score < 0.80:
                print(f"Low face detection score: {det_score} .. Skipping face")
                continue
            
            if image_color_space == "BGR":
                # converting BGR to RGB
                image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = img

            image_pil = Image.fromarray(image_rgb)

            ## crop face area and save first face image
            x1, y1 = int(bbox[0]), int(bbox[1])
            x2, y2 = int(bbox[2]), int(bbox[3])
        
            crop_face_area = image_rgb[y1:y2, x1:x2]
            return (faces, crop_face_area)
    return (None, None)

def find_get_top_face(img):
    """
    Finds and returns the top face detected in the given image using the RetinaFace algorithm.

    Args:
        image: The input image.

    Returns:
        If a face is detected, the function returns a dictionary containing information about the top face.
        If no face is detected, it returns False.

    Raises:
        Exception: If there is an error while finding the face or processing the image.

    """
    faces = face_detector.get(img)
    if faces:
        try:
            return faces[0]
        except Exception as e:
            err_msg = trace_error()
            logger.error(f'Face area finding failed.. {err_msg}')
            return None
    else:
        logger.warning('No face detected!')
    return False

def get_face_coordinates(face: Dict, format="xy-wh"):
    """
    Retrieves the coordinates of a face from the given face dictionary.

    Args:
        face: A dictionary containing information about a face, typically obtained from a face detection algorithm.
        format: The format of the output coordinates. Valid values are "xy-wh" (default) and "x1y1-x2y2".

    Returns:
        If the format is "xy-wh", the function returns a tuple (x, y, w, h) representing the top-left coordinates (x, y)
        of the bounding box and its width (w) and height (h).
        If the format is "x1y1-x2y2", the function returns a tuple (x1, y1, x2, y2) representing the top-left (x1, y1)
        and bottom-right (x2, y2) coordinates of the bounding box.

    """
    bbox = face.get('bbox')
    x1, y1, x2, y2 = bbox

    if format == "x1y1-x2y2":
        return (x1, y1, x2, y2)
    else:
        # xywh format
        w = int(abs(x2 - x1))
        h = int(abs(y2 - y1))

        x = int(x1)
        y = int(y1)

        return (x, y, w, h)

def draw_detected_face(image, face=None):
    """
    Draws a rectangle around the detected face in the given image.

    Args:
        image: The input image.

    Returns:
        None. The function modifies the image in-place and displays it.

    """
    if not face:
        face = find_get_top_face(image)

    image_cpy = image.copy()
    # face box visualize
    x, y, w, h = get_face_coordinates(face)
    cv2.rectangle(image_cpy, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # landmarks visualize
    visualized_image_pil = Image.fromarray(image_cpy)
    return visualized_image_pil

def calculate_face_orientation(landmarks: Dict[str, Tuple[int, int]]) -> Tuple[float, str]:
    """
    Calculates the orientation of a face based on the given landmarks.
    
    Args:
        landmarks (Dict[str, Tuple[int, int]]): A dictionary containing landmark positions.
            Expected keys: "left_eye", "right_eye", "nose", "mouth_left", "mouth_right".
    
    Returns:
        Tuple[float, str]: A tuple containing the calculated angle and determined orientation.
            The angle is in degrees and represents the rotation of the face.
            The orientation can be one of the following values: "Upside-Down", "Rotated-Left",
            "Rotated-Right", or "Upright".
    """
    logger.debug(f"landmarks: {landmarks}")
    # from IPython import embed; embed()

    if isinstance(landmarks, (list, np.ndarray)):
        if landmarks[1][0] - landmarks[0][0] > landmarks[4][0] - landmarks[3][0]:
            left_eye = landmarks[0]
            right_eye = landmarks[1]
            nose = landmarks[2]
            mouth_left = landmarks[3]
            mouth_right = landmarks[4]
        else:
            # upside face
            left_eye = landmarks[4]
            right_eye = landmarks[3]
            nose = landmarks[2]
            mouth_left = landmarks[1]
            mouth_right = landmarks[0]
    else:
        left_eye = landmarks["left_eye"]
        right_eye = landmarks["right_eye"]
        nose = landmarks["nose"]
        mouth_left = landmarks["mouth_left"]
        mouth_right = landmarks["mouth_right"]

    # Calculate the angle between the eyes
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = math.degrees(math.atan2(dy, dx))

    # Determine the face orientation based on the position of the mouth
    if mouth_left[1] > nose[1] and mouth_right[1] > nose[1]:
        orientation = "Upside-Down"
    elif mouth_left[0] < nose[0] and mouth_right[0] < nose[0]:
        orientation = "Rotated-Left"
    elif mouth_left[0] > nose[0] and mouth_right[0] > nose[0]:
        orientation = "Rotated-Right"
    else:
        orientation = "Upside-Right"

    # from IPython import embed; embed()
    return angle, orientation

def get_face_det_res_json_filepath(img_filepath):
    """
    Add postfix to file stem and change extension to json and return the 
    resulting filepath.

    Args:
        img_filepath (str): The filepath of the input image.

    Returns:
        str: The filepath of the JSON output file.

    """
    p = Path(img_filepath)
    json_out_filepath = p.with_name(f"{p.stem}_insight_face_buffalo_s.json")
    return json_out_filepath


class NumpyArrayEncoder(json.JSONEncoder):
    """
    JSON encoder class for encoding NumPy arrays.

    This encoder extends the default JSONEncoder class to handle encoding of NumPy arrays.
    It converts NumPy integers to Python integers, NumPy floats to Python floats,
    and NumPy arrays to Python lists.

    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyArrayEncoder, self).default(obj)

def store_face_det_result(json_out_filepath, result):
    """
    Store the insight face detection result list as a JSON file.

    Args:
        json_out_filepath (str): The filepath to store the JSON file.
        faces (list): The list containing the all faces information.

    Returns:
        bool: True if the JSON file was successfully stored, False otherwise.

    """
    with open(json_out_filepath, "w") as outfile:
        json.dump(result, outfile, indent=4, cls=NumpyArrayEncoder)
    return True

def read_face_det_json_annotation_file(json_annotation_filepath):
    """
    Read and load the face detection results from a JSON annotation file.

    Args:
        json_annotation_filepath (str): The filepath of the JSON annotation file.

    Returns:
        list: A list containing the all face detection results.

    """
    face_detection_result = {}
    with open(json_annotation_filepath, "r") as f:
        face_detection_result = json.load(f)
    return face_detection_result

def add_margin_to_rect(
        rect, 
        left_margin_percent, 
        right_margin_percent, 
        top_margin_percent, 
        bottom_margin_percent, 
        image_shape
    ):
    """
    Adds margins to the rectangle coordinates on all four sides in percentage and corrects the rectangle
    if it goes outside the specified image shape.

    Args:
        rect (tuple): A tuple containing the original rectangle coordinates (x, y, w, h).
        left_margin_percent (float): The margin percentage to be added to the left side.
        right_margin_percent (float): The margin percentage to be added to the right side.
        top_margin_percent (float): The margin percentage to be added to the top side.
        bottom_margin_percent (float): The margin percentage to be added to the bottom side.
        image_shape (tuple): A tuple containing the shape of the image (height, width).

    Returns:
        tuple: A new tuple containing the modified rectangle coordinates (x, y, w, h) with the added margins.

    """
    x, y, w, h = rect
    image_height, image_width, c = image_shape

    # Calculate the margins in pixels
    left_margin = int(w * left_margin_percent)
    right_margin = int(w * right_margin_percent)
    top_margin = int(h * top_margin_percent)
    bottom_margin = int(h * bottom_margin_percent)

    # Calculate the new coordinates with margins
    x_new = max(x - left_margin, 0)
    y_new = max(y - top_margin, 0)
    w_new = min(w + left_margin + right_margin, image_width - x_new)
    h_new = min(h + top_margin + bottom_margin, image_height - y_new)

    return x_new, y_new, w_new, h_new

def main():
    import time
    start_time = time.monotonic()

    img = ins_get_image('t1')
    faces = face_detector.get(img)
    print(f"faces: {faces}")

    # import ipdb;ipdb.set_trace()
    print(f'Elapsed seconds: {time.monotonic() - start_time}')

    rimg = face_detector.draw_on(img, faces)
    os.makedirs("./output", exist_ok=True)
    cv2.imwrite("./output/t1_output.jpg", rimg)

    ## Crop face and save
    _, crop_face_area = detect_crop_face(img, faces)
    face_pil = Image.fromarray(crop_face_area)
    out_cropped_face_file = f"./output/cropped_first_face.jpg"
    os.makedirs(os.path.dirname(out_cropped_face_file), exist_ok=True)
    face_pil.save(out_cropped_face_file)
    logger.debug(f"Face ROI result has saved in: {out_cropped_face_file}")

if __name__ == "__main__":
    main()
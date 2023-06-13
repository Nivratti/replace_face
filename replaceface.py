import os
import argparse
from pathlib import Path
import json
from json import JSONEncoder
import math
import random

import numpy as np
import cv2
from PIL import Image

from loguru import logger
from nb_utils.error_handling import trace_error

from typing import Dict, Tuple
from retinaface import RetinaFace


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyArrayEncoder, self).default(obj)

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

def find_get_top_face_retina_face(image):
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
    resp = RetinaFace.detect_faces(image)
    if resp:
        try:
            face_key = max(resp, key=lambda face_number: resp[face_number]['score'])
            face = resp[face_key]
            return face
        except Exception as e:
            err_msg = trace_error()
            logger.error(f'Face area finding failed.. {err_msg}')
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
    facial_area = face.get('facial_area')
    x1, y1, x2, y2 = facial_area

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
        face = find_get_top_face_retina_face(image)

    x, y, w, h = get_face_coordinates(face)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    Image.fromarray(image).show()

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
    logger.info(f"landmarks: {landmarks}")
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
        orientation = "Upright"

    # from IPython import embed; embed()
    return angle, orientation

def get_retina_face_json_result_filepath(img_filepath):
    """
    Add postfix to file stem and change extension to json and return
    """
    p = Path(img_filepath)
    json_out_filepath = p.with_name(f"{p.stem}_retinaface.json")
    return json_out_filepath

def store_retina_face_result(json_out_filepath, single_face_dict):
    with open(json_out_filepath, "w") as outfile:
        json.dump(single_face_dict, outfile, indent=4, cls=NumpyArrayEncoder)
    return True

def read_retina_face_json_annotation_file(json_annotation_filepath):
    face_detection_result = {}
    with open(json_annotation_filepath, "r") as f:
        face_detection_result = json.load(f)

    return face_detection_result

def replace_face(
        source_cropped_face_path, 
        target_image_path, 
        target_face_dict={},
        is_add_margin=True,
        is_random_margin=False,
        clone=False
    ):
    # read
    source_cropped_face = cv2.imread(str(source_cropped_face_path))
    target_image = cv2.imread(str(target_image_path))

    if target_image is None:
        logger.error(f"error reading target image..")
        return 
    
    if not target_face_dict:
        json_annotation_filepath = get_retina_face_json_result_filepath(target_image_path)
        if os.path.exists(json_annotation_filepath):
            target_face_dict = read_retina_face_json_annotation_file(json_annotation_filepath)
            logger.info(f"loaded face detection result from .json file")
        else:
            target_face_dict = find_get_top_face_retina_face(target_image)
            # save result as a json
            store_retina_face_result(json_annotation_filepath, target_face_dict)
            logger.info(f"Stored face detection result in .json")
                                 
    if target_face_dict:
        coordinates = get_face_coordinates(target_face_dict)

        # check orientation
        face_angle, orientation = calculate_face_orientation(target_face_dict["landmarks"])
        logger.debug(f"face_angle: {face_angle}")
        logger.debug(f"orientation: {orientation}")

        if orientation == "Upside-Right":
            source_cropped_face = cv2.rotate(source_cropped_face, cv2.ROTATE_180)
            orientation == "Upside-Right" # in case face angle condition true
        elif orientation == "Rotated-Right":
            # Using cv2.ROTATE_90_COUNTERCLOCKWISE
            # rotate by 270 degrees clockwise
            source_cropped_face = cv2.rotate(source_cropped_face, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif orientation == "Rotated-Left":
            source_cropped_face = cv2.rotate(source_cropped_face, cv2.ROTATE_90_CLOCKWISE)
        else:
            "Upside-Down"
            pass

        # margin
        if is_add_margin:
            if is_random_margin:
                left_margin_percent = random.uniform(0.2, 0.4)
                right_margin_percent = random.uniform(0.2, 0.4)
                top_margin_percent = random.uniform(0.1, 0.3)
                bottom_margin_percent = random.uniform(0.1, 0.3)
            else:
                if orientation in ["Rotated-Right", "Rotated-Left"]:
                    left_margin_percent = 0.15
                    right_margin_percent = 0.15
                    top_margin_percent = 0.7
                    bottom_margin_percent = 0.5
                elif orientation == "Upside-Right":
                    left_margin_percent = 0.6 
                    right_margin_percent = 0.6
                    top_margin_percent = 0.5 
                    bottom_margin_percent = 0.5
                else:
                    # Upside-Down
                    left_margin_percent = 0.4 
                    right_margin_percent = 0.4
                    top_margin_percent = 0.3 
                    bottom_margin_percent = 0.3
            
            coordinates = add_margin_to_rect(
                rect=coordinates, 
                left_margin_percent=left_margin_percent, 
                right_margin_percent=right_margin_percent, 
                top_margin_percent=top_margin_percent, 
                bottom_margin_percent=bottom_margin_percent, 
                image_shape=target_image.shape
            )

        # resize source face as per target face area
        x, y, w, h = coordinates
        resized_source_face = cv2.resize(source_cropped_face, (w, h))

        if clone:
            resized_source_face_mask = 250 * np.ones(resized_source_face.shape, resized_source_face.dtype) # white mask
            center = (x+w//2,y+h//2)
            # Clone seamlessly.
            output = cv2.seamlessClone(
                resized_source_face, target_image, resized_source_face_mask, 
                center, cv2.NORMAL_CLONE
            )
            return output
        
        target_image[y:y+h,x:x+w] = resized_source_face[:,:]

        return target_image
    else:
        return None
    
if __name__ == '__main__':

    BASE_DIR = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser()
    parser.add_argument('-s','--source',help='Path to source image',required=True)
    parser.add_argument('-t','--target',help='Path to target image',required=True)
    parser.add_argument('-o','--output',help='Path to output directory',default='./outputs')
    parser.add_argument('--clone',help='Whether to use seamless cloning or not', action='store_true')
    args = parser.parse_args()

    # Grab commands arguments
    target_path = Path(args.target)
    source_path = Path(args.source)

    output_dir_path = Path(args.output)
    os.makedirs(output_dir_path, exist_ok=True)

    use_clone = False
    if args.clone:
        use_clone = True

    # Face replacement
    replaced = replace_face(source_path, target_path, clone=use_clone)
    
    if np.any(replaced):
        # Save output in output_dir
        output_name = f'{source_path.stem}_by_{target_path.stem}{source_path.suffix}'
        output_path = output_dir_path / output_name
        cv2.imwrite(str(output_path), replaced)
        logger.info(f"Outfile stored in {str(output_path)}")
    
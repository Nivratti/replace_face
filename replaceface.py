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

try:
    from replace_face.face_detection import FaceDetector
except Exception as e:
    from face_detection import FaceDetector

## Init face detector
face_detector = FaceDetector(is_warmup=False)

def perform_face_replacement(
        source_image_path, 
        target_image_path, 
        target_face_dict={},
        is_detect_source_face=True,
        is_add_margin=True,
        is_random_margin=False,
        use_seamless_clone=True,
        is_paste_on_transparent_container=False,
    ):
    """
    Replace a detected top most face in a target image with a source face.

    Args:
        source_image_path (str): The filepath of the source face image. It may be aligned and some margin added. 
                                        So you can directly put in target image.

        target_image_path (str): The filepath of the target image.
        target_face_dict (dict, optional): The dictionary containing the face detection results of the target image.
            If not provided, the function will try to load the face detection results from a JSON file associated with the target image.
            If the JSON file doesn't exist, the function will perform face detection on the target image.
            Defaults to an empty dictionary.
        is_add_margin (bool, optional): Flag to add a margin around the replaced face. Defaults to True.
        is_random_margin (bool, optional): Flag to add a random margin. Defaults to False.
        clone (bool, optional): Flag to clone the source face seamlessly onto the target image using cv2.seamlessClone. Defaults to False.

    Returns:
        numpy.ndarray or None: The modified target image with the replaced face. Returns None if the face detection results are not available.

    """
    # read
    source_image_bgr = cv2.imread(str(source_image_path))
    source_image = cv2.cvtColor(source_image_bgr, cv2.COLOR_BGR2RGB)

    target_image_bgr = cv2.imread(str(target_image_path))
    target_image = cv2.cvtColor(target_image_bgr, cv2.COLOR_BGR2RGB)

    # source_face = np.array(Image.open(source_face_path))
    # target_image = np.array(Image.open(target_image_path))

    if source_image is None:
        logger.error(f"error reading source image..")
        return 
    
    if target_image is None:
        logger.error(f"error reading target image..")
        return 
    
    if is_detect_source_face:
        source_faces = face_detector.detect_faces_wrapper(str(source_image_path))
        if not source_faces:
            logger.debug(f"No face detected in source image skipping.. {source_image_path}")
            return None

    if not target_face_dict:
        faces = face_detector.detect_faces_wrapper(str(target_image_path))

    if faces:
        target_face_dict = faces[0]

    if target_face_dict:
        coordinates = face_detector.get_face_coordinates(target_face_dict)

        # # check orientation
        # face_angle, orientation = face_detector.calculate_face_orientation(target_face_dict["kps"])
        # logger.debug(f"face_angle: {face_angle}")
        # logger.debug(f"orientation: {orientation}")

        # if orientation == "Upside-Right":
        #     source_cropped_face = cv2.rotate(source_cropped_face, cv2.ROTATE_180)
        #     orientation == "Upside-Right" # in case face angle condition true
        # elif orientation == "Rotated-Right":
        #     source_cropped_face = cv2.rotate(source_cropped_face, cv2.ROTATE_90_CLOCKWISE)
        # elif orientation == "Rotated-Left":
        #     # Using cv2.ROTATE_90_COUNTERCLOCKWISE
        #     # rotate by 270 degrees clockwise
        #     source_cropped_face = cv2.rotate(source_cropped_face, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # else:
        #     "Upside-Down"
        #     pass

        # margin
        if is_add_margin:
            if is_random_margin:
                left_margin_percent = random.uniform(0.2, 0.4)
                right_margin_percent = random.uniform(0.2, 0.4)
                top_margin_percent = random.uniform(0.1, 0.3)
                bottom_margin_percent = random.uniform(0.1, 0.3)
            else:
                # if orientation in ["Rotated-Left"]:
                #     left_margin_percent = 0.3
                #     right_margin_percent = 0.2
                #     top_margin_percent = 0.4
                #     bottom_margin_percent = 0.4
                # elif orientation == "Rotated-Right":
                #     left_margin_percent = 0.2
                #     right_margin_percent = 0.2
                #     top_margin_percent = 0.4
                #     bottom_margin_percent = 0.4

                # elif orientation == "Upside-Right":
                #     left_margin_percent = 0.4
                #     right_margin_percent = 0.4
                #     top_margin_percent = 0.2
                #     bottom_margin_percent = 0.2
                # else:

                # Upside-Down
                left_margin_percent = 0.35 
                right_margin_percent = 0.35
                top_margin_percent = 0.4
                bottom_margin_percent = 0.4
            
            coordinates = face_detector.add_margin_to_rect(
                rect=coordinates, 
                left_margin_percent=left_margin_percent, 
                right_margin_percent=right_margin_percent, 
                top_margin_percent=top_margin_percent, 
                bottom_margin_percent=bottom_margin_percent, 
                image_shape=target_image.shape
            )
        
        if is_detect_source_face:
            source_coordinates = face_detector.get_face_coordinates(source_faces[0])
            if is_add_margin:
                source_coordinates = face_detector.add_margin_to_rect(
                    rect=source_coordinates, 
                    left_margin_percent=left_margin_percent, 
                    right_margin_percent=right_margin_percent, 
                    top_margin_percent=top_margin_percent, 
                    bottom_margin_percent=bottom_margin_percent, 
                    image_shape=source_image.shape
                )
            source_cropped_face = face_detector.crop_face(
                source_image, source_coordinates,
                image_color_space="RGB"
            )
        else:
            source_cropped_face = source_image # if already cropped face input given
            
        # import ipdb; ipdb.set_trace()
        
        # resize source face as per target face area
        x1, y1, x2, y2 = coordinates
        x, y, w, h = int(x1), int(y1), int((x2 -x1)), int(y2 - y1)
        resized_source_face = cv2.resize(source_cropped_face, (w, h))

        if use_seamless_clone:
            resized_source_face_mask = 250 * np.ones(resized_source_face.shape, resized_source_face.dtype) # white mask
            center = (x+w//2,y+h//2)
            # Clone seamlessly.
            output = cv2.seamlessClone(
                resized_source_face, target_image, resized_source_face_mask, 
                center, cv2.NORMAL_CLONE
            )
            return output
        
        if is_paste_on_transparent_container:
            height, width = target_image.shape[:2]

            # Create a transparent RGBA image
            transparent_image = np.zeros((height, width, 4), dtype=np.uint8)

            # Ensure resized_source_face has an alpha channel
            if resized_source_face.shape[2] == 3:
                resized_source_face = np.dstack((resized_source_face, np.where((resized_source_face == [255, 255, 255]).all(axis=-1), 0, 255)))

            # Replace the region in the target image
            transparent_image[y:y+h, x:x+w] = resized_source_face[:,:]
            return transparent_image
        else:
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
    parser.add_argument('--use_seamless_clone',help='Whether to use seamless cloning or not', action='store_true')
    args = parser.parse_args()

    # Grab commands arguments
    target_path = Path(args.target)
    source_path = Path(args.source)

    output_dir_path = Path(args.output)
    os.makedirs(output_dir_path, exist_ok=True)

    use_seamless_clone = False
    if args.use_seamless_clone:
        use_seamless_clone = True

    # Face replacement
    replaced = perform_face_replacement(source_path, target_path, clone=use_seamless_clone)
    
    if np.any(replaced):
        # Save output in output_dir
        output_name = f'{source_path.stem}_by_{target_path.stem}{source_path.suffix}'
        output_path = output_dir_path / output_name
        cv2.imwrite(str(output_path), replaced)
        logger.info(f"Outfile stored in {str(output_path)}")
    
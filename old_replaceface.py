#! usr/bin/env

import argparse
import math
import os
from pathlib import Path
from retinaface import RetinaFace
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import math

def calculate_face_orientation(landmarks):
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
    elif mouth_left[0] < nose[0] and mouth_right[0] > nose[0]:
        orientation = "Rotated-Left"
    elif mouth_left[0] > nose[0] and mouth_right[0] < nose[0]:
        orientation = "Rotated-Right"
    else:
        orientation = "Upright"

    return angle, orientation

def find_face_coordinates(image ,x_eps=0.38, y_eps=0.38):
    """
      Find and return coodinates of unique relevant face in image

      x_eps,y_eps: coefficients for face area expansion
    """

    resp = RetinaFace.detect_faces(image)
    try:
        face_key = max(resp , key=lambda face_number: resp[face_number]['score'])
        face = resp[face_key]
        facial_area = face.get('facial_area')

        x1,y1,x2,y2 = facial_area
        w = abs(x2 - x1)
        h = abs(y2 - y1)
        x1 -= int(x_eps*w)
        y1 -= int(y_eps*h)
        w = int(w*(1+2*x_eps))
        h = int(h*(1+2*y_eps))

        # clip coordinates
        img_h, img_w, c = image.shape

        x1 = max(x1, 0)
        y1 = max(y1 ,0)
        w = min(w, img_w)
        h = min(h, img_h)

        return (x1,y1,w,h)
    except AttributeError:
        print('Face area finding failed!')
      
    except TypeError:
        print('No face area detected!')

def draw_dectected_face(image):
    """
    """
    x,y,w,h = find_face_coordinates(image)
    cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    plt.imshow(image)

def align_face(image):
  """
    Detect unique relevant face and align it by rotating image
  """
  try:
    resp = RetinaFace.detect_faces(image)
    face_key = max(resp , key=lambda face_number: resp[face_number]['score'])
    face = resp[face_key]
    x1 , y1 = face['landmarks']['right_eye']
    x2 , y2 = face['landmarks']['left_eye']
    a = abs(y1 - y2)
    b = abs(x2 - x1)
    c = math.sqrt(a**2 + b**2)
    cos_alpha = (b**2 + c**2 - a**2) / (2*b*c)
    alpha = np.arccos(cos_alpha)
    alpha_degrees = np.degrees(alpha)
    pillow_img = Image.fromarray(image)
    aligned_img = np.array(pillow_img.rotate(alpha_degrees))

    return aligned_img
  except:
    print('Alignement failed')

def replace_face(source,target,*, align=True , clone=False):
    """
        Detect and replace unique relevant face in source by target image
        return source with replaced face
    """
    if align:
      target = align_face(target)

    coordinates = find_face_coordinates(source)
    if not coordinates:
        print('No face found on source image')
        return []
    x,y,w,h = coordinates
    resized_target = cv.resize(target,(w,h),interpolation = cv.INTER_AREA)

    if clone:
      resized_target_mask = 250 * np.ones(resized_target.shape, resized_target.dtype) # white mask
      center = (x+w//2,y+h//2)
      # Clone seamlessly.
      output = cv.seamlessClone(resized_target, source, resized_target_mask, center, cv.NORMAL_CLONE)
      return output
    
    # import ipdb; ipdb.set_trace()

    # assert(resized_target.shape[0] == h and resized_target.shape[1] == w )
    source[y:y+h,x:x+w] = resized_target[:,:]
    return source


if __name__ == '__main__':

    BASE_DIR = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--target',help='Path to target image',required=True)
    parser.add_argument('-s','--source',help='Path to source image',required=True)
    parser.add_argument('-o','--output',help='Path to output directory',default='outputs')
    parser.add_argument('--clone',help='Whether to use seamless cloning or not', action='store_true')
    args = parser.parse_args()
    # Grab commands arguments
    target_path = Path(args.target)
    if not target_path.is_absolute():
        target_path = BASE_DIR / args.target

    source_path = Path(args.source)
    if not source_path.is_absolute():    
        source_path = BASE_DIR / args.source

    if args.output:
        output_dir_path = Path(args.output)
        if not output_dir_path.is_absolute():
            output_dir_path = BASE_DIR / args.output
    os.makedirs(output_dir_path, exist_ok=True)

    use_clone = False
    if args.clone:
        use_clone = True
    # Face replacement
    target = cv.imread(str(target_path))
    source = cv.imread(str(source_path))
    
    replaced = replace_face(source,target,clone=use_clone)
    
    if len(replaced) != 0:
        # Save output in output_dir
        output_name = f'{source_path.stem}_by_{target_path.stem}{source_path.suffix}'
        output_path = output_dir_path / output_name
        cv.imwrite(str(output_path), replaced)
    
"""
Face and 5 points landmarks (keypoints) detection using insightface

================================================
Performance on T4 GPU with 12 cores:
-------------------------------------------
1) TensorRt -- 11 milliseconds
    Elapesed time: 0.011443780735135078 Second

2) GPU: -- 13 milliseconds
    0.013077417388558388

3) CPU:
    0.22224561870098114 Seconds i.e 222 milliseconds
"""
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
# from utilities.custom_json_encoder import NumpyArrayEncoder

import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from insightface.utils import face_align


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


class FaceDetector:
    def __init__(self, model_name="buffalo_l", is_warmup=True):
        """
        Initializes the FaceDetector with the specified model name.

        Args:
            model_name (str): The name of the face detection model.

        """
        self.model_name = model_name
        self.check_model_files()
        self.face_detector = FaceAnalysis(
            name=model_name,
            allowed_modules=['detection'],
            providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'],
        )
        self.face_detector.prepare(ctx_id=0, det_size=(640, 640))
        if is_warmup:
            self.warmup_model() # perform warmup after creating instance

    def check_model_files(self) -> bool:
        """
        Checks the existence of model files in the home directory and copies them if missing.

        This method checks whether the model files for the specified `model_name` exist in the home directory. If the model files
        are not found, it looks for them in the `resources/models/insightface/models/{self.model_name}` directory and copies the 
        entire folder to the home directory.

        Returns:
            bool: True if the model files are available or successfully copied, False otherwise.
        """
        from shutil import copytree
        dir_home = os.path.expanduser("~")
        dir_model_home = os.path.join(dir_home, ".insightface", "models", self.model_name)
        if not os.path.exists(dir_model_home):
            dir_model_resources = f"resources/models/insightface/models/{self.model_name}"
            if os.path.exists(dir_model_resources):
                # copy folder to home
                copytree(dir_model_resources, dir_model_home, dirs_exist_ok=True)
        return True
    
    def warmup_model(self, image_shape=(640, 640, 3), num_iterations=10):
        """
        Warm-up the face detection model by performing inference with dummy data.

        This function simulates the inference process on the face detection model using dummy data.
        The warm-up process initializes the model's weights and optimizes the execution paths,
        which can result in improved inference performance during real data inference.

        Args:
            image_shape (tuple, optional): The shape of the dummy input image to be used for warm-up.
                The default is (640, 640, 3), which corresponds to a height of 640, width of 640,
                and 3 channels (RGB).
            num_iterations (int, optional): The number of warm-up iterations to perform.
                Each iteration runs a batch of dummy data through the model.
                The default is 10.

        Returns:
            None

        Example:
            # Create an instance of the FaceDetector class
            face_detector = FaceDetector()

            # Perform warm-up with default settings (image_shape=(640, 640, 3), num_iterations=10)
            face_detector.warmup_model()

            # Alternatively, you can customize the warm-up settings
            custom_image_shape = (128, 128, 3)
            custom_num_iterations = 5
            face_detector.warmup_model(image_shape=custom_image_shape, num_iterations=custom_num_iterations)
        """
        dummy_data = np.random.randint(0, 256, size=image_shape, dtype=np.uint8)
        for _ in range(num_iterations):
            _ = self.detect_faces(dummy_data)

    def detect_faces(self, img, drop_score=0.80):
        """
        Detects faces in the image and returns a list of detected faces.

        Args:
            img (np.ndarray): Numpy array of the image in RGB color.

        Returns:
            list: A list of faces, each containing the bounding box and key points.

        """
        faces = self.face_detector.get(img)
        if faces and drop_score > 0:
            filtered_faces = []
            for idx, face in enumerate(faces):
                # bbox = face["bbox"]
                # key_points = face["kps"]
                det_score = face["det_score"]

                if det_score < drop_score:
                    # print(f"Low face detection score: {det_score} .. Skipping face")
                    continue
                else:
                    filtered_faces.append(face)
            return filtered_faces
        
        return faces

    def detect_faces_wrapper(self, image_path, drop_score=0.80):
        json_annotation_filepath = self.get_face_det_res_json_filepath(image_path)
        if os.path.exists(json_annotation_filepath):
            faces = self.read_face_det_json_annotation_file(json_annotation_filepath)
            logger.debug(f"loaded face detection result from .json file: {json_annotation_filepath}")
        else:
            img = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = self.detect_faces(img)
            # save result as a json
            self.store_face_det_result(json_annotation_filepath, faces)
            logger.debug(f"Stored face detection result in .json {json_annotation_filepath}")
        return faces
    
    def estimate_norm(self, landmark, image_size=112, mode='arcface'):
        """
        Estimate the normalization transformation matrix (M) to align facial landmarks.

        Parameters:
            landmark (numpy.ndarray): A 2D array of shape (5, 2) representing the facial
                                    landmarks. Each row contains the (x, y) coordinates
                                    of a facial landmark point.
            image_size (int, optional): The desired size of the output aligned image.
                                        Should be a multiple of 112 or 128. Default is 112.
            mode (str, optional): The normalization mode. Currently, only "arcface" is supported.
                                Default is "arcface".

        Returns:
            M (numpy.ndarray): A 2x3 transformation matrix used for face alignment.

        Notes:
            The function estimates the transformation matrix (M) required to align facial
            landmarks to a standard configuration. It is primarily used in face alignment
            and normalization tasks before face recognition or other facial analysis tasks.

            The 'landmark' parameter should be a 2D array of shape (5, 2) representing the
            coordinates of five facial landmarks: left eye, right eye, nose, left mouth corner,
            and right mouth corner.

            The 'image_size' parameter specifies the desired size of the output aligned image.
            It should be a multiple of 112 or 128, as these are common sizes used in deep learning
            models for face recognition.

            The 'mode' parameter defines the normalization mode. Currently, only "arcface" is
            supported, which applies alignment constraints suitable for ArcFace face recognition.
            In future versions, more normalization modes may be introduced.

            Example usage:
                landmark = faces[0]['kps']
                M = self.estimate_norm(landmark, image_size=112, mode="arcface")
        """
        M = face_align.estimate_norm(
            lmk=landmark, 
            image_size=image_size, 
            mode=mode
        )
        return M
        
    def get_alignment_angle(self, M):
        """
        Calculate the rotation angle and scales from the transformation matrix M.

        Parameters:
            M (numpy.ndarray): A 2x3 transformation matrix obtained from face alignment.

        Returns:
            rotation_deg (float): The rotation angle in degrees, representing the rotation
                                of the face with respect to the vertical axis after alignment.
            scale_x (float): The scaling factor along the x-axis applied during alignment.
            scale_y (float): The scaling factor along the y-axis applied during alignment.

        Notes:
            The function extracts the rotation and scaling components from the transformation
            matrix to provide insight into the face alignment process. The rotation angle is
            returned in degrees, while the scaling factors represent the changes along the x
            and y axes applied to align the face to a standard orientation.

            The rotation angle can be in the range [-180, 180] degrees. To get the angle in
            the range [0, 360], you can use the following code:
            if rotation_deg < 0:
                rotation_deg += 360
        """
        # Extract scale and rotation components from the transformation matrix
        scale_x = np.sqrt(M[0][0] * M[0][0] + M[0][1] * M[0][1])
        scale_y = np.sqrt(M[1][0] * M[1][0] + M[1][1] * M[1][1])
        rotation_rad = np.arctan2(M[1][0], M[0][0])

        # Convert rotation from radians to degrees
        rotation_deg = np.degrees(rotation_rad)

        return rotation_deg, scale_x, scale_y

    def normlize_crop_face(self, img, landmark, image_size=112, mode='arcface', borderValue=0.0):
        """
        Crop and align a face from the input image based on the provided facial landmarks.

        Parameters:
            img (numpy.ndarray): The input image from which the face will be cropped and aligned.
            landmark (numpy.ndarray): A 2D array of shape (5, 2) representing the facial landmarks.
                                      Each row contains the (x, y) coordinates of a facial landmark point.
            image_size (int, optional): The desired size of the output aligned face image. Should be
                                        a multiple of 112 or 128. Default is 112.
            mode (str, optional): The normalization mode. Currently, only "arcface" is supported.
                                  Default is "arcface".
            borderValue (float, optional): The value to be used for padding if the face goes beyond
                                           the image boundaries. Default is 0.0.

        Returns:
            warped (numpy.ndarray): The cropped and aligned face image.
            M (numpy.ndarray): A 2x3 transformation matrix used for face alignment.

        Notes:
            This function performs face alignment and cropping based on the provided facial landmarks.
            It estimates a transformation matrix (M) to align the landmarks to a standard configuration
            and then applies the transformation to crop and align the face in the input image.

            The 'landmark' parameter should be a 2D array of shape (5, 2) representing the coordinates
            of five facial landmarks: left eye, right eye, nose, left mouth corner, and right mouth corner.

            The 'image_size' parameter specifies the desired size of the output aligned face image.
            It should be a multiple of 112 or 128, as these are common sizes used in deep learning
            models for face recognition.

            The 'mode' parameter defines the normalization mode. Currently, only "arcface" is supported,
            which applies alignment constraints suitable for ArcFace face recognition. In future versions,
            more normalization modes may be introduced.

            The 'borderValue' parameter sets the value to be used for padding if the face goes beyond the
            image boundaries during cropping.
        """
        M = self.estimate_norm(landmark, image_size, mode)
        warped = cv2.warpAffine(
            img, M, 
            (image_size, image_size), 
            borderValue=borderValue
        )
        return warped, M
    
    def correct_image_orientation(self, img, rotation_deg, face_info_list=None):
        """
        Correct the image orientation based on the provided rotation angle.

        Parameters:
            img (numpy.ndarray): The input image to be corrected.
            rotation_deg (float): The rotation angle in degrees, in the range of -360 to +360.
            face_info_list (list, optional): A list of dictionaries containing face information.
                                            Each dictionary should have 'bbox', 'kps', and 'det_score' keys.

        Returns:
            numpy.ndarray: The corrected image after applying the rotation and filling extra region with black.
            list, optional: A list of corrected face information dictionaries, each containing the updated 'bbox' and 'kps'.

        Notes:
            This function corrects the orientation of the input image based on the provided
            rotation angle. The rotation angle should be in degrees and can be in the range
            of -360 to +360.

            The input 'img' should be a NumPy array representing the image.

            The 'rotation_deg' parameter represents the rotation angle applied to the image
            in degrees. Positive values indicate counter-clockwise rotation, and negative
            values indicate clockwise rotation.

            The function applies the rotation to the image and fills the extra region with
            black color to keep the whole image in view. The resulting image may have blank
            regions due to rotation, and the image size may change.

            The 'face_info_list' parameter is an optional list of dictionaries, each containing face
            information for a single face in the image. Each dictionary must have the following keys:
            - 'bbox': A list containing four elements [x1, y1, x2, y2] representing the bounding box coordinates.
            - 'kps': A list of facial landmark points as a 2D array of shape (n_points, 2).
                    Each row contains the (x, y) coordinates of a facial landmark point.
            - 'det_score': The detection score for the face.

            The function applies the rotation to the facial landmark points and the bounding boxes
            to keep them aligned with the rotated image.

            Example usage:
                corrected_img = correct_image_orientation(img, rotation_deg)
                corrected_img, corrected_faces = correct_image_orientation(img, rotation_deg, face_info_list)
        """
        if rotation_deg != 0:
            # Get the image center
            height, width = img.shape[:2]
            center = (width // 2, height // 2)

            # Calculate the new image size after rotation
            rotation_rad = np.radians(rotation_deg)
            new_width = int((np.abs(height * np.sin(rotation_rad)) + np.abs(width * np.cos(rotation_rad))))
            new_height = int((np.abs(height * np.cos(rotation_rad)) + np.abs(width * np.sin(rotation_rad))))

            # Calculate the translation to keep the rotated image centered
            # grab the rotation matrix (applying the negative of the
            # angle to rotate clockwise), then grab the sine and cosine
            # (i.e., the rotation components of the matrix)
            rotation_matrix = cv2.getRotationMatrix2D(center, -rotation_deg, 1.0)
            rotation_matrix[0, 2] += (new_width / 2) - center[0]
            rotation_matrix[1, 2] += (new_height / 2) - center[1]

            # Apply the rotation to the image and fill extra region with black
            corrected_img = cv2.warpAffine(
                img, 
                rotation_matrix, 
                (new_width, new_height), 
                borderMode=cv2.BORDER_CONSTANT, 
                borderValue=(0, 0, 0)
            )

            if face_info_list is not None:
                corrected_faces = []
                for face_info in face_info_list:
                    # Apply the rotation to the facial landmark points
                    face_points = face_info['kps']
                    num_points = len(face_points)
                    homogenous_points = np.hstack(
                        (np.array(face_points), np.ones((num_points, 1)))
                    )
                    transformed_points = np.dot(homogenous_points, rotation_matrix.T)
                    corrected_face_points = transformed_points[:, :2]

                    # Apply the rotation to the bounding box coordinates
                    bbox = face_info['bbox']
                    bbox = np.array(bbox)
                    bbox_points = np.array(
                        [[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]]]
                    )
                    homogenous_bbox_points = np.hstack((bbox_points, np.ones((4, 1))))
                    transformed_bbox_points = np.dot(homogenous_bbox_points, rotation_matrix.T)
                    corrected_bbox = [
                        min(transformed_bbox_points[:, 0]), min(transformed_bbox_points[:, 1]),
                        max(transformed_bbox_points[:, 0]), max(transformed_bbox_points[:, 1])
                    ]

                    # Append the corrected face information to the list
                    corrected_faces.append({
                        'bbox': corrected_bbox,
                        'kps': corrected_face_points.tolist(),
                        'det_score': face_info['det_score']
                    })

                return corrected_img, corrected_faces
            else:
                return corrected_img
        else:
            return img

    def crop_face(self, img, bbox, image_color_space="BGR"):
        """
        Crop the face area from the input image based on the provided bounding box.

        Parameters:
            img (numpy.ndarray): The input image from which to crop the face area.
            bbox (list): A list containing four elements [x1, y1, x2, y2] representing the bounding box coordinates.
                         The bounding box defines the region of interest to crop the face area.
            image_color_space (str, optional): The color space of the input image. Default is "BGR".
                                               If "BGR", the function will convert the image to RGB before cropping.
                                               If "RGB" or any other value, the image is assumed to be in RGB format.

        Returns:
            numpy.ndarray: The cropped face area as a NumPy array.

        Notes:
            This function crops the face area from the input image based on the provided bounding box.
            The bounding box is defined by the coordinates [x1, y1, x2, y2], where (x1, y1) and (x2, y2)
            represent the top-left and bottom-right corners of the bounding box, respectively.

            The function supports two color spaces for the input image:
            - "BGR": The function assumes the input image is in BGR format (OpenCV default).
            - "RGB": The function assumes the input image is in RGB format.

            If the input image is in BGR format, the function will convert it to RGB before cropping.

            Example usage:
                # Assuming 'detector' is an instance of YourFaceDetectorClass
                face_area = detector.crop_face(image, [x1, y1, x2, y2], image_color_space="BGR")
        """
        if image_color_space == "BGR":
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = img

        # Crop face area and save the first face image
        x1, y1 = int(bbox[0]), int(bbox[1])
        x2, y2 = int(bbox[2]), int(bbox[3])

        crop_face_area = image_rgb[y1:y2, x1:x2]
        return crop_face_area

    def find_get_top_face(self, img):
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
        faces = self.detect_faces(img)
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

    def get_face_coordinates(self, face: Dict, box_format="x1y1-x2y2"):
        """
        Retrieves the coordinates of a face from the given face dictionary.

        Args:
            face: A dictionary containing information about a face, typically obtained from a face detection algorithm.
            box_format: The format of the output coordinates. Valid values are "xy-wh" and "x1y1-x2y2" (default).

        Returns:
            If the box_format is "xy-wh", the function returns a tuple (x, y, w, h) representing the top-left coordinates (x, y)
            of the bounding box and its width (w) and height (h).
            If the box_format is "x1y1-x2y2", the function returns a tuple (x1, y1, x2, y2) representing the top-left (x1, y1)
            and bottom-right (x2, y2) coordinates of the bounding box.

        """
        bbox = face.get('bbox')
        x1, y1, x2, y2 = bbox

        if box_format == "x1y1-x2y2":
            return (x1, y1, x2, y2)
        else:
            # xywh format
            w = int(abs(x2 - x1))
            h = int(abs(y2 - y1))

            x = int(x1)
            y = int(y1)

            return (x, y, w, h)

    def visualize_faces_and_landmarks(
            self,
            img, 
            faces_info, 
            bbox_color=(0, 255, 0), 
            kps_color=(0, 0, 255)
        ):
        """
        Visualize detected faces and their landmarks on the input image.

        Parameters:
            img (numpy.ndarray): The input image on which to draw faces and landmarks.
            faces_info (list): A list of dictionaries containing face information.
                            Each dictionary should have 'bbox' and 'kps' keys.
            bbox_color (tuple, optional): The color of the bounding box. Default is (0, 255, 0) (green).
            kps_color (tuple, optional): The color of the facial landmark points. Default is (0, 0, 255) (red).

        Returns:
            numpy.ndarray: The image with bounding boxes and landmarks drawn.

        Notes:
            This function takes an input image and a list of dictionaries containing face information.
            Each dictionary in the list must have the following keys:
            - 'bbox': A list containing four elements [x1, y1, x2, y2] representing the bounding box coordinates.
            - 'kps': A list of facial landmark points as a 2D array of shape (n_points, 2).
                    Each row contains the (x, y) coordinates of a facial landmark point.
            - bbox_color and kps_color are optional parameters that allow you to specify the color of the bounding box
            and facial landmark points, respectively.

            The function draws bounding boxes around the detected faces and plots the facial landmark points on the image
            using the specified colors.

            Example usage:
                # Assuming 'img' is the input image and 'faces_info' contains face information
                result_img = visualize_faces_and_landmarks(img, faces_info)
                cv2.imshow('Detected Faces and Landmarks', result_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        """
        for face_info in faces_info:
            bbox = face_info['bbox']
            kps = face_info['kps']

            # Draw bounding box around the face
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(img, (x1, y1), (x2, y2), bbox_color, 2)

            # Draw facial landmark points with different colors
            for i, kp in enumerate(kps):
                x, y = map(int, kp)
                color = kps_color if i < 5 else (255, 0, 0)  # First 5 points in red, rest in blue
                cv2.circle(img, (x, y), 2, color, -1)

        return img

    def get_face_det_res_json_filepath(self, img_filepath):
        """
        Add postfix to file stem and change extension to json and return the 
        resulting filepath.

        Args:
            img_filepath (str): The filepath of the input image.

        Returns:
            str: The filepath of the JSON output file.

        """
        p = Path(img_filepath)
        json_out_filepath = p.with_name(f"{p.stem}_insight_face_buffalo_l.json")
        return json_out_filepath

    def store_face_det_result(self, json_out_filepath, result):
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

    def read_face_det_json_annotation_file(self, json_annotation_filepath):
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
            self,
            rect, 
            left_margin_percent, 
            right_margin_percent, 
            top_margin_percent, 
            bottom_margin_percent, 
            image_shape,
            box_format="x1y1-x2y2",
        ):
        """
        Adds margins to the rectangle coordinates on all four sides in percentage and corrects the rectangle
        if it goes outside the specified image shape.

        Args:
            rect (tuple): A tuple containing the original rectangle coordinates.
            left_margin_percent (float): The margin percentage to be added to the left side.
            right_margin_percent (float): The margin percentage to be added to the right side.
            top_margin_percent (float): The margin percentage to be added to the top side.
            bottom_margin_percent (float): The margin percentage to be added to the bottom side.
            image_shape (tuple): A tuple containing the shape of the image (height, width).

        Returns:
            tuple: A new tuple containing the modified rectangle coordinates with the added margins.

        """
        if box_format == "x1y1-x2y2":
            x1, y1, x2, y2 = rect
            x, y = x1, y1
            w, h = (x2- x1), (y2 - y1)
        else:
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

        if box_format == "x1y1-x2y2":
            ## Add new width and height in new x1, y1 point
            x2_new = x_new + w_new
            y2_new = y_new + h_new
            return (x_new, y_new, x2_new, y2_new)
        else:
            return (x_new, y_new, w_new, h_new)

    def detect_crop_top_face_with_margin(
            self, 
            image_path,
            faces=None,
            left_margin_percent=0.2, 
            right_margin_percent=0.2, 
            top_margin_percent=0.1, 
            bottom_margin_percent=0.1, 
            is_random_margin=False
        ):
        """
        Crop the top face from an image with specified margins.

        Args:
            self: The object instance.
            image_path (str): The path to the image file.
            left_margin_percent (float, optional): The percentage of the left margin to add to the face bounding box. Defaults to 0.2.
            right_margin_percent (float, optional): The percentage of the right margin to add to the face bounding box. Defaults to 0.2.
            top_margin_percent (float, optional): The percentage of the top margin to add to the face bounding box. Defaults to 0.1.
            bottom_margin_percent (float, optional): The percentage of the bottom margin to add to the face bounding box. Defaults to 0.1.

        Returns:
            cropped_pil (PIL.Image.Image): The cropped PIL Image containing the top face.

        Notes:
            - The method performs face detection using a pre-trained model to locate the faces in the image.
            - If a precomputed face detection result file exists, it is used; otherwise, face detection is performed and the result is saved as a JSON file.
            - If no face is detected in the image, False is returned.
        """
        # read image
        image_pil = Image.open(image_path).convert("RGB")
        image = np.array(image_pil)

        if not faces:
            # perform face detection
            ## first check if result file exists
            face_detection_json_result_filepath = self.get_face_det_res_json_filepath(
                image_path
            )
            if os.path.exists(face_detection_json_result_filepath):
                faces = self.read_face_det_json_annotation_file(
                    face_detection_json_result_filepath
                )
            else:
                faces = self.detect_faces(image)
                # save result as a json
                self.store_face_det_result(face_detection_json_result_filepath, faces)
        
        if faces:
            # print(faces[0])
            face_box = faces[0]["bbox"]

            if is_random_margin:
                import random
                left_margin_percent = random.uniform(left_margin_percent - 0.1, left_margin_percent + 0.1)
                right_margin_percent = random.uniform(right_margin_percent - 0.1, right_margin_percent + 0.1)
                top_margin_percent = random.uniform(top_margin_percent - 0.1, top_margin_percent + 0.1)
                bottom_margin_percent = random.uniform(bottom_margin_percent - 0.1, bottom_margin_percent + 0.1)

            face_box_new = self.add_margin_to_rect(
                face_box,
                left_margin_percent=left_margin_percent,
                right_margin_percent=right_margin_percent,
                top_margin_percent=top_margin_percent,
                bottom_margin_percent=bottom_margin_percent,
                image_shape=image.shape
            )
            cropped_pil = image_pil.crop(face_box_new)
            return cropped_pil
        else:
            # logger.warning(f"Skipping.. No any face detected in image {image_path}..")
            return False
        
def main():
    from timeit import default_timer as timer

    ## init face detector
    face_detector = FaceDetector(is_warmup=False)

    img = ins_get_image('t1')
    # img = cv2.imread("./resources/images/pan-card-500x500-face-oriented-left-side.jpg")

    ## perform warmup
    face_detector.warmup_model()

    start_time = timer()
    faces = face_detector.detect_faces(img, drop_score=0.7)
    end_time = timer()
    elasped_time = end_time - start_time
    print(f"Elapesed time: {elasped_time}")

    print(f"Detected faces: {faces}")

    if faces:
        # get landmarks
        landmark = faces[0]['kps']

        # set output dir
        out_dir = "/tmp/ray/session_latest/aaa/"
        os.makedirs(out_dir, exist_ok=True)

        ## visualize original image
        visualized_image = face_detector.visualize_faces_and_landmarks(img.copy(), faces)
        cv2.imwrite(os.path.join(out_dir, "original_visulized_image.jpg"), visualized_image)

        ## crop from original image -- align face
        cropped_wrapped_face, M = face_detector.normlize_crop_face(img, landmark)
        cv2.imwrite(os.path.join(out_dir, "original_cropped_face.jpg"), cropped_wrapped_face)

        ## make image orientation correction
        ## estimate rotation angle from M i.e landmark transformation matrix
        rotation_deg, scale_x, scale_y = face_detector.get_alignment_angle(M)
        print(f"rotation_deg: {rotation_deg}")

        ## perform image orientation correction with faces
        corrected_img, corrected_faces = face_detector.correct_image_orientation(
            img, rotation_deg, face_info_list=faces
        )
        cv2.imwrite(os.path.join(out_dir, "corrected_image.jpg"), corrected_img)

        visualized_corrected_img = face_detector.visualize_faces_and_landmarks(
            corrected_img.copy(), corrected_faces
        )
        cv2.imwrite(os.path.join(out_dir, "corrected_visualized_image.jpg"), visualized_corrected_img)

        ## crop from corrected image
        bbox = corrected_faces[0]["bbox"]
        correted_cropped_face = face_detector.crop_face(corrected_img, bbox)
        cv2.imwrite(os.path.join(out_dir, "corrected_cropped_face.jpg"), correted_cropped_face)

    else:
        print(f"No any face detected in image...")

    # from ipdb import set_trace; set_trace()
    
if __name__ == "__main__":
    main()
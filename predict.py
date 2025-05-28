# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
import os
import cv2
import sys
import time
import torch
import subprocess
import numpy as np
from PIL import Image
from typing import List, Dict, Any
import polyline  # New import
from pycocotools import mask as coco_mask_util  # New import for RLE decoding
import json # Added for parsing points/labels strings
# Add /tmp/sa2 to sys path
sys.path.extend("/sa2")
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_image_predictor import SAM2ImagePredictor # Added for single mode

WEIGHTS_CACHE = "./checkpoints"
MODEL_NAME = "sam2.1_hiera_large.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
WEIGHTS_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
#MODEL_NAME = "sam2_hiera_large.pt"
#MODEL_CFG = "sam2_hiera_l.yaml"
#WEIGHTS_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["wget", "-O", dest, url], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        os.chdir("/sa2")
        # Get path to model
        model_cfg = MODEL_CFG
        model_path = WEIGHTS_CACHE + "/" +MODEL_NAME
        # Download weights
        if not os.path.exists(model_path):
            download_weights(WEIGHTS_URL, model_path)
        # Setup SAM2
        self.sam2 = build_sam2(config_file=model_cfg, ckpt_path=model_path, device='cuda', apply_postprocessing=False)
        # turn on tfloat32 for Ampere GPUs
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def predict(
        self,
        image: Path = Input(description="Input image"),
        mode: str = Input(
            default="auto", 
            choices=["auto", "single"], 
            description="Prediction mode: 'auto' for automatic mask generation, 'single' for mask prediction based on point prompts."
        ),
        point_coords_str: str = Input(
            default="[]", 
            description="JSON string of point coordinates (list of [x,y] pairs, e.g., \"[[100,200],[150,250]]\") for 'single' mode. Active only if mode is 'single'."
        ),
        point_labels_str: str = Input(
            default="[]", 
            description="JSON string of point labels (list of 0 or 1, e.g., \"[1,0]\") for 'single' mode. Active only if mode is 'single'."
        ),
        mask_limit: int = Input(
            default=-1, description="maximum number of masks to return. If -1 or None, all masks will be returned. NOTE: The masks are sorted by predicted_iou."),
        points_per_side: int = Input(
            default=64, description="The number of points to be sampled along one side of the image."),
        points_per_batch: int = Input(
            default=128, description="Sets the number of points run simultaneously by the model"),
        pred_iou_thresh: float = Input(
            default=0.7, description="A filtering threshold in [0,1], using the model's predicted mask quality."),
        stability_score_thresh: float = Input(
            default=0.92, description="A filtering threshold in [0,1], using the stability of the mask under changes to the cutoff used to binarize the model's mask predictions."),
        stability_score_offset: float = Input(
            default=0.7, description="The amount to shift the cutoff when calculated the stability score."),
        crop_n_layers: int = Input(
            default=1, description="If >0, mask prediction will be run again on crops of the image"),
        box_nms_thresh: float = Input(
            default=0.7, description="The box IoU cutoff used by non-maximal suppression to filter duplicate masks."),
        crop_n_points_downscale_factor: int = Input(
            default=2, description="The number of points-per-side sampled in layer n is scaled down by crop_n_points_downscale_factor**n."),
        min_mask_region_area: float = Input(
            default=25.0, description="If >0, postprocessing will be applied to remove disconnected regions and holes in masks with area smaller than min_mask_region_area."),
        mask_2_mask: bool = Input(
            default=True, description="Whether to add a one step refinement using previous mask predictions."),
        multimask_output: bool = Input(
            default=False, description="Whether to output multimask at each point of the grid."),
    ) -> List[Dict[str, Any]]: # Modified return type
        """Run a single prediction on the model"""
        # Convert input image
        image_rgb = Image.open(image).convert('RGB')
        image_arr = np.array(image_rgb)

        if mode == "auto":
            # Setup the predictor and image for auto mode
            mask_generator = SAM2AutomaticMaskGenerator(
                model=self.sam2,
                points_per_side=points_per_side,
                points_per_batch=points_per_batch,
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=stability_score_thresh,
                stability_score_offset=stability_score_offset,
                crop_n_layers=crop_n_layers,
                box_nms_thresh=box_nms_thresh,
                crop_n_points_downscale_factor=crop_n_points_downscale_factor,
                min_mask_region_area=min_mask_region_area,
                use_m2m=mask_2_mask,
                multimask_output=multimask_output,
                output_mode="coco_rle", 
            )
            sam_output = mask_generator.generate(image_arr)

            # Process each annotation to add the polygon
            for ann in sam_output:
                rle_segmentation = ann['segmentation']
                
                # Decode RLE to binary mask
                # rle_segmentation is a dict like {'size': [H, W], 'counts': b'RLE_str'}
                binary_mask = coco_mask_util.decode(rle_segmentation)

                # Ensure binary_mask is uint8 for cv2.findContours
                if binary_mask.dtype != np.uint8:
                    binary_mask = binary_mask.astype(np.uint8)
                
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                ann['polygon'] = "" # Default to empty string

                if contours:
                    # Filter out small contours if necessary, or select the largest one
                    # For simplicity, let's assume the largest contour is the one we want.
                    # If multiple disjoint objects are in one mask, this will only get the largest.
                    main_contour = max(contours, key=cv2.contourArea)

                    if main_contour.size > 0: # Check if contour is not empty
                        # Approximate contour to polygon. Epsilon is 0.5% of arc length.
                        # A smaller epsilon means more points and a more precise polygon.
                        # A larger epsilon means fewer points and a more simplified polygon.
                        epsilon = 0.005 * cv2.arcLength(main_contour, True)
                        approx_polygon_cv = cv2.approxPolyDP(main_contour, epsilon, True) # Shape: (N, 1, 2)

                        # approx_polygon_cv contains points as [[x1, y1]], [[x2, y2]], ...
                        # polyline.encode expects a list of (y, x) tuples or lists.
                        if approx_polygon_cv is not None and approx_polygon_cv.ndim == 3 and \
                        approx_polygon_cv.shape[1] == 1 and approx_polygon_cv.shape[2] == 2 and \
                        approx_polygon_cv.shape[0] > 0: # Check if polygon has points
                            
                            # Convert points to list of (y, x) for polyline.encode
                            # OpenCV points are (x,y). Polyline expects (latitude, longitude) i.e. (y,x).
                            polyline_coords = []
                            for point_wrapper in approx_polygon_cv:
                                x, y = point_wrapper[0]
                                polyline_coords.append((float(y), float(x))) # Ensure float
                            
                            if polyline_coords:
                                try:
                                    encoded_polyline = polyline.encode(polyline_coords)
                                    ann['polygon'] = encoded_polyline
                                except Exception as e:
                                    print(f"Warning: Could not encode polyline for a mask: {e}")
                                    # ann['polygon'] remains ""
                            # else: ann['polygon'] remains "" (no valid coordinates after processing)
                    # else: ann['polygon'] remains "" (main_contour was empty)
                # else: ann['polygon'] remains "" (no contours found)

            # Sort sam_output (which now includes 'polygon') by 'predicted_iou' and limit the number of masks
            sorted_sam_output = sorted(sam_output, key=lambda x: x['predicted_iou'], reverse=True)
            if mask_limit != -1 and mask_limit is not None:
                return sorted_sam_output[:mask_limit]
            return sorted_sam_output

        elif mode == "single":
            np_point_coords = None
            np_point_labels = None

            try:
                raw_coords = json.loads(point_coords_str)
                if raw_coords: # If not an empty list
                    if not isinstance(raw_coords, list) or not all(isinstance(p, list) and len(p) == 2 for p in raw_coords):
                        raise ValueError("point_coords_str must be a JSON string of a list of [x,y] pairs, e.g., \"[[100,200],[150,250]]\".")
                    np_point_coords = np.array(raw_coords, dtype=np.float32)

                raw_labels = json.loads(point_labels_str)
                if raw_labels: # If not an empty list
                    if not isinstance(raw_labels, list) or not all(isinstance(l, int) for l in raw_labels):
                        raise ValueError("point_labels_str must be a JSON string of a list of integers, e.g., \"[1,0]\".")
                    np_point_labels = np.array(raw_labels, dtype=np.int32)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in point_coords_str or point_labels_str: {e}")

            if np_point_coords is not None:
                if np_point_labels is None:
                    raise ValueError("point_labels_str must be provided if point_coords_str is provided and not empty.")
                if len(np_point_coords) != len(np_point_labels):
                    raise ValueError("point_coords_str and point_labels_str must have the same number of elements.")
            elif np_point_labels is not None: # Labels provided but no coords
                 raise ValueError("point_coords_str must be provided if point_labels_str is provided and not empty.")

            image_predictor = SAM2ImagePredictor(self.sam2)
            
            start_embedding_time = time.time() # Start timer
            image_predictor.set_image(image_arr)
            embedding_time_taken = time.time() - start_embedding_time # End timer
            print(f"Image embedding calculation took: {embedding_time_taken:.4f} seconds") # Log time

            masks_np, iou_predictions_np, _ = image_predictor.predict(
                point_coords=np_point_coords,
                point_labels=np_point_labels,
                multimask_output=True # Default for SAM2ImagePredictor, can be made configurable if needed
            )

            output_annotations = []
            num_masks_generated = masks_np.shape[0]
            for i in range(num_masks_generated):
                binary_mask = masks_np[i] 
                if binary_mask.dtype != np.uint8:
                    binary_mask = binary_mask.astype(np.uint8)

                rle_segmentation = coco_mask_util.encode(np.asfortranarray(binary_mask))
                # Ensure counts is decoded if it's bytes, for JSON compatibility, though Cog might handle bytes.
                # The existing code for auto mode implies direct use of rle_segmentation dict.
                # Let's ensure 'counts' is a string for broader compatibility if it's bytes.
                if isinstance(rle_segmentation['counts'], bytes):
                    rle_segmentation['counts'] = rle_segmentation['counts'].decode('utf-8')


                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                encoded_polyline = ""
                if contours:
                    main_contour = max(contours, key=cv2.contourArea)
                    if main_contour.size > 0:
                        epsilon = 0.005 * cv2.arcLength(main_contour, True)
                        approx_polygon_cv = cv2.approxPolyDP(main_contour, epsilon, True)
                        if approx_polygon_cv is not None and approx_polygon_cv.ndim == 3 and \
                           approx_polygon_cv.shape[1] == 1 and approx_polygon_cv.shape[2] == 2 and \
                           approx_polygon_cv.shape[0] > 0:
                            polyline_coords = [(float(p[0][1]), float(p[0][0])) for p in approx_polygon_cv]
                            if polyline_coords:
                                try:
                                    encoded_polyline = polyline.encode(polyline_coords)
                                except Exception as e:
                                    print(f"Warning: Could not encode polyline for a mask (single mode): {e}")
                
                current_mask_area = coco_mask_util.area(rle_segmentation)
                current_mask_bbox = coco_mask_util.toBbox(rle_segmentation)

                ann = {
                    'segmentation': rle_segmentation,
                    'predicted_iou': float(iou_predictions_np[i]),
                    'polygon': encoded_polyline,
                    'area': float(current_mask_area),
                    'bbox': [float(x) for x in current_mask_bbox],
                }
                output_annotations.append(ann)

            sorted_output = sorted(output_annotations, key=lambda x: x['predicted_iou'], reverse=True)
            if mask_limit != -1 and mask_limit is not None:
                return sorted_output[:mask_limit]
            return sorted_output
        else:
            raise ValueError(f"Invalid mode: {mode}. Choose 'auto' or 'single'.")

        # Sort sam_output (which now includes 'polygon') by 'predicted_iou' and limit the number of masks
        # This part is now handled within each mode's block.
        # sorted_sam_output = sorted(sam_output, key=lambda x: x['predicted_iou'], reverse=True)
        # if mask_limit != -1 and mask_limit is not None:
        #     return sorted_sam_output[:mask_limit]
        # return sorted_sam_output

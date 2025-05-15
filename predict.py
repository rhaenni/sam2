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
# Add /tmp/sa2 to sys path
sys.path.extend("/sa2")
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

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

        # Setup the predictor and image
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
            output_mode="coco_rle", # Changed output_mode
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

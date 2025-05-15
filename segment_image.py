import argparse
import os
import torch
import numpy as np
import cv2
from PIL import Image
import random
from pathlib import Path

# Try importing sam2 components
try:
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
except ImportError:
    print("Error: Failed to import sam2 components.")
    print("Please ensure the sam2 library is installed correctly:")
    print("pip install git+https://github.com/facebookresearch/sam2.git")
    exit(1)

# Define the target aspect ratio for a Pokemon card (height / width)
TARGET_ASPECT_RATIO = 8.8 / 6.3
DEFAULT_MIN_AREA = 5000 # Default minimum pixel area for a mask to be considered a card

# --- Default values from SAM2AutomaticMaskGenerator ---
DEFAULT_POINTS_PER_SIDE = 32
DEFAULT_POINTS_PER_BATCH = 64
DEFAULT_PRED_IOU_THRESH = 0.0
DEFAULT_STABILITY_SCORE_THRESH = 0.95
DEFAULT_STABILITY_SCORE_OFFSET = 1.0
DEFAULT_MASK_THRESHOLD = 0.0
DEFAULT_BOX_NMS_THRESH = 0.7
DEFAULT_CROP_N_LAYERS = 0
DEFAULT_CROP_NMS_THRESH = 0.7
DEFAULT_CROP_OVERLAP_RATIO = 512 / 1500
DEFAULT_CROP_N_POINTS_DOWNSCALE_FACTOR = 1
DEFAULT_MIN_MASK_REGION_AREA = 0
DEFAULT_USE_M2M = False
# apply_postprocessing default is False via action='store_true'

# --- New Preprocessing Default ---
DEFAULT_APPLY_CLAHE = False
DEFAULT_REMOVE_HIGHLIGHTS = False # Added

def segment_image_native(
    image_path,
    output_filename="segmented.png",
    checkpoint_path="checkpoints/sam2.1_hiera_large.pt",
    model_cfg_path="configs/sam2.1/sam2.1_hiera_l.yaml", # Required: Path to the model config yaml
    device_str='auto',
    # Mask Generation Parameters (defaults match SAM2AutomaticMaskGenerator)
    points_per_side=DEFAULT_POINTS_PER_SIDE,
    points_per_batch=DEFAULT_POINTS_PER_BATCH,
    pred_iou_thresh=DEFAULT_PRED_IOU_THRESH,
    stability_score_thresh=DEFAULT_STABILITY_SCORE_THRESH,
    stability_score_offset=DEFAULT_STABILITY_SCORE_OFFSET,
    mask_threshold=DEFAULT_MASK_THRESHOLD, # Added
    box_nms_thresh=DEFAULT_BOX_NMS_THRESH,
    crop_n_layers=DEFAULT_CROP_N_LAYERS,
    crop_nms_thresh=DEFAULT_CROP_NMS_THRESH, # Added
    crop_overlap_ratio=DEFAULT_CROP_OVERLAP_RATIO, # Added
    crop_n_points_downscale_factor=DEFAULT_CROP_N_POINTS_DOWNSCALE_FACTOR,
    min_mask_region_area=DEFAULT_MIN_MASK_REGION_AREA,
    use_m2m=DEFAULT_USE_M2M, # Added
    apply_postprocessing=False, # Default handled by argparse action
    # Filtering Parameters
    ratio_threshold=0.15,
    min_area=DEFAULT_MIN_AREA,
    show_all_masks=False,
    # Preprocessing Parameters
    apply_clahe=DEFAULT_APPLY_CLAHE, # Added
    remove_highlights=DEFAULT_REMOVE_HIGHLIGHTS # Added
):
    """
    Segments an image using native SAM2 AutomaticMaskGenerator, filters masks, and saves the result.

    Args:
        image_path (str): Path to the input image file.
        output_filename (str): Desired filename for the output segmented image.
        checkpoint_path (str): Path to the SAM2 model weights (.pt) file.
        model_cfg_path (str): Path to the SAM2 model configuration (.yaml) file.
        device_str (str): Device to use ('cuda', 'cpu', 'mps', 'auto').
        # Mask Generation Args...
        points_per_side (int): Grid density for sampling points.
        points_per_batch (int): Batch size for point processing.
        pred_iou_thresh (float): Predicted IoU threshold for filtering masks.
        stability_score_thresh (float): Stability score threshold for filtering masks.
        stability_score_offset (float): Stability score offset.
        box_nms_thresh (float): Box NMS threshold for deduplication.
        crop_n_layers (int): Number of crop layers.
        crop_n_points_downscale_factor (int): Downscale factor for points in cropped regions.
        min_mask_region_area (float): Minimum area for postprocessing removal.
        apply_postprocessing (bool): Whether to apply postprocessing to remove small regions/holes.
        # Filtering Args...
        ratio_threshold (float): Allowable deviation from the target aspect ratio.
        min_area (int): Minimum bounding box area (pixels) for a mask to be kept.
        show_all_masks (bool): If True, skip filtering and show all generated masks as outlines.
        apply_clahe (bool): If True, apply CLAHE preprocessing to the input image.
        remove_highlights (bool): If True, attempt to remove specular highlights using inpainting.
    """

    if not os.path.exists(checkpoint_path):
        print(f"Error: Model checkpoint not found at {checkpoint_path}")
        return
    if not os.path.exists(image_path):
        print(f"Error: Input image not found at {image_path}")
        return

    # --- Device Setup ---
    if device_str == 'auto':
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Warning: MPS support is preliminary and may have issues.")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)
    print(f"Using device: {device}")

    try:
        # --- Load Image ---
        print(f"Loading image from {image_path}...")
        image_pil = Image.open(image_path).convert("RGB")
        image_np_rgb = np.array(image_pil)

        # --- Optional Preprocessing ---
        if apply_clahe:
            print("Applying CLAHE preprocessing...")
            # Convert RGB to LAB
            image_np_lab = cv2.cvtColor(image_np_rgb, cv2.COLOR_RGB2LAB)
            # Split channels
            l_channel, a_channel, b_channel = cv2.split(image_np_lab)
            # Create CLAHE object (parameters can be tuned)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            # Apply CLAHE to L channel
            cl_l = clahe.apply(l_channel)
            # Merge channels back
            merged_lab = cv2.merge((cl_l, a_channel, b_channel))
            # Convert LAB back to RGB
            image_np_rgb = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)
            print("CLAHE applied.")
            # Optional: Save preprocessed image for debugging
            # cv2.imwrite("preprocessed_clahe.png", cv2.cvtColor(image_np_rgb, cv2.COLOR_RGB2BGR))

        # --- Optional Highlight Removal (Inpainting) ---
        if remove_highlights:
            print("Attempting highlight removal via inpainting...")
            try:
                # Convert to grayscale to find highlights
                image_gray = cv2.cvtColor(image_np_rgb, cv2.COLOR_RGB2GRAY)
                # Threshold to create a mask of very bright pixels (tune the threshold value 240-250)
                highlight_thresh = 245
                _, highlight_mask = cv2.threshold(image_gray, highlight_thresh, 255, cv2.THRESH_BINARY)

                # Check if any highlights were found
                if np.sum(highlight_mask > 0) > 0:
                    # Dilate the mask slightly to ensure edges are covered
                    # kernel = np.ones((3,3),np.uint8)
                    # highlight_mask = cv2.dilate(highlight_mask, kernel, iterations = 1)

                    # Inpaint using the mask (Telea's method often works well)
                    # The '3' is the inpaint radius - pixels around the masked area used for filling
                    inpainted_image = cv2.inpaint(image_np_rgb, highlight_mask, 3, cv2.INPAINT_TELEA)
                    image_np_rgb = inpainted_image # Update the image
                    print(f"Highlights above {highlight_thresh} inpainted.")
                    # Optional: Save inpainted image for debugging
                    # cv2.imwrite("preprocessed_inpainted.png", cv2.cvtColor(image_np_rgb, cv2.COLOR_RGB2BGR))
                else:
                    print(f"No highlights found above threshold {highlight_thresh}.")
            except Exception as e:
                print(f"Error during highlight removal: {e}")

        # --- Build Model and Generator ---
        print(f"Loading SAM2 model from checkpoint: {checkpoint_path} and config: {model_cfg_path}...")
        # Use the original relative path for the config file
        # Note: build_sam2 apply_postprocessing seems redundant if generator handles it? Keep for now.
        sam2 = build_sam2(model_cfg_path, checkpoint_path, device=device, apply_postprocessing=True)
        print("Model built.")

        print("Initializing AutomaticMaskGenerator...")
        mask_generator = SAM2AutomaticMaskGenerator(
            model=sam2,
            points_per_side=points_per_side,
            points_per_batch=points_per_batch,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            stability_score_offset=stability_score_offset,
            mask_threshold=mask_threshold, # Added
            box_nms_thresh=box_nms_thresh,
            crop_n_layers=crop_n_layers,
            crop_nms_thresh=crop_nms_thresh, # Added
            crop_overlap_ratio=crop_overlap_ratio, # Added
            crop_n_points_downscale_factor=crop_n_points_downscale_factor,
            min_mask_region_area=min_mask_region_area,
            apply_postprocessing=apply_postprocessing,
            use_m2m=use_m2m # Added
            # Add other parameters if needed
        )
        print("Generator initialized with parameters:")
        # Print all relevant parameters, using the actual values from the function arguments
        print(f"  points_per_side={points_per_side}, points_per_batch={points_per_batch}, "
              f"pred_iou_thresh={pred_iou_thresh}, stability_score_thresh={stability_score_thresh}, "
              f"stability_score_offset={stability_score_offset}, mask_threshold={mask_threshold}, " # Added
              f"box_nms_thresh={box_nms_thresh}, crop_n_layers={crop_n_layers}, "
              f"crop_nms_thresh={crop_nms_thresh}, crop_overlap_ratio={crop_overlap_ratio:.4f}, " # Added
              f"crop_n_points_downscale_factor={crop_n_points_downscale_factor}, "
              f"min_mask_region_area={min_mask_region_area}, apply_postprocessing={apply_postprocessing}, "
              f"use_m2m={use_m2m}") # Added

        # --- Generate Masks ---
        print(f"Running mask generation on {image_path}...")
        # Note: Consider torch.autocast for performance on CUDA if needed
        raw_masks_info = mask_generator.generate(image_np_rgb)
        print(f"Generated {len(raw_masks_info)} raw masks.")

        if not raw_masks_info:
            print("Warning: No masks generated by the model.")
            cv2.imwrite(output_filename, cv2.cvtColor(image_np_rgb, cv2.COLOR_RGB2BGR))
            print(f"Original image saved as {output_filename} (no masks generated)")
            return

        # --- Prepare Data for Filtering ---
        # Extract necessary info: segmentation mask and bbox [x, y, w, h]
        masks_data = [ann['segmentation'] for ann in raw_masks_info] # boolean numpy arrays
        boxes_data = [ann['bbox'] for ann in raw_masks_info] # [x, y, w, h]

        # --- Filtering Logic ---
        final_masks = []
        if show_all_masks:
            print("Skipping custom filtering, using all generated masks.")
            final_masks = masks_data
        else:
            H, W, _ = image_np_rgb.shape

            # Helper function to calculate aspect ratio deviation
            def aspect_ratio_deviation(box):
                x, y, w, h = box
                if w == 0 or h == 0: return float('inf')
                return abs((h / w) - TARGET_ASPECT_RATIO)

            # Step 1: Initial Filtering (Area AND Aspect Ratio)
            initial_indices = []
            initial_boxes = []
            passed_initial_filter = 0
            for i, box in enumerate(boxes_data):
                x, y, w, h = box
                area = w * h
                if area >= min_area:
                    deviation = aspect_ratio_deviation(box)
                    if deviation <= ratio_threshold:
                        passed_initial_filter += 1
                        initial_indices.append(i)
                        initial_boxes.append(box)
            print(f"{passed_initial_filter} masks passed initial area ({min_area}px) AND aspect ratio (dev<={ratio_threshold:.3f}) filters.")

            # --- Save Intermediate Output (Pass 1) ---
            if passed_initial_filter > 0:
                print("Saving intermediate result after initial filtering to segmented_pass1.png...")
                img_pass1 = cv2.cvtColor(image_np_rgb.copy(), cv2.COLOR_RGB2BGR)
                for idx in initial_indices:
                    mask = masks_data[idx].astype(np.uint8) * 255
                    color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(img_pass1, contours, -1, color, thickness=2)
                cv2.imwrite("segmented_pass1.png", img_pass1)
                print("Intermediate result saved.")
            else:
                print("Skipping intermediate save as no masks passed initial filters.")
            # ------------------------------------------

            refined_masks_data = list(masks_data) # Create a mutable copy for refinement
            refined_boxes_data = list(boxes_data) # Create a mutable copy for refinement
            refined_count = 0

            # Step 2: Recursive Refinement Attempt (operates on initially filtered masks)
            print("\nAttempting recursive refinement for initial candidates...")
            padding = 5 # Restore smaller padding
            for i in initial_indices: # Iterate through indices that passed initial filters
                original_box = refined_boxes_data[i] # Use potentially refined box data if looping multiple times (though not currently)
                original_deviation = aspect_ratio_deviation(original_box)
                print(f"\nConsidering mask index {i} for refinement (Original deviation: {original_deviation:.4f})")

                # Extract sub-image based on original bbox + padding
                x1, y1 = max(0, original_box[0] - padding), max(0, original_box[1] - padding)
                x2, y2 = min(W, original_box[0] + original_box[2] + padding), min(H, original_box[1] + original_box[3] + padding)
                # --- Ensure coordinates are integers for slicing ---
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # --------------------------------------------------
                sub_image_h, sub_image_w = y2 - y1, x2 - x1

                # --- No Center Crop - Use original box for tighter refinement ---
                # crop_x = int(sub_image_w * 0.02)
                # crop_y = int(sub_image_h * 0.02)
                # x1_crop, y1_crop = x1 + crop_x, y1 + crop_y
                # x2_crop, y2_crop = x2 - crop_x, y2 - crop_y
                # # Ensure cropped dimensions are valid
                # if x1_crop >= x2_crop or y1_crop >= y2_crop:
                #     print(f"  Skipping refinement for mask {i}: 2% crop resulted in invalid dimensions.")
                #     continue
                # ---------------------------

                # Original check for invalid dimensions (based on uncropped coords)
                # if sub_image_h_orig <= 0 or sub_image_w_orig <= 0:
                if sub_image_h <= 0 or sub_image_w <= 0:
                    continue # Skip if bbox is invalid or too small

                # Extract the UNPADDED sub-image
                # Extract the PADDED sub-image (fixed comment)
                sub_image = image_np_rgb[y1:y2, x1:x2]
                print(f"  Extracted sub-image of size {sub_image.shape[1]}x{sub_image.shape[0]} at ({x1},{y1})")

                # --- Refinement using OpenCV Contours --- Start --- 
                sub_masks_info = [] # Placeholder if needed downstream, but we directly create the best mask
                best_sub_mask_idx = -1
                best_sub_mask_bool = None
                best_sub_box = None
                min_sub_deviation = original_deviation # Initialize deviation check

                try:
                    print(f"  Running OpenCV contour detection for refinement...")
                    sub_h, sub_w = sub_image.shape[:2]
                    sub_gray = cv2.cvtColor(sub_image, cv2.COLOR_RGB2GRAY)
                    sub_blur = cv2.GaussianBlur(sub_gray, (5, 5), 0)
                    # Apply Otsu's threshold
                    _, sub_thresh = cv2.threshold(sub_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    
                    # Find contours
                    contours, _ = cv2.findContours(sub_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    print(f"    Found {len(contours)} raw contours.")

                    potential_contours = []
                    min_contour_area = 0.1 * sub_w * sub_h # Require contour to be at least 10% of sub-image area
                    
                    for cnt in contours:
                        area = cv2.contourArea(cnt)
                        if area >= min_contour_area:
                            potential_contours.append(cnt)
                    print(f"    Found {len(potential_contours)} contours passing area threshold ({min_contour_area:.0f}px)." )

                    best_contour = None
                    
                    # Find contour with best aspect ratio
                    if potential_contours:
                        for cnt in potential_contours:
                            x_sub, y_sub, w_sub, h_sub = cv2.boundingRect(cnt)
                            current_sub_deviation = aspect_ratio_deviation([x_sub, y_sub, w_sub, h_sub])

                            print(f"      Contour @ ({x_sub},{y_sub}) {w_sub}x{h_sub}: deviation={current_sub_deviation:.4f}, required < {min_sub_deviation:.4f} & < {original_deviation * 0.9:.4f}") # Debug
                            # Check if this contour is better
                            if current_sub_deviation < min_sub_deviation and current_sub_deviation < original_deviation * 0.9:
                                print(f"        -> New best candidate contour for refinement.") # Debug
                                min_sub_deviation = current_sub_deviation
                                best_contour = cnt
                                best_sub_box = [x_sub, y_sub, w_sub, h_sub] # Store the best bbox

                    # If a best contour was found, create its boolean mask
                    if best_contour is not None:
                        print(f"    Selected best contour with deviation {min_sub_deviation:.4f}.")
                        best_sub_mask_idx = 0 # Indicate success

                        # --- Visualize selected contour (for debugging) ---
                        sub_image_vis_cv = cv2.cvtColor(sub_image.copy(), cv2.COLOR_RGB2BGR)
                        cv2.drawContours(sub_image_vis_cv, [best_contour], -1, (0, 255, 0), thickness=2)
                        save_path_sub_cv = f"segmented_refinement_mask{i}.png"
                        cv2.imwrite(save_path_sub_cv, sub_image_vis_cv)
                        print(f"    Saved selected contour visualization to {save_path_sub_cv}")
                        # --------------------------------------------------

                        # Create mask by drawing filled contour
                        # best_sub_mask_bool = np.zeros((sub_h, sub_w), dtype=bool)
                        # cv2.drawContours(best_sub_mask_bool, [best_contour], -1, (True), thickness=cv2.FILLED)
                        # Create uint8 mask, draw, then convert to boolean
                        mask_uint8 = np.zeros((sub_h, sub_w), dtype=np.uint8)
                        cv2.drawContours(mask_uint8, [best_contour], -1, 255, thickness=cv2.FILLED)
                        best_sub_mask_bool = mask_uint8.astype(bool)
                    else:
                        print("    No suitable contour found for refinement.")

                except Exception as e_cv:
                    print(f"    Error during OpenCV refinement: {e_cv}")
                # --- Refinement using OpenCV Contours --- End --- 

                # If a significantly better sub-mask was found (by OpenCV), replace the original
                if best_sub_mask_idx != -1:
                    refined_count += 1
                    # Get the best sub-mask boolean data
                    best_sub_mask_bool = best_sub_mask_bool

                    # Create a new full-size mask canvas
                    new_mask_full = np.zeros((H, W), dtype=bool)

                    # Calculate where the sub-mask lives in the original image coords
                    # --- Adjust absolute coordinates based on the center crop offset ---
                    # sub_x_abs = best_sub_box[0] + x1_crop # Use x1_crop offset
                    # sub_y_abs = best_sub_box[1] + y1_crop # Use y1_crop offset
                    # ---------------------------------------------------------------
                    # --- Use original offsets (no crop) ---
                    sub_x_abs = best_sub_box[0] + x1
                    sub_y_abs = best_sub_box[1] + y1
                    # -------------------------------------
                    sub_w_abs, sub_h_abs = best_sub_box[2], best_sub_box[3]

                    # --- Ensure coordinates are integers for slicing ---
                    sub_x_abs, sub_y_abs = int(sub_x_abs), int(sub_y_abs)
                    sub_w_abs, sub_h_abs = int(sub_w_abs), int(sub_h_abs)
                    # --------------------------------------------------

                    # Determine the slicing for placement, handling boundary conditions
                    place_y1, place_y2 = sub_y_abs, sub_y_abs + sub_h_abs
                    place_x1, place_x2 = sub_x_abs, sub_x_abs + sub_w_abs
                    slice_y1, slice_y2 = 0, best_sub_mask_bool.shape[0]
                    slice_x1, slice_x2 = 0, best_sub_mask_bool.shape[1]

                    # Ensure indices are within bounds of both source and destination
                    if place_y1 < H and place_x1 < W:
                        # Adjust placement region and slice region if submask goes out of bounds (shouldn't happen often with padding)
                        actual_h = min(best_sub_mask_bool.shape[0], H - place_y1)
                        actual_w = min(best_sub_mask_bool.shape[1], W - place_x1)
                        slice_y2 = slice_y1 + actual_h
                        slice_x2 = slice_x1 + actual_w
                        place_y2 = place_y1 + actual_h
                        place_x2 = place_x1 + actual_w

                        # Place the boolean mask
                        new_mask_full[place_y1:place_y2, place_x1:place_x2] = best_sub_mask_bool[slice_y1:slice_y2, slice_x1:slice_x2]

                    # Update the mask and box data in our refined lists
                    refined_masks_data[i] = new_mask_full
                    refined_boxes_data[i] = [sub_x_abs, sub_y_abs, sub_w_abs, sub_h_abs]
                    # print(f"  Refined mask {i} with sub-mask {best_sub_mask_idx} (dev: {min_sub_deviation:.3f} vs {original_deviation:.3f})") # Debug

            print(f"\n{refined_count} masks were refined using recursive segmentation.")

            # Step 3: Aspect Ratio Filtering on potentially refined masks
            candidate_indices = []
            candidate_boxes = []
            passed_ratio_filter = 0
            for i in initial_indices: # Iterate using original indices, but check refined data
                box = refined_boxes_data[i]
                deviation = aspect_ratio_deviation(box)
                if deviation <= ratio_threshold:
                    passed_ratio_filter += 1
                    candidate_indices.append(i) # Store original index
                    candidate_boxes.append(box) # Store the (potentially refined) box
            print(f"{passed_ratio_filter} masks passed aspect ratio filter (threshold: {ratio_threshold:.3f}).")

            if not candidate_indices:
                 print("No masks passed filters after potential refinement. Saving original image.")
                 cv2.imwrite(output_filename, cv2.cvtColor(image_np_rgb, cv2.COLOR_RGB2BGR))
                 print(f"Original image saved as {output_filename}")
                 return

            # --- Overlap Refinement (using potentially refined boxes) ---
            # Now operates on the initial_indices that passed Step 1
            indices_to_keep = list(range(len(initial_indices))) # Indices relative to the initial_indices list
            indices_to_discard_relative = set()

            print(f"\nApplying overlap refinement to {len(initial_indices)} candidates...")
            for i in range(len(initial_indices)):
                if i in indices_to_discard_relative: continue
                idx_i_original = initial_indices[i]
                x_i, y_i, w_i, h_i = refined_boxes_data[idx_i_original] # Use refined box
                area_i = w_i * h_i
                cx_i = x_i + w_i / 2
                cy_i = y_i + h_i / 2

                for j in range(i + 1, len(initial_indices)):
                    if j in indices_to_discard_relative: continue
                    idx_j_original = initial_indices[j]
                    x_j, y_j, w_j, h_j = refined_boxes_data[idx_j_original] # Use refined box
                    area_j = w_j * h_j
                    cx_j = x_j + w_j / 2
                    cy_j = y_j + h_j / 2

                    center_i_in_j = (x_j <= cx_i <= x_j + w_j) and (y_j <= cy_i <= y_j + h_j)
                    center_j_in_i = (x_i <= cx_j <= x_i + w_i) and (y_i <= cy_j <= y_i + h_i)

                    if center_i_in_j or center_j_in_i:
                        if area_i > area_j:
                            indices_to_discard_relative.add(i)
                            break
                        else:
                            indices_to_discard_relative.add(j)

            # Collect final masks
            kept_count = 0
            for i in indices_to_keep:
                if i not in indices_to_discard_relative:
                    original_index = initial_indices[i] # Get the original index
                    final_masks.append(refined_masks_data[original_index]) # Append the potentially refined mask data
                    kept_count += 1
            print(f"{kept_count} masks remaining after overlap refinement.")

        # --- Draw Masks ---
        if not final_masks:
            print("No masks available to draw. Saving original image.")
            cv2.imwrite(output_filename, cv2.cvtColor(image_np_rgb, cv2.COLOR_RGB2BGR))
            print(f"Original image saved as {output_filename}")
            return

        img_to_draw_on = cv2.cvtColor(image_np_rgb, cv2.COLOR_RGB2BGR) # Convert to BGR for OpenCV
        alpha = 0.5 # Transparency for filled masks

        if show_all_masks:
            print("Drawing outlines for generated masks...")
            for mask in final_masks:
                color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
                # Mask is already boolean, convert to uint8 for findContours
                mask_uint8 = mask.astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img_to_draw_on, contours, -1, color, thickness=2)
        else: # Draw filled masks (either filtered or all)
            print("Drawing filled masks...")
            overlay = img_to_draw_on.copy()
            for mask in final_masks:
                color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
                # Mask is boolean, apply color directly
                overlay[mask] = color
            cv2.addWeighted(overlay, alpha, img_to_draw_on, 1 - alpha, 0, img_to_draw_on)

        # --- Save Result ---
        save_mode = "raw outlines" if show_all_masks else "filtered"
        print(f"Saving {save_mode} segmentation result to {output_filename}...")
        cv2.imwrite(output_filename, img_to_draw_on)
        print(f"Result saved successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment image using native SAM2, filter masks, and save.")

    # --- Input/Output ---
    parser.add_argument("--image", type=str, required=True, help="Path to the input image file.")
    parser.add_argument("--output", type=str, default="segmented.png", help="Filename for the output segmented image.")

    # --- Model ---
    parser.add_argument("--checkpoint", type=str, default="checkpoints/sam2.1_hiera_large.pt", help="Path to SAM2 checkpoint (.pt).")
    parser.add_argument("--model_cfg", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml", help="Path to SAM2 model config (.yaml).")
    parser.add_argument("--device", type=str, default='auto', choices=['auto', 'cuda', 'cpu', 'mps'], help="Device to run inference on.")

    # --- Mask Generation Parameters (Defaults match SAM2AutomaticMaskGenerator) ---
    parser.add_argument("--points_per_side", type=int, default=DEFAULT_POINTS_PER_SIDE)
    parser.add_argument("--points_per_batch", type=int, default=DEFAULT_POINTS_PER_BATCH)
    parser.add_argument("--pred_iou_thresh", type=float, default=DEFAULT_PRED_IOU_THRESH)
    parser.add_argument("--stability_score_thresh", type=float, default=DEFAULT_STABILITY_SCORE_THRESH)
    parser.add_argument("--stability_score_offset", type=float, default=DEFAULT_STABILITY_SCORE_OFFSET)
    parser.add_argument("--mask_threshold", type=float, default=DEFAULT_MASK_THRESHOLD, help="Threshold for binarizing masks.") # Added
    parser.add_argument("--box_nms_thresh", type=float, default=DEFAULT_BOX_NMS_THRESH)
    parser.add_argument("--crop_n_layers", type=int, default=DEFAULT_CROP_N_LAYERS)
    parser.add_argument("--crop_nms_thresh", type=float, default=DEFAULT_CROP_NMS_THRESH, help="NMS threshold for crops.") # Added
    parser.add_argument("--crop_overlap_ratio", type=float, default=DEFAULT_CROP_OVERLAP_RATIO, help="Overlap ratio for crops.") # Added
    parser.add_argument("--crop_n_points_downscale_factor", type=int, default=DEFAULT_CROP_N_POINTS_DOWNSCALE_FACTOR)
    parser.add_argument("--min_mask_region_area", type=float, default=DEFAULT_MIN_MASK_REGION_AREA, help="Min area for postprocessing removal (requires opencv).") # Changed type hint to float
    parser.add_argument("--apply_postprocessing", action='store_true', help="Apply postprocessing to remove small holes/regions from masks (requires opencv).")
    parser.add_argument("--use_m2m", action='store_true', help="Use mask-to-mask refinement step.") # Added

    # --- Custom Filtering ---
    parser.add_argument("--ratio_threshold", type=float, default=0.15, help="Allowable absolute deviation from target aspect ratio H/W=1.397.")
    parser.add_argument("--min_area", type=int, default=DEFAULT_MIN_AREA, help=f"Minimum bounding box area (pixels) to keep a mask (default: {DEFAULT_MIN_AREA}).")

    # --- Debugging/Visualization ---
    parser.add_argument("--show_all_masks", action='store_true', help="Skip custom filtering and show all masks generated by SAM2.")

    # --- Preprocessing ---
    parser.add_argument("--apply_clahe", action='store_true', default=DEFAULT_APPLY_CLAHE, help="Apply CLAHE preprocessing to the input image.") # Added
    parser.add_argument("--remove_highlights", action='store_true', default=DEFAULT_REMOVE_HIGHLIGHTS, help="Attempt to remove specular highlights using inpainting.") # Added


    args = parser.parse_args()

    # Note: Convert min_mask_region_area back to int if needed by generator, float allows 0 default easily
    segment_image_native(
        image_path=args.image,
        output_filename=args.output,
        checkpoint_path=args.checkpoint,
        model_cfg_path=args.model_cfg,
        device_str=args.device,
        points_per_side=args.points_per_side,
        points_per_batch=args.points_per_batch,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        stability_score_offset=args.stability_score_offset,
        mask_threshold=args.mask_threshold, # Added
        box_nms_thresh=args.box_nms_thresh,
        crop_n_layers=args.crop_n_layers,
        crop_nms_thresh=args.crop_nms_thresh, # Added
        crop_overlap_ratio=args.crop_overlap_ratio, # Added
        crop_n_points_downscale_factor=args.crop_n_points_downscale_factor,
        min_mask_region_area=int(args.min_mask_region_area), # Ensure int type
        apply_postprocessing=args.apply_postprocessing,
        use_m2m=args.use_m2m, # Added
        ratio_threshold=args.ratio_threshold,
        min_area=args.min_area,
        show_all_masks=args.show_all_masks,
        apply_clahe=args.apply_clahe, # Added
        remove_highlights=args.remove_highlights # Added
    ) 
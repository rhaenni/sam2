\
import argparse
import json
import os
import random
import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as mask_util
import polyline # Added import

def draw_masks_on_image(image_path, json_path, output_path, draw_polygon_outlines: bool = False): # Added draw_polygon_outlines parameter
    """
    Draws segmentation masks or polygon outlines from a JSON file onto an image.
    Ensuring semi-transparency is correctly blended for masks.

    Args:
        image_path (str): Path to the input image.
        json_path (str): Path to the JSON file containing annotations.
        output_path (str): Path to save the output image.
        draw_polygon_outlines (bool): If True, draw polygon outlines. Otherwise, draw RLE masks.
    """
    try:
        # Load the base image
        base_pil_image = Image.open(image_path)
        # Convert to RGBA to serve as the main canvas for compositing
        canvas_rgba = base_pil_image.convert("RGBA")
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return

    try:
        # Load JSON data
        with open(json_path, 'r') as f:
            data_from_json = json.load(f) # Changed variable name for clarity
    except FileNotFoundError:
        print(f"Error: Image file not found at {json_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_path}")
        return
    except Exception as e:
        print(f"Error loading JSON file {json_path}: {e}")
        return

    if not isinstance(data_from_json, dict) or 'output' not in data_from_json:
        print(f"Error: JSON data should be a dictionary with an 'output' key. Got {type(data_from_json)}")
        return

    annotations = data_from_json['output'] # Get the list from the 'output' key

    if not isinstance(annotations, list):
        print(f"Error: The 'output' field in JSON should be a list of annotations. Got {type(annotations)}")
        return

    # Process each annotation
    for i, ann in enumerate(annotations):
        if draw_polygon_outlines:
            if 'polygon' not in ann or not isinstance(ann['polygon'], str) or not ann['polygon']:
                print(f"Warning: Annotation {i} is missing 'polygon' field, it's not a non-empty string, or flag --polygon is set. Skipping.")
                continue

            encoded_polyline_str = ann['polygon']
            try:
                # polyline.decode gives list of (lat, lon) which we've set as (y, x) in predict.py
                decoded_points_yx = polyline.decode(encoded_polyline_str)

                if len(decoded_points_yx) < 2:
                    print(f"Warning: Polygon {i} has less than 2 points after decoding. Skipping. Data: {encoded_polyline_str}")
                    continue

                # Convert to (x, y) for PIL
                # Ensure coordinates are float for precision, then int for drawing if necessary,
                # but ImageDraw.line can handle float coordinates.
                decoded_points_xy = [(float(x_coord), float(y_coord)) for y_coord, x_coord in decoded_points_yx]

                # To close the polygon for line drawing, add the first point to the end of the list
                # if it's not already closed.
                if decoded_points_xy[0] != decoded_points_xy[-1]:
                     decoded_points_xy.append(decoded_points_xy[0])
                
                if len(decoded_points_xy) < 2 : # Should not happen if initial check passed, but as safeguard
                    print(f"Warning: Polygon {i} resulted in < 2 points for drawing. Skipping.")
                    continue

                draw = ImageDraw.Draw(canvas_rgba)
                outline_color_rgba = (255, 0, 0, 255)  # Bright red, fully opaque
                outline_width = 2
                draw.line(decoded_points_xy, fill=outline_color_rgba, width=outline_width)

            except Exception as e:
                print(f"Error processing polygon for annotation {i}: {e}. Polygon data: {encoded_polyline_str}")
                continue
        else:
            # Existing RLE mask drawing logic
            if 'segmentation' not in ann:
                print(f"Warning: Annotation {i} is missing 'segmentation' field. Skipping.")
                continue

            segmentation_data = ann['segmentation']
            if not isinstance(segmentation_data, dict) or \
               'counts' not in segmentation_data or \
               'size' not in segmentation_data:
                print(f"Warning: Annotation {i} has invalid 'segmentation' format. Skipping. Details: {segmentation_data}")
                continue
            
            rle = segmentation_data

            try:
                binary_mask_np = mask_util.decode(rle)
            except Exception as e:
                print(f"Error decoding RLE for annotation {i}: {e}. Segmentation data: {rle}")
                continue

            if binary_mask_np.shape[0] != canvas_rgba.height or binary_mask_np.shape[1] != canvas_rgba.width:
                print(f"Warning: Mask {i} dimensions ({binary_mask_np.shape[0]}H x {binary_mask_np.shape[1]}W) "
                      f"do not match image dimensions ({canvas_rgba.height}H x {canvas_rgba.width}W). Skipping.")
                continue

            # Generate a random RGB color for the mask
            mask_color_rgb = (
                random.randint(70, 220),  # R
                random.randint(70, 220),  # G
                random.randint(70, 220)   # B
            )
            mask_alpha_value = 128  # Alpha for semi-transparency (0-255)
            paint_color_rgba = (*mask_color_rgb, mask_alpha_value)

            # Create a PIL 'L' mode mask for the current segment's shape
            shape_mask_pil = Image.fromarray(binary_mask_np.astype(np.uint8) * 255, mode='L')

            # Create an RGBA overlay for the current mask
            current_mask_overlay_rgba = Image.new('RGBA', canvas_rgba.size, (0, 0, 0, 0))
            
            solid_color_for_mask_shape = Image.new('RGBA', canvas_rgba.size, paint_color_rgba)
            current_mask_overlay_rgba.paste(solid_color_for_mask_shape, (0,0), shape_mask_pil)

            canvas_rgba = Image.alpha_composite(canvas_rgba, current_mask_overlay_rgba)
        
    try:
        # Save the output image
        original_ext = os.path.splitext(image_path)[1].lower()
        
        if original_ext in ['.jpg', '.jpeg']:
            # For JPEGs, convert the blended RGBA image to RGB.
            # The transparency effect is now baked into the RGB pixel values.
            final_image_to_save = canvas_rgba.convert("RGB")
            print(f"Original format is JPEG. Saving blended image as RGB to {output_path}")
            final_image_to_save.save(output_path)
        else:
            # For other formats (like PNG), save the RGBA image to preserve true alpha.
            # Ensure the output path for non-JPEGs uses .png
            base_output, _ = os.path.splitext(output_path)
            png_output_path = f"{base_output}.png"
            final_image_to_save = canvas_rgba
            print(f"Original format is not JPEG. Saving with transparency to {png_output_path}")
            final_image_to_save.save(png_output_path)
            
        print(f"Successfully saved masked image.")
    except Exception as e:
        print(f"Error saving image: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Draw COCO RLE segmentation masks or polygon outlines from a JSON file onto an image." # Updated description
    )
    parser.add_argument(
        "image_path", 
        type=str, 
        help="Path to the input image file."
    )
    parser.add_argument(
        "json_path", 
        type=str, 
        help="Path to the JSON file containing COCO RLE mask annotations or polygon data." # Updated help
    )
    parser.add_argument( # New argument
        "--polygon",
        action="store_true", # Makes it a boolean flag
        help="If set, draw polygon outlines from the 'polygon' property instead of RLE masks."
    )
    
    args = parser.parse_args()

    # Determine output path
    base, ext = os.path.splitext(args.image_path)
    output_image_path = f"{base}_masked{ext}" # Default output name

    # Adjust output name if drawing polygons to avoid overwriting mask images by default
    if args.polygon:
        output_image_path = f"{base}_polygons_outlined{ext}"


    # The actual output path (extension) will be finalized in draw_masks_on_image
    # based on whether the original was JPEG or not.
    # For JPEGs, output_image_path will be used as is.
    # For non-JPEGs, .png will be enforced.
    
    draw_masks_on_image(args.image_path, args.json_path, output_image_path, args.polygon) # Pass new polygon flag

if __name__ == "__main__":
    print("Executing script to draw COCO RLE masks or polygon outlines.") # Updated print
    print("Ensure 'Pillow', 'numpy', 'pycocotools', and 'polyline' are installed.") # Updated print
    print("  pip install Pillow numpy pycocotools polyline") # Updated print
    print("-" * 50)
    main()

"""
Merges multiple Label Studio COCO JSON exports (one per syringe run folder)
into a single unified COCO JSON file, with full resolved image paths.

"""

import json
import os

# CONFIG 
JSON_DIR = r"C:\Users\HP\Downloads\Exported JSONs"
IMAGES_ROOT = r"C:\Users\HP\Downloads\Syringe Data Collection\syringe_images\B AND W MARKS (43 syringes)"
OUTPUT_PATH = r"C:\Users\HP\Downloads\Exported JSONs\merged_annotations.json"

def strip_hash_prefix(filename):
    """Remove Label Studio's hash prefix e.g. 'e5b1d669-img_012.png' -> 'img_012.png'"""
    basename = os.path.basename(filename)
    if '-' in basename:
        return basename.split('-', 1)[1]
    return basename


def merge_coco_jsons(json_dir, images_root, output_path):
    merged = {
        "images": [],
        "annotations": [],
        "categories": [],
        "info": {
            "description": "Merged COCO annotations from all syringe run folders",
            "version": "1.0",
            "year": 2026,
            "contributor": "merge_coco_annotations.py"
        }
    }

    categories_set = False
    image_id_offset = 0
    annotation_id_offset = 0
    missing_files = []

    json_files = sorted([f for f in os.listdir(json_dir) if f.endswith('.json')])

    if not json_files:
        print(f"No JSON files found in: {json_dir}")
        return

    print(f"Found {len(json_files)} JSON files to merge...\n")

    for json_file in json_files:
        json_path = os.path.join(json_dir, json_file)

        # Run folder name is derived from the JSON filename
        # e.g. run_20260611_113754.json -> run_20260611_113754
        run_folder_name = os.path.splitext(json_file)[0]
        run_folder_path = os.path.join(images_root, run_folder_name)

        with open(json_path, 'r') as f:
            data = json.load(f)

        # Set categories from first file (assumed consistent across all)
        if not categories_set:
            merged["categories"] = data.get("categories", [])
            categories_set = True

        # Build mapping from old image_id -> new image_id
        old_to_new_image_id = {}

        for image in data.get("images", []):
            new_image_id = image["id"] + image_id_offset
            old_to_new_image_id[image["id"]] = new_image_id

            # Recover original filename by stripping Label Studio hash prefix
            original_filename = strip_hash_prefix(image["file_name"])

            # Build full resolved path to original image on disk
            full_image_path = os.path.join(run_folder_path, original_filename)

            # Warn if image doesn't exist at resolved path
            if not os.path.exists(full_image_path):
                missing_files.append(full_image_path)

            merged["images"].append({
                "id": new_image_id,
                "width": image["width"],
                "height": image["height"],
                "file_name": original_filename,       # clean original filename
                "full_path": full_image_path,         # full resolved path for DataLoader
                "run_folder": run_folder_name         # syringe identity
            })

        for annotation in data.get("annotations", []):
            new_annotation_id = annotation["id"] + annotation_id_offset
            new_image_id = old_to_new_image_id[annotation["image_id"]]

            merged["annotations"].append({
                "id": new_annotation_id,
                "image_id": new_image_id,
                "category_id": annotation["category_id"],
                "bbox": annotation["bbox"],
                "area": annotation["area"],
                "segmentation": annotation.get("segmentation", []),
                "iscrowd": annotation.get("iscrowd", 0),
                "ignore": annotation.get("ignore", 0)
            })

        num_images = len(data.get("images", []))
        num_annotations = len(data.get("annotations", []))
        print(f"  {json_file}: {num_images} images, {num_annotations} annotations")

        image_id_offset += num_images
        annotation_id_offset += len(data.get("annotations", []))

    with open(output_path, 'w') as f:
        json.dump(merged, f, indent=2)

    print(f"\nDone.")
    print(f"Total images:      {len(merged['images'])}")
    print(f"Total annotations: {len(merged['annotations'])}")
    print(f"Saved to:          {output_path}")

    if missing_files:
        print(f"\nWARNING: {len(missing_files)} image(s) not found on disk:")
        for f in missing_files:
            print(f"  {f}")
        print("Check that JSON filenames match their run folder names exactly.")


if __name__ == "__main__":
    merge_coco_jsons(JSON_DIR, IMAGES_ROOT, OUTPUT_PATH)

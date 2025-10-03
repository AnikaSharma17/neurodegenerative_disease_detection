import cv2
import os
import numpy as np

input_folder = r"D:\Neurodegenerative_Disease_Detection\alzheimer_mri_dataset"
output_folder = r"D:\Neurodegenerative_Disease_Detection\alzheimer_mask"

def process_image(img_path, save_path):
    img = cv2.imread(img_path) 
    if img is None:
        print(f"Could not read {img_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.medianBlur(gray, 5)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    _, brain_mask = cv2.threshold(enhanced, 10, 255, cv2.THRESH_BINARY)
    brain = cv2.bitwise_and(enhanced, brain_mask)

    _, lesion = cv2.threshold(enhanced, 180, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    lesion = cv2.morphologyEx(lesion, cv2.MORPH_OPEN, kernel, iterations=2)

    # Save only the lesion mask
    cv2.imwrite(save_path, lesion)

# Walk through all subfolders and process images
for root, dirs, files in os.walk(input_folder):
    for filename in files:
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            img_path = os.path.join(root, filename)

            # replicate folder structure in output
            rel_path = os.path.relpath(root, input_folder)
            save_dir = os.path.join(output_folder, rel_path)
            os.makedirs(save_dir, exist_ok=True)

            save_path = os.path.join(save_dir, "mask_" + filename)
            process_image(img_path, save_path)
            print(f"Saved: {save_path}")

import cv2
import os
import numpy as np

input_folder = r"D:\Neurodegenerative_Disease_Detection\parkinsons_mri_dataset"
output_folder = r"D:\Neurodegenerative_Disease_Detection\parkinson_mask"

os.makedirs(output_folder, exist_ok=True)

def process_image(img_path, save_path):
    # Read image
    img = cv2.imread(img_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Denoising
    denoised = cv2.medianBlur(gray, 5)

    # Contrast enhancement (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # Brain extraction (threshold + mask)
    _, brain_mask = cv2.threshold(enhanced, 10, 255, cv2.THRESH_BINARY)
    brain = cv2.bitwise_and(enhanced, brain_mask)

    # Lesion segmentation (higher threshold)
    _, lesion = cv2.threshold(enhanced, 180, 255, cv2.THRESH_BINARY)

    # Morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    lesion = cv2.morphologyEx(lesion, cv2.MORPH_OPEN, kernel, iterations=2)

    # Save results
    result = np.hstack([gray, enhanced, brain, lesion])
    cv2.imwrite(save_path, result)

# Process all images
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        img_path = os.path.join(input_folder, filename)
        save_path = os.path.join(output_folder, "processed_" + filename)
        process_image(img_path, save_path)
        print(f"Processed & saved: {save_path}")

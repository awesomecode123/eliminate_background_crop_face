import os
import dlib
import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO("yolov8x-seg.pt")  # Load an official model

# Path to the input and output folder
input_folder = "input_pictures"  # Change this to your input folder path
output_folder = "output_pictures"  # Change this to your output folder path

# Create output directory if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Initialize Dlib's face detector
detector = dlib.get_frontal_face_detector()

# Process each image in the input folder
for image_name in os.listdir(input_folder):
    image_path = os.path.join(input_folder, image_name)
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for i, face in enumerate(faces):
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())

        # Predict using YOLO model
        results = model.predict(img, classes=[0], conf=0.2, verbose=False)
        result = results[0]

        # Check if masks are available
        if result.masks:
            mask = result.masks[0].xy  # Get the mask as polygon points

            # Create a blank binary mask
            mask_image = np.zeros(img.shape[:2], dtype=np.uint8)

            # Fill the mask polygon
            cv2.fillPoly(mask_image, [np.array(mask, dtype=np.int32)], 255)

            # Create a 3-channel binary mask
            mask_3ch = cv2.merge([mask_image, mask_image, mask_image])

            # Apply mask to the image
            masked_img = cv2.bitwise_and(img, mask_3ch)

            # Define the region of interest based on the face coordinates
            ymin = int(max(y - 0.45 * h, 0))
            ymax = int(min(y + 1.3 * h, img.shape[0]))
            xmin = int(max(x - 0.3 * w, 0))
            xmax = int(min(x + 1.3 * w, img.shape[1]))

            # Crop the face region from the masked image
            cropped_face = masked_img[ymin:ymax, xmin:xmax]

            if cropped_face is not None:
                output_path = os.path.join(output_folder, f"{i}_{image_name}")
                cv2.imwrite(output_path, cropped_face)
                print(f"Cropped face saved to {output_path}")
            else:
                print(f"No face detected in {image_path}")
        else:
            print(f"No mask found in {image_path}")

print("Processing complete.")

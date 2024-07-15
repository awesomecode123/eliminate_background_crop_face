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

            # Create an alpha channel from the mask
            alpha_channel = np.where(mask_image == 255, 255, 0).astype(np.uint8)

            # Merge the original image with the alpha channel
            b, g, r = cv2.split(img)
            img_with_alpha = cv2.merge([b, g, r, alpha_channel])

            # Define the region of interest based on the face coordinates
            ymin = int(max(y - 0.45 * h, 0))
            ymax = int(min(y + 1.3 * h, img.shape[0]))
            xmin = int(max(x - 0.3 * w, 0))
            xmax = int(min(x + 1.3 * w, img.shape[1]))

            # Crop the face region from the image with alpha channel
            cropped_face = img_with_alpha[ymin:ymax, xmin:xmax]

            if cropped_face is not None:
                output_path = os.path.join(output_folder, f"{os.path.splitext(image_name)[0]}_{i}.png")
                cv2.imwrite(output_path, cropped_face)
                print(f"Cropped face saved to {output_path}")
            else:
                print(f"No face detected in {image_path}")
        else:
            print(f"No mask found in {image_path}")

print("Processing complete.")

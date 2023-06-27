import cv2
import numpy as np
import fitz
import os

'''Visual Representation of the rgb difference between the two pdfs'''

# Specify the paths of the original and printed/scanned PDFs
original_pdf_path = "Comparison/originalpg12.pdf"
printed_scanned_pdf_path = "Comparison/printscannedpg12.pdf"

# Create a directory to store the comparison images
comparison_directory = "Comparison/Comparison_Images"
if not os.path.exists(comparison_directory):
    os.makedirs(comparison_directory)

def compare_bubbles(original_pdf_path, printed_scanned_pdf_path):
    original_images = convert_pdf_to_images(original_pdf_path)
    printed_scanned_images = convert_pdf_to_images(printed_scanned_pdf_path)

    for i in range(len(original_images)):
        original_image = original_images[i].astype(np.uint8)
        printed_scanned_image = printed_scanned_images[i].astype(np.uint8)

        # Ensure that the original and printed/scanned images have the same dimensions
        if original_image.shape != printed_scanned_image.shape:
            raise ValueError("Images have different dimensions.")

        # Calculate the pixel-wise difference between the original and printed/scanned images
        difference = np.abs(original_image - printed_scanned_image)

        # Convert the difference image to grayscale
        difference_gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)

        # Threshold the grayscale difference image to create a binary mask
        _, threshold = cv2.threshold(difference_gray, 0, 255, cv2.THRESH_BINARY)

        # Save the comparison image
        comparison_image_path = os.path.join(comparison_directory, f"comparison_{i}.png")
        cv2.imwrite(comparison_image_path, threshold)

    print("Bubbles comparison completed.")

def convert_pdf_to_images(pdf_path):
    images = []
    with fitz.open(pdf_path) as doc:
        for i in range(len(doc)):
            page = doc.load_page(i)
            pix = page.get_pixmap()
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            images.append(img)
    return images

# Call the compare_bubbles function
compare_bubbles(original_pdf_path, printed_scanned_pdf_path)

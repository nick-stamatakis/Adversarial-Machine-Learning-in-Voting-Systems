import cv2
import numpy as np
import fitz
import os
import matplotlib.pyplot as plt
import PyPDF2


def align_images(reference_image, image):
    # Convert images to grayscale
    reference_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find features and match keypoints
    orb = cv2.ORB_create()
    keypoints_reference, descriptors_reference = orb.detectAndCompute(reference_gray, None)
    keypoints_image, descriptors_image = orb.detectAndCompute(image_gray, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors_reference, descriptors_image)
    matches = sorted(matches, key=lambda x: x.distance)

    # Select top matches for alignment
    num_matches = 30
    selected_matches = matches[:num_matches]

    # Extract corresponding keypoints
    points_reference = np.float32([keypoints_reference[m.queryIdx].pt for m in selected_matches]).reshape(-1, 1, 2)
    points_image = np.float32([keypoints_image[m.trainIdx].pt for m in selected_matches]).reshape(-1, 1, 2)

    # Estimate affine transformation
    transformation_matrix, _ = cv2.estimateAffinePartial2D(points_image, points_reference)

    # Apply affine transformation to align image with reference image
    aligned_image = cv2.warpAffine(image, transformation_matrix, (reference_image.shape[1], reference_image.shape[0]))

    return aligned_image


def convert_pdf_to_png(pdf_path, output_dir, reference_page):
    # Load the PDF files
    doc_reference = fitz.open(reference_page)
    doc = fitz.open(pdf_path)

    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)

    # Render the reference page as an image
    pix_reference = doc_reference.load_page(0).get_pixmap()

    # Convert pixmap to numpy array
    reference_image = np.frombuffer(pix_reference.samples, dtype=np.uint8).reshape(pix_reference.h, pix_reference.w, pix_reference.n)

    # Save the reference image as PNG
    reference_image_path = os.path.join(output_dir, "reference.png")
    cv2.imwrite(reference_image_path, reference_image)

    print("Reference image saved as PNG: " + reference_image_path)

    for i, page in enumerate(doc.pages(), start=1):
        # Render the page as an image
        pix = page.get_pixmap()

        # Convert pixmap to numpy array
        image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)

        # Save the image as PNG
        image_path = os.path.join(output_dir, f"page_{i}.png")
        cv2.imwrite(image_path, image)

        print(f"Page {i} saved as PNG: {image_path}")

    print("PDF conversion to PNG completed successfully.")


def perform_image_alignment(pdf_path, output_dir, reference_page):
    # Convert the PDF to PNG images and extract the reference image
    convert_pdf_to_png(pdf_path, output_dir, reference_page)

    # Load the PNG images and perform image alignment
    png_files = sorted([f for f in os.listdir(output_dir) if f.endswith(".png")])

    # Load the reference image
    reference_image_path = os.path.join(output_dir, "reference.png")
    reference_image = cv2.imread(reference_image_path)

    # Register the reference image
    registered_image = reference_image.copy()

    # Save the registered reference image
    registered_image_path = os.path.join(output_dir, "registered_reference.png")
    cv2.imwrite(registered_image_path, registered_image)

    print("Registered reference image created.")

    pixel_diffs = []

    for i, png_file in enumerate(png_files, start=1):
        if png_file != "reference.png":
            # Load the image to be aligned
            page_image_path = os.path.join(output_dir, png_file)
            page_image = cv2.imread(page_image_path)

            # Align the images
            aligned_image = align_images(reference_image, page_image)

            # Calculate the difference in pixel values
            pixel_diff = np.abs(aligned_image.astype(np.int16) - reference_image.astype(np.int16))
            pixel_diffs.append(np.mean(pixel_diff))

            # Save the aligned image
            aligned_image_path = os.path.join(output_dir, f"aligned_page_{i}.png")
            cv2.imwrite(aligned_image_path, aligned_image)

            print(f"Aligned image created for page {i}.")

            if i == 1:
                # Calculate the difference image
                diff_image = np.abs(page_image.astype(np.int16) - aligned_image.astype(np.int16))

                # Normalize the difference image to the range [0, 1]
                diff_image_normalized = cv2.normalize(diff_image, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

                # Create a heatmap representation
                plt.imshow(diff_image_normalized, cmap='hot', interpolation='nearest')
                plt.colorbar()
                plt.title("Difference Heatmap for Page 1")
                plt.show()

                # Save the difference image
                diff_image_path = os.path.join(output_dir, "difference_page_1.png")
                cv2.imwrite(diff_image_path, diff_image)

                print("Difference image created for page 1.")

    print("Image alignment completed successfully.")

    # Create a graph of pixel value differences
    plt.plot(range(1, len(pixel_diffs) + 1), pixel_diffs)
    plt.xlabel("Page Number")
    plt.ylabel("Pixel Value Difference")
    plt.title("Difference in Pixel Values between Aligned Images and Reference Image")
    plt.show()


# Specify the PDF path and output directory
pdf_path = "Comparison/printscannedpg12.pdf"
output_dir = "image_alignment_output"
reference_page = "Comparison/originalpg12.pdf"

# Perform image alignment and create the graph
perform_image_alignment(pdf_path, output_dir, reference_page)


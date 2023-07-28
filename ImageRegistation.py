import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def align_images(reference_image, image):
    # Find features and match keypoints
    sift = cv2.SIFT_create()
    keypoints_reference, descriptors_reference = sift.detectAndCompute(reference_image, None)
    keypoints_image, descriptors_image = sift.detectAndCompute(image, None)

    # Match keypoints using Brute-Force matcher
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(descriptors_reference, descriptors_image, k=2)

    # Apply ratio test to select good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 4:
        raise ValueError("Insufficient matches found for alignment.")

    # Extract corresponding keypoints
    points_reference = np.float32([keypoints_reference[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    points_image = np.float32([keypoints_image[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Estimate perspective transformation
    transformation_matrix, _ = cv2.findHomography(points_image, points_reference, cv2.RANSAC, 5.0)

    # Warp the image to align with the reference image
    aligned_image = cv2.warpPerspective(image, transformation_matrix,
                                        (reference_image.shape[1], reference_image.shape[0]))

    return aligned_image


def align_png_images(output_dir, reference_image_path):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the reference image
    reference_image = cv2.imread(reference_image_path)

    # Register the reference image
    registered_image = reference_image.copy()

    # Save the registered reference image
    registered_image_path = os.path.join(output_dir, "registered_reference.png")
    cv2.imwrite(registered_image_path, registered_image)

    print("Registered reference image created.")

    png_files = sorted([f for f in os.listdir(output_dir) if f.endswith(".png") and f != "reference.png"])
    pixel_diffs = []

    for i, png_file in enumerate(png_files, start=1):
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

            # Normalize the difference image to the range [0, 255]
            diff_image_normalized = cv2.normalize(diff_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            # Create a heatmap representation
            plt.imshow(diff_image_normalized, cmap='hot', interpolation='nearest')
            plt.colorbar()
            plt.title("Difference Heatmap for Page 1")
            plt.show()

            # Save the difference image
            diff_image_path = os.path.join(output_dir, "difference_page_1.png")
            cv2.imwrite(diff_image_path, diff_image_normalized)

            print("Difference image created for page 1.")

    print("Image alignment completed successfully.")

    # Create a graph of pixel value differences
    plt.plot(range(1, len(pixel_diffs) + 1), pixel_diffs)
    plt.xlabel("Page Number")
    plt.ylabel("Pixel Value Difference")
    plt.title("Difference in Pixel Values between Aligned Images and Reference Image")
    plt.show()


def align_and_save_image(input_image_path, reference_image_path, output_dir):
    # Load the input image
    input_image = cv2.imread(input_image_path)

    # Load the reference image
    reference_image = cv2.imread(reference_image_path)

    # Align the images
    aligned_image = align_images(reference_image, input_image)

    # Save the aligned image
    aligned_image_path = os.path.join(output_dir, os.path.basename(input_image_path))
    cv2.imwrite(aligned_image_path, aligned_image)

    print(f"Aligned image saved: {aligned_image_path}")


for i in range(1,12):
    # Specify the input PNG image path, reference image path, and output directory
    input_image_path = f"PostVALEX/PostVALEX_P{i}.png"
    reference_image_path = f"PreImagesVAL/output_sheet_{i}.png"
    output_dir = "AlignedImagesVAL"

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Align and save the input image
    align_and_save_image(input_image_path, reference_image_path, output_dir)

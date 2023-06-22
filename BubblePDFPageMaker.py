import os
import shutil
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from skimage.io import imread, imsave
from skimage.transform import resize
import numpy as np

# Specify the directory containing the saved bubble images
bubble_directory = "bubble_images"
# Specify the number of bubbles to include per PDF page
bubbles_per_pdf_page = 200

def create_pdf_pages(bubble_dir, bubbles_per_page):
    bubble_files = sorted([f for f in os.listdir(bubble_dir) if f.endswith(".png")])
    total_bubbles = len(bubble_files)

    total_pages = total_bubbles // bubbles_per_page + 1

    pdf_dir = "pdf_pages"

    # Remove the existing directory if it exists
    if os.path.exists(pdf_dir):
        shutil.rmtree(pdf_dir)

    # Create a new directory
    os.makedirs(pdf_dir)

    pdf_paths = []  # List to store the paths of created PDF pages

    for page in range(total_pages):
        start_idx = page * bubbles_per_page
        end_idx = min(start_idx + bubbles_per_page, total_bubbles)
        images = []

        for idx in range(start_idx, end_idx):
            image_path = os.path.join(bubble_dir, bubble_files[idx])
            image = imread(image_path)
            resized_image = resize(image, (40, 50), anti_aliasing=True)  # Resize the image to approximately 40x50 pixels
            resized_image = (255 * resized_image).astype(np.uint8)  # Convert to uint8 data type
            resized_image_path = f"resized_{idx}.png"  # Save the resized image
            imsave(resized_image_path, resized_image)
            images.append(resized_image_path)

        pdf_path = os.path.join(pdf_dir, f"page_{page + 1}.pdf")
        c = canvas.Canvas(pdf_path, pagesize=letter)

        x_offset = 72  # 1 inch margin
        y_offset = 72  # 1 inch margin
        image_width = (letter[0] - 2 * x_offset) / 10
        image_height = (letter[1] - 2 * y_offset) / (bubbles_per_page / 10)

        x_separation = (letter[0] - 2 * x_offset - 10 * image_width) / 9  # Calculate separation between bubbles in the x direction
        y_separation = (letter[1] - 2 * y_offset - (bubbles_per_page // 10) * image_height) / (bubbles_per_page // 10 - 1)  # Calculate separation between bubbles in the y direction

        for i, image_path in enumerate(images):
            row = i // 10
            col = i % 10

            x = x_offset + col * (image_width + x_separation)
            y = y_offset + (bubbles_per_page // 10 - row - 1) * (image_height + y_separation)

            c.drawImage(image_path, x, y, width=image_width, height=image_height)

        c.save()

        pdf_paths.append(pdf_path)  # Store the path of the created PDF page

        # Remove the temporary resized images
        for image_path in images:
            os.remove(image_path)

    print(f"PDF pages created successfully in the '{pdf_dir}' directory.")

    return pdf_paths

pdf_pages = create_pdf_pages(bubble_directory, bubbles_per_pdf_page)

for pdf_path in pdf_pages:
    print(f"PDF page: {pdf_path}")

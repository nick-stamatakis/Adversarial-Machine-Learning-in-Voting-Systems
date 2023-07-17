import os
import shutil
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from skimage.io import imread, imsave
from skimage.transform import resize
import numpy as np

'''Creates a pdf of png images from the 'ExtractBubblesFromHDF5.py' file'''

# Specify the directory containing the saved bubble images
bubble_directory = "BalCombined_DenseNet_0.031PGD_Adv_Images SECOND FIFTY"

# Specify the number of bubbles to include per PDF page
bubbles_per_pdf_page = 200

def create_pdf_pages(bubble_dir, bubbles_per_page):
    bubble_files = sorted([f for f in os.listdir(bubble_dir) if f.endswith(".png")])
    total_bubbles = len(bubble_files)

    total_pages = total_bubbles // bubbles_per_page + 1

    pdf_dir = "PreImages"

    # # Remove the existing directory if it exists
    # if os.path.exists(pdf_dir):
    #     shutil.rmtree(pdf_dir)
    #
    # # Create a new directory
    # os.makedirs(pdf_dir)

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
        grid_width = 10  # Number of columns in the grid
        grid_height = 20  # Number of rows in the grid

        image_width = (letter[0] - 2 * x_offset) / grid_width
        image_height = (letter[1] - 2 * y_offset) / grid_height

        x_separation = (letter[0] - 2 * x_offset - grid_width * image_width) / (grid_width - 1)  # Calculate separation between bubbles in the x direction
        y_separation = (letter[1] - 2 * y_offset - grid_height * image_height) / (grid_height - 1)  # Calculate separation between bubbles in the y direction
        print(str(x_separation) + " " + str(y_separation))
        bubble_index = 0  # Index to keep track of the bubbles

        for row in range(grid_height):
            for col in range(grid_width):
                x = x_offset + col * (image_width + x_separation)
                y = y_offset + (grid_height - row - 1) * (image_height + y_separation)

                if row % 2 == 0 and col % 2 == 0 and bubble_index < len(images):
                    # Display the bubble image if there shouldn't be whitespace
                    c.drawImage(images[bubble_index], x, y, width=image_width, height=image_height)
                    bubble_index += 1
                else:
                    # Display white rectangle for whitespace or alternate rows
                    c.setFillColorRGB(1, 1, 1)  # Set fill color to white
                    c.rect(x, y, image_width, image_height, fill=1)

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
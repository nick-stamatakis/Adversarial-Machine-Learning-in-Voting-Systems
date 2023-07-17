import os
import shutil
from PIL import Image

bubbles_per_png_page = 200
x_separation = 0
y_separation = 0

def extract_bubbles_from_png(png_path, output_directory):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Open the PNG image
    with Image.open(png_path) as image:
        image_width, image_height = image.size

        x_offset = 72  # 1 inch margin
        y_offset = 72  # 1 inch margin
        bubble_width = (image_width - 2 * x_offset) / 10
        bubble_height = (image_height - 2 * y_offset) / (bubbles_per_png_page / 10)

        # Iterate through the bubbles in the image
        for bubble_num in range(bubbles_per_png_page):
            row = bubble_num // 10
            col = bubble_num % 10

            x = x_offset + col * (bubble_width + x_separation)
            y = y_offset + (bubbles_per_png_page // 10 - row - 1) * (bubble_height + y_separation)

            # Extract the bubble from the image
            bubble = image.crop((x, y, x + bubble_width, y + bubble_height))

            # Save the bubble as a PNG file
            bubble_filename = f"bubble_{os.path.splitext(os.path.basename(png_path))[0]}_{bubble_num}.png"
            bubble_path = os.path.join(output_directory, bubble_filename)
            bubble.save(bubble_path)

            print(f"Extracted bubble: {bubble_path}")

# Specify the directory containing the PNG images
png_directory = "alignedPngs"

# Specify the output directory for the extracted bubbles
output_directory = "EXTRACTED AlignedDenseNet_Val_SECOND FIFTY"

# Remove the existing output directory if it exists
if os.path.exists(output_directory):
    shutil.rmtree(output_directory)

# Create a new output directory
os.makedirs(output_directory)

# Iterate through the PNG images
for png_file in os.listdir(png_directory):
    png_path = os.path.join(png_directory, png_file)
    extract_bubbles_from_png(png_path, output_directory)

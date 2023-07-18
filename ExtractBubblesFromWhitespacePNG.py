import os
from PIL import Image

# Open the output sheet image
output_sheet_path = "AlignedImages/AlignedDenseNet_Adv_P2.png"
output_sheet = Image.open(output_sheet_path)

# Constants for bubble layout
PAGE_WIDTH = int(8.5 * 200)  # 8.5 inches converted to pixels at 200 dpi
PAGE_HEIGHT = int(11 * 200)  # 11 inches converted to pixels at 200 dpi
MARGIN = int(1 * 200)  # 1 inch margin converted to pixels at 200 dpi
BUBBLE_WIDTH = 50  # Bubble width in pixels
BUBBLE_HEIGHT = 40  # Bubble height in pixels
SPACING = int(0.5 * 200)  # 0.5 inch spacing converted to pixels at 200 dpi

# Calculate the number of bubbles that can fit on the sheet
num_columns = int((PAGE_WIDTH - 2 * MARGIN + SPACING) / (BUBBLE_WIDTH + SPACING))
num_rows = int((PAGE_HEIGHT - 2 * MARGIN + SPACING) / (BUBBLE_HEIGHT + SPACING))
total_bubbles = num_columns * num_rows

# Directory to save the extracted bubbles
extracted_bubbles_directory = "DenseNet_Adv_ExtractedBubbles"

# Create the directory if it doesn't exist
os.makedirs(extracted_bubbles_directory, exist_ok=True)

# Get the number of existing bubbles in the directory
existing_bubbles = len(os.listdir(extracted_bubbles_directory))

# Extract the individual bubbles from the sheet
x = MARGIN
y = MARGIN

for row in range(num_rows):
    for col in range(num_columns):
        bubble_box = (x, y, x + BUBBLE_WIDTH, y + BUBBLE_HEIGHT)
        bubble = output_sheet.crop(bubble_box)

        # Save the extracted bubble
        bubble_index = existing_bubbles + row * num_columns + col + 1
        bubble_filename = f"bubble_{bubble_index}.png"
        bubble_path = os.path.join(extracted_bubbles_directory, bubble_filename)
        bubble.save(bubble_path)

        x += BUBBLE_WIDTH + SPACING

    # Move to the next row
    x = MARGIN
    y += BUBBLE_HEIGHT + SPACING

print("Individual bubbles have been extracted and saved in the 'ExtractedBubbles' directory.")

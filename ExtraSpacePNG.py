import os
from PIL import Image, ImageDraw

# Constants for sheet layout
PAGE_WIDTH = int(8.5 * 200)  # 8.5 inches converted to pixels at 200 dpi
PAGE_HEIGHT = int(11 * 200)  # 11 inches converted to pixels at 200 dpi
MARGIN = int(1 * 200)  # 1 inch margin converted to pixels at 200 dpi
SPACING = int(0.5 * 200)  # 0.5 inch spacing converted to pixels at 200 dpi
BUBBLE_WIDTH = 50  # Bubble width in pixels
BUBBLE_HEIGHT = 40  # Bubble height in pixels

# Directory containing the bubble PNGs
bubble_directory = "Printing_Val_vs_Adv/BalCombined_DenseNet_PGD_Adv_Images"

# Create the PreImages directory if it doesn't exist
os.makedirs("PreImages", exist_ok=True)

# Get the list of PNG files in the bubble directory
png_files = [file for file in os.listdir(bubble_directory) if file.endswith(".png")]

# Calculate the number of bubbles that can fit on the sheet
num_columns = int((PAGE_WIDTH - 2 * MARGIN + SPACING) / (BUBBLE_WIDTH + SPACING))
num_rows = int((PAGE_HEIGHT - 2 * MARGIN + SPACING) / (BUBBLE_HEIGHT + SPACING))
total_bubbles = num_columns * num_rows

# Iterate over the PNG files and place them on the sheets
sheet_index = 1
bubbles_placed = 0

while bubbles_placed < len(png_files):
    # Create a new blank sheet
    sheet = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), "white")
    draw = ImageDraw.Draw(sheet)

    # Place the bubbles on the sheet
    x = MARGIN
    y = MARGIN

    for index in range(bubbles_placed, min(bubbles_placed + total_bubbles, len(png_files))):
        if index % num_columns == 0 and index != 0:
            # Move to the next row
            x = MARGIN
            y += BUBBLE_HEIGHT + SPACING

        bubble_file = png_files[index]
        bubble_path = os.path.join(bubble_directory, bubble_file)
        bubble = Image.open(bubble_path)

        sheet.paste(bubble, (x, y))

        x += BUBBLE_WIDTH + SPACING

    # Save the sheet in the PreImages directory
    output_path = f"PreImages/output_sheet_{sheet_index}.png"
    sheet.save(output_path)

    # Update the counters
    sheet_index += 1
    bubbles_placed += total_bubbles

print(f"{sheet_index - 1} sheets have been generated in the PreImages directory.")

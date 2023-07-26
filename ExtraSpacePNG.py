import os
from PIL import Image, ImageDraw
import re

# Constants for sheet layout
PAGE_WIDTH = int(8.5 * 200)  # 8.5 inches converted to pixels at 200 dpi
PAGE_HEIGHT = int(11 * 200)  # 11 inches converted to pixels at 200 dpi
MARGIN = int(1 * 200)  # 1 inch margin converted to pixels at 200 dpi
SPACING = int(0.5 * 200)  # 0.5 inch spacing converted to pixels at 200 dpi
BUBBLE_WIDTH = 50  # Bubble width in pixels
BUBBLE_HEIGHT = 40  # Bubble height in pixels

# Directory containing the bubble PNGs
bubble_directory = "Printed_Val_vs_Adv/BalCombined_SimpleCNN_PGD_Val_Images"

# Create the Val directory if it doesn't exist
os.makedirs("PreImagesSimpleCNN_Val", exist_ok=True)

# Function to extract batch_index and example_index from the filename
def get_batch_and_example_indices(filename):
    pattern = r"__(\d+)th Batch (\d+)th Example__(Vote|Non-Vote).png"
    match = re.match(pattern, filename)

    if match:
        batch_index = int(match.group(1))
        example_index = int(match.group(2))
        return batch_index, example_index
    else:
        raise ValueError(f"Invalid filename format: {filename}")


# Get the list of PNG files in the bubble directory, sort them based on batch and example indices
png_files = [file for file in os.listdir(bubble_directory) if file.endswith(".png")]
png_files.sort(key=get_batch_and_example_indices)

# Calculate the number of bubbles that can fit on each sheet
available_width = PAGE_WIDTH - 2 * MARGIN
available_height = PAGE_HEIGHT - 2 * MARGIN
num_columns = available_width // (BUBBLE_WIDTH + SPACING)
num_rows = available_height // (BUBBLE_HEIGHT + SPACING)
total_bubbles_per_sheet = num_columns * num_rows

# Initialize counters
sheet_index = 1
bubbles_placed = 0

# Initialize an empty list to store the labels (0 for "Non-Vote", 1 for "Vote")
labels = []

# Function to extract batch_index, example_index, and label from the filename
def get_batch_example_and_label(filename):
    pattern = r"__(\d+)th Batch (\d+)th Example__(Vote|Non-Vote).png"
    match = re.match(pattern, filename)

    if match:
        batch_index = int(match.group(1))
        example_index = int(match.group(2))
        label = match.group(3)
        return batch_index, example_index, label
    else:
        # Return default values in case the label is not present in the filename
        return -1, -1, "Unknown"

while bubbles_placed < len(png_files):
    # Create a new blank sheet
    sheet = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), "white")
    draw = ImageDraw.Draw(sheet)

    # Place the bubbles on the sheet
    x = MARGIN
    y = MARGIN

    for index in range(bubbles_placed, min(bubbles_placed + total_bubbles_per_sheet, len(png_files))):
        if (index - bubbles_placed) % num_columns == 0 and index != bubbles_placed:
            # Move to the next row
            x = MARGIN
            y += BUBBLE_HEIGHT + SPACING

        bubble_file = png_files[index]
        bubble_path = os.path.join(bubble_directory, bubble_file)
        bubble = Image.open(bubble_path)

        sheet.paste(bubble, (x, y))

        # Get the label from the filename and add it to the labels list
        _, _, label = get_batch_example_and_label(bubble_file)
        label_value = 0 if label == "Vote" else 1
        labels.append(label_value)

        x += BUBBLE_WIDTH + SPACING

    # Save the sheet in the PreImagesSimpleCNN_Val directory
    output_path = f"PreImagesSimpleCNN_Val/output_sheet_{sheet_index}.png"
    sheet.save(output_path)

    # Update the counters
    sheet_index += 1
    bubbles_placed += total_bubbles_per_sheet

print(f"{sheet_index - 1} sheets have been generated in the PreImagesSimpleCNN_Val directory.")
print("Labels array:", labels)


# Adversarial example pattern code
# import os
# from PIL import Image, ImageDraw
# import re
# 
# # Constants for sheet layout
# PAGE_WIDTH = int(8.5 * 200)  # 8.5 inches converted to pixels at 200 dpi
# PAGE_HEIGHT = int(11 * 200)  # 11 inches converted to pixels at 200 dpi
# MARGIN = int(1 * 200)  # 1 inch margin converted to pixels at 200 dpi
# SPACING = int(0.5 * 200)  # 0.5 inch spacing converted to pixels at 200 dpi
# BUBBLE_WIDTH = 50  # Bubble width in pixels
# BUBBLE_HEIGHT = 40  # Bubble height in pixels
# 
# # Directory containing the bubble PNGs
# bubble_directory = "Printed_Val_vs_Adv/BalCombined_SimpleCNN_PGD_Adv_Images"
# 
# # Create the Val directory if it doesn't exist
# os.makedirs("PreImagesSimpleCNN_Adv", exist_ok=True)
# 
# # Function to extract batch_index, example_index, correct_or_misclassified, and vote_or_non_vote from the filename
# def get_batch_example_and_labels(filename):
#     pattern = r"(\d+)th Batch (\d+)th Example__(Correct|Misclassified)_(Vote|Non-Vote).png"
#     match = re.match(pattern, filename)
# 
#     if match:
#         batch_index = int(match.group(1))
#         example_index = int(match.group(2))
#         correct_or_misclassified = match.group(3)
#         vote_or_non_vote = match.group(4)
#         return batch_index, example_index, correct_or_misclassified, vote_or_non_vote
#     else:
#         # Return default values in case the labels are not present in the filename
#         return -1, -1, "Unknown", "Unknown"
# 
# # Get the list of PNG files in the bubble directory, sort them based on batch and example indices
# png_files = [file for file in os.listdir(bubble_directory) if file.endswith(".png")]
# png_files.sort(key=get_batch_example_and_labels)
# 
# # Calculate the number of bubbles that can fit on each sheet
# available_width = PAGE_WIDTH - 2 * MARGIN
# available_height = PAGE_HEIGHT - 2 * MARGIN
# num_columns = available_width // (BUBBLE_WIDTH + SPACING)
# num_rows = available_height // (BUBBLE_HEIGHT + SPACING)
# total_bubbles_per_sheet = num_columns * num_rows
# 
# # Initialize counters
# sheet_index = 1
# bubbles_placed = 0
# 
# # Initialize an empty list to store the labels (0 for "Non-Vote", 1 for "Vote")
# correct_or_misclassified_labels = []
# vote_or_non_vote_labels = []

# print(f"{sheet_index - 1} sheets have been generated in the PreImagesSimpleCNN_Val directory.")
# print("Labels array:", labels)

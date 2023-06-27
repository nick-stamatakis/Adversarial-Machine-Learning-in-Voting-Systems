import fitz
import os
from PIL import Image

'''Extracts pngs from pdf created in 'BubblePDFPagemaker.py' '''

def extract_bubbles_with_positions(pdf_path, bubbles_per_page, image_width, image_height, x_offset, y_offset,
                                   x_separation, y_separation):
    doc = fitz.open(pdf_path)
    bubble_images = []

    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        for i in range(bubbles_per_page):
            row = i // 10
            col = i % 10

            x = x_offset + col * (image_width + x_separation)
            y = y_offset + (bubbles_per_page // 10 - row - 1) * (image_height + y_separation)

            bubble_image = img.crop((x, y, x + image_width, y + image_height))
            bubble_images.append((bubble_image, (x, y)))

    return bubble_images


# Parameters for original PDF
original_pdf_path = "Comparison/originalpg12.pdf"
original_output_dir = "original_extracted_bubbles"
original_bubbles_per_pdf_page = 200
original_image_width = 46.8
original_image_height = 32.4
original_x_offset = 72
original_y_offset = 72
original_x_separation = 0
original_y_separation = 0

# Parameters for print-scanned PDF
printScanned_pdf_path = "Comparison/printscannedpg12.pdf"
printScanned_output_dir = "printScanned_extracted_bubbles"
printScanned_bubbles_per_pdf_page = 200
printScanned_image_width = 46.8
printScanned_image_height = 32.4
printScanned_x_offset = 72
printScanned_y_offset = 72
printScanned_x_separation = 0
printScanned_y_separation = 0

# Extract bubbles from original PDF
original_extracted_bubbles = extract_bubbles_with_positions(original_pdf_path, original_bubbles_per_pdf_page, original_image_width, original_image_height, original_x_offset, original_y_offset, original_x_separation, original_y_separation)

# Create a directory to store the extracted bubble images from the original PDF
os.makedirs(original_output_dir, exist_ok=True)

# Access individual bubble images and positions from the original PDF
for i, (bubble_image, position) in enumerate(original_extracted_bubbles):
    output_path = os.path.join(original_output_dir, f"extracted_bubble_{i}.png")
    bubble_image.save(output_path)
    print(f"Extracted bubble {i+1} from original PDF at position: {position}")

# Extract bubbles from print-scanned PDF
printScanned_extracted_bubbles = extract_bubbles_with_positions(printScanned_pdf_path, printScanned_bubbles_per_pdf_page, printScanned_image_width, printScanned_image_height, printScanned_x_offset, printScanned_y_offset, printScanned_x_separation, printScanned_y_separation)

# Create a directory to store the extracted bubble images from the print-scanned PDF
os.makedirs(printScanned_output_dir, exist_ok=True)

# Access individual bubble images and positions from the print-scanned PDF
for i, (bubble_image, position) in enumerate(printScanned_extracted_bubbles):
    output_path = os.path.join(printScanned_output_dir, f"extracted_bubble_{i}.png")
    bubble_image.save(output_path)
    print(f"Extracted bubble {i+1} from print-scanned PDF at position: {position}")

print("Bubbles extracted successfully.")

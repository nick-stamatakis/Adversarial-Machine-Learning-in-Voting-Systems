from PIL import Image

def calculate_image_dimensions(image_path):
    with Image.open(image_path) as image:
        width, height = image.size
        return width, height

# Example usage
image_path = "DenseNet_Adv_ExtractedBubbles/bubble_5.png"
width, height = calculate_image_dimensions(image_path)
print(f"Image dimensions: {width} x {height} pixels")

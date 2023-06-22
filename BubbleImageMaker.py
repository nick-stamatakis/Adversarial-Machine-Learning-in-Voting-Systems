import os
import urllib.request
import h5py
import numpy as np
from PIL import Image

num_batches = 1  # Total batches is 1 (batch 94 only)
inf_color_model = [[[[[] for k in range(3)] for x in range(num_batches)] for i in range(3)] for y in range(2)]
image_data = [[[[] for x in range(num_batches)] for i in range(3)] for y in range(2)]

print('Downloading datafile')
if not os.path.isfile("data_Blank_Vote_Questionable.h5"):
    urllib.request.urlretrieve("http://puf-data.engr.uconn.edu/data/data_Blank_Vote_Questionable.h5",
                               "data_Blank_Vote_Questionable.h5")

f = h5py.File("data_Blank_Vote_Questionable.h5", "r")

batches = [94]  # Specify the batch number(s) you want to read
rgb = ['r', 'g', 'b']
dset = ['COLOR', 'POSITIONAL']
dset_type = ['VOTE', 'BLANK', 'QUESTIONABLE']
X = []
Y = []

'''
    Reading in the entire dataset
'''

print("Reading entire dataset")
for c_1, d in enumerate(dset):
    for c_2, d_t in enumerate(dset_type):
        for c_3, b in enumerate(batches):
            if d_t in f[d] and str(b) in f[d][d_t]:
                image_data[c_1][c_2][c_3].extend(list(f[d][d_t][str(b)][:]))
                for c_4, r in enumerate(rgb):
                    if d_t in f['INFORMATION'][d] and str(b) + str(r) in f['INFORMATION'][d][d_t]:
                        inf_color_model[c_1][c_2][c_3][c_4].extend(list(f['INFORMATION'][d][d_t][str(b) + str(r)][:]))

'''
    Collapses an RGB 3-dimensional image to 1D using color model coefficients
'''


def collapse(img, b_r, b_g, b_b):
    collapsed_img = np.zeros(shape=(40, 50), dtype=np.float32)

    img_copy = np.copy(img)  # Create a copy of the image to avoid modifying the original image

    img_copy[:, :, 0] = img[:, :, 0] * (0.02383815) + (b_r * (-0.01898671))
    img_copy[:, :, 1] = img[:, :, 1] * (0.00010994) + (b_g * (-0.001739))
    img_copy[:, :, 2] = img[:, :, 2] * (0.00178155) + (b_b * (-0.00044142))

    collapsed_img = np.sum(img_copy, axis=2)

    return collapsed_img, img_copy


def save_bubble_images():
    save_dir = "bubble_images"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        # Clear existing images in the directory
        file_list = [f for f in os.listdir(save_dir) if f.endswith(".png")]
        for f in file_list:
            os.remove(os.path.join(save_dir, f))

    for x in range(len(dset_type) - 1):
        for y in range(len(batches)):
            for k in range(len(inf_color_model[1][x][y][0])):
                img = np.array(image_data[1][x][y][k], dtype=np.float32)
                img = img[:, :, ::-1]

                b_r = inf_color_model[1][x][y][0][k]
                b_g = inf_color_model[1][x][y][1][k]
                b_b = inf_color_model[1][x][y][2][k]

                collapsed_img, _ = collapse(img, b_r, b_g, b_b)

                # Normalize pixel values to the range [0, 255]
                normalized_img = (collapsed_img - collapsed_img.min()) / (
                        collapsed_img.max() - collapsed_img.min()) * 255
                normalized_img = normalized_img.astype(np.uint8)

                # Create a PIL image from the numpy array
                pil_image = Image.fromarray(normalized_img)

                # Save the image as PNG
                save_path = os.path.join(save_dir, f"bubble_{dset_type[x]}_{x}_{y}_{k}.png")
                pil_image.save(save_path)

    print("Bubble images saved successfully.")


save_bubble_images()

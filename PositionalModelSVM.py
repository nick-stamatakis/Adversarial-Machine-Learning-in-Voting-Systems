import os
import urllib.request
import h5py
import numpy as np
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt

'''
    Dataset structure: 

            COLOR  -   POSITIONAL  - INFORMATION
            /            /                /
        B/V/Q           B/V/Q          COLOR/POSITIONAL
         /              /                   /
       IMAGE          IMAGE               B/V/Q
                                            /
                                    BACKGROUND RGB VALUES

    Images divided into 'batches' as in the ground truth
'''

num_batches = 31
inf_color_model = [[[[[] for k in range(3)] for x in range(num_batches)] for i in range(3)] for y in range(2)]
image_data = [[[[] for x in range(num_batches)] for i in range(3)] for y in range(2)]

print('Downloading datafile')
if not os.path.isfile("data_Blank_Vote_Questionable.h5"):
    urllib.request.urlretrieve("http://puf-data.engr.uconn.edu/data/data_Blank_Vote_Questionable.h5",
                               "data_Blank_Vote_Questionable.h5")

f = h5py.File("data_Blank_Vote_Questionable.h5", "r")

batches = np.arange(num_batches)
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
            image_data[c_1][c_2][c_3].extend(list(f[d][d_t][str(b)][:]))
            for c_4, r in enumerate(rgb):
                inf_color_model[c_1][c_2][b][c_4].extend(list(f['INFORMATION'][d][d_t][str(b) + str(r)][:]))

'''
    Collapses a RGB 3 dimensional image to 1D using color model coeffecients
'''


def collapse(img, b_r, b_g, b_b):
    collapsed_img = np.zeros(shape=(40, 50), dtype=np.float32)

    img[:, :, 0] = img[:, :, 0] * (0.02383815) + (b_r * (-0.01898671))
    img[:, :, 1] = img[:, :, 1] * (0.00010994) + (b_g * (-0.001739))
    img[:, :, 2] = img[:, :, 2] * (0.00178155) + (b_b * (-0.00044142))

    collapsed_img = np.sum(img, axis=2)
    stacked = np.stack((collapsed_img,) * 3, axis=-1)

    return collapsed_img.flatten()


def load_positional_data():
    positional_image_data = []
    positional_ground_truth = []
    for x in range(len(dset_type) - 1):
        for y in range(num_batches):
            for k in range(len(inf_color_model[1][x][y][0])):
                small_x = []

                img = image_data[1][x][y][k]
                img = img[:, :, ::-1]

                b_r = inf_color_model[1][x][y][0][k]
                b_g = inf_color_model[1][x][y][1][k]
                b_b = inf_color_model[1][x][y][2][k]

                img = collapse(img, b_r, b_g, b_b)
                positional_image_data.append(img)
                positional_ground_truth.append(x)

    return positional_image_data, positional_ground_truth

X, Y = load_positional_data()

print("Training positional model")
print("Currently this positional model uses hard coded color model params, not the parameters just produced")
clf = LinearSVC(random_state=0, max_iter=10000, dual=False, C=0.00000001,
                tol=0.0000001, penalty='l2', class_weight='balanced', intercept_scaling=1000)

clf.fit(X, Y)

print(len(X))
print(len(Y))

print("Coeffecient array shape " + str(clf.coef_.shape))
print("Coefficients " + str(clf.coef_))
print("Training accuracy " + str(clf.score(X, Y)))


# Assuming you have already trained the positional model and obtained the coefficients (clf.coef_) and the bubble data (X, Y)

def create_heatmap(position_model, bubble_data):
    # Reshape the coefficient array to match the image dimensions
    coef_array = position_model.coef_.reshape((40, 50))
    plt.imshow(coef_array, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Coefficient Heatmap')
    plt.show()

    # Iterate over each bubble in the data
    for i, bubble in enumerate(bubble_data):
        # Reshape the bubble data to match the image dimensions
        bubble_img = bubble.reshape((40, 50))

        # Apply the coefficient values to the bubble image
        heatmap_img = bubble_img * coef_array

        # Create a heatmap plot
        plt.figure()
        plt.imshow(heatmap_img, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title('Bubble Heatmap - Bubble ' + str(i + 1))
        plt.show()


# Call the function to create heatmaps for all bubbles
create_heatmap(clf, X)


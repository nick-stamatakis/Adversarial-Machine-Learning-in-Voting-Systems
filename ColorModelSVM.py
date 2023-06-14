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

print('Downloading datafile')
if not os.path.isfile("data_Blank_Vote_Questionable.h5"):
    urllib.request.urlretrieve("http://puf-data.engr.uconn.edu/data/data_Blank_Vote_Questionable.h5",
                               "data_Blank_Vote_Questionable.h5")

f = h5py.File("data_Blank_Vote_Questionable.h5", "r")

num_batches = 2
batches = np.arange(num_batches)
rgb = ['r', 'g', 'b']
dset = ['COLOR', 'POSITIONAL']
dset_type = ['VOTE', 'BLANK', 'QUESTIONABLE']
X = []
Y = []

image_data = [[[[] for x in range(num_batches)] for i in range(3)] for y in range(2)]
inf_color_model = [[[[[] for k in range(3)] for x in range(num_batches)] for i in range(3)] for y in range(2)]

print("Reading entire dataset")
for c_1, d in enumerate(dset):
    for c_2, d_t in enumerate(dset_type):
        for c_3, b in enumerate(batches):
            image_data[c_1][c_2][c_3].extend(list(f[d][d_t][str(b)][:]))
            for c_4, r in enumerate(rgb):
                inf_color_model[c_1][c_2][b][c_4].extend(list(f['INFORMATION'][d][d_t][str(b) + str(r)][:]))

def within_var(img):
    # Rearrange dimensions of the image array
    im = np.einsum('abc->cab', img)
    mean = []
    var = []

    # Calculate mean and variance for each color channel (r, g, b)
    for x in range(3):
        mean.append(np.mean(im[x]))  # Compute mean of the x-th color channel
        var.append(np.var(im[x]))  # Compute variance of the x-th color channel

        # Check if the variance is greater than 0.5 times the mean
        if var[x] > 0.5 * mean[x]:
            return 1  # Return 1 if variance exceeds the threshold

    return 0  # Return 0 if variance is within the threshold


# Defines what a color is through rgb values
def colorDef(r, b, g):
    r = int(r)
    b = int(b)
    g = int(g)

    # Blue
    if 181 > r > 174:
        return 0

    # Green
    if 190 > r > 180:
        return 1

    # White
    if r > 250 and b > 250 and g > 250:
        return 2

    # Yellow
    if r > 200 > g and b > 200:
        return 3

    # Pink
    if r > 200 and 160 > b > 140:
        return 4

    # Salmon
    if r > 200 and 180 > b > 160:
        return 5

    print("No match found " + str(r) + " " + str(b) + " " + str(g))

def load_color_data():
    for x in range(len(dset_type) - 1):
        for y in range(len(batches)):
            for k in range(len(inf_color_model[0][x][y][0])):
                small_x = []
                img = image_data[0][x][y][k]
                img = img[:, :, ::-1]

                small_x.append(inf_color_model[0][x][y][0][k])
                small_x.append(inf_color_model[0][x][y][1][k])
                small_x.append(inf_color_model[0][x][y][2][k])

                small_x.extend(np.mean(img, axis=(0, 1)))

                X.append(small_x)
                color = colorDef(inf_color_model[0][x][y][0][k], inf_color_model[0][x][y][1][k],
                                 inf_color_model[0][x][y][2][k])

                Y.append(x)

    return X, Y

def plot_heat_map(bubble_index):
    if bubble_index >= len(X):
        print("Invalid bubble index.")
        return

    features = X[bubble_index]
    label = Y[bubble_index]

    img_data = np.array(image_data[0][0][bubble_index])
    if img_data.size == 0:
        print("Empty image data for the bubble.")
        return
    #Add check to make dimensions the same
    img_data = img_data.reshape((img_data.shape[0], -1, 3))
    img_data = img_data[:, :, ::-1]

    plt.figure()
    plt.imshow(img_data, cmap='hot')
    plt.colorbar(label='Intensity')
    plt.title("Heatmap for Bubble " + str(bubble_index) + " (" + get_label_text(label) + ")")
    plt.show()

def get_label_text(label):
    if label == 0:
        return "Vote"
    elif label == 1:
        return "Non-Vote"
    elif label == 2:
        return "Questionable"
    else:
        return "Unknown"

X, Y = load_color_data()

print("Training Color model")
clf = LinearSVC(penalty='l2', dual=False, tol=0.0000001, C=1.0, multi_class='ovr', fit_intercept=True,
                intercept_scaling=1000,
                class_weight='balanced', random_state=0, max_iter=10000)

clf.fit(X, Y)

print(len(X))
print(len(Y))

print("Coefficient array shape " + str(clf.coef_.shape))
print("Coefficients " + str(clf.coef_))
print("Coefficients " + str(clf.get_params()))
print("Training accuracy " + str(clf.score(X, Y)))


for bubble_index in range(len(X)):
    plot_heat_map(bubble_index)

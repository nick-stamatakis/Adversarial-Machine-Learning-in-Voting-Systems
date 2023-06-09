import os
import urllib.request
import h5py
import numpy as np
from sklearn.svm import LinearSVC

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
inf_color_model = [[[[[] for k in range(3)] for x in range(108)] for i in range(3)] for y in range(2)]
image_data = [[[[] for x in range(108)] for i in range(3)] for y in range(2)]

print('Downloading datafile')
if not os.path.isfile("data_Blank_Vote_Questionable.h5"):
    urllib.request.urlretrieve("http://puf-data.engr.uconn.edu/data/data_Blank_Vote_Questionable.h5",
                               "data_Blank_Vote_Questionable.h5")

f = h5py.File("data_Blank_Vote_Questionable.h5", "r")

batches = np.arange(108)
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

'''The function serves as a criterion to determine whether the image has sufficient variation in color, 
based on the variance-to-mean ratio.Regenerate response'''

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


''' It creates a feature vector small_x that consists of the color model values and the mean values 
of the RGB channels of the image. This feature vector is then appended to the X dataset.
Prepares the feature vectors and corresponding labels for the color data, 
combining color model values and mean RGB values, to be used for training a model.'''


def load_color_data():
    # Iterate over the different data types (VOTE, BLANK, QUESTIONABLE)
    for x in range(len(dset_type) - 1):
        # Iterate over the batches
        for y in range(len(batches)):
            # Iterate over the samples within each batch
            for k in range(len(inf_color_model[0][x][y][0])):
                small_x = []  # Initialize a list to store the feature values for each sample

                # Obtain the image data for the current sample
                img = image_data[0][x][y][k]
                img = img[:, :, ::-1]  # Reverse the order of color channels (from BGR to RGB)

                # Append the color model values for the current sample to the feature list
                small_x.append(inf_color_model[0][x][y][0][k])
                small_x.append(inf_color_model[0][x][y][1][k])
                small_x.append(inf_color_model[0][x][y][2][k])

                # Append the mean values of the RGB channels of the image to the feature list
                small_x.extend(np.mean(img, axis=(0, 1)))

                X.append(small_x)  # Append the feature list to the X dataset

                # Determine the color (vote/non-vote) of the current sample based on color model values
                color = colorDef(inf_color_model[0][x][y][0][k], inf_color_model[0][x][y][1][k],
                                 inf_color_model[0][x][y][2][k])

                # Append the color (vote/non-vote) label to the Y dataset
                Y.append(x)

    return X, Y

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

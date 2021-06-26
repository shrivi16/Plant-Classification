import pandas as pd
import ImageMethods
import os
import cv2

# The labels of the dataset (y_train and y_test)
plant_class = []

# These are the Haralick features and Hu Moments(x_train and x_test)
h1 = []
h2 = []
h3 = []
h4 = []
h5 = []
h6 = []
h7 = []
h8 = []
h9 = []
h10 = []
h11 = []
h12 = []
h13 = []
hu1 = []
hu2 = []
hu3 = []
hu4 = []
hu5 = []
hu6 = []
hu7 = []

# Accessing all images of the Dataset
i = 0

# Setting the directory to the folder that contains the image from the field category of the leafsnap dataset
directory = r'C:\Users\Shrivi\Documents\CAMPUS(YEAR 4) - Honors\Semester 1\COMP702 - Image Processing and Computer Vision\Project\Source Code\Plant Images'

# Looping through all images in the folder
for filename in os.listdir(directory):
    for leaf_image in os.listdir(fr'C:\Users\Shrivi\Documents\CAMPUS(YEAR 4) - Honors\Semester 1\COMP702 - Image Processing and Computer Vision\Project\Source Code\Plant Images\{filename}'):
        if leaf_image.endswith(".jpg"):
            image_path = fr'C:\Users\Shrivi\Documents\CAMPUS(YEAR 4) - Honors\Semester 1\COMP702 - Image Processing and Computer Vision\Project\Source Code\Plant Images\{filename}' + '\\' + os.path.basename(leaf_image)

            print(f'Species {i}') # This was used to determine how far the dataset has calculated while the program executes

            # Read the image in Grayscale mode
            img = cv2.imread(fr'{image_path}', 0)

            # We use the current folder as an ID to identify the plant
            plant_class.append(i)

            # Using the method from the ImageMethods.py file to compute Hu moments
            hu = ImageMethods.hu_moments(img)
            hu1.append(hu[0])
            hu2.append(hu[1])
            hu3.append(hu[2])
            hu4.append(hu[3])
            hu5.append(hu[4])
            hu6.append(hu[5])
            hu7.append(hu[6])

            # Using the method from the ImageMethods.py file to compute Haralick Features
            haralick = ImageMethods.haralick_moments(img)
            h1.append(haralick[0])
            h2.append(haralick[1])
            h3.append(haralick[2])
            h4.append(haralick[3])
            h5.append(haralick[4])
            h6.append(haralick[5])
            h7.append(haralick[6])
            h8.append(haralick[7])
            h9.append(haralick[8])
            h10.append(haralick[9])
            h11.append(haralick[10])
            h12.append(haralick[11])
            h13.append(haralick[12])

        else:
            continue

    # Update the plant species ID once done with all images of this folder
    i = i + 1

# converting the above data into a dataframe
data = {
    'Speceis': plant_class,
    'hu1': hu1,
    'hu2': hu1,
    'hu3': hu1,
    'hu4': hu1,
    'hu5': hu1,
    'hu6': hu1,
    'hu7': hu1,
    'haralick1': h1,
    'haralick2': h2,
    'haralick3': h3,
    'haralick4': h4,
    'haralick5': h5,
    'haralick6': h6,
    'haralick7': h7,
    'haralick8': h8,
    'haralick9': h9,
    'haralick10': h10,
    'haralick11': h11,
    'haralick12': h12,
    'haralick13': h13
}

# Converting the data frame into a csv that can be used for training and testing the model
df = pd.DataFrame(data)
df.to_csv('dataset_of_features.csv', index=False)

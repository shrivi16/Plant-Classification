import cv2
import numpy as np
import mahotas as mt

# Stage 1 of Preprocessing - Cropping and Resizing

# Cropping the image

def crop(image):

    # Getting resolution of image
    img_y_axis = image.shape[0]
    img_x_axis = image.shape[1]

    # Only focus on ROI
    img_x_axis = img_x_axis - 170   # This removes the paint bar from the R.H.S

    # For y we have variable lengths. Mostly between the range of 600 - 900
    if 700 < img_y_axis < 770:
        img_y_axis = img_y_axis - 160
    elif img_y_axis >= 770:
        img_y_axis = img_y_axis - 185
    else:
        img_y_axis = img_y_axis - 145

    # Cropping the image and resizing it while maintaining aspect ration using cv2.INTER_AREA
    cropped_img = image[0:img_y_axis, 0:img_x_axis]
    cropped_img = cv2.resize(cropped_img, (500, 500), interpolation=cv2.INTER_AREA)

    return cropped_img


# Stage 2 of Preprocessing - Filtering out noise with the use of filters

def meanF(image):    # Mean filter
    img = crop(image)
    im2 = cv2.blur(img, (5, 5))
    return im2


def gaussianF(image):   # Gaussian Filter
    img = crop(image)
    im2 = cv2.GaussianBlur(img, (5, 5), 0)
    return im2


def medianF(image):     # Median Filter
    img = crop(image)
    im2 = cv2.medianBlur(img, 5)
    return im2


def bilateralF(image):  # Bilateral Filter
    img = crop(image)
    im2 = cv2.bilateralFilter(img, 9, 5, 5)
    return im2

	
# Stage 3 of Preprocessing - Image enhancement with Histogram Equalization

def histogram_equalization(image):      # Normal Histogram Equalization Method
    filtered_image = bilateralF(image)
    equ = cv2.equalizeHist(filtered_image)
    return equ


def clahe_equalization_Image(image):    # Adaptive Histogram Equalization
    filtered_image = bilateralF(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(20, 20))
    cl1 = clahe.apply(filtered_image)
    return cl1


# Stage 1 of Image Segmentation - Image Thresholding ------------------------------------------------------------------

def simple_thresholding(image):
    img = clahe_equalization_Image(image)
    mean = img.mean()
    ret3, thresh = cv2.threshold(img, mean/2+10, 255, cv2.THRESH_BINARY_INV)
    return thresh


def otsu_binary_thresholding(image):
    img = clahe_equalization_Image(image)
    mean = img.mean()
    ret3, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh


def closing(image):
    img = simple_thresholding(image)
    kernel = np.ones((40, 40), np.uint8)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    closing = cv2.morphologyEx(img, cv2.MORPH_DILATE, np.ones((5, 5), np.uint8))
    return closing


# Stage 2 of Image Segmentation - Edge Detection

def sobel_edge_detection(image):    # Using Sobel method
    processed_image = closing(image)
    sobelx = cv2.Sobel(processed_image, cv2.CV_64F, 1, 0, ksize=5)
    abs_sobelx = np.absolute(sobelx)
    sobel_unit = np.uint8(abs_sobelx)
    sobely = cv2.Sobel(processed_image, cv2.CV_64F, 0, 1, ksize=5)
    abs_sobely = np.absolute(sobely)
    sobel_unit = np.uint8(abs_sobely)
    return abs_sobelx, abs_sobely


def canny_edge_detection(image):    # Using Canny Edge Detection
    processed_image = closing(image)
    edges = cv2.Canny(processed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(processed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours[0]


# Stage 1 of Feature Extraction ---------------------------------------------------------------------------------------

def hu_moments(image):    # Hu's 7 Moments Approach
    img = closing(image)
    hu = cv2.HuMoments(cv2.moments(img)).flatten()
    return hu


# Stage 2 of Feature Extraction

def haralick_moments(image):    # Haralick Moments
    img = clahe_equalization_Image(image)
    textures = mt.features.haralick(img)
    ht_mean = textures.mean(axis=0)
    return ht_mean

# Stage 3 of feature extraction - shape features (contour features)

def shape_features(image):
    shape_features = []
    cnt = canny_edge_detection(image)

    area = cv2.contourArea(cnt)
    shape_features.append(area)

    perimeter = cv2.arcLength(cnt, True)
    shape_features.append(perimeter)

    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / h
    shape_features.append(aspect_ratio)

    rectangularity = w * h / area
    shape_features.append(rectangularity)

    circularity = (perimeter ** 2) / area
    shape_features.append(circularity)

    equi_diameter = np.sqrt(4 * area / np.pi)
    shape_features.append(equi_diameter)

    return shape_features

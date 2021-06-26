import cv2
import numpy as np
import mahotas as mt

# These are all the methods used to segment and enhance the image

img = cv2.imread('SamplePlant.jpg', 0)

# Stage 1 of Preprocessing - Filtering out noise with the use of filters

def meanF(image):    # Mean filter
    im2 = cv2.blur(image, (3, 3))
    # cv2.imshow('Original Vs Mean Filtered Image', np.hstack((image, im2)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.waitKey(1)
    return im2


def gaussianF(image):   # Gaussian Filter
    im2 = cv2.GaussianBlur(image, (15, 15), 0)
    # cv2.imshow('Original Vs Gaussian Filtered Image', np.hstack((image, im2)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.waitKey(1)
    return im2


def medianF(image):     # Median Filter
    im2 = cv2.medianBlur(image, 15)
    # cv2.imshow('Original Vs Median Filtered Image', np.hstack((image, im2)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.waitKey(1)
    return im2


def bilateralF(image):  # Bilateral Filter
    im2 = cv2.bilateralFilter(image, 9, 80, 80)
    # cv2.imshow('Original Vs Bilateral Filtered Image', np.hstack((image, im2)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.waitKey(1)
    return im2


# meanF(img)
# gaussianF(img)
# medianF(img)
# bilateralF(img)

# Stage 2 of Preprocessing - Image enhancement with Histogram Equalization

def histogram_equalization(image):      # Normal Histogram Equalization Method
    filtered_image = meanF(image)
    equ = cv2.equalizeHist(filtered_image)
    # cv2.imshow('Histogram Equalization Applied', np.hstack((filtered_image, equ)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.waitKey(1)
    return equ


def clahe_equalization_Image(image):    # Adaptive Histogram Equalization
    filtered_image = meanF(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(20, 20))
    cl1 = clahe.apply(filtered_image)
    # cv2.imshow('CLAHE Applied', np.hstack((filtered_image, cl1)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.waitKey(1)
    return cl1


# clahe_equalization_Image(img)
# histogram_equalization(img)

# Stage 1 of Image Segmentation - Image Thresholding ------------------------------------------------------------------

def simple_thresholding(image):
    img = meanF(image)
    mean = img.mean()
    ret3, thresh = cv2.threshold(img, mean/2, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow('Binary Thresholding Applied', np.hstack((img, thresh)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.waitKey(1)
    return thresh


def otsu_binary_thresholding(image):
    img = clahe_equalization_Image(image)
    mean = img.mean()
    ret3, thresh = cv2.threshold(img, mean/2, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # cv2.imshow('OTSU Binary Thresholding', np.hstack((img, thresh)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.waitKey(1)
    return thresh


def closing(image):
    img = otsu_binary_thresholding(image)
    kernel = np.ones((10, 10), np.uint8)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow('Morphological Closing', np.hstack((img, closing)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.waitKey(1)
    return closing


# simple_thresholding(img)
# otsu_binary_thresholding(img)
# closing(img)

# Stage 2 of Image Segmentation - Edge Detection

def sobel_edge_detection(image):    # Using Sobel method
    processed_image = closing(image)
    sobelx = cv2.Sobel(processed_image, cv2.CV_64F, 1, 0, ksize=5)
    abs_sobelx = np.absolute(sobelx)
    sobel_unit = np.uint8(abs_sobelx)
    sobely = cv2.Sobel(processed_image, cv2.CV_64F, 0, 1, ksize=5)
    abs_sobely = np.absolute(sobely)
    sobel_unit = np.uint8(abs_sobely)
    # cv2.imshow('Sobel Edge Detection', np.hstack((processed_image, abs_sobelx)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.waitKey(1)
    return abs_sobelx, abs_sobely


def canny_edge_detection(image):    # Using Canny Edge Detection
    processed_image = closing(image)
    edges = cv2.Canny(processed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.imshow('Canny Edge Detection', np.hstack((processed_image, edges)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.waitKey(1)
    contours, hierarchy = cv2.findContours(processed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    final_image = cv2.drawContours(processed_image, contours, -1, (255, 255, 255), 10)
    # cv2.imshow('Canny Edge Detection', np.hstack((image, final_image)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.waitKey(1)
    return contours, final_image


#sobel_edge_detection(img)
#canny_edge_detection(img)

def final_image(image):
    cnts = canny_edge_detection(image)
    contour = cnts[0]
    finImage = cv2.drawContours(meanF(image), contour, -1, (0, 0, 0), 10)
    # cv2.imshow('Canny Edge Detection', np.hstack((image, finImage)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.waitKey(1)
    return finImage, contour


# Stage 1 of Feature Extraction ---------------------------------------------------------------------------------------

def hu_moments(image):    # Hu's 7 Moments Approach
    img, cnts = final_image(image)
    M = cv2.HuMoments(cv2.moments(image)).flatten()
    return M


# Stage 2 of Feature Extraction

def haralick_moments(image):    # Haralick Moments
    img, cnts = final_image(image)
    textures = mt.features.haralick(img)
    ht_mean = textures.mean(axis=0)
    return ht_mean

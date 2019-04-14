import argparse
import copy
import os

import cv2
import numpy as np
from scipy.spatial import distance


def parse_args():

    # Argument parser (taken from the first project, thanks!)
    parser = argparse.ArgumentParser(description="image stitching project")
    parser.add_argument(
        "--dir_path", type=str, default="",
        help="path to the images used for edge detection")
    parser.add_argument(
        "--dir_path2", type=str, default="",
        help="path to the images used for edge detection")
    args = parser.parse_args()
    return args

def findMatches(source_img, dest_img):

    # Detecting features using BRISK algorithm
    brisk = cv2.BRISK_create()
    kp1, des1 = brisk.detectAndCompute(source_img, None)
    kp2, des2 = brisk.detectAndCompute(dest_img, None)

    # Finding distance between each feature detected
    data = []
    for i in range(len(des1)):
        threshold = .4
        for j in range(len(des2)):
            feature_distance = distance.hamming(des1[i],des2[j])
            if feature_distance < threshold:
                threshold = feature_distance
                reserve = j
        if threshold < .4:
            data.append([kp1[i],kp2[reserve]])
    return data

def findHomography(data):

    # Get four random indexes
    index = np.arange(len(data))
    np.random.shuffle(index)
    inliers = index[:4]

    # Extracting data in np arrays from randomized index for further processing
    # This block separates matches into source and transform lists
    source = []
    transform = []
    for i in inliers:
        source.append(data[i][0])
        transform.append(data[i][1])

    # This block takes the source and transform keyPoint lists and extracts the x and y values into a numpy array
    source_points = np.ndarray(shape=(4,2), dtype=np.float32)
    transform_points = np.ndarray(shape=(4,2), dtype=np.float32)
    for i in range(len(source_points)):
        source_points[i] = [source[i].pt[0], source[i].pt[1]]
        transform_points[i] = [transform[i].pt[0], transform[i].pt[1]]

    # Finding the homography and returning
    homography = cv2.getPerspectiveTransform(source_points, transform_points, cv2.DECOMP_SVD)

    # Calculating translation homography so that values that are on the negative side of the axis are included

    translation_h = np.zeros((3, 3), dtype=np.float32)
    translation_h[0] = [1., 0., -1000]
    translation_h[1] = [0., 0., -1000]
    translation_h[2] = [0., 0., 1.]
    return(homography)

def main():

    # Parsing arguments
    args = parse_args()
    files = []
    for filename in os.listdir(args.dir_path):
        files.append(filename)
    # Deduping the image list
    dedup_files = []
    for i in range(len(files)):
        image = cv2.imread(os.path.join(args.dir_path,files[i]), 1)
        dedup_files.append(image)
    img1 = dedup_files[0]   
    for image in dedup_files:
        # Finding the matches
        matches = findMatches(image, img1)

        # Finding the homography
        h = findHomography(matches)

        # Warping image
        height, width, channels = image.shape
        im_out = cv2.warpPerspective(image, h, (width, height))

        # Display image
        #cv2.imshow("Source Image", img1)
        #cv2.imshow("Destination Image", img2)
        #cv2.imshow("Warped Source Image", im_out)
        img1 = cv2.addWeighted(img1, .5, im_out, .5, 1)

    resized_final = cv2.resize(img1, (0,0), fx=0.5, fy=0.5)
    cv2.imshow("Blended image",resized_final)
    cv2.waitKey()

if __name__ == "__main__":
    main()

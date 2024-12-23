import numpy as np
import cv2 as cv
import glob
import os

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Define the square size (mm)
square_size = 40

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(3,5,0)
objp = np.zeros((4*6, 3), np.float32)

# Here, we multiply the 2D coordinates by the square size to give the real-world coordinates
objp[:, :2] = np.mgrid[0:4, 0:6].T.reshape(-1, 2) * square_size

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

images = glob.glob('img/*.jpg')

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (4, 6), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv.drawChessboardCorners(img, (4, 6), corners2, ret)
        cv.imshow('img', img)
        
        # Wait for the user to press 'n' to move to the next image
        key = cv.waitKey(0)  # Wait indefinitely for a key press
        if key == ord('n'):  # Check if 'n' is pressed
            continue  # Proceed to the next image
        else:
            break  # Exit if any other key is pressed
    
    else:
        print(f"Pattern not found in {fname}. Deleting image.")
        os.remove(fname)

if objpoints:
    # Calibrate the camera using the object points and image points
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print("Camera matrix : \n")  # Camera matrix
    print(mtx)
    print("dist : \n")  # Lens distortion coefficients
    print(dist)
    print("rvecs : \n")  # Rotation vectors
    print(rvecs)
    print("tvecs : \n")  # Translation vectors
    print(tvecs)

    # Now, let's undistort the images and show the comparison
    images = glob.glob('img/*.jpg')
    for fname in images:
        img = cv.imread(fname)
        # Undistort the image
        undistorted_img = cv.undistort(img, mtx, dist, None, mtx)

        # Stack the original and undistorted images side by side
        combined_img = np.hstack((img, undistorted_img))

        # Display the original and undistorted image side by side
        cv.imshow('Original vs Undistorted', combined_img)

        # Wait for the user to press 'n' to proceed to the next image
        key = cv.waitKey(0)  # Wait indefinitely for a key press
        if key == ord('n'):  # Check if 'n' is pressed
            continue  # Proceed to the next image
        else:
            break  # Exit if any other key is pressed

cv.destroyAllWindows()

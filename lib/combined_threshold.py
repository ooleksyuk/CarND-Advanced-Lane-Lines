import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
print("Imposted All Combined Threshold Libraries!")

def abs_sobel_thresh(img, orient='x', thresh_min=20, thresh_max=100):
    """
    Takes an image, gradient orientation, and threshold min/max values
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
        # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(30, 100)):
    """
    Return the magnitude of the gradient
    for a given sobel kernel size and threshold values
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    """
    Return the direction of the gradient
    for a given sobel kernel size and threshold values
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output

def hls_sthresh(img, thresh=(100, 255)):
    """
    Convert RGB to HLS and threshold to binary image using S channel
    """
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output

def lab_bthresh(img, thresh=(190,255)):
    # 1) Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab_b = lab[:,:,2]
    # don't normalize if there are no yellows in the image
    if np.max(lab_b) > 175:
        lab_b = lab_b*(255/np.max(lab_b))
    # 2) Apply a threshold to the L channel
    binary_output = np.zeros_like(lab_b)
    binary_output[((lab_b > thresh[0]) & (lab_b <= thresh[1]))] = 1
    # 3) Return a binary image of threshold result
    return binary_output

def hls_lthresh(img, thresh=(220, 255)):
    """
    Convert RGB to HLS and threshold to binary image using L channel
    """
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    hls_l = hls[:,:,1]
    hls_l = hls_l*(255/np.max(hls_l))
    # 2) Apply a threshold to the L channel
    binary_output = np.zeros_like(hls_l)
    binary_output[(hls_l > thresh[0]) & (hls_l <= thresh[1])] = 1
    # 3) Return a binary image of threshold result
    return binary_output

def combined_thresh(img):
    abs_bin = abs_sobel_thresh(img, orient='x', thresh_min=50, thresh_max=255)
    mag_bin = mag_thresh(img, sobel_kernel=3, mag_thresh=(50, 255))
    dir_bin = dir_threshold(img, sobel_kernel=15, thresh=(0.7, 1.3))
    hls_s_bin = hls_sthresh(img, thresh=(170, 255))
    lab_b_bin = lab_bthresh(img, thresh=(190,255))
    hls_l_bin = hls_lthresh(img, thresh=(220, 255))

    combined = np.zeros_like(dir_bin)
    combined[(lab_b_bin | (hls_l_bin | (abs_bin == 1 | ((mag_bin == 1) & (dir_bin == 1))))) | hls_s_bin == 1] = 1
    
    combined2 = np.zeros_like(dir_bin)
    combined2[(abs_bin == 1 | ((mag_bin == 1) & (dir_bin == 1))) | hls_s_bin == 1] = 1

    return combined, combined2, abs_bin, mag_bin, dir_bin, hls_s_bin, lab_b_bin, hls_l_bin  # DEBUG

if __name__ == '__main__':
    img_file = 'test_images/straight_lines1.jpg'
    img_file = 'test_images/test5.jpg'

    with open('output_images/calibrate_camera.p', 'rb') as f:
        save_dict = pickle.load(f)
    mtx = save_dict['mtx']
    dist = save_dict['dist']

    img = mpimg.imread(img_file)
    img = cv2.undistort(img, mtx, dist, None, mtx)

    combined, combined2, abs_bin, mag_bin, dir_bin, hls_s_bin, lab_b_bin, hls_l_bin = combined_thresh(img)
    
    plt.subplots(3, 3, figsize=(20,20))
    plt.subplot(3, 3, 1)
    plt.imshow(abs_bin, cmap='gray', vmin=0, vmax=1)
    plt.title("Sobel Threshold", fontsize=20)
    plt.subplot(3, 3, 2)
    plt.imshow(mag_bin, cmap='gray', vmin=0, vmax=1)
    plt.title("Magnitude of the Gradient", fontsize=20)
    plt.subplot(3, 3, 3)
    plt.imshow(dir_bin, cmap='gray', vmin=0, vmax=1)
    plt.title("Direction of the Gradient", fontsize=20)
    plt.subplot(3, 3, 4)
    plt.imshow(lab_b_bin, cmap='gray', vmin=0, vmax=1)
    plt.title("Lab B Color Space", fontsize=20)
    plt.subplot(3, 3, 5)
    plt.imshow(hls_l_bin, cmap='gray', vmin=0, vmax=1)
    plt.title("HLS L Color Space", fontsize=20)
    plt.subplot(3, 3, 6)
    plt.imshow(hls_s_bin, cmap='gray', vmin=0, vmax=1)
    plt.title("HLS S Color Space", fontsize=20)
    plt.subplot(3, 3, 7)
    plt.imshow(img)
    plt.title("Original Image", fontsize=20)
    plt.subplot(3, 3, 8)
    plt.imshow(combined2, cmap='gray', vmin=0, vmax=1)
    plt.title("Combined 2 Thresholds", fontsize=20)
    plt.subplot(3, 3, 9)
    plt.imshow(combined, cmap='gray', vmin=0, vmax=1)
    plt.title("Combined Final Thresholds", fontsize=20)

    plt.tight_layout()
    plt.savefig('output_images/combined_threshold.jpg')
    #plt.show()
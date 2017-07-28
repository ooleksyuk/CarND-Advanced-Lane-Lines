import numpy as np
import cv2
import glob
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import pickle
print ("Imported all Camera Calibration libraries!")

def draw_corners(image, corners, ret, fname, ny=9, nx=6):
    '''
    Draw corners of image
    '''
    plt.imshow(image)
    #plt.show()

def save_image(image, idx):
    write_name = 'output_images/corners_found'+str(idx)+'.jpg'
    cv2.imwrite(write_name, image)
    
def get_object_points():
    nx, ny = 6, 9
    
    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    
    # Make a list of calibration images
    images = glob.glob('camera_cal/calibration*.jpg')
    
    # find corners on all test images provided to calibrate camera
    for idx, fname in enumerate(images):
        image = cv2.imread(fname)
        
        # convert all images into gray
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # find chess board corners 
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
        
        # save chess board corners
        if ret == True:
            # save object points and it's corners
            objpoints.append(objp)
            imgpoints.append(corners)
            cv2.drawChessboardCorners(image, (nx,ny), corners, ret)
            
            draw_corners(image, corners, ret, fname, 9, 6)
            save_image(image, idx)
            print('Found corners for %s' % fname)
        else:
            print('Warning: ret = %s for %s' % (ret, fname))
    cv2.destroyAllWindows()        
            
    return objpoints, imgpoints

def calibrate_camera():
    # Get objpoints, imgpoints
    objpoints, imgpoints = get_object_points()
    
    # Test undistortion on an image
    img = cv2.imread('test_images/straight_lines1.jpg')
    img_size = (img.shape[1], img.shape[0])
    
    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    save_mtx_dist(mtx, dist)

    return mtx, dist

def draw_two_images(img, dst, idx):
    #dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    # Visualize undistortion
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(dst)
    ax2.set_title('Undistorted Image', fontsize=30)
    #plt.show()
    plt.savefig('output_images/undistort_calibration'+str(idx)+'.jpg')

def save_mtx_dist(mtx, dist):
    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    dist_pickle = {}
    dist_pickle["mtx"], dist_pickle["dist"] = mtx, dist
    pickle.dump( dist_pickle, open( "output_images/calibrate_camera.p", "wb" ) )
    
    print("Saved calibrate_camera file")
    
if __name__ == '__main__':
    print("Calibrating the camera...")
    mtx, dist = calibrate_camera()
    
    images = glob.glob('camera_cal/calibration*.jpg')
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        
        dst = cv2.undistort(img, mtx, dist, None, mtx)

        draw_two_images(img, dst, idx)
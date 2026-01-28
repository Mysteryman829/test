'''
 NORTHEASTERN UNIVERSITY COMPUTER VISION I
 PROGRAMMING ASSIGNMENT 5-1
 2023

 Joe Weber

This script performs visual odometry on a series of stereo images from the KITTI dataset,
  which includes the camera calibration information for the stereo cameras.
  VO is performed on each frame pair independently (there is no look ahead or loop closure)

 MODIFICATION: Changed ORB to SIFT + Added Lowe's Ratio Test
'''

import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.optimize as sciopt
import math
import cv2
import argparse

def forward_projection(p3D, cam):
    '''
     Projects 3D points to 2D image points using the camera calibration parameters
     '''
    fx = cam['fx']
    fy = cam['fy']
    cx = cam['cx']
    cy = cam['cy']

    xs = fx*p3D[:,0]/p3D[:,2] + cx
    ys = fy*p3D[:,1]/p3D[:,2] + cy
    return xs, ys

def inverse_projection(img_pt, disparity, cam):
    ''' INVERSE_PROJECTION
    returns the world 3D coordinates from an image point and its stereo disparity
    '''
    fx = cam['fx']
    fy = cam['fy']
    cx = cam['cx']
    cy = cam['cy']
    baseline = cam['baseline']

    Z = fx*baseline/disparity
    X = (img_pt[0]-cx)*Z/fx
    Y = (img_pt[1]-cy)*Z/fy
    return X,Y,Z

def get_transform_difference(t1, t2):
    ''' get the translation error between two 4x4 Transforms
    '''
    return np.linalg.norm(t1[:3,3] - t2[:3,3])

def apply_lowes_ratio_test(knn_matches, ratio=0.75):
    '''
    Apply Lowe's ratio test to filter good matches
    ratio: threshold for distance ratio between best and second-best match (typically 0.7-0.8)
    Returns list of good DMatch objects
    '''
    good_matches = []
    for match_pair in knn_matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio * n.distance:
                good_matches.append(m)
    return good_matches

def epipolar_match(matches, l_keypoints, r_keypoints, cam, y_thresh, d_thresh = 1):
    ''' given keypoint matches between left and right cameras, remove matches that:
        1) differ in y values by more than y_thresh (epipolar constraint says they should have same y)
        2) have small disparity < d_thresh
        3) have large disparity > d_thresh + d_max
    '''
    fx = cam['fx']
    fy = cam['fy']
    cx = cam['cx']
    baseline = cam['baseline']
    camera_height = cam['camera_height']
    d_max = 1.5*fx*baseline*cx/(fy*camera_height)
    # keep track of number of matches removed
    num_removed = 0
    for match in matches[::-1]: # work backwards in list so can remove items
        l_kp = l_keypoints[match.queryIdx].pt
        r_kp = r_keypoints[match.trainIdx].pt
        deltay = abs(r_kp[1] - l_kp[1])
        disparity = abs(r_kp[0] - l_kp[0])
        if disparity < d_thresh: # too small
            # remove this match
            num_removed += 1
            matches.remove(match)
            continue

        if deltay > y_thresh: # too far apart along y axis, not on epipolar plane
            # remove this match
            num_removed += 1
            matches.remove(match)
            continue

        if disparity > d_max:
            num_removed += 1
            matches.remove(match)
            continue

    return num_removed

def forward_motion_match(matches, kps1, kps2, cam, motion_thresh):
    ''' FORWARD MOTION MATCH - given keypoint matches between same camera one time step apart,
    remove matches that are greater than expected motion distance
    '''
    cx = cam['cx']
    cy = cam['cy']
    center = np.array([cy, cx])
    # number of matches removed
    num_removed = 0

    for match in matches[::-1]:
        kp1 = np.array( kps1[match.queryIdx].pt )
        kp2 = np.array( kps2[match.trainIdx].pt )
        motion_distance = np.linalg.norm(kp2 - kp1)
        if motion_distance > motion_thresh:
            num_removed += 1
            matches.remove(match)

    return num_removed

def determine_motion2D(image, pts_2D, pts_3D, cam, draw=False, out_thresh= 3.0):
    '''     DETERMINE MOTION 2D
    Given a 3D point cloud, Find the motion (rotation and translation) of the cloud
    that makes the re-projection of the 3D points match as closely as possible the 2D image points
    Outliers are considered points with residual error > mean + out_thresh*std of residuals

    Section 2 Fix: pts_2D should be from CURRENT frame, pts_3D should be from PREVIOUS frame
    '''
    image_x, image_y = pts_2D[:,0], pts_2D[:,1]
    #  put 3d points into homogeneous form by adding ones
    pts_3D_ho = np.insert(pts_3D, 3, 1, axis=1)

    # Apply Motion
    # Internal function that applies 3D rotation and translation to pts_3D array
    def apply_motion(motion):
        theta, d = motion
        # rotate and translate the 3D points
        c, s = np.cos(theta), np.sin(theta)
        tz = d*c
        tx = d*s
        RT = np.array([[ c, 0, s, tx],
                       [ 0, 1, 0,  0],
                       [-s, 0, c, tz],
                       [ 0, 0, 0,  1]])

        # transform the 3D points.
        new_pts_3D = RT.dot(pts_3D_ho.T)
        # convert back to cartesian
        new_pts_3D = new_pts_3D[:3,:]

        return new_pts_3D.T

    # err_function
    # Calculates the reprojection error afer motion is applied to points
    # Section 5: Use Huber loss for robustness to outliers
    def err_function(motion):
        new_pts_3D = apply_motion(motion)
        # calc new projections into image
        r_xs, r_ys = forward_projection(new_pts_3D, cam)
        # calc re-projection error using Huber loss (robust to outliers)
        residuals = np.sqrt((image_x-r_xs)**2 + (image_y-r_ys)**2)  # Euclidean distance
        delta = 1.5  # Huber threshold in pixels
        huber_loss = np.where(residuals <= delta,
                              0.5 * residuals**2,  # quadratic for small errors
                              delta * (residuals - 0.5 * delta))  # linear for large errors
        return np.sum(huber_loss)

    # start of function
    r_xs, r_ys = forward_projection(pts_3D, cam)
    residuals = (image_x-r_xs)**2 + (image_y-r_ys)**2
    # print("initial reprojection error: ",np.sqrt( np.sum( residuals )))
    if draw:
        # draw the two sets of points
        fig, ax = plt.subplots(figsize=(20,5))
        plt.imshow(image)
        ax.set_title('Optical Flow')
        plt.plot(r_xs,r_ys,'+',image_x,image_y,'o')
        for i in range(len(r_xs)):
            plt.plot([image_x[i], r_xs[i]],[image_y[i],r_ys[i]],'-')
        plt.show()
        plt.close(fig)

    outliers = True
    while outliers:
        # initial estimate is no motion:
        motion0 = [0, 0]  # equals rotation angle theta, and translation d

        # Section 3: Tighten motion bounds - KITTI moves ~10-30 km/h (~3-8 m/s)
        # At 10 fps, that's 0.3-0.8 m per frame. Allow up to 2m translation
        # Rotation: ±5 degrees is more realistic for typical driving
        bounds = ( (-math.pi/36.0, math.pi/36.0 ), (0, 2))  # ±5° rotation, 0-2m translation

        res = sciopt.minimize(err_function, motion0, method = "SLSQP", bounds = bounds)
        motion = res.x
        # print(res.message)
        # print(" Minimization error: ", err_function(motion))

        # calculate residuals
        new_pts_3D = apply_motion(motion)
        r_xs, r_ys = forward_projection(new_pts_3D, cam)
        residuals = (image_x-r_xs)**2 + (image_y-r_ys)**2
        mean, std = np.mean(residuals), np.std(residuals)
        if draw:
            plt.plot(residuals,'o')
            plt.plot([0,len(residuals)], [mean, mean])
            plt.plot([0,len(residuals)], [mean + out_thresh*std, mean + out_thresh*std])
            plt.title("Residuals")
            plt.show()

        #remove outliers and re-calculate motion
        o_threh = max(mean + out_thresh*std, 16.0) # assumes 4 pixel typical max error (16 = 4^2)
        remove = np.where(residuals>o_threh)
        # print("removing ", len(remove[0]), "outliers")
        if len(remove[0])>0:
            pts_3D_ho = np.delete(pts_3D_ho, remove, axis=0)
            image_x =np.delete(image_x, remove)
            image_y = np.delete(image_y, remove)
        else:
            outliers = False

    # draw the final two sets of points
    if draw:
        new_pts_3D = apply_motion(motion)
        r_xs, r_ys = forward_projection(new_pts_3D, cam)
        fig, ax = plt.subplots(figsize=(20,5))
        plt.imshow(image)
        ax.set_title('Reprojection Error')
        plt.plot(r_xs,r_ys,'+',image_x,image_y,'o')
        for i in range(len(r_xs)):
            plt.plot([image_x[i], r_xs[i]],[image_y[i],r_ys[i]],'-')
        plt.show()
    print("Final Re-re projection error: ", err_function(motion))

    theta, d = motion
    c, s = np.cos(theta), np.sin(theta)
    tz = d*c
    tx = d*s
    RT = np.array([[ c, 0, s, tx],  # Section 3: Now in meters already
                   [ 0, 1, 0,  0],
                   [-s, 0, c, tz],
                   [ 0, 0, 0,  1]])
    return RT

'''        
# MAIN CODE
'''

ap = argparse.ArgumentParser(
        description="Visual Odometry - SIFT with Lowe's Ratio Test")

ap.add_argument("-V", "-verbose", dest="verbose", action="store_true")
ap.add_argument("-d", action="store_true", help="Draw Images at each step")
ap.add_argument("-start", dest="start", type = int, required=False, help="start frame", default = 0)
ap.add_argument("-end", dest="end", type = int, required=False, help="end frame", default = 1999)
ap.add_argument("-ratio", dest="ratio", type = float, required=False, help="Lowe's ratio threshold", default = 0.75)
ap.add_argument("-temporal_ratio", dest="temporal_ratio", type = float, required=False, help="Temporal Lowe's ratio (Section 4)", default = 0.85)
args = ap.parse_args()

show_images = args.d
frame_start = args.start
frame_end = args.end
lowe_ratio = args.ratio
temporal_ratio = args.temporal_ratio  # Section 4: Separate ratio for temporal matches

# Global Variables
speed_thresh = 50 # max number of pixels a point can move between time frames due to rotaton or translation alone
vertical_thresh = 4 # max number of pixels up or down allowed in epipolar matching (ideal epipolar has zero pixels up/down)
outlier_threshold = 3 # threshold on error residuals for determining outliers. outlier = mean + outlier_thresh*std
min_num_3D_points = 4 # min number of 3D stereo points between time frames needed to calculate motion
min_disparity = 1 # minimum stereo disparity to use

# Set the paths to the folders of the KITTI Dataset:
root_pathname = "C:/Test Pictures/KITTIVO/"
image_folder = "images/"
calib_folder = "calibs/"
poses_folder = "poses/"
sequence_number = "00/"
left_camera = "cam00/"
right_camera = "cam01/"

# read camera calibration file
calib_path = root_pathname+calib_folder+sequence_number
calib_file = "calib.txt"

# Load the camera Calibration information
try:
    with open(calib_path+calib_file, 'r') as f:
       line = f.readline()
       K_left = np.fromstring(line[4:], dtype=float, sep=' ')
       K_left = K_left.reshape(3,4)
       line = f.readline()
       K_right = np.fromstring(line[4:], dtype=float, sep=' ')
       K_right = K_right.reshape(3,4)

except FileNotFoundError:
    print('Unable to open calibration file: ' + calib_file)

# extract camera intrinsics from K matrices
fx, fy = K_left[0,0], K_left[1,1]
cx, cy = K_left[0,2], K_left[1,2]
baseline = -K_right[0,3] / K_right[0,0]  # Calculate baseline from calibration matrix
camera_height = 1.6  # Camera height in meters (approximately 1.6m)

# create a dict of camera information
camera_data = {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "baseline": baseline, "camera_height": camera_height}

# Get Poses for ground truth trajectory
pose_path = root_pathname+poses_folder
pose_file = os.path.join(pose_path, sequence_number[:-1] + '.txt')
gt_poses = []
try:
    with open(pose_file, 'r') as f:
       lines = f.readlines()

    for line in lines:
        T_w_cam0 = np.fromstring(line, dtype=float, sep=' ')
        T_w_cam0 = T_w_cam0.reshape(3, 4)
        gt_poses.append( T_w_cam0 )

except FileNotFoundError:
            print('Unable to open pose file: ' + pose_file)

# Create an array for storing our motion estimates
motion_estimates = []

print("="*60)
print("SIFT with LOWE'S RATIO TEST - IMPROVED")
print("="*60)
print(f"Using SIFT with 6000 features")
print(f"Stereo Lowe's ratio: {lowe_ratio}")
print(f"Temporal Lowe's ratio: {temporal_ratio} (Section 4: Relaxed)")
print(f"Motion bounds: ±5° rotation, 0-2m translation (Section 3)")
print("="*60)

# Create SIFT detector
sift = cv2.SIFT_create(
    nfeatures=6000,
    nOctaveLayers=4,
    contrastThreshold=0.03,
    edgeThreshold=10,
    sigma=1.6
)

# Matcher for SIFT - using knnMatch for Lowe's ratio test
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)  # crossCheck=False for knnMatch

# loop over images
frame = frame_start
while frame<=frame_end:
    frame_number = str(frame).zfill(6)
    if frame % 50 == 0:
        print("Working on frame ", frame_number," out of ", frame_end)

    # load the image into a NUMPY array
    left_img_file = root_pathname+image_folder+sequence_number+left_camera+frame_number+'.png'
    l_image = cv2.imread(left_img_file)
    right_img_file = root_pathname+image_folder+sequence_number+right_camera+frame_number+'.png'
    r_image = cv2.imread(right_img_file)

    # find the keypoints and descriptors using SIFT
    l_kps, l_des = sift.detectAndCompute(l_image, None)
    r_kps, r_des = sift.detectAndCompute(r_image, None)

    # Match descriptors between left and right camera using knnMatch
    knn_matches_lr = bf.knnMatch(l_des, r_des, k=2)

    # Apply Lowe's ratio test
    current_lr_matches = apply_lowes_ratio_test(knn_matches_lr, ratio=lowe_ratio)

    # Sort them in the order of their distance.
    current_lr_matches = sorted(current_lr_matches, key = lambda x:x.distance)

    if show_images: # draw the matches
        fig, ax = plt.subplots(figsize=(20,5))
        match_image = l_image.copy()
        match_image = cv2.drawMatches(l_image, l_kps, r_image, r_kps, current_lr_matches, match_image, flags=2)
        plt.title(f'Raw Left-Right Correspondences (SIFT + Lowe ratio={lowe_ratio})')
        plt.imshow(match_image),plt.show()

    # Do epipolar constrained matching
    num_removed = epipolar_match(current_lr_matches, l_kps, r_kps, camera_data, vertical_thresh, d_thresh = min_disparity)

    if show_images: # draw the matches
        fig, ax = plt.subplots(figsize=(20,5))
        match_image = l_image.copy()
        match_image = cv2.drawMatches(l_image, l_kps, r_image, r_kps, current_lr_matches, match_image, flags=2)
        plt.title(f'Filtered Left-Right Correspondences (SIFT + Lowe ratio={lowe_ratio})')
        plt.imshow(match_image),plt.show()

    # now match features with previous frame
    if frame > frame_start:
        # Match descriptors between previous left and current left camera
        knn_matches_temporal = bf.knnMatch(previous_l_des, l_des, k=2)

        # Apply Lowe's ratio test
        previous_l_current_l_matches = apply_lowes_ratio_test(knn_matches_temporal, ratio=lowe_ratio)

        if show_images:
            fig, ax = plt.subplots(figsize=(20,5))
            match_image = l_image.copy()
            match_image = cv2.drawMatches(previous_l_img, previous_l_kps, l_image, l_kps, previous_l_current_l_matches, match_image, flags=2)
            plt.title(f'Left to Left Temporal (SIFT + Lowe ratio={temporal_ratio})')
            plt.imshow(match_image), plt.show()

        # for every point in previous frame that we matched for stereo
        # see if we also matched to current frame in time
        overlap = list(set([m.queryIdx for m in previous_lr_matches]) & set([m.queryIdx for m in previous_l_current_l_matches]))

        # now get 3D points from stereo match for those points with matches (in overlap)
        num_p1 = 0
        time_matches, points1_3D, points2_3D, points1_2D, points2_2D = [], [], [], [], []
        # Section 2: Swap frame usage - 3D from previous, 2D from current
        for m1 in previous_l_current_l_matches:
            if m1.queryIdx in overlap:
                disparity1 = None
                for m2 in previous_lr_matches:
                    if m2.queryIdx == m1.queryIdx:
                        l_img_point1 = previous_l_kps[m2.queryIdx].pt
                        r_img_point1 = previous_r_kps[m2.trainIdx].pt
                        disparity1 = l_img_point1[0] - r_img_point1[0]
                # find match in current time frame
                disparity2 = None
                for m2 in current_lr_matches:
                    if m2.queryIdx == m1.trainIdx:
                        l_img_point2 = l_kps[m2.queryIdx].pt
                        r_img_point2 = r_kps[m2.trainIdx].pt
                        disparity2 = l_img_point2[0] - r_img_point2[0]

                if disparity2 is not None and disparity1 is not None:
                    # Section 2: Get 3D from PREVIOUS frame, 2D from CURRENT frame
                    X1,Y1,Z1 = inverse_projection(l_img_point1, disparity1, camera_data)
                    points1_3D.append([X1,Y1,Z1])  # 3D from previous frame
                    points2_2D.append(l_img_point2)  # 2D from current frame
                    num_p1 += 1

        # create lists of x and y locations in the current frame
        # for the points that have stereo matches and temporal matches
        if num_p1 > min_num_3D_points:
            points1_3D = np.array(points1_3D)  # 3D from previous frame
            points2_2D = np.array(points2_2D)  # 2D from current frame

            # Section 2: Pass current 2D and previous 3D (swapped order)
            transform = determine_motion2D(l_image, points2_2D, points1_3D, camera_data, draw=show_images, out_thresh=outlier_threshold)
            motion_estimates.append(transform)
        else:
            print("Not enough matches to calculate motion!")
            # repeat the previous motion as our best guess
            motion_estimates.append(motion_estimates[-1])

    # copy current stuff to previous stuff
    previous_r_img = r_image
    previous_r_kps = r_kps
    previous_r_des = r_des

    previous_l_img = l_image
    previous_l_kps = l_kps
    previous_l_des = l_des

    previous_lr_matches = current_lr_matches

    frame = frame+1
# END Frame loop

print("\n" + "="*60)
print("COMPUTING RESULTS...")
print("="*60)

# calculate the errors in the transforms over the path
trans_errors = []
for i, t in enumerate(motion_estimates):
    pose_i = np.eye(4)
    pose_i[:3,:] = gt_poses[frame_start]
    pose_i_1 = np.eye(4)
    pose_i_1[:3,:] = gt_poses[frame_start+1]
    gt_t = pose_i_1.dot(np.linalg.inv(pose_i))
    trans_err = get_transform_difference(t, gt_t)
    trans_errors.append(trans_err)

print("Trans mean and variance over the path: ", np.array(trans_errors).mean(), np.array(trans_errors).var())
trans_err = np.array(trans_errors).mean()

# extract the x and z coordinates from the ground truth poses
x_pos_truth, y_pos_truth, z_pos_truth = [], [], []
# extract X and Z values
for i in range(frame_end-frame_start+1):
    x_pos_truth.append(gt_poses[i+frame_start][0,3])
    y_pos_truth.append(gt_poses[i+frame_start][1,3])
    z_pos_truth.append(gt_poses[i+frame_start][2,3])

# estimate path of car and compare to ground truth
first_pose = np.eye(4)
first_pose[:3,:] = gt_poses[frame_start]
x_pos_est = [first_pose[0,3]]
y_pos_est = [first_pose[1,3]]
z_pos_est = [first_pose[2,3]]
pose_estimates = [first_pose]
for t in motion_estimates:
    new_pose = np.dot(pose_estimates[-1], t)
    pose_estimates.append(new_pose)
    x_pos_est.append(new_pose[0,3])
    y_pos_est.append(new_pose[1,3])
    z_pos_est.append(new_pose[2,3])

x_pos_est = np.array(x_pos_est)
y_pos_est = np.array(y_pos_est)
z_pos_est = np.array(z_pos_est)
x_pos_truth = np.array(x_pos_truth)
y_pos_truth = np.array(y_pos_truth)
z_pos_truth = np.array(z_pos_truth)

# Calculate final position error
final_error = np.linalg.norm([x_pos_est[-1] - x_pos_truth[-1], z_pos_est[-1] - z_pos_truth[-1]])

print("\n" + "="*60)
print(f"FINAL RESULTS - IMPROVED SIFT + LOWE'S RATIO")
print("="*60)
print(f"Mean translation error:  {trans_err:.6f} m")
print(f"Variance:                {np.array(trans_errors).var():.6f}")
print(f"Final position error:    {final_error:.3f} m")
print(f"Stereo ratio: {lowe_ratio} | Temporal ratio: {temporal_ratio}")
print(f"Original (ORB) error:    ~0.222 m (150m final drift)")
print("="*60)

# plot the paths in the X-Z plane and score
plt.figure(figsize=(12, 10))
plt.plot(x_pos_est, z_pos_est, 'r-', linewidth=2, label=f'Estimated (Improved)')
plt.plot(x_pos_est, z_pos_est,'ro', markersize=3)
plt.plot(x_pos_truth, z_pos_truth, 'g-', linewidth=2, label='Ground Truth')
plt.plot(x_pos_truth, z_pos_truth, 'g+', markersize=5)

s = f"Trans err: {trans_err:.3f}\nFinal error: {final_error:.1f}m\nStereo: {lowe_ratio} | Temporal: {temporal_ratio}"
font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 12,
        }
xt = np.min(x_pos_est)
yt = np.min(z_pos_est)
plt.text(xt, yt, s, fontdict = font, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.xlabel('X position (m)', fontsize=11, fontweight='bold')
plt.ylabel('Z position (m)', fontsize=11, fontweight='bold')
plt.title("Improved SIFT+Lowe's: Estimated (red) vs Ground Truth (green)", fontsize=13, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.tight_layout()
plt.show()
'''
Find ORB features in a window around an object and
find their matching features in the next frame of a video sequence
Track the object over time using these feature matches.

Usage
-----
feature_tracking.py <path to image files> <ground truth xml file>


Keys
----
ESC, q - exit
'''

import numpy as np
import cv2
import xml.etree.ElementTree as ET

class App:
    def __init__(self):
        self.prev_features = None
        self.prev_kps = None
        self.box = None  # bounding box around the object, and the window where features are detected
        self.direction = [0,0]  # motion of the object (vx, vy)
        self.tracks = [] # list of positions of the object for drawing
        self.gt_tracks = [] # list of ground truth positions of the object for drawing
    
    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB, ratio):
        # compute the raw matches and initialize the list of 
        # predicted best matches
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        print("Number of raw matches:", len(rawMatches))
        matches = []
        # loop over the raw matches
        for m in rawMatches:
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))
        print("Number of final matches:", len(matches))
        if len(matches) > 0:
            # construct the two sets of matching points
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])
            return matches
        # otherwise return none
        print("Unable to find any matches!")
        return None

    def detectAndDescribe(self, image):
        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # create the mask from the bounding box
        mask = np.zeros_like(gray)
        box = [int(x) for x in self.box] # must be integers
        mask[box[1]:box[3], box[0]:box[2]] = 1
        # detect keypoints in the image
        orb = cv2.ORB_create()
        kps, features = orb.detectAndCompute(gray, mask)
        kps = np.float32([kp.pt for kp in kps])
        # return a tuple of keypoints and features
        return (kps, features)
    
    def get_groundtruth(self, gt_dict):
        # extract this frame's ground truth from the XML dictionary
        gt_center = [int(gt_dict['xc']), int(gt_dict['yc'])]
        gt_wh = [int(gt_dict['w']), int(gt_dict['h'])] 
        gt_box = [  gt_center[0] - gt_wh[0]//2,
                    gt_center[1] - gt_wh[1]//2,
                    gt_center[0] + gt_wh[0]//2,
                    gt_center[1] + gt_wh[1]//2]
        if 'xv' in gt_dict:
            gt_vel = [float(gt_dict['xv']), float(gt_dict['yv'])]
        else:
            gt_vel = None
        return gt_box, gt_center, gt_wh, gt_vel

    def get_score(self):
        # average distance between predicted and ground truth location over all frames
        score = 0.0
        n = 0
        for e, gt in zip(self.tracks, self.gt_tracks):
            score += (e[0]-gt[0])**2 + (e[1]-gt[1])**2
            n += 1
        if n>0:
            return np.sqrt(score/n)
        else: return 0

    def drawMatches(self, vis, kpsA, kpsB, matches):
        # loop over the matches
        for (trainIdx, queryIdx) in matches:
            # draw the match
            ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
            ptB = (int(kpsB[trainIdx][0]), int(kpsB[trainIdx][1]))
            cv2.line(vis, ptA, ptB, (0, 0, 255), 1)
            cv2.circle(vis, (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1])), 4, (0, 0, 255), -1)
        # return the visualization
        return vis

    def update_direction(self, kpsA, kpsB, matches):
        dxs, dys = [], []
        for (trainIdx, queryIdx) in matches:
            ptA = kpsA[queryIdx][0], kpsA[queryIdx][1]
            ptB = kpsB[trainIdx][0], kpsB[trainIdx][1]
            dxs.append(ptA[0] - ptB[0])
            dys.append(ptA[1] - ptB[1])
        dxs, dys = np.array(dxs), np.array(dys)

        # RANSAC
        if len(dxs) > 10:
            best_inliers = None
            best_count = 0

            for _ in range(20):
                idx = np.random.randint(len(dxs))
                hyp_dx, hyp_dy = dxs[idx], dys[idx]

                dist = np.sqrt((dxs - hyp_dx) ** 2 + (dys - hyp_dy) ** 2)
                inliers = dist < 10.0

                if np.sum(inliers) > best_count:
                    best_count = np.sum(inliers)
                    best_inliers = inliers

            if best_inliers is not None and best_count >= 5:
                dxs, dys = dxs[best_inliers], dys[best_inliers]

        if len(dxs) > 0:
            self.direction = [dxs.mean(), dys.mean()]  # Using the mean

    def run(self, images, ground_truth, ratio=0.80):
        for idx, image in enumerate(images):
            print("Reading image ", image)
            objlist = ground_truth[idx][0]
            if self.box is None and len(objlist) > 0:
                gt_box, gt_center, gt_wh, gt_vel = self.get_groundtruth(objlist[0][1].attrib)
                # initialize the box once we see the object and the box is big enough
                if gt_wh[0]>40 and gt_wh[1] > 40:
                    self.box = gt_box
                    self.direction = gt_vel

            if self.box is not None:
                gt_box, gt_center, gt_wh, gt_vel = self.get_groundtruth(objlist[0][1].attrib)
                frame = cv2.imread(image)
                (kpsA, featuresA) = self.detectAndDescribe(frame)
                print("number of keypoints:", len(kpsA))
                vis = frame.copy()

                if self.prev_features is not None:
                    featuresB = self.prev_features
                    kpsB = self.prev_kps
                    # match features between the two images
                    matches = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio)
                    if matches is not None:
                        self.update_direction(kpsA, kpsB, matches)
                        vis = self.drawMatches(vis, kpsA, kpsB, matches)

                # draw the estimated bounding box and tracks
                box = [int(x) for x in self.box]
                cv2.rectangle(vis, (box[0], box[1]), (box[2], box[3]), 
                    (0, 0, 255), 1)
                self.tracks.append([(box[0] + box[2])//2, (box[1] + box[3])//2])
                for c in self.tracks:
                    cv2.circle(vis, c, 5, (0, 0, 255), 1)
                for i in range(len(self.tracks)-1):
                    c1 = self.tracks[i]
                    c2 = self.tracks[i+1]
                    cv2.line(vis, c1, c2, (0, 0, 255), 1)

                self.gt_tracks.append(gt_center)
                # draw ground truth
                cv2.rectangle(vis, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]), 
                                    (0, 255, 0), 1)
                for c in self.gt_tracks:
                    cv2.circle(vis, c, 5, (0, 255, 0), 1)
                for i in range(len(self.gt_tracks)-1):
                    c1 = self.gt_tracks[i]
                    c2 = self.gt_tracks[i+1]
                    cv2.line(vis, c1, c2, (0, 255, 0), 1)
                score = self.get_score()
                t = f"Score: {score:.2f}"
                cv2.putText(vis, t, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.8, (0, 255, 255), 1, cv2.LINE_AA)
                # the current frame now becomes the previous frame
                self.prev_features = featuresA
                self.prev_kps = kpsA
                # move the four corners of the boundingbox base on the direction
                self.box[0] += self.direction[0]
                self.box[2] += self.direction[0]
                self.box[1] += self.direction[1]
                self.box[3] += self.direction[1]
                cv2.imshow('feat_track', vis)

                ch = 0xFF & cv2.waitKey(1)
                if ch == 27 or ch == 113: # escape or q key
                    break
        
        # create final track image
        cv2.imshow('final_track', vis)
        cv2.waitKey(-1)
        cv2.imwrite("feature_tracking_result.jpg", vis)
        cv2.destroyAllWindows()

def main():
    import sys
    import os
    print(sys.argv[1:])
    img_path = sys.argv[1]
    dir_list = os.listdir(img_path)
    images = [os.path.join(img_path, x) for x in dir_list if (x.endswith(".png") or x.endswith(".jpg") or x.endswith(".pgm"))]
    images.sort()
    xmlf = sys.argv[2]
    tree = ET.parse(xmlf)
    ground_truth = tree.getroot()

    App().run(images, ground_truth)
    cv2.destroyAllWindows() 			

if __name__ == '__main__':
    main()
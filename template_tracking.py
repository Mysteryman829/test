'''
Use a template/window around an object find it in the next frame of a video sequence
Track the object over time using the highest correlation of the template

Usage
-----
template_tracking.py <path to image files> <ground truth xml file>


Keys
----
ESC, q - exit
'''

import numpy as np
import cv2
import xml.etree.ElementTree as ET

class App:
    def __init__(self, max_search, use_gaussian=True):
        self.template = None
        self.box = None
        self.center = None
        self.direction = [0, 0]
        self.tracks = []
        self.max_search = max_search
        self.gt_tracks = []
        self.use_gaussian = use_gaussian

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

    def update_state(self, cx, cy, gray, update_template=False):
        """update_templateé»˜è®¤æ”¹ä¸ºFalse"""
        self.direction = [cx - self.center[0], cy - self.center[1]]
        self.center = [cx, cy]
        self.box[0] += self.direction[0]
        self.box[2] += self.direction[0]
        self.box[1] += self.direction[1]
        self.box[3] += self.direction[1]

        if update_template:
            box = [int(x) for x in self.box]
            self.template = gray[box[1]:box[3], box[0]:box[2]]

    def run(self, images, ground_truth):
        res = None

        for idx, image in enumerate(images):
            print("Reading image ", image)
            frame = cv2.imread(image)
            vis = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            objlist = ground_truth[idx][0]

            if self.template is not None:
                gt_box, gt_center, gt_wh, gt_vel = self.get_groundtruth(objlist[0][1].attrib)

                th, tw = self.template.shape
é‡
                if self.use_gaussian:
                    y, x = np.ogrid[:th, :tw]
                    cy, cx = th / 2, tw / 2


                    sigma_y = th / 10.0
                    sigma_x = tw / 10.0

                    weights = np.exp(-((x - cx) ** 2 / (2 * sigma_x ** 2) +
                                       (y - cy) ** 2 / (2 * sigma_y ** 2)))


                    weights = np.float32(weights)

                    res = cv2.matchTemplate(np.float32(gray),
                                            np.float32(self.template),
                                            cv2.TM_CCOEFF_NORMED,
                                            mask=weights)
                else:
                    res = cv2.matchTemplate(gray, self.template, cv2.TM_CCOEFF_NORMED)

                hg, wg = res.shape


                predicted_cx = self.center[0] + self.direction[0]
                predicted_cy = self.center[1] + self.direction[1]


                predicted_left = predicted_cx - tw / 2
                predicted_top = predicted_cy - th / 2

                ms_box = [
                    int(max(0, predicted_left - self.max_search[0])),
                    int(max(0, predicted_top - self.max_search[1])),
                    int(min(wg, predicted_left + self.max_search[0])),
                    int(min(hg, predicted_top + self.max_search[1])),
                ]


                mask = np.zeros_like(res)
                if ms_box[2] > ms_box[0] and ms_box[3] > ms_box[1]:
                    mask[ms_box[1]:ms_box[3], ms_box[0]:ms_box[2]] = 1
                else:
                    mask = np.ones_like(res)

                res_masked = res * mask

                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res_masked)
                print(f"Max correlation: {max_val:.3f}")

                top_left = max_loc
                cx = top_left[0] + tw / 2
                cy = top_left[1] + th / 2

                self.update_state(cx, cy, gray, update_template=False)

                box = [int(x) for x in self.box]
                cv2.rectangle(vis, (box[0], box[1]), (box[2], box[3]),
                              (0, 0, 255), 1)

                self.tracks.append([self.center[0], self.center[1]])

                for c in self.tracks:
                    cv2.circle(vis, (int(c[0]), int(c[1])), 5, (0, 0, 255), 1)
                for i in range(len(self.tracks) - 1):
                    c1 = (int(self.tracks[i][0]), int(self.tracks[i][1]))
                    c2 = (int(self.tracks[i + 1][0]), int(self.tracks[i + 1][1]))
                    cv2.line(vis, c1, c2, (0, 0, 255), 1)

                self.gt_tracks.append(gt_center)
                cv2.rectangle(vis, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]),
                              (0, 255, 0), 1)
                for c in self.gt_tracks:
                    cv2.circle(vis, c, 5, (0, 255, 0), 1)
                for i in range(len(self.gt_tracks) - 1):
                    c1 = self.gt_tracks[i]
                    c2 = self.gt_tracks[i + 1]
                    cv2.line(vis, c1, c2, (0, 255, 0), 1)

                score = self.get_score()
                t = f"Score: {score:.2f}"
                cv2.putText(vis, t, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 255), 1, cv2.LINE_AA)

                cv2.imshow('temp_track', vis)
                cv2.imshow('template', self.template)
                if res is not None:
                    cv2.imshow('correlation', res_masked)

                ch = 0xFF & cv2.waitKey(1)
                if ch == 27 or ch == 113:
                    break
            else:
                if len(objlist) > 0:
                    gt_box, gt_center, gt_wh, gt_vel = self.get_groundtruth(objlist[0][1].attrib)
                    if gt_wh[0] > 40 and gt_wh[1] > 40:
                        self.template = gray[gt_box[1]:gt_box[3], gt_box[0]:gt_box[2]]
                        self.box = gt_box[:]
                        self.direction = gt_vel if gt_vel is not None else [0, 0]
                        self.center = [gt_center[0], gt_center[1]]

        cv2.imshow('final_track', vis)
        cv2.waitKey(-1)
        cv2.imwrite("template_tracking_result.jpg", vis)
        cv2.destroyAllWindows()

def main():
    import sys
    import os
    print(sys.argv[1])
    path = sys.argv[1]
    dir_list = os.listdir(path)
    images = [os.path.join(path, x) for x in dir_list if
        (x.endswith(".png") or x.endswith(".jpg") or x.endswith(".pgm"))]
    images.sort()
    xmlf = sys.argv[2]
    tree = ET.parse(xmlf)
    ground_truth = tree.getroot()

    if "walk" in path.lower() or "wk" in xmlf.lower():
        max_search = [32, 32]
        use_gaussian = False 
        print("Detected Walk3 dataset: max_search=[32, 32], Gaussian=OFF")
    else:
        max_search = [100, 100]
        use_gaussian = True  
        print("Detected fish dataset: max_search=[100, 100], Gaussian=ON")

    App(max_search, use_gaussian).run(images, ground_truth)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

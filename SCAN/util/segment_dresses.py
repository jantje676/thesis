from math import floor
import numpy as np

def segment_dresses(img):
    segments = {}
    H, W, C = img.shape
    # 1=x_1, 2=y_1, 3=x_2, 4 =y_2 linkerbovenhoek=(x_1, y_1) rechteronderhoek=(x_2, y_2)
    bboxes = np.array([[0,0,W, floor(0.35*H)],[0,floor(0.35*H),W,H],[0,floor(0.35*H),W,floor(0.75*H)],
                      [0,0,W,floor(0.2*H)],[0,0,floor(0.5*W),floor(0.5*H)],[floor(0.5*W),0,W,floor(0.5*H)]])

    # segment dress in seven parts according to laenen
    segments["top"] = img[: floor(0.35*H) , : , :]
    segments["full_skirt"] = img[floor(0.35*H):  , : , :]
    segments["skirt_above_knee"] = img[floor(0.35*H): floor(0.75*H) , : , :]
    segments["neckline"] = img[: floor(0.2*H) , : , :]
    segments["left_sleeve"] = img[: floor(0.5*H) , : floor(0.5*W) , :]
    segments["right_sleeve"] = img[: floor(0.5*H) , floor(0.5*W): , :]
    segments["full"] = img

    return segments, bboxes

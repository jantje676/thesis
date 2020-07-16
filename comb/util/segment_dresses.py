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

def segment_dresses_tile(img):
    segments = {}
    H, W, C = img.shape
    # 1=x_1, 2=y_1, 3=x_2, 4 =y_2 linkerbovenhoek=(x_1, y_1) rechteronderhoek=(x_2, y_2)
    bboxes = np.array([[0,0,W, floor(0.35*H)],[0,floor(0.35*H),W,H],[0,floor(0.35*H),W,floor(0.75*H)],
                      [0,0,W,floor(0.2*H)],[0,0,floor(0.5*W),floor(0.5*H)],[floor(0.5*W),0,W,floor(0.5*H)]])

    segments["top"] = img[: floor(0.33*H) , : floor(0.5*W) , :]
    segments["full_skirt"] = img[: floor(0.33*H) ,  floor(0.5*W): , :]
    segments["skirt_above_knee"] = img[floor(0.33*H): floor(0.66*H) , : floor(0.5*W) , :]
    segments["neckline"] = img[floor(0.33*H): floor(0.66*H) , floor(0.5*W):  , :]
    segments["left_sleeve"] = img[floor(0.66*H):  , : floor(0.5*W) , :]
    segments["right_sleeve"] = img[floor(0.66*H): , floor(0.5*W):  , :]
    segments["full"] = img

    return segments, bboxes



def segment_dresses_tile_nine(img):
    H, W, C = img.shape
    segments = {}

    bboxes = np.array([[0,0,W, floor(0.35*H)],[0,floor(0.35*H),W,H],[0,floor(0.35*H),W,floor(0.75*H)],
                      [0,0,W,floor(0.2*H)],[0,0,floor(0.5*W),floor(0.5*H)],[floor(0.5*W),0,W,floor(0.5*H)],
                      [floor(0.5*W),0,W,floor(0.5*H)], [floor(0.5*W),0,W,floor(0.5*H)], [floor(0.5*W),0,W,floor(0.5*H)] ])

    segments["one"] = img[: floor(0.33*H) , : floor(0.33*W) , :]
    segments["two"] = img[: floor(0.33*H) ,  floor(0.33*W): floor(0.66*W) , :]
    segments["three"] = img[:floor(0.33*H) ,floor(0.66*W) : , :]
    segments["four"] = img[floor(0.33*H): floor(0.66*H) , : floor(0.33*W) , :]
    segments["five"] = img[floor(0.33*H): floor(0.66*H) ,floor(0.33*W) : floor(0.66*W) , :]
    segments["six"] = img[floor(0.33*H): floor(0.66*H) ,  floor(0.66*W): , :]
    segments["seven"] = img[floor(0.66*H): , : floor(0.33*W) , :]
    segments["eight"] = img[floor(0.66*H): , floor(0.33*W): floor(0.66*W) , :]
    segments["nine"] = img[floor(0.66*H): , floor(0.66*W):  , :]
    segments["full"] = img
    return segments, bboxes

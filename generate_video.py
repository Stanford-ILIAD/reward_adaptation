import numpy as np
from PIL import Image
import cv2
import sys

def process_traj(traj_file):
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('output.mp4',fourcc, 10.0, (400,400))
    lines = open(traj_file).readlines()
    radius = 0.125
    height = 400
    width = 400
    obstacle = []
    for i in range(40):
        for j in range(40):
            if ((i-20)**2+(j-20)**2) < 40 * 0.125:
                obstacle.append([j, i])
    img = np.zeros([41, 41, 3]).astype(np.uint8)
    for pair in obstacle:
            img[pair[0], pair[1], :] = 255
    prototype = img
    for line in lines[0:5000]:
        img = np.array(prototype)
        values = [float(value) for value in line.split()]
        values_pos = []
        for i in range((len(values)-1)//2):
            values_pos.append([values[i*2], values[i*2+1]])
        ret = values[-1]
        for value in values_pos:
            x_pos = int(value[0] * 20 + 20)
            y_pos = int((-value[1]) * 20 + 20)
            img[y_pos, x_pos, :] = 255
        img = cv2.resize(img, (width, height))
        out.write(img)
    out.release()
    cv2.destroyAllWindows()

process_traj(sys.argv[1])

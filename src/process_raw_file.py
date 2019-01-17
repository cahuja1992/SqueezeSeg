import sys
import os.path
import numpy as np
from PIL import Image


def filter(x, y, z, fov, fov_type='h'):
    """Filter the points in the fov and the """
    d = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    if fov_type == 'h':
        return np.logical_and(np.arctan2(y, x) > (-fov[1] * np.pi / 180), \
                                np.arctan2(y, x) < (-fov[0] * np.pi / 180))
    elif fov_type == 'v':
        return np.logical_and(np.arctan2(z, d) < (fov[1] * np.pi / 180), \
                                np.arctan2(z, d) > (fov[0] * np.pi / 180))
    else:
        raise NameError("fov type must be set between 'h' and 'v' ")

def get_depth_map(points, H=64, W=512, C=5, dtheta=np.radians(0.4), dphi=np.radians(90./512.0)):
        x, y, z, i = points[:, 0], points[:, 1], points[:, 2], points[:, 3]
        d = np.sqrt(x ** 2 + y ** 2 + z**2)
        r = np.sqrt(x ** 2 + y ** 2)
        d[d==0] = 0.000001
        r[r==0] = 0.000001
        phi = np.radians(45.) - np.arcsin(y/r)
        phi_ = (phi/dphi).astype(int)
        phi_[phi_<0] = 0
        phi_[phi_>=512] = 511

        theta = np.radians(2.) - np.arcsin(z/d)
        theta_ = (theta/dtheta).astype(int)
        theta_[theta_<0] = 0
        theta_[theta_>=64] = 63

        depth_map = np.zeros((H, W, C))
        
        depth_map[theta_, phi_, 0] = x
        depth_map[theta_, phi_, 1] = y
        depth_map[theta_, phi_, 2] = z
        depth_map[theta_, phi_, 3] = i
        depth_map[theta_, phi_, 4] = d
        return depth_map

if __name__ == "__main__":
    lidar_raw_path = os.path.join('./data', 'lidar_raw')
    lidar_2d = os.path.join('./data', 'lidar_2d') 
    with open('./data/ImageSet/test.txt','rb') as f:
        for line in f.readlines():
            line = line.decode("utf-8")
            binary_path = os.path.join(lidar_raw_path, line+'.bin')
            depth_frame = os.path.join(lidar_2d, line+'.npy')

            points = np.fromfile(binary_path, dtype=np.float32).reshape(-1, 4)
            points = points[filter(x=points[:, 0], y=points[:, 1], z=points[:, 2], fov=[-45, 45])]
            lidar = get_depth_map(points).astype(np.float32)
            np.save(depth_frame, lidar) 



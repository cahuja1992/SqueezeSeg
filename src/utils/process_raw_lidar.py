import sys
import os.path
import numpy as np
from PIL import Image

import tensorflow as tf

import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import Image as ImageMsg
from std_msgs.msg import Header

from ..config import *
from ..nets import SqueezeSeg

def _normalize(x):
    return (x - x.min()) / (x.max() - x.min())

class SegmentationNode():

    def __init__(self,
                 sub_topic, pub_topic, pub_feature_map_topic, pub_label_map_topic,
                 FLAGS):
        os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

        self._mc = kitti_squeezeSeg_config()
        self._mc.LOAD_PRETRAINED_MODEL = False
        
        self._mc.BATCH_SIZE = 1
        self._model = SqueezeSeg(self._mc)
        self._saver = tf.train.Saver(self._model.model_params)

        self._session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self._saver.restore(self._session, FLAGS.checkpoint)

        self._sub = rospy.Subscriber(sub_topic, PointCloud2, self.point_cloud_callback, queue_size=1)
        self._pub = rospy.Publisher(pub_topic, PointCloud2, queue_size=1)
        self._feature_map_pub = rospy.Publisher(pub_feature_map_topic, ImageMsg, queue_size=1)
        self._label_map_pub = rospy.Publisher(pub_label_map_topic, ImageMsg, queue_size=1)

        rospy.spin()

    def point_cloud_callback(self, cloud_msg):
        pc = pc2.read_points(cloud_msg, skip_nans=False, field_names=("x", "y", "z","intensity"))
        
        np_p = np.array(list(pc))
        
        cond = self.hv_in_range(x=np_p[:, 0],
                                y=np_p[:, 1],
                                z=np_p[:, 2],
                                fov=[-45, 45])
        
        np_p_ranged = np_p[cond]

        # get depth map
        lidar = self.pto_depth_map(velo_points=np_p_ranged, C=5)
        lidar_f = lidar.astype(np.float32)

        # to perform prediction
        lidar_mask = np.reshape(
            (lidar[:, :, 4] > 0),
            [self._mc.ZENITH_LEVEL, self._mc.AZIMUTH_LEVEL, 1]
        )
        lidar_f = (lidar_f - self._mc.INPUT_MEAN) / self._mc.INPUT_STD
        pred_cls = self._session.run(
            self._model.pred_cls,
            feed_dict={
                self._model.lidar_input: [lidar_f],
                self._model.keep_prob: 1.0,
                self._model.lidar_mask: [lidar_mask]
            }
        )
        label = pred_cls[0]

        # # generated depth map from LiDAR data
        depth_map = Image.fromarray(
            (255 * _normalize(lidar[:, :, 3])).astype(np.uint8))
      
        label_3d = np.zeros((label.shape[0], label.shape[1], 3))
        label_3d[np.where(label==0)] = [1., 1., 1.]
        label_3d[np.where(label==1)] = [0., 1., 0.]
        label_3d[np.where(label==2)] = [1., 1., 0.]
        label_3d[np.where(label==3)] = [0., 1., 1.]

        x = lidar[:, :, 0].reshape(-1)
        y = lidar[:, :, 1].reshape(-1)
        z = lidar[:, :, 2].reshape(-1)
        i = lidar[:, :, 3].reshape(-1)
        label = label.reshape(-1)
        # cond = (label!=0)
        # print(cond)
        cloud = np.stack((x, y, z, i, label))
        # cloud = np.stack((x, y, z, i))

        label_map = Image.fromarray(
            (255 * _normalize(label_3d)).astype(np.uint8))

        header = Header()
        header.stamp = rospy.Time()
        header.frame_id = "velodyne"
        # feature map & label map
        msg_feature = ImageConverter.to_ros(depth_map)
        msg_feature.header = header
        msg_label = ImageConverter.to_ros(label_map)
        msg_label.header = header

        # point cloud segments
        # 4 PointFields as channel description
        msg_segment = pc2.create_cloud(header=header,
                                       fields=_make_point_field(cloud.shape[0]),
                                       points=cloud.T)

        self._feature_map_pub.publish(msg_feature)
        self._label_map_pub.publish(msg_label)
        self._pub.publish(msg_segment)
        rospy.loginfo("Point cloud processed. Took %.6f ms.", clock.takeRealTime())

    def hv_in_range(self, x, y, z, fov, fov_type='h'):
        d = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        if fov_type == 'h':
            return np.logical_and(np.arctan2(y, x) > (-fov[1] * np.pi / 180), \
                                  np.arctan2(y, x) < (-fov[0] * np.pi / 180))
        elif fov_type == 'v':
            return np.logical_and(np.arctan2(z, d) < (fov[1] * np.pi / 180), \
                                  np.arctan2(z, d) > (fov[0] * np.pi / 180))
        else:
            raise NameError("fov type must be set between 'h' and 'v' ")

    def pto_depth_map(self, velo_points, H=64, W=512, C=5, dtheta=np.radians(0.4), dphi=np.radians(90./512.0)):
        x, y, z, i = velo_points[:, 0], velo_points[:, 1], velo_points[:, 2], velo_points[:, 3]
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
        # 5 channels according to paper
        if C == 5:
            depth_map[theta_, phi_, 0] = x
            depth_map[theta_, phi_, 1] = y
            depth_map[theta_, phi_, 2] = z
            depth_map[theta_, phi_, 3] = i
            depth_map[theta_, phi_, 4] = d
        else:
            depth_map[theta_, phi_, 0] = i
        return depth_map
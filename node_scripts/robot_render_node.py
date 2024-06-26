#!/usr/bin/env python

from collections import OrderedDict
import numpy as np
import cv2
from pyvirtualdisplay import Display

import rospy
import tf
from sensor_msgs.msg import Image, CameraInfo, JointState
from image_geometry import PinholeCameraModel
from cv_bridge import CvBridge

import urdfpy
import pyrender
from pyrender import OffscreenRenderer

from utils import create_raymond_lights, get_processed_urdf_path


class RobotRenderNode(object):
    def __init__(self):
        super(RobotRenderNode, self).__init__()
        self.display = Display(visible=0, size=(640, 480))
        self.display.start()
        self.cv_bridge = CvBridge()

        self.fps = rospy.get_param("~fps", 30)
        self.base_frame = rospy.get_param("~base_frame", "base_link")
        try:
            img_msg = rospy.wait_for_message("~input_image", Image, timeout=10)
            self.image_frame = img_msg.header.frame_id
            self.viewport_height, self.viewport_width = self.cv_bridge.imgmsg_to_cv2(img_msg, "bgr8").shape[:2]
        except rospy.ROSException:
            rospy.logerr("Failed to get image size from {}".format("~input_image"))
        self.renderer = OffscreenRenderer(viewport_width=self.viewport_width, viewport_height=self.viewport_height)
        self.scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))

        # Set lights
        for light_node in create_raymond_lights():
            self.scene.add_node(light_node)

        # set robot
        self.robot_urdf = get_processed_urdf_path(rospy.get_param("~robot_urdf"))
        self.robot = urdfpy.URDF.load(self.robot_urdf)
        self.fk = self.robot.visual_trimesh_fk()  # dict, key: trimesh, value: 4x4 matrix
        self.node_map = OrderedDict()

        # add mesh to scene
        for tm in self.fk:
            pose = self.fk[tm]
            mesh = pyrender.Mesh.from_trimesh(tm, smooth=False)
            node = self.scene.add(mesh, pose=pose)
            self.node_map[tm] = node

        self.joint_names = [joint.name for joint in self.robot.joints]

        # Set camera
        self.camera_info = rospy.wait_for_message("~camera_info", CameraInfo, timeout=10)
        self.camera_model = PinholeCameraModel()
        self.camera_model.fromCameraInfo(self.camera_info)
        self.camera = pyrender.IntrinsicsCamera(
            fx=self.camera_model.fx(), fy=self.camera_model.fy(), cx=self.camera_model.cx(), cy=self.camera_model.cy()
        )
        self.camera_node = self.scene.add(self.camera, pose=np.eye(4))

        # set subscriber and publisher
        self.pub_img = rospy.Publisher("~render_image", Image, queue_size=1)
        self.pub_mask = rospy.Publisher("~render_mask", Image, queue_size=1)
        self.tf_listener = tf.TransformListener()
        self.sub_joint = rospy.Subscriber("~joint_states", JointState, self.joint_callback)
        self.sub_img = rospy.Subscriber("~input_image", Image, self.img_callback, queue_size=1, buff_size=2**24)
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.fps), self.timer_callback)

    def joint_callback(self, msg):
        self.joint_states = {name: angle for name, angle in zip(msg.name, msg.position)}

    def img_callback(self, msg):
        self.image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")

    def timer_callback(self, event):
        try:
            # get camera pose and apply to camera node
            camera_trans, camera_quat = self.tf_listener.lookupTransform(
                self.base_frame, self.image_frame, rospy.Time(0)
            )
            camera_rot = tf.transformations.quaternion_matrix(camera_quat)[:3, :3]
            camera_frame = np.identity(4)
            camera_frame[:3, 3] = camera_trans
            camera_frame[:3, :3] = camera_rot
            coordinate_transform = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            camera_frame = camera_frame @ coordinate_transform
            self.camera_node.matrix = camera_frame
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return

        # get joint states for update rendered robot joints
        filtered_joint_names = [name for name in self.joint_names if name in self.joint_states]
        joint_angles = {joint_name: self.joint_states[joint_name] for joint_name in filtered_joint_names}

        # apply forward kinematics to mesh
        self.fk = self.robot.visual_trimesh_fk(cfg=joint_angles)
        for mesh in self.fk:
            pose = self.fk[mesh]
            self.node_map[mesh].matrix = pose

        # render robot on image
        rgba, depth = self.renderer.render(self.scene, flags=pyrender.RenderFlags.RGBA)
        rgb = cv2.cvtColor(rgba[:, :, :3].copy(), cv2.COLOR_RGB2BGR)
        alpha = rgba[:, :, 3].copy().astype(np.float32) / 255.0
        render_image = rgb * alpha[..., None] + self.image * (1 - alpha[..., None])
        render_image = render_image.astype(np.uint8)

        # publish
        render_img_msg = self.cv_bridge.cv2_to_imgmsg(render_image, "bgr8")
        render_img_msg.header.stamp = rospy.Time.now()
        render_img_msg.header.frame_id = self.image_frame
        self.pub_img.publish(render_img_msg)

        render_mask_msg = self.cv_bridge.cv2_to_imgmsg((alpha * 255).astype(np.uint8), "mono8")
        render_mask_msg.header.stamp = rospy.Time.now()
        render_mask_msg.header.frame_id = self.image_frame
        self.pub_mask.publish(render_mask_msg)


if __name__ == "__main__":
    rospy.init_node("robot_render_node")
    node = RobotRenderNode()
    rospy.spin()
    node.display.stop()

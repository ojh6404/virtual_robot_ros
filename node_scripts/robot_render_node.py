#!/usr/bin/env python

import os
import numpy as np
import cv2
from collections import OrderedDict
from pyvirtualdisplay import Display

import rospy
from sensor_msgs.msg import Image, CameraInfo, JointState
from image_geometry import PinholeCameraModel
from cv_bridge import CvBridge

import urdfpy
import pyrender
import skrobot

from virtual_robot_ros.utils import create_raymond_lights, get_processed_urdf_path


if os.environ.get("PYOPENGL_PLATFORM") is None:
    os.environ["PYOPENGL_PLATFORM"] = "egl"


class RobotRenderNode(object):
    def __init__(self):
        super(RobotRenderNode, self).__init__()
        self.display = Display(visible=0, size=(640, 480))
        self.display.start()
        self.cv_bridge = CvBridge()

        self.fps = rospy.get_param("~fps", 30)
        self.base_frame = rospy.get_param("~base_frame", "base_footprint")
        self.camera_frame = rospy.get_param("~camera_frame", "head_mount_kinect_rgb_optical_frame")

        try:
            img_msg = rospy.wait_for_message("~input_image", Image, timeout=10)
            self.image_frame = img_msg.header.frame_id
            self.viewport_height, self.viewport_width = self.cv_bridge.imgmsg_to_cv2(img_msg, "bgr8").shape[:2]
        except rospy.ROSException:
            rospy.logerr("Failed to get image size from {}".format("~input_image"))
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=self.viewport_width, viewport_height=self.viewport_height
        )
        self.scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))
        self.image = None

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

        # set skrobot model
        self.skrobot_model = skrobot.models.urdf.RobotModelFromURDF(urdf_file=self.robot_urdf)
        self.skrobot_joint_names = [joint.name for joint in self.skrobot_model.joint_list]
        self.skrobot_link_names = [link.name for link in self.skrobot_model.link_list]

        # set subscriber and publisher
        self.joint_states = {}
        self.pub_img = rospy.Publisher("~render_image", Image, queue_size=1)
        self.pub_mask = rospy.Publisher("~render_mask", Image, queue_size=1)
        self.sub_joint = rospy.Subscriber("~joint_states", JointState, self.joint_callback)
        self.sub_img = rospy.Subscriber("~input_image", Image, self.img_callback, queue_size=1, buff_size=2**24)

        while not rospy.is_shutdown():
            if self.image is not None:
                # update skrobot model with real joint states
                for _ in self.joint_states:
                    angle_vector = np.array([self.joint_states[name] for name in self.skrobot_joint_names])
                    self.skrobot_model.angle_vector(angle_vector)

                # TODO add transform from base to camera
                camera_frame = (
                    self.skrobot_model.link_list[self.skrobot_link_names.index(self.camera_frame)].worldcoords().T()
                )
                coordinate_transform = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
                camera_frame = camera_frame @ coordinate_transform
                self.camera_node.matrix = camera_frame

                # get joint states for update rendered robot joints
                try:
                    filtered_joint_names = [name for name in self.joint_names if name in self.joint_states]
                    joint_angles = {joint_name: self.joint_states[joint_name] for joint_name in filtered_joint_names}
                except AttributeError:
                    rospy.logwarn("Failed to get joint states")
                    continue

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

    def joint_callback(self, msg):
        self.joint_states.update({name: angle for name, angle in zip(msg.name, msg.position)})

    def img_callback(self, msg):
        self.image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")


if __name__ == "__main__":
    rospy.init_node("robot_render_node")
    node = RobotRenderNode()
    rospy.spin()
    node.display.stop()

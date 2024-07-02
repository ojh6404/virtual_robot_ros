#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script is used to process rosbag files and save the processed results without roscore.
"""

import message_filters

from scipy import signal
from std_msgs.msg import String
from sensor_msgs.msg import CameraInfo, Image, CompressedImage, JointState
from geometry_msgs.msg import PoseStamped
import rospy
import rosbag
import argparse
from tqdm import tqdm
from cv_bridge import CvBridge
import numpy as np
import cv2
import pyrender
from pyrender import OffscreenRenderer
from pyvirtualdisplay import Display
import urdfpy

from collections import OrderedDict
import tf
import tf2_ros

from utils import create_raymond_lights, get_processed_urdf_path


from eus_imitation_msgs.msg import FloatVector
from eus_imitation_msgs.srv import RobotIK, RobotIKRequest, RobotIKResponse
from copy import deepcopy

import os

if os.environ.get("PYOPENGL_PLATFORM") is None:
    os.environ["PYOPENGL_PLATFORM"] = "egl"


bridge = CvBridge()


current_joint_states = None


class RealtimeLowPassFilter:
    def __init__(self, cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        self.b, self.a = signal.butter(order, normal_cutoff, btype="low", analog=False)
        self.z = signal.lfilter_zi(self.b, self.a)

    def filter(self, data):
        filtered, self.z = signal.lfilter(self.b, self.a, [data], zi=self.z)
        return filtered[0]


def main(args):
    # caching tf messages
    tf_buffer = tf2_ros.Buffer(rospy.Duration(1000000.0))
    with rosbag.Bag(args.rosbag) as input_bag:
        for topic, msg, t in input_bag.read_messages(topics=["/tf", "/tf_static"]):
            for msg_tf in msg.transforms:
                if topic == "/tf_static":
                    tf_buffer.set_transform_static(msg_tf, "default_authority")
                else:
                    tf_buffer.set_transform(msg_tf, "default_authority")

    with rosbag.Bag(args.output, "w") as outbag:
        display = Display(visible=0, size=(640, 480))
        display.start()

        renderer = OffscreenRenderer(viewport_width=640, viewport_height=480)
        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))

        # Set lights
        for light_node in create_raymond_lights():
            scene.add_node(light_node)

        # set robot
        robot_urdf = get_processed_urdf_path("/home/leus/pr2.urdf")
        robot = urdfpy.URDF.load(robot_urdf)
        fk = robot.visual_trimesh_fk()  # dict, key: trimesh, value: 4x4 matrix
        node_map = OrderedDict()

        # add mesh to scene
        for tm in fk:
            pose = fk[tm]
            mesh = pyrender.Mesh.from_trimesh(tm, smooth=False)
            node = scene.add(mesh, pose=pose)
            node_map[tm] = node
        joint_names = [joint.name for joint in robot.joints]
        joint_limit_cfgs = robot.joint_limit_cfgs  # tuple of dict, key: JointObject, value: min or max
        max_joints = joint_limit_cfgs[0]
        min_joints = joint_limit_cfgs[1]
        gripper_joints = [
            joint for joint in robot.actuated_joints if "gripper" in joint.name and joint.joint_type != "continuous"
        ]
        gripper_max_cfg = {joint: max_joints[joint] for joint in gripper_joints}
        gripper_min_cfg = {joint: min_joints[joint] for joint in gripper_joints}

        # gripper
        hand_close = 0.01
        hand_open = 0.16
        scale = {
            joint: (gripper_max_cfg[joint] - gripper_min_cfg[joint]) / (hand_open - hand_close)
            for joint in gripper_joints
        }

        # Set camera
        camera = pyrender.IntrinsicsCamera(fx=525.0, fy=525.0, cx=319.5, cy=239.5)
        camera_node = scene.add(camera, pose=np.eye(4))

        # ==================================================================== #
        # topics = [
        #     "/inpaint/compressed",
        #     "/joint_states",
        #     "/left_hand_gripper_frame",
        #     "/right_hand_gripper_frame",
        # ]
        topics = [
            "/kinect_head/rgb/image_rect_color/compressed",
            "/joint_states",
            "/left_hand/grasp_pose",
            "/right_hand/grasp_pose",
        ]
        msgs = [CompressedImage, CameraInfo, JointState, PoseStamped, PoseStamped]
        subscribers = {topic: message_filters.Subscriber(topic, msg) for topic, msg in zip(topics, msgs)}
        ts = message_filters.ApproximateTimeSynchronizer(
            subscribers.values(),
            queue_size=1,
            slop=0.1,
            allow_headerless=False,
        )

        def callback(img_msg, joint_state_msg, left_hand_pose_msg, right_hand_pose_msg):
            image = (
                bridge.compressed_imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
                if "Compressed" in img_msg._type
                else bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
            )
            camera_frame = img_msg.header.frame_id

            global current_joint_states
            if current_joint_states is None:
                current_joint_states = deepcopy(joint_state_msg)

            def request_ik():
                rospy.wait_for_service("/eus_ik_solver")
                ik_client = rospy.ServiceProxy("/eus_ik_solver", RobotIK)
                request = RobotIKRequest()
                request.current_joint_states = current_joint_states
                request.target_coords = [right_hand_pose_msg.pose, left_hand_pose_msg.pose]  # mirror left hand pose
                request.move_end_coords = [String(data="left"), String(data="right")]

                response = ik_client(request)
                return response.target_joint_states

            current_joint_states.position = list(current_joint_states.position)
            target_joint_states = request_ik()
            target_joint_states.header.stamp = joint_state_msg.header.stamp

            # update current joint states with target joint states. target joint states include partial joint states
            for i, name in enumerate(target_joint_states.name):
                idx = current_joint_states.name.index(name)
                current_joint_states.position[idx] = target_joint_states.position[i]

            try:
                # get camera pose and apply to camera node
                transform = tf_buffer.lookup_transform(args.target_frame, camera_frame, joint_state_msg.header.stamp)
                camera_trans = np.array(
                    [
                        transform.transform.translation.x,
                        transform.transform.translation.y,
                        transform.transform.translation.z,
                    ]
                )
                camera_quat = np.array(
                    [
                        transform.transform.rotation.x,
                        transform.transform.rotation.y,
                        transform.transform.rotation.z,
                        transform.transform.rotation.w,
                    ]
                )
                camera_rot = tf.transformations.quaternion_matrix(camera_quat)[:3, :3]
                camera_frame = np.identity(4)
                camera_frame[:3, 3] = camera_trans
                camera_frame[:3, :3] = camera_rot
                coordinate_transform = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
                camera_frame = camera_frame @ coordinate_transform
                camera_node.matrix = camera_frame
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                return

            # get joint states for update rendered robot joints
            joint_states = {name: angle for name, angle in zip(target_joint_states.name, target_joint_states.position)}

            # gripper
            hand_dist = 0.07
            for joint in gripper_joints:
                joint_states[joint.name] = hand_dist * scale[joint] + min_joints[joint]
            # joint_states["r_gripper_motor_slider_joint"] = hand_dist * scale[0] + min_joints[0]
            # joint_states["r_gripper_l_finger_joint"] = hand_dist * scale[1] + min_joints[1]
            # joint_states["r_gripper_joint"] = hand_dist * scale[2] + min_joints[2]

            filtered_joint_names = [name for name in joint_names if name in joint_states]
            joint_angles = {joint_name: joint_states[joint_name] for joint_name in filtered_joint_names}

            # apply forward kinematics to mesh
            fk = robot.visual_trimesh_fk(cfg=joint_angles)
            for mesh in fk:
                pose = fk[mesh]
                node_map[mesh].matrix = pose

            # render robot on image
            rgba, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
            rgb = cv2.cvtColor(rgba[:, :, :3].copy(), cv2.COLOR_RGB2BGR)
            alpha = rgba[:, :, 3].copy().astype(np.float32) / 255.0
            render_image = rgb * alpha[..., None] + image * (1 - alpha[..., None])
            render_image = render_image.astype(np.uint8)

            render_msg = bridge.cv2_to_imgmsg(render_image, "bgr8")
            render_msg.header = img_msg.header
            render_msg.header.frame_id = img_msg.header.frame_id

            # robot state is left hand xyz, roll pitch yaw, right hand xyz, roll pitch yaw
            robot_state_msg = FloatVector()
            robot_state_msg.header.stamp = joint_state_msg.header.stamp
            left_hand_pos = [
                left_hand_pose_msg.pose.position.x,
                left_hand_pose_msg.pose.position.y,
                left_hand_pose_msg.pose.position.z,
            ]
            left_hand_quat = [
                left_hand_pose_msg.pose.orientation.x,
                left_hand_pose_msg.pose.orientation.y,
                left_hand_pose_msg.pose.orientation.z,
                left_hand_pose_msg.pose.orientation.w,
            ]
            left_hand_rpy = tf.transformations.euler_from_quaternion(left_hand_quat)
            right_hand_pos = [
                right_hand_pose_msg.pose.position.x,
                right_hand_pose_msg.pose.position.y,
                right_hand_pose_msg.pose.position.z,
            ]
            right_hand_quat = [
                right_hand_pose_msg.pose.orientation.x,
                right_hand_pose_msg.pose.orientation.y,
                right_hand_pose_msg.pose.orientation.z,
                right_hand_pose_msg.pose.orientation.w,
            ]
            right_hand_rpy = tf.transformations.euler_from_quaternion(right_hand_quat)
            robot_state_msg.data = left_hand_pos + left_hand_rpy + right_hand_pos + right_hand_rpy

            outbag.write("/rendered_image", render_msg, img_msg.header.stamp)
            outbag.write("/target_joint_states", current_joint_states, joint_state_msg.header.stamp)
            outbag.write("/eus_imitation/robot_state", robot_state_msg, joint_state_msg.header.stamp)

        ts.registerCallback(callback)

        print("Processing rosbags...")
        bag_reader = rosbag.Bag(args.rosbag, skip_index=True)

        # message filter
        for message_idx, (topic, msg, t) in tqdm(enumerate(bag_reader.read_messages(topics=topics))):
            subscriber = subscribers[topic]
            if subscriber:
                subscriber.signalMessage(msg)

        # write original messages
        print("Writing original messages...")
        with rosbag.Bag(args.rosbag, "r") as input_bag:
            for topic, msg, t in tqdm(input_bag.read_messages()):
                outbag.write(topic, msg, t)

        # write some timer msg at 10hz
        print("Writing timer messages...")
        with rosbag.Bag(args.rosbag, "r") as input_bag:
            start_time = input_bag.get_start_time()
            end_time = input_bag.get_end_time()

            for t in np.arange(start_time, end_time, 1.0 / args.fps):
                timer_msg = String()
                timer_msg.data = "timer"
                outbag.write("/timer", timer_msg, rospy.Time(t))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process rosbag file to render robot on image")
    parser.add_argument("-bag", "--rosbag", type=str, default=None, help="rosbag file to process")
    parser.add_argument("-o", "--output", type=str, default="virtual.bag", help="output rosbag file or mp4 file")
    parser.add_argument(
        "-sf", "--source_frame", type=str, default="head_mount_kinect_rgb_optical_frame", help="source frame"
    )
    parser.add_argument("-tf", "--target_frame", type=str, default="base_footprint", help="target frame")
    parser.add_argument("-f", "--fps", type=int, default=10, help="fps for timer message")

    args = parser.parse_args()
    rospy.init_node("rosbag_processor")
    main(args)

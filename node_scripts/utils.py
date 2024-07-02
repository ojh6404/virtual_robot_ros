import os
import rospkg
import numpy as np
import pyrender
import xml.etree.ElementTree as ET


cache_dir = os.path.expanduser("~/.ros")


def create_raymond_lights():
    """
    Return raymond light nodes for the scene.
    """
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

    nodes = []

    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)

        z = np.array([xp, yp, zp])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        matrix = np.eye(4)
        matrix[:3, :3] = np.c_[x, y, z]
        nodes.append(pyrender.Node(light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0), matrix=matrix))

    return nodes


def package_to_absolute_path(package_path):
    if package_path.startswith("package://"):
        # Remove 'package://'
        path = package_path[10:]
        # Split the path into package name and the relative path
        splits = path.split("/", 1)
        package_name = splits[0]
        relative_path = splits[1] if len(splits) > 1 else ""

        # Get the absolute path of the package
        rospack = rospkg.RosPack()
        package_path = rospack.get_path(package_name)

        # Combine the package path with the relative path
        absolute_path = os.path.join(package_path, relative_path)
        return absolute_path
    else:
        return package_path


def convert_element_paths(element):
    for attr in element.attrib:
        if "filename" in attr or "uri" in attr or "path" in attr:
            element.set(attr, package_to_absolute_path(element.get(attr)))
    for child in element:
        convert_element_paths(child)


def remove_transmission_elements(root):
    for transmission in root.findall("transmission"):
        root.remove(transmission)


def remove_gazebo_elements(root):
    for transmission in root.findall("gazebo"):
        root.remove(transmission)


def add_virtual_prefix_to_specific_elements(root):
    for link in root.findall(".//link"):
        if "name" in link.attrib:
            link.set("name", "virtual_" + link.get("name"))
    for joint in root.findall(".//joint"):
        for tag in ["child", "parent"]:
            for elem in joint.findall(tag):
                if "link" in elem.attrib:
                    elem.set("link", "virtual_" + elem.get("link"))


def convert_urdf_package_to_absolute(urdf_file):
    tree = ET.parse(urdf_file)
    root = tree.getroot()

    # Remove transmission elements
    remove_transmission_elements(root)

    # Remove gazebo elements
    remove_gazebo_elements(root)

    # Add 'virtual_' prefix to link names
    # add_virtual_prefix_to_specific_elements(root)

    # Write the modified URDF file for ROS
    new_urdf_file = os.path.join(cache_dir, (os.path.basename(urdf_file) + ".ros"))
    tree.write(new_urdf_file)
    print(f"Converted URDF saved as {new_urdf_file}")

    # Convert paths in remaining elements
    convert_element_paths(root)

    # Write the modified URDF file for urdfpy
    new_urdf_file = os.path.join(cache_dir, os.path.basename(urdf_file))
    tree.write(new_urdf_file)
    print(f"Converted URDF saved as {new_urdf_file}")


def get_processed_urdf_path(urdf_path):
    new_urdf_path = os.path.join(cache_dir, os.path.basename(urdf_path))
    if not os.path.exists(new_urdf_path):
        convert_urdf_package_to_absolute(urdf_path)
    return new_urdf_path

import cv2
import numpy as np
import k4a
import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped, Pose, Point, Quaternion
import tf.transformations as tft
from visualization_msgs.msg import Marker
from getQRPose import getQRPose
from collections import deque
from math import sqrt

# Filter to smooth the poses using a moving average
class PoseFilter:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.pose_buffer = deque(maxlen=window_size)

    def add_pose(self, pose):
        self.pose_buffer.append(pose)

    def get_filtered_pose(self):
        if len(self.pose_buffer) == 0:
            return None

        # Compute the average position
        avg_position = np.mean([pose[0] for pose in self.pose_buffer], axis=0)
        # Compute the average quaternion (using averaging of quaternions)
        avg_quaternion = np.mean([pose[1] for pose in self.pose_buffer], axis=0)

        # Convert quaternion to Euler angles for comparison and debugging
        avg_euler = tft.euler_from_quaternion(avg_quaternion)

        return avg_position, avg_quaternion

# Function to calculate the object TF from the QR code poses
def calculate_object_tf(qr_tf_list, marker_id, z_offset):
    object_tf = np.eye(4)
    tf_marker = next((qr['tf'] for qr in qr_tf_list if qr['id'] == marker_id), None)
    
    if tf_marker is not None:
        rotation_matrix = tf_marker[:3, :3]
        origin = tf_marker[:3, 3] + rotation_matrix @ np.array([0, 0, z_offset])
        object_tf[:3, :3] = rotation_matrix
        object_tf[:3, 3] = origin
        return object_tf
    else:
        print(f"QR code with ID {marker_id} not detected.")
        return None

# Function to publish a transform
def publish_tf(tf_broadcaster, tf_matrix, parent_frame, child_frame):
    t = TransformStamped()
    t.header.stamp = rospy.Time.now()
    t.header.frame_id = parent_frame
    t.child_frame_id = child_frame

    # Get translation and rotation (quaternion) from the transformation matrix
    t.transform.translation.x = tf_matrix[0, 3]
    t.transform.translation.y = tf_matrix[1, 3]
    t.transform.translation.z = tf_matrix[2, 3]

    # Convert rotation matrix to quaternion
    qx, qy, qz, qw = tft.quaternion_from_matrix(tf_matrix)
    t.transform.rotation.x = qx
    t.transform.rotation.y = qy
    t.transform.rotation.z = qz
    t.transform.rotation.w = qw

    tf_broadcaster.sendTransform(t)

# Function to load the static TF from the npz file
def load_static_tf(file_path):
    try:
        data = np.load(file_path)
        calibrated_tf = data['calibrated_tf']
        return calibrated_tf
    except FileNotFoundError:
        print("File saved_tfs.npz not found.")
        return None
    except Exception as e:
        print("Error loading saved_tfs.npz:", e)
        return None

# Function to create and publish the cuboid marker for visualization
def publish_cuboid_marker(marker_pub, frame_id, length=0.195, width=0.195, height=0.16):
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.Time.now()
    marker.ns = "cuboid"
    marker.id = 0
    marker.type = Marker.CUBE
    marker.action = Marker.ADD

    # Define the cuboid's pose (position and orientation)
    pose = Pose()
    pose.position.x = 0.0
    pose.position.y = 0.0
    pose.position.z = 0.0
    pose.orientation = Quaternion(*tft.quaternion_from_euler(0, 0, 0))  # No rotation

    marker.pose = pose

    # Define the cuboid's size
    marker.scale.x = length
    marker.scale.y = width
    marker.scale.z = height

    # Set the color of the cuboid
    marker.color.r = 0.0
    marker.color.g = 1.0  # Green
    marker.color.b = 0.0
    marker.color.a = 1.0  # Full opacity

    marker_pub.publish(marker)
    
    # Create a Marker message to represent the sphere (radius should cover the cuboid)
    marker_sphere = Marker()
    marker_sphere.header.frame_id = "object_frame"  # The frame where the sphere exists
    marker_sphere.header.stamp = rospy.Time.now()
    marker_sphere.ns = "sphere"
    marker_sphere.id = 1
    marker_sphere.type = Marker.SPHERE
    marker_sphere.action = Marker.ADD

    # Define the sphere's pose (position and orientation)
    pose_sphere = Pose()
    pose_sphere.position.x = 0.0  # Set position (x, y, z) relative to the object frame
    pose_sphere.position.y = 0.0
    pose_sphere.position.z = 0.0
    pose_sphere.orientation = Quaternion(*tft.quaternion_from_euler(0, 0, 0))  # No rotation

    marker_sphere.pose = pose_sphere

    # Calculate radius of sphere to cover cuboid
    radius = sqrt((length/2)**2 + (width/2)**2 + (height/2)**2)

    # Define the sphere's size (radius)
    marker_sphere.scale.x = radius * 2  # Diameter (2 * radius)
    marker_sphere.scale.y = radius * 2
    marker_sphere.scale.z = radius * 2

    # Set the color of the sphere (RGBA format)
    marker_sphere.color.r = 1.0  # Red color for the sphere
    marker_sphere.color.g = 0.0
    marker_sphere.color.b = 0.0
    marker_sphere.color.a = 0.5  # Half opacity

    # Publish the sphere marker
    marker_pub.publish(marker_sphere)

# Main function
def main():
    rospy.init_node('visualize_and_publish_tf')
    
    tf_broadcaster = tf2_ros.TransformBroadcaster()
    marker_pub = rospy.Publisher('visualization_marker', Marker, queue_size=10)
    
    # Initialize the pose filter
    pose_filter = PoseFilter(window_size=10)

    # Load static TF from the saved file
    static_tf = load_static_tf('calibrated_tf.npz')
    
    kinect = k4a.Device.open()
    device_config = k4a.DEVICE_CONFIG_BGRA32_1080P_NFOV_UNBINNED_FPS30
    kinect.start_cameras(device_config)

    camera_matrix = np.array([[918.97827975, 0.,          956.74460955],
                              [0.,           920.3574322, 552.8364351 ],
                              [0.,           0.,          1.        ]])
    dist_coeffs = np.array([8.09622039e-02, -3.71595305e-02,  7.71880774e-05, -3.45432002e-04, 6.97244572e-03])

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()
    
    rate = rospy.Rate(100)  # 100 Hz
    
    try:
        while not rospy.is_shutdown():
            qr_tf_list = getQRPose(camera=kinect,
                                   camera_matrix=camera_matrix,
                                   dist_coeffs=dist_coeffs,
                                   aruco_dict=aruco_dict,
                                   aruco_param=parameters,
                                   marker_length=0.12,
                                   plot_img=False)
            
            if qr_tf_list:
                # Calculate object TF from QR code with ID 1
                object_tf = calculate_object_tf(qr_tf_list, marker_id=1, z_offset=-0.08)
                if object_tf is not None:
                    # Filter the pose using the PoseFilter
                    position = object_tf[:3, 3]
                    quaternion = tft.quaternion_from_matrix(object_tf)
                    pose_filter.add_pose((position, quaternion))
                    
                    filtered_position, filtered_quaternion = pose_filter.get_filtered_pose()

                    # Create a new transform matrix with the filtered position and quaternion
                    filtered_object_tf = np.eye(4)
                    filtered_object_tf[:3, 3] = filtered_position
                    filtered_object_tf[:3, :3] = tft.quaternion_matrix(filtered_quaternion)[:3, :3]

                    # Publish the object's transform based on the filtered pose
                    publish_tf(tf_broadcaster, filtered_object_tf, parent_frame="camera_link", child_frame="object_frame")
                    # Visualize the cuboid in RViz
                    publish_cuboid_marker(marker_pub, frame_id="object_frame")
            
            if static_tf is not None:
                # Publish the static TF (from 'panda_link0' to 'camera_link')
                publish_tf(tf_broadcaster, static_tf, parent_frame="panda_link0", child_frame="camera_link")
            
            rate.sleep()
    
    finally:
        kinect.stop_cameras()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

import cv2
import numpy as np
import k4a

def getQRPose(camera: k4a.Device,
              camera_matrix: np.array,
              dist_coeffs: np.array,
              aruco_dict: dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250),
              aruco_param: cv2.aruco.DetectorParameters = cv2.aruco.DetectorParameters(),
              marker_length: float = 0.05,
              plot_img: bool = True) -> list:
    
    if camera_matrix is None:
        # Get camera intrinsics for pose estimation
        calibration = camera.get_calibration(color_resolution=k4a.EColorResolution.RES_1080P, depth_mode=k4a.EDepthMode.OFF)
        color_camera_intrinsics = calibration.color_cam_cal.intrinsics.parameters.param
        camera_matrix = np.array([[color_camera_intrinsics.fx, 0, color_camera_intrinsics.cx],
                                  [0, color_camera_intrinsics.fy, color_camera_intrinsics.cy],
                                  [0, 0, 1]])
    
    if dist_coeffs is None:
        calibration = camera.get_calibration(color_resolution=k4a.EColorResolution.RES_1080P, depth_mode=k4a.EDepthMode.OFF)
        color_camera_intrinsics = calibration.color_cam_cal.intrinsics.parameters.param
        dist_coeffs = np.array([color_camera_intrinsics.k1,
                                color_camera_intrinsics.k2,
                                color_camera_intrinsics.p1,
                                color_camera_intrinsics.p2,
                                color_camera_intrinsics.k3
                                ])
    
    # Capture a frame from the Kinect camera
    capture = camera.get_capture(timeout_ms=1000)

    # Check if capture is valid
    if capture.color is None:
        print("Failed to get capture")
        return None
    
    # Convert Kinect BGRA image to BGR for OpenCV
    color_image_bgr = cv2.cvtColor(capture.color.data, cv2.COLOR_BGRA2BGR)

    # Detect ArUco markers
    corners, ids, _ = cv2.aruco.detectMarkers(color_image_bgr, aruco_dict, parameters=aruco_param)

    qr_tf_list = []
    
    # Loop through detected markers
    if ids is not None:
        for corner, marker_id in zip(corners, ids.flatten()):
            # Estimate the pose of the marker
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corner, marker_length, camera_matrix, dist_coeffs)

            # Draw the marker's frame
            cv2.drawFrameAxes(color_image_bgr, camera_matrix, dist_coeffs, rvec[0], tvec[0], marker_length)

            # Get the origin position
            origin_position = tvec[0].flatten()

            # Convert rotation vector to rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(rvec[0])
            
            # Construct the transformation matrix
            transformation_matrix = np.eye(4)  # Initialize 4x4 identity matrix
            transformation_matrix[:3, :3] = rotation_matrix  # Set rotation matrix
            transformation_matrix[:3, 3] = tvec[0].flatten()  # Set translation vector
            
            # Add tf to list
            qr_tf_list.append({"id": marker_id, 
                               "tf": transformation_matrix})
            
            if plot_img:
                # Determine the position to place the text on the image
                top_left = tuple(corner[0][0].astype(int))  # Top-left corner of the marker
                text_position = (top_left[0], top_left[1] - 10)  # Slightly above the marker

                # Put marker ID
                cv2.putText(color_image_bgr, f"ID: {marker_id}", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Put position (RGB for X, Y, Z)
                text_position_x = (text_position[0], text_position[1] + 20)
                text_position_y = (text_position[0], text_position[1] + 40)
                text_position_z = (text_position[0], text_position[1] + 60)

                cv2.putText(color_image_bgr, f"X: {origin_position[0]:.2f}", text_position_x, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # Red for X
                cv2.putText(color_image_bgr, f"Y: {origin_position[1]:.2f}", text_position_y, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Green for Y
                cv2.putText(color_image_bgr, f"Z: {origin_position[2]:.2f}", text_position_z, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # Blue for Z
    
    if plot_img:
        # Display the image
        cv2.imshow("Kinect Stream", color_image_bgr)
        
    return qr_tf_list

if __name__ == "__main__":
    # Open Kinect Azure device
    kinect = k4a.Device.open()

    # Set the camera to capture at 1080p resolution
    device_config = k4a.DEVICE_CONFIG_BGRA32_1080P_NFOV_UNBINNED_FPS30
    kinect.start_cameras(device_config)

    camera_matrix = np.array([[918.97827975, 0.,          956.74460955],
                              [0.,           920.3574322, 552.8364351 ],
                              [0.,           0.,          1.        ]])
    dist_coeffs = np.array([8.09622039e-02, -3.71595305e-02,  7.71880774e-05, -3.45432002e-04, 6.97244572e-03])

    # Initialize ArUco marker detector
    board_type = cv2.aruco.DICT_6X6_250
    aruco_dict = cv2.aruco.getPredefinedDictionary(board_type)
    parameters = cv2.aruco.DetectorParameters()
    
    # Define the size of the ArUco marker in meters
    marker_length = 0.12  # Adjust this based on your marker's actual size (50 mm)
    
    # Create a window to display the video
    cv2.namedWindow("Kinect Stream", cv2.WINDOW_NORMAL)
    
    try:
        while True:
            qr_tf_list = getQRPose(camera=kinect,
                                camera_matrix=camera_matrix,
                                dist_coeffs=dist_coeffs,
                                aruco_dict=aruco_dict,
                                aruco_param=parameters,
                                marker_length=marker_length,
                                plot_img=True)
            
            if qr_tf_list is not None:
                for qr_tf in qr_tf_list:
                    # Display the transformation matrix
                    print(f"Marker ID: {qr_tf['id']}")
                    print("Transformation Matrix:")
                    print(qr_tf["tf"])
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Stop the Kinect cameras and close the OpenCV window
        kinect.stop_cameras()
        cv2.destroyAllWindows()
import rospy
from geometry_msgs.msg import PoseStamped, TransformStamped
import cv2
import numpy as np
import k4a
import threading
import sys
import select
import time
import os

from scipy.linalg import sqrtm
from numpy.linalg import inv

# OS에 따라 다른 모듈 임포트
if os.name == 'nt':  # Windows
    import msvcrt
else:  # Unix/Linux
    import termios
    import tty

class TFCollector:
    def __init__(self):
        # 리스트 초기화
        self.list_tf_base2ee = []
        self.list_tf_cam2qr = []
        self.latest_tf_base2ee = None
        self.latest_tf_cam2qr = None
        self.lock = threading.Lock()

        # 초기 모드 설정 (0: 데이터 수집, 1: 캘리브레이션 및 발행)
        self.mode = 0
        self.mode_lock = threading.Lock()

        # ROS 초기화 및 구독자 설정
        rospy.init_node('tf_collector_node', anonymous=True)
        rospy.Subscriber('mpcc/ee_pose', PoseStamped, self.ros_callback)

        # ROS 퍼블리셔 설정 (캘리브레이션 결과 발행)
        self.tf_pub = rospy.Publisher('calibrated_tf', TransformStamped, queue_size=10)
        self.cam_pose_pub = rospy.Publisher('cam_pose', PoseStamped, queue_size=10)

        # Kinect Azure 초기화
        try:
            self.kinect = k4a.Device.open()
            device_config = k4a.DEVICE_CONFIG_BGRA32_1080P_NFOV_UNBINNED_FPS30
            self.kinect.start_cameras(device_config)
        except Exception as e:
            print(f"Kinect Azure 초기화 실패: {e}")
            sys.exit(1)

        # 카메라 내부 파라미터 설정
        self.camera_matrix = np.array([[918.97827975, 0.,          956.74460955],
                                       [0.,           920.3574322, 552.8364351 ],
                                       [0.,           0.,          1.        ]])
        self.dist_coeffs = np.array([8.09622039e-02, -3.71595305e-02,  7.71880774e-05, -3.45432002e-04, 6.97244572e-03])

        # ArUco 마커 설정
        self.board_type = cv2.aruco.DICT_6X6_250
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(self.board_type)
        self.parameters = cv2.aruco.DetectorParameters()

        # ArUco 마커의 실제 크기 (미터 단위)
        self.marker_length = 0.04 # 50mm

        # Calibration 결과 변수
        self.calibrated_tf = None

        # 시작 메시지
        print("프로그램 시작")
        print("모드를 선택하세요 (0: 데이터 수집, 1: 캘리브레이션 및 발행): ", end='', flush=True)

    def ros_callback(self, msg):
        # PoseStamped 메시지를 4x4 변환 행렬로 변환
        position = msg.pose.position
        orientation = msg.pose.orientation

        # 쿼터니언을 회전 행렬로 변환
        q = np.array([orientation.x, orientation.y, orientation.z, orientation.w])
        rotation_matrix = self.quaternion_to_rotation_matrix(q)

        # 변환 행렬 구성
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = [position.x, position.y, position.z]

        with self.lock:
            self.latest_tf_base2ee = transformation_matrix

    def quaternion_to_rotation_matrix(self, q):
        # 쿼터니언을 회전 행렬로 변환
        x, y, z, w = q
        rotation_matrix = np.array([
            [1 - 2*(y**2 + z**2),     2*(x*y - z*w),         2*(x*z + y*w)],
            [2*(x*y + z*w),           1 - 2*(x**2 + z**2),   2*(y*z - x*w)],
            [2*(x*z - y*w),           2*(y*z + x*w),         1 - 2*(x**2 + y**2)]
        ])
        return rotation_matrix

    def capture_camera_qr_tf(self, color_image_bgr):
        # ArUco 마커 감지
        corners, ids, _ = cv2.aruco.detectMarkers(color_image_bgr, self.aruco_dict, parameters=self.parameters)

        camera_qr_tf = None

        if ids is not None:
            for corner, marker_id in zip(corners, ids.flatten()):
                # 마커의 자세 추정
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corner, self.marker_length, self.camera_matrix, self.dist_coeffs)

                # 마커의 축 그리기
                cv2.drawFrameAxes(color_image_bgr, self.camera_matrix, self.dist_coeffs, rvec[0], tvec[0], self.marker_length)

                # 마커의 회전 벡터를 회전 행렬로 변환
                rotation_matrix, _ = cv2.Rodrigues(rvec[0])

                # 변환 행렬 구성
                transformation_matrix = np.eye(4)
                transformation_matrix[:3, :3] = rotation_matrix
                transformation_matrix[:3, 3] = tvec[0].flatten()

                # 변환 행렬 저장 (여기서는 첫 번째 마커만 저장)
                camera_qr_tf = transformation_matrix

                # 하나의 마커만 처리하려면 break
                break

        return camera_qr_tf

    def create_tf_image(self, panda_tf, camera_qr_tf):
        # 텍스트를 표시할 빈 이미지 생성
        tf_image = np.ones((400, 800, 3), dtype=np.uint8) * 255  # 흰색 배경

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (0, 0, 0)  # 검정색
        thickness = 1
        line_height = 20

        y_offset = 20

        # Panda EE TF 표시
        cv2.putText(tf_image, "Panda Link0 to EE Frame TF:", (10, y_offset), font, font_scale, color, thickness, cv2.LINE_AA)
        y_offset += line_height

        if panda_tf is not None:
            for row in panda_tf[:3]:
                text = ' '.join([f"{val:.4f}" for val in row])
                cv2.putText(tf_image, text, (10, y_offset), font, font_scale, color, thickness, cv2.LINE_AA)
                y_offset += line_height
        else:
            cv2.putText(tf_image, "No data available.", (10, y_offset), font, font_scale, color, thickness, cv2.LINE_AA)
            y_offset += line_height

        y_offset += line_height  # 추가 공간

        # Camera QR TF 표시
        cv2.putText(tf_image, "Camera to QR Code Frame TF:", (10, y_offset), font, font_scale, color, thickness, cv2.LINE_AA)
        y_offset += line_height

        if camera_qr_tf is not None:
            for row in camera_qr_tf[:3]:
                text = ' '.join([f"{val:.4f}" for val in row])
                cv2.putText(tf_image, text, (10, y_offset), font, font_scale, color, thickness, cv2.LINE_AA)
                y_offset += line_height
        else:
            cv2.putText(tf_image, "No data available.", (10, y_offset), font, font_scale, color, thickness, cv2.LINE_AA)
            y_offset += line_height

        return tf_image

    def logR(self, T):
        R = T[0:3, 0:3]
        trace = np.trace(R)
        theta = np.arccos((trace - 1)/2)
        if np.sin(theta) == 0:
            return np.zeros(3)
        logr = np.array([R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]]) * theta / (2*np.sin(theta))
        return logr

    def calibrate_AX_XB(self, A_list, B_list):
        n_data = len(A_list)
        M = np.zeros((3,3))
        C = np.zeros((3*n_data, 3))
        d = np.zeros((3*n_data, 1))

        for i in range(n_data):
            alpha = self.logR(A_list[i])
            beta = self.logR(B_list[i])

            M += np.outer(beta, alpha)

        # Compute theta
        try:
            M_inv = inv(M.T @ M)
            theta = sqrtm(M_inv) @ M.T
        except np.linalg.LinAlgError:
            print("M 행렬의 역행렬을 계산할 수 없습니다.")
            return None, None

        for i in range(n_data):
            rot_a = A_list[i][0:3, 0:3]
            trans_a = A_list[i][0:3, 3]
            trans_b = B_list[i][0:3, 3]

            C[3*i:3*i+3, :] = np.eye(3) - rot_a
            d[3*i:3*i+3, 0] = trans_a - theta @ trans_b

        try:
            b_x = inv(C.T @ C) @ (C.T @ d)
        except np.linalg.LinAlgError:
            print("C 행렬의 역행렬을 계산할 수 없습니다.")
            return None, None

        return theta, b_x

    def run(self):
        # 터미널 입력 스레드 시작
        input_thread = threading.Thread(target=self.listen_terminal)
        input_thread.daemon = True
        input_thread.start()

        try:
            while not rospy.is_shutdown():
                # Kinect에서 프레임 캡처
                capture = self.kinect.get_capture(timeout_ms=1000)

                if capture.color is None:
                    print("Failed to get capture")
                    continue

                # BGRA 이미지를 BGR로 변환
                color_image_bgr = cv2.cvtColor(capture.color.data, cv2.COLOR_BGRA2BGR)

                # ArUco 마커를 통해 카메라 -> QR 코드 변환 행렬 얻기
                camera_qr_tf = self.capture_camera_qr_tf(color_image_bgr)

                with self.lock:
                    self.latest_tf_cam2qr = camera_qr_tf

                # 이미지 표시
                cv2.imshow("Kinect Stream", color_image_bgr)

                # TF 정보를 이미지로 생성
                with self.lock:
                    panda_tf = self.latest_tf_base2ee.copy() if self.latest_tf_base2ee is not None else None
                    camera_tf = self.latest_tf_cam2qr.copy() if self.latest_tf_cam2qr is not None else None

                tf_image = self.create_tf_image(panda_tf, camera_tf)
                cv2.imshow("TF Information", tf_image)

                # OpenCV 창 키 입력 감지
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("종료 키 'q'가 눌렸습니다.")
                    break
                elif key == ord('s'):
                    with self.mode_lock:
                        current_mode = self.mode
                    if current_mode == 0:
                        with self.lock:
                            if self.latest_tf_base2ee is not None and self.latest_tf_cam2qr is not None:
                                self.list_tf_base2ee.append(self.latest_tf_base2ee.copy())
                                self.list_tf_cam2qr.append(self.latest_tf_cam2qr.copy())
                                print(f"데이터 저장됨. 현재 수집된 데이터 수: {len(self.list_tf_base2ee)}")
                            else:
                                print("최신 TF 데이터가 아직 준비되지 않았습니다.")
                    else:
                        print("현재 모드가 데이터 수집 모드가 아닙니다.")

        finally:
            # 리소스 정리
            self.kinect.stop_cameras()
            cv2.destroyAllWindows()
            rospy.signal_shutdown("TF Collector stopped.")

    def listen_terminal(self):
        self.listen_terminal_unix()

    def listen_terminal_unix(self):
        # 터미널 설정 변경
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(sys.stdin.fileno())
            while not rospy.is_shutdown():
                dr, dw, de = select.select([sys.stdin], [], [], 0.1)
                if dr:
                    ch = sys.stdin.read(1)
                    if ch == '0':
                        with self.mode_lock:
                            self.mode = 0
                        print("\n모드 0: 데이터 수집 모드로 전환됨.")
                    elif ch == '1':
                        with self.mode_lock:
                            self.mode = 1
                        print("\n모드 1: 캘리브레이션 및 발행 모드로 전환됨.")
                        self.perform_calibration_and_publish()
                    else:
                        print(f"\n잘못된 입력입니다. '0' 또는 '1'을 입력하세요.")
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def perform_calibration_and_publish(self):
        with self.lock:
            if len(self.list_tf_base2ee) < 2 or len(self.list_tf_cam2qr) < 2:
                print("캘리브레이션을 수행하기 위한 데이터가 충분하지 않습니다. 최소 2개 이상의 데이터가 필요합니다.")
                return

            # A와 B 리스트 생성
            A_list = []
            B_list = []
            p = len(self.list_tf_base2ee)
            for i in range(p - 1):
                A = self.list_tf_base2ee[i + 1] @ inv(self.list_tf_base2ee[i])
                B = self.list_tf_cam2qr[i + 1] @ inv(self.list_tf_cam2qr[i])

                A_list.append(A)
                B_list.append(B)

        # 캘리브레이션 수행
        theta, b_x = self.calibrate_AX_XB(A_list, B_list)
        if theta is None or b_x is None:
            print("캘리브레이션 실패.")
            return

        # 변환 행렬 구성
        X = np.eye(4)
        X[0:3, 0:3] = theta.real  # sqrtm은 복소수를 반환할 수 있으므로 실수부만 사용
        X[0:3, 3] = b_x.flatten()

        print("캘리브레이션 결과 변환 행렬 X:")
        print(X)

        # ROS TransformStamped 메시지 생성
        tf_msg = TransformStamped()
        tf_msg.header.stamp = rospy.Time.now()
        tf_msg.header.frame_id = "panda_link0"
        tf_msg.child_frame_id = "camera_calibrated"

        # 회전 행렬을 쿼터니언으로 변환
        rotation_matrix = X[0:3, 0:3]
        quat = self.rotation_matrix_to_quaternion(rotation_matrix)

        tf_msg.transform.rotation.x = quat[0]
        tf_msg.transform.rotation.y = quat[1]
        tf_msg.transform.rotation.z = quat[2]
        tf_msg.transform.rotation.w = quat[3]

        # 번역 벡터 설정
        tf_msg.transform.translation.x = X[0, 3]
        tf_msg.transform.translation.y = X[1, 3]
        tf_msg.transform.translation.z = X[2, 3]

        # 퍼블리시
        self.tf_pub.publish(tf_msg)
        print("캘리브레이션된 TF가 'camera_calibrated' 프레임으로 퍼블리시되었습니다.")
        
        # ROS PoseStamped 메시지 생성
        cam_pose_msg = PoseStamped()
        cam_pose_msg.header.stamp = rospy.Time.now()
        cam_pose_msg.header.frame_id = "panda_link0"

        cam_pose_msg.pose.orientation.x = quat[0]
        cam_pose_msg.pose.orientation.y = quat[1]
        cam_pose_msg.pose.orientation.z = quat[2]
        cam_pose_msg.pose.orientation.w = quat[3]

        # 번역 벡터 설정
        cam_pose_msg.pose.position.x = X[0, 3]
        cam_pose_msg.pose.position.y = X[1, 3]
        cam_pose_msg.pose.position.z = X[2, 3]

        # 퍼블리시
        self.cam_pose_pub.publish(cam_pose_msg)
        print("camera pose 'camera_calibrated' 프레임으로 퍼블리시되었습니다.")

        # 캘리브레이션 결과 저장 (옵션)
        np.savez("calibrated_tf.npz", calibrated_tf=X)
        print("캘리브레이션 결과가 'calibrated_tf.npz' 파일로 저장되었습니다.")

    def rotation_matrix_to_quaternion(self, R):
        # 회전 행렬을 쿼터니언으로 변환
        q = np.empty((4, ))
        trace = np.trace(R)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            q[3] = 0.25 / s
            q[0] = (R[2,1] - R[1,2]) * s
            q[1] = (R[0,2] - R[2,0]) * s
            q[2] = (R[1,0] - R[0,1]) * s
        else:
            if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
                s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
                q[3] = (R[2,1] - R[1,2]) / s
                q[0] = 0.25 * s
                q[1] = (R[0,1] + R[1,0]) / s
                q[2] = (R[0,2] + R[2,0]) / s
            elif R[1,1] > R[2,2]:
                s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
                q[3] = (R[0,2] - R[2,0]) / s
                q[0] = (R[0,1] + R[1,0]) / s
                q[1] = 0.25 * s
                q[2] = (R[1,2] + R[2,1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
                q[3] = (R[1,0] - R[0,1]) / s
                q[0] = (R[0,2] + R[2,0]) / s
                q[1] = (R[1,2] + R[2,1]) / s
                q[2] = 0.25 * s
        return q

if __name__ == "__main__":
    collector = TFCollector()
    collector.run()

    # 저장된 리스트를 파일로 저장하거나 추가 처리를 할 수 있습니다.
    # 예시로 numpy 파일로 저장:
    if len(collector.list_tf_base2ee) > 0 and len(collector.list_tf_cam2qr) > 0:
        np.savez("saved_tfs.npz", 
                 tf_base2ee=np.array(collector.list_tf_base2ee),
                 tf_cam2qr=np.array(collector.list_tf_cam2qr))
        print("저장된 TF 리스트를 'saved_tfs.npz' 파일로 저장했습니다.")

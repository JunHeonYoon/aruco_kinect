import time
import cv2
import numpy as np
import k4a

def capture_and_save_images(camera: k4a.Device):
    """
    Function to capture and save images when user presses 'c'.
    It saves both the original 1080p and resized 640x360 images.
    """
    index = 0  # To keep track of image index

    while True:
        # Capture a frame from the camera
        capture = camera.get_capture(timeout_ms=1000)
        
        if capture.color is None:
            print("No color image captured, skipping frame.")
            continue
        
        # Convert BGRA to BGR (OpenCV format)
        color_image_bgr = cv2.cvtColor(capture.color.data, cv2.COLOR_BGRA2BGR)
        
        # Resize the image to 640x360 to maintain 16:9 aspect ratio
        color_image_resized = cv2.resize(color_image_bgr, (640, 360))
        
        # Wait for the user to press 'c' to capture an image
        cv2.imshow('Azure Kinect Image Capture', color_image_bgr)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):  # Capture when 'c' is pressed
            # Save the 1080p image
            filename_1080p = f'img/index_{index}.jpg'
            cv2.imwrite(filename_1080p, color_image_bgr)
            print(f"Saved 1080p image as {filename_1080p}")
            
            # Increment index for next image
            index += 1
        
        elif key == ord('q'):  # Press 'q' to exit the loop
            break

    # Clean up when done
    cv2.destroyAllWindows()

def main():
    # Initialize Azure Kinect device
    kinect = k4a.Device.open()

    # Set the camera to capture at 1080p resolution
    device_config = k4a.DEVICE_CONFIG_BGRA32_1080P_NFOV_UNBINNED_FPS30
    kinect.start_cameras(device_config)

    try:
        # Capture and save images when user presses 'c'
        capture_and_save_images(kinect)
    finally:
        # Stop the camera when done
        kinect.stop_cameras()
        kinect.close()

if __name__ == "__main__":
    main()

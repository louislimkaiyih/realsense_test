import pyrealsense2 as rs
import numpy as np
import cv2

point = None
last_reported_point = None


def on_mouse(event, x, y, flags, param):
    global point
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)

# Create a pipeline object (this manages the camera stream)
pipeline = rs.pipeline()

# Create a config object (this stores which streams/settings we want)
config = rs.config()

# Conservative settings for USB 2.x
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Create a colorizer object (this will help us visualize depth data)
colorizer = rs.colorizer()
align = rs.align(rs.stream.color)

# Start the camera
pipeline.start(config)

try:
    while True:
        # Wait for a new set of frames from the camera
        frames = pipeline.wait_for_frames()

        # NEW: perform alignment here
        aligned_frames = align.process(frames)

        # Get depth and color frames
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # If either frame is missing, skip this loop and wait for the next one
        if not depth_frame or not color_frame:
            continue

        # Convert camera frame objects into NumPy arrays
        depth_color_frame = colorizer.colorize(depth_frame)
        depth_colormap = np.asanyarray(depth_color_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        if point is not None:
            px, py = point
            distance_m = depth_frame.get_distance(px, py)

            if point != last_reported_point:
                print(f"Pixel ({px}, {py}) -> {distance_m:.3f} m")
                last_reported_point = point

            cv2.circle(color_image, (px, py), 5, (0, 0, 255), -1)
            cv2.putText(
                color_image,
                f"({px}, {py})  {distance_m:.3f} m",
                (px + 10, py - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
        
        # Show both windows
        cv2.imshow("Color", color_image)
        cv2.setMouseCallback("Color", on_mouse)
        cv2.imshow("Depth", depth_colormap)

        # Press ESC to quit
        key = cv2.waitKey(1)
        if key == 27:
            break

finally:
    # Clean shutdown
    pipeline.stop()
    cv2.destroyAllWindows()

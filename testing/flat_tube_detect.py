import pyrealsense2 as rs
import numpy as np
import cv2


def build_contour_mask(contour, mask_shape):
    contour_mask = np.zeros(mask_shape, dtype=np.uint8)
    cv2.drawContours(contour_mask, [contour], -1, 255, thickness=-1)
    return contour_mask


def get_dominant_color(blue_mask, green_mask, yellow_mask, contour, mask_shape):
    contour_mask = build_contour_mask(contour, mask_shape)

    blue_count = cv2.countNonZero(cv2.bitwise_and(blue_mask, contour_mask))
    green_count = cv2.countNonZero(cv2.bitwise_and(green_mask, contour_mask))
    yellow_count = cv2.countNonZero(cv2.bitwise_and(yellow_mask, contour_mask))

    counts = {
        "BLUE": blue_count,
        "GREEN": green_count,
        "YELLOW": yellow_count,
    }
    label = max(counts, key=counts.get)

    if label == "BLUE":
        box_color = (255, 0, 0)
    elif label == "GREEN":
        box_color = (0, 255, 0)
    else:
        box_color = (0, 255, 255)

    return label, box_color


def get_window_median_depth_m(depth_image, depth_scale, cx, cy, window_size):
    half = window_size // 2
    img_h, img_w = depth_image.shape

    x1 = max(0, cx - half)
    x2 = min(img_w, cx + half + 1)
    y1 = max(0, cy - half)
    y2 = min(img_h, cy + half + 1)

    depth_window = depth_image[y1:y2, x1:x2]
    valid_depths = depth_window[depth_window > 0]

    if valid_depths.size == 0:
        return 0.0

    return float(np.median(valid_depths) * depth_scale)


def get_long_axis_angle_deg(rect):
    (_, _), (width, height), angle = rect

    if width < height:
        angle += 90.0

    if angle < 0:
        angle += 180.0
    if angle >= 180.0:
        angle -= 180.0

    return angle


def normalize_axis_angle_diff_deg(angle_a, angle_b):
    diff = abs(angle_a - angle_b) % 180.0
    if diff > 90.0:
        diff = 180.0 - diff
    return diff


def average_axis_angle_deg(angle_a, angle_b):
    angle_a_rad = np.deg2rad(angle_a * 2.0)
    angle_b_rad = np.deg2rad(angle_b * 2.0)

    x = np.cos(angle_a_rad) + np.cos(angle_b_rad)
    y = np.sin(angle_a_rad) + np.sin(angle_b_rad)

    average_angle = 0.5 * np.rad2deg(np.arctan2(y, x))
    if average_angle < 0:
        average_angle += 180.0
    return average_angle


def get_axis_vectors(angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    axis_u = np.array([np.cos(angle_rad), np.sin(angle_rad)], dtype=np.float32)
    perp_u = np.array([-axis_u[1], axis_u[0]], dtype=np.float32)
    return axis_u, perp_u


def get_projection_interval(points_xy, axis_u):
    projections = points_xy @ axis_u
    return float(np.min(projections)), float(np.max(projections))


def get_projection_gap(min_a, max_a, min_b, max_b):
    if max_a < min_b:
        return min_b - max_a
    if max_b < min_a:
        return min_a - max_b
    return 0.0


def compute_shape_metrics(contour):
    area = cv2.contourArea(contour)
    rect = cv2.minAreaRect(contour)
    (_, _), (rect_w, rect_h), _ = rect

    if rect_w <= 1 or rect_h <= 1:
        return None

    long_side = max(rect_w, rect_h)
    short_side = min(rect_w, rect_h)
    aspect_ratio = long_side / short_side
    rect_area = rect_w * rect_h
    fill_ratio = area / rect_area
    angle_deg = get_long_axis_angle_deg(rect)
    axis_u, perp_u = get_axis_vectors(angle_deg)

    points_xy = contour.reshape(-1, 2).astype(np.float32)
    projection_min, projection_max = get_projection_interval(points_xy, axis_u)
    line_start = (np.array(rect[0], dtype=np.float32) - axis_u * (long_side * 0.5))
    line_end = (np.array(rect[0], dtype=np.float32) + axis_u * (long_side * 0.5))

    return {
        "area": area,
        "rect": rect,
        "rect_w": rect_w,
        "rect_h": rect_h,
        "long_side": long_side,
        "short_side": short_side,
        "aspect_ratio": aspect_ratio,
        "fill_ratio": fill_ratio,
        "angle_deg": angle_deg,
        "axis_u": axis_u,
        "perp_u": perp_u,
        "points_xy": points_xy,
        "projection_min": projection_min,
        "projection_max": projection_max,
        "line_start": line_start,
        "line_end": line_end,
    }


def passes_shape_filters(shape_metrics):
    if shape_metrics is None:
        return False
    if shape_metrics["area"] < min_contour_area or shape_metrics["area"] > max_contour_area:
        return False
    if shape_metrics["aspect_ratio"] < min_aspect_ratio or shape_metrics["aspect_ratio"] > max_aspect_ratio:
        return False
    if shape_metrics["fill_ratio"] < min_fill_ratio:
        return False
    return True


def build_raw_candidate(contour, depth_image, depth_scale, depth_intrin, blue_mask, green_mask, yellow_mask, mask_shape):
    shape_metrics = compute_shape_metrics(contour)
    if not passes_shape_filters(shape_metrics):
        return None

    rect_cx, rect_cy = shape_metrics["rect"][0]
    cx = int(round(rect_cx))
    cy = int(round(rect_cy))

    img_h, img_w = mask_shape
    if (
        cx <= margin or cy <= margin or
        cx >= img_w - margin or cy >= img_h - margin
    ):
        return None

    distance_m = get_window_median_depth_m(
        depth_image,
        depth_scale,
        cx,
        cy,
        depth_window_size,
    )
    if distance_m <= 0:
        return None

    point_3d = rs.rs2_deproject_pixel_to_point(depth_intrin, [cx, cy], distance_m)
    X_m, Y_m, Z_m = point_3d

    label, box_color = get_dominant_color(
        blue_mask,
        green_mask,
        yellow_mask,
        contour,
        mask_shape,
    )

    return {
        "contour": contour,
        "label": label,
        "box_color": box_color,
        "cx": cx,
        "cy": cy,
        "X_m": X_m,
        "Y_m": Y_m,
        "Z_m": Z_m,
        "distance_m": distance_m,
        **shape_metrics,
    }


def should_merge_candidates(candidate_a, candidate_b):
    if candidate_a["label"] != candidate_b["label"]:
        return False

    angle_diff = normalize_axis_angle_diff_deg(candidate_a["angle_deg"], candidate_b["angle_deg"])
    if angle_diff > merge_angle_diff_deg:
        return False

    if abs(candidate_a["Z_m"] - candidate_b["Z_m"]) > merge_depth_diff_m:
        return False

    average_angle = average_axis_angle_deg(candidate_a["angle_deg"], candidate_b["angle_deg"])
    axis_u, perp_u = get_axis_vectors(average_angle)

    min_a, max_a = get_projection_interval(candidate_a["points_xy"], axis_u)
    min_b, max_b = get_projection_interval(candidate_b["points_xy"], axis_u)
    axis_gap = get_projection_gap(min_a, max_a, min_b, max_b)
    if axis_gap > merge_axis_gap_px:
        return False

    center_delta = np.array(
        [candidate_b["cx"] - candidate_a["cx"], candidate_b["cy"] - candidate_a["cy"]],
        dtype=np.float32,
    )
    perpendicular_offset = abs(float(center_delta @ perp_u))
    if perpendicular_offset > merge_perp_offset_px:
        return False

    return True


def group_raw_candidates(raw_candidates):
    groups = []
    visited = set()

    for start_idx in range(len(raw_candidates)):
        if start_idx in visited:
            continue

        queue = [start_idx]
        component = []

        while queue:
            current_idx = queue.pop()
            if current_idx in visited:
                continue

            visited.add(current_idx)
            component.append(raw_candidates[current_idx])

            for next_idx in range(len(raw_candidates)):
                if next_idx in visited or next_idx == current_idx:
                    continue
                if should_merge_candidates(raw_candidates[current_idx], raw_candidates[next_idx]):
                    queue.append(next_idx)

        groups.append(component)

    return groups


def build_merged_contour(candidate_group, mask_shape):
    merged_mask = np.zeros(mask_shape, dtype=np.uint8)

    all_points = np.vstack([candidate["points_xy"] for candidate in candidate_group]).astype(np.int32)
    if len(candidate_group) == 1:
        cv2.drawContours(merged_mask, [candidate_group[0]["contour"]], -1, 255, thickness=-1)
    else:
        hull = cv2.convexHull(all_points)
        cv2.drawContours(merged_mask, [hull], -1, 255, thickness=-1)

    contours, _ = cv2.findContours(merged_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    return max(contours, key=cv2.contourArea)


def analyze_cap_geometry(contour, mask_shape, angle_deg):
    contour_mask = build_contour_mask(contour, mask_shape)
    ys, xs = np.nonzero(contour_mask)
    if len(xs) == 0:
        return None

    coords_xy = np.column_stack((xs, ys)).astype(np.float32)
    center_xy = np.mean(coords_xy, axis=0)
    axis_u, perp_u = get_axis_vectors(angle_deg)

    centered_xy = coords_xy - center_xy
    axis_values = centered_xy @ axis_u
    perp_values = centered_xy @ perp_u

    axis_min = float(np.min(axis_values))
    axis_max = float(np.max(axis_values))
    tube_length = axis_max - axis_min
    if tube_length <= 1.0:
        return None

    axis_edges = np.linspace(axis_min, axis_max, width_profile_bin_count + 1)
    width_profile = np.full(width_profile_bin_count, np.nan, dtype=np.float32)

    for idx in range(width_profile_bin_count):
        if idx == width_profile_bin_count - 1:
            in_bin = (axis_values >= axis_edges[idx]) & (axis_values <= axis_edges[idx + 1])
        else:
            in_bin = (axis_values >= axis_edges[idx]) & (axis_values < axis_edges[idx + 1])

        if np.count_nonzero(in_bin) < min_pixels_per_width_bin:
            continue

        bin_perp_values = perp_values[in_bin]
        width_profile[idx] = float(np.max(bin_perp_values) - np.min(bin_perp_values))

    end_bin_count = max(3, int(round(width_profile_bin_count * end_width_fraction)))
    left_widths = width_profile[:end_bin_count]
    right_widths = width_profile[-end_bin_count:]

    left_valid = left_widths[~np.isnan(left_widths)]
    right_valid = right_widths[~np.isnan(right_widths)]

    if len(left_valid) < max(2, end_bin_count // 2):
        return None
    if len(right_valid) < max(2, end_bin_count // 2):
        return None

    left_avg = float(np.mean(left_valid))
    right_avg = float(np.mean(right_valid))
    left_std = float(np.std(left_valid))
    right_std = float(np.std(right_valid))

    smaller_avg = min(left_avg, right_avg)
    larger_avg = max(left_avg, right_avg)
    if smaller_avg <= 0:
        return None

    width_ratio = larger_avg / smaller_avg
    if width_ratio < min_cap_width_ratio:
        return None

    left_score = left_avg + cap_consistency_weight * left_std
    right_score = right_avg + cap_consistency_weight * right_std

    if left_score < right_score:
        cap_end = "axis_min"
        cap_tip_axis_value = axis_min
        grasp_axis_value = axis_min + grasp_inset_fraction * tube_length
    else:
        cap_end = "axis_max"
        cap_tip_axis_value = axis_max
        grasp_axis_value = axis_max - grasp_inset_fraction * tube_length

    band_half_width = max(4.0, tube_length * grasp_band_fraction)
    in_grasp_band = np.abs(axis_values - grasp_axis_value) <= band_half_width
    if np.count_nonzero(in_grasp_band) < min_pixels_in_grasp_band:
        return None

    band_perp_values = perp_values[in_grasp_band]
    grasp_perp_value = 0.5 * (float(np.min(band_perp_values)) + float(np.max(band_perp_values)))
    grasp_xy = center_xy + axis_u * grasp_axis_value + perp_u * grasp_perp_value

    in_tip_band = np.abs(axis_values - cap_tip_axis_value) <= band_half_width
    if np.count_nonzero(in_tip_band) < min_pixels_in_grasp_band:
        tip_perp_value = grasp_perp_value
    else:
        tip_perp_values = perp_values[in_tip_band]
        tip_perp_value = 0.5 * (float(np.min(tip_perp_values)) + float(np.max(tip_perp_values)))

    cap_tip_xy = center_xy + axis_u * cap_tip_axis_value + perp_u * tip_perp_value
    cap_confidence = width_ratio - 1.0

    return {
        "contour_mask": contour_mask,
        "cap_end": cap_end,
        "cap_confidence": cap_confidence,
        "cap_tip_xy": cap_tip_xy,
        "grasp_xy": grasp_xy,
        "tube_length_px": tube_length,
    }


def build_final_candidate(candidate_group, merged_contour, depth_image, depth_scale, depth_intrin, blue_mask, green_mask, yellow_mask, mask_shape):
    shape_metrics = compute_shape_metrics(merged_contour)
    if not passes_shape_filters(shape_metrics):
        return None

    rect_cx, rect_cy = shape_metrics["rect"][0]
    cx = int(round(rect_cx))
    cy = int(round(rect_cy))

    img_h, img_w = mask_shape
    if (
        cx <= margin or cy <= margin or
        cx >= img_w - margin or cy >= img_h - margin
    ):
        return None

    distance_m = get_window_median_depth_m(
        depth_image,
        depth_scale,
        cx,
        cy,
        depth_window_size,
    )
    if distance_m <= 0:
        return None

    point_3d = rs.rs2_deproject_pixel_to_point(depth_intrin, [cx, cy], distance_m)
    X_m, Y_m, Z_m = point_3d

    label, box_color = get_dominant_color(
        blue_mask,
        green_mask,
        yellow_mask,
        merged_contour,
        mask_shape,
    )

    cap_geometry = analyze_cap_geometry(merged_contour, mask_shape, shape_metrics["angle_deg"])
    if cap_geometry is None:
        return None

    grasp_px = int(round(cap_geometry["grasp_xy"][0]))
    grasp_py = int(round(cap_geometry["grasp_xy"][1]))
    if (
        grasp_px <= margin or grasp_py <= margin or
        grasp_px >= img_w - margin or grasp_py >= img_h - margin
    ):
        return None

    grasp_distance_m = get_window_median_depth_m(
        depth_image,
        depth_scale,
        grasp_px,
        grasp_py,
        depth_window_size,
    )
    if grasp_distance_m <= 0:
        return None

    grasp_point_3d = rs.rs2_deproject_pixel_to_point(depth_intrin, [grasp_px, grasp_py], grasp_distance_m)
    grasp_X_m, grasp_Y_m, grasp_Z_m = grasp_point_3d

    box_points = cv2.boxPoints(shape_metrics["rect"])
    box_points = np.intp(box_points)

    cap_tip_px = int(round(cap_geometry["cap_tip_xy"][0]))
    cap_tip_py = int(round(cap_geometry["cap_tip_xy"][1]))

    return {
        "label": label,
        "box_color": box_color,
        "contour": merged_contour,
        "box_points": box_points,
        "cx": cx,
        "cy": cy,
        "X_m": X_m,
        "Y_m": Y_m,
        "Z_m": Z_m,
        "angle_deg": shape_metrics["angle_deg"],
        "axis_u": shape_metrics["axis_u"],
        "long_side": shape_metrics["long_side"],
        "grasp_px": grasp_px,
        "grasp_py": grasp_py,
        "grasp_X_m": grasp_X_m,
        "grasp_Y_m": grasp_Y_m,
        "grasp_Z_m": grasp_Z_m,
        "cap_end": cap_geometry["cap_end"],
        "cap_confidence": cap_geometry["cap_confidence"],
        "cap_tip_px": cap_tip_px,
        "cap_tip_py": cap_tip_py,
        "part_count": len(candidate_group),
        "area": shape_metrics["area"],
        "aspect_ratio": shape_metrics["aspect_ratio"],
        "fill_ratio": shape_metrics["fill_ratio"],
    }


def draw_detection(color_image, detection):
    box_color = detection["box_color"]
    box_points = detection["box_points"]

    cv2.drawContours(color_image, [box_points], 0, box_color, 2)
    cv2.circle(color_image, (detection["cx"], detection["cy"]), 4, (0, 0, 255), -1)

    axis_u = detection["axis_u"]
    line_half_length = int(detection["long_side"] * 0.35)
    dx = int(axis_u[0] * line_half_length)
    dy = int(axis_u[1] * line_half_length)
    cv2.line(
        color_image,
        (detection["cx"] - dx, detection["cy"] - dy),
        (detection["cx"] + dx, detection["cy"] + dy),
        box_color,
        2,
    )

    cv2.circle(color_image, (detection["cap_tip_px"], detection["cap_tip_py"]), 6, (255, 255, 255), 2)
    cv2.circle(color_image, (detection["grasp_px"], detection["grasp_py"]), 6, (0, 0, 255), -1)
    cv2.putText(
        color_image,
        "G",
        (detection["grasp_px"] + 8, detection["grasp_py"] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    text_x = int(np.min(box_points[:, 0]))
    text_y = int(np.min(box_points[:, 1])) - 12

    cv2.putText(
        color_image,
        detection["label"],
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        box_color,
        2,
    )

    cv2.putText(
        color_image,
        f"A={detection['angle_deg']:.1f} deg",
        (detection["cx"] + 10, detection["cy"] - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        box_color,
        2,
    )

    cv2.putText(
        color_image,
        f"GZ={detection['grasp_Z_m']:.3f} m",
        (detection["cx"] + 10, detection["cy"] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255),
        2,
    )

    cv2.putText(
        color_image,
        f"GX={detection['grasp_X_m']:.3f}, GY={detection['grasp_Y_m']:.3f}",
        (detection["cx"] + 10, detection["cy"] + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        2,
    )

    cv2.putText(
        color_image,
        f"parts={detection['part_count']} conf={detection['cap_confidence']:.2f}",
        (detection["cx"] + 10, detection["cy"] + 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
    )


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
pipeline_profile = pipeline.start(config)
depth_sensor = pipeline_profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

frame_count = 0
frame_skip_count = 50

last_target_key = None
stable_count = 0
required_stable_frames = 3

locked_target = None
lost_target_count = 0
max_lost_frames = 5

near_threshold_m = 0.50
combined_close_size = (7, 7)
foreground_close_size = (9, 9)
foreground_dilate_size = (3, 3)
min_contour_area = 2500
max_contour_area = 14000
min_aspect_ratio = 1.8
max_aspect_ratio = 5.5
min_fill_ratio = 0.45
margin = 8
depth_window_size = 5

merge_angle_diff_deg = 20.0
merge_depth_diff_m = 0.03
merge_perp_offset_px = 20.0
merge_axis_gap_px = 80.0

width_profile_bin_count = 24
end_width_fraction = 0.20
min_cap_width_ratio = 1.12
cap_consistency_weight = 0.35
grasp_inset_fraction = 0.12
grasp_band_fraction = 0.04
min_pixels_per_width_bin = 10
min_pixels_in_grasp_band = 20

try:
    while True:
        # Wait for a new set of frames from the camera
        frames = pipeline.wait_for_frames()
        frame_count += 1
        if frame_count < frame_skip_count:
            continue

        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

        depth_color_frame = colorizer.colorize(depth_frame)
        depth_colormap = np.asanyarray(depth_color_frame.get_data())

        color_image = np.asanyarray(color_frame.get_data())
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

        lower_blue = np.array([95, 90, 60])
        upper_blue = np.array([140, 255, 255])

        lower_green = np.array([35, 60, 50])
        upper_green = np.array([85, 255, 255])

        lower_yellow = np.array([20, 120, 120])
        upper_yellow = np.array([35, 255, 255])

        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        combined_mask = cv2.bitwise_or(blue_mask, green_mask)
        combined_mask = cv2.bitwise_or(combined_mask, yellow_mask)

        combined_close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, combined_close_size)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, combined_close_kernel)

        near_threshold_raw = int(near_threshold_m / depth_scale)
        near_mask = np.where(
            (depth_image > 0) & (depth_image < near_threshold_raw),
            255,
            0,
        ).astype(np.uint8)

        foreground_mask = cv2.bitwise_and(combined_mask, near_mask)

        foreground_close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, foreground_close_size)
        foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, foreground_close_kernel)

        foreground_dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, foreground_dilate_size)
        foreground_mask = cv2.dilate(foreground_mask, foreground_dilate_kernel, iterations=1)

        contours, _ = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        raw_candidates = []
        for contour in contours:
            candidate = build_raw_candidate(
                contour,
                depth_image,
                depth_scale,
                depth_intrin,
                blue_mask,
                green_mask,
                yellow_mask,
                foreground_mask.shape,
            )
            if candidate is not None:
                raw_candidates.append(candidate)

        grouped_candidates = group_raw_candidates(raw_candidates)
        detections = []

        for candidate_group in grouped_candidates:
            merged_contour = build_merged_contour(candidate_group, foreground_mask.shape)
            if merged_contour is None:
                continue

            detection = build_final_candidate(
                candidate_group,
                merged_contour,
                depth_image,
                depth_scale,
                depth_intrin,
                blue_mask,
                green_mask,
                yellow_mask,
                foreground_mask.shape,
            )
            if detection is None:
                continue

            detections.append(detection)
            draw_detection(color_image, detection)

        target = None

        if detections:
            print("Detections:")
            for detection in detections:
                print(
                    f'{detection["label"]}  parts={detection["part_count"]}  '
                    f'cx={detection["cx"]}  cy={detection["cy"]}  '
                    f'GX={detection["grasp_X_m"]:.3f}  GY={detection["grasp_Y_m"]:.3f}  '
                    f'GZ={detection["grasp_Z_m"]:.3f}  angle={detection["angle_deg"]:.1f}  '
                    f'cap={detection["cap_end"]}  conf={detection["cap_confidence"]:.2f}'
                )

            candidate = min(detections, key=lambda detection: detection["grasp_Z_m"])

            target_key = (
                candidate["label"],
                int(candidate["grasp_px"] / 30),
                int(candidate["grasp_py"] / 30),
            )

            if target_key == last_target_key:
                stable_count += 1
            else:
                stable_count = 1
                last_target_key = target_key

            if stable_count >= required_stable_frames:
                locked_target = candidate
                lost_target_count = 0

            print("CANDIDATE:")
            print(
                f'{candidate["label"]}  grasp_px={candidate["grasp_px"]}  grasp_py={candidate["grasp_py"]}  '
                f'GX={candidate["grasp_X_m"]:.3f}  GY={candidate["grasp_Y_m"]:.3f}  '
                f'GZ={candidate["grasp_Z_m"]:.3f}  angle={candidate["angle_deg"]:.1f}  '
                f'parts={candidate["part_count"]}  stable_count={stable_count}'
            )
        else:
            stable_count = 0
            last_target_key = None

            if locked_target is not None:
                lost_target_count += 1
                if lost_target_count > max_lost_frames:
                    locked_target = None
                    lost_target_count = 0

        if locked_target is not None:
            target = locked_target

            print("TARGET:")
            print(
                f'{target["label"]}  grasp_px={target["grasp_px"]}  grasp_py={target["grasp_py"]}  '
                f'GX={target["grasp_X_m"]:.3f}  GY={target["grasp_Y_m"]:.3f}  '
                f'GZ={target["grasp_Z_m"]:.3f}  angle={target["angle_deg"]:.1f}  '
                f'parts={target["part_count"]}'
            )
            print("-" * 50)

            cv2.putText(
                color_image,
                "TARGET",
                (target["grasp_px"] + 10, target["grasp_py"] + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

        cv2.imshow("Color", color_image)
        cv2.imshow("Combined Mask", combined_mask)
        cv2.imshow("Near Mask", near_mask)
        cv2.imshow("Foreground Mask", foreground_mask)
        cv2.imshow("Depth", depth_colormap)

        key = cv2.waitKey(1)
        if key == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

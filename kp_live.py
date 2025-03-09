import argparse
import cv2
import numpy as np
from numba import njit

@njit
def to_gray(img):
    h, w = img.shape[0], img.shape[1]
    gray = np.empty((h, w), dtype=img.dtype)
    for i in range(h):
        for j in range(w):
            gray[i, j] = 0.07 * img[i, j, 0] + 0.72 * img[i, j, 1] + 0.21 * img[i, j, 2]
    return gray

@njit
def minmax_normalize(img):
    mn = np.min(img)
    mx = np.max(img)
    return (img - mn) / (mx - mn + 1e-8)

def mask_stuff(img, omega=0.8):
    imgvec = img.reshape(-1, 3)
    x_RGB = np.mean(imgvec, axis=0)
    x_mean = np.repeat(x_RGB[np.newaxis, np.newaxis, :], img.shape[0], axis=0)
    x_mean = np.repeat(x_mean, img.shape[1], axis=1)

    scat_basis = x_mean / np.maximum(
        np.sqrt(np.sum(x_mean**2, axis=2, keepdims=True)), 0.001
    )
    fog_basis = img / np.maximum(np.sqrt(np.sum(img**2, axis=2, keepdims=True)), 0.001)
    cs_sim = np.sum(scat_basis * fog_basis, axis=2, keepdims=True)

    scattering_light = (
        cs_sim
        * (
            np.sum(img, axis=2, keepdims=True)
            / np.maximum(np.sum(x_mean, axis=2, keepdims=True), 0.001)
        )
        * x_mean
    )

    T = 1 - omega * scattering_light
    T_m = to_gray(T**2)

    gaussian1 = cv2.GaussianBlur(T_m, (0, 0), sigmaX=1)
    gaussian2 = cv2.GaussianBlur(T_m, (0, 0), sigmaX=21)
    dog = gaussian1 - gaussian2

    dog = minmax_normalize(dog)
    dog = dog**2 + (T_m - gaussian1)

    fog_removed = dog
    return np.clip(fog_removed, 0, 1)

def sort_points(pts):
    centroid = np.mean(pts, axis=0)
    diff = pts - centroid
    angles = np.arctan2(diff[:, 1], diff[:, 0])
    sorted_indices = np.argsort(-angles)
    sorted_pts = pts[sorted_indices]
    return sorted_pts

def extract_keypoints(keypoints):
    pts = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    if pts.shape[0] == 0:
        return None
    if pts.shape[0] <= 4:
        return pts
    hull = cv2.convexHull(pts)
    epsilon = 0.03 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)
    if len(approx) <= 4:
        return approx.reshape(-1, 2)
    else:
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)
        return np.array(
            [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
            dtype=np.float32,
        )

def process_image(img, detector, draw_flag=False):
    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    result = mask_stuff(img_rgb)
    result = (result * 255).astype(np.uint8)
    result[result < np.percentile(result, 10)] = 0

    keypoints, _desc = detector.detectAndCompute(result, None)
    four_pts = extract_keypoints(keypoints)

    # Always draw detected keypoints as red circles
    if four_pts is not None:
        for pt in four_pts:
            pt_int = tuple(np.round(pt).astype(int))
            cv2.circle(img, pt_int, 5, (0, 0, 255), -1)

    if four_pts is None or len(four_pts) != 4:
        return img, None, None

    try:
        sorted_pts = sort_points(four_pts)
    except:
        return img, None, None

    object_pts = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0]
    ], dtype=np.float32)

    fx = w
    fy = h
    cx = w / 2
    cy = h / 2
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1))

    success, rvec, tvec = cv2.solvePnP(object_pts, sorted_pts.astype(np.float32), camera_matrix, dist_coeffs)

    if not success:
        return img, None, None

    if draw_flag:
        axis_points = np.array([
            [0, 0, 0],
            [0.5, 0, 0],
            [0, 0.5, 0],
            [0, 0, 0.5]
        ], dtype=np.float32)
        imgpts, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)
        imgpts = imgpts.reshape(-1, 2).astype(int)
        origin = tuple(imgpts[0])
        x_end = tuple(imgpts[1])
        y_end = tuple(imgpts[2])
        z_end = tuple(imgpts[3])
        cv2.line(img, origin, x_end, (0, 0, 255), 2)
        cv2.line(img, origin, y_end, (0, 255, 0), 2)
        cv2.line(img, origin, z_end, (255, 0, 0), 2)

    return img, rvec, tvec

def main():
    parser = argparse.ArgumentParser(description="Estimate object pose from video and save results")
    parser.add_argument("--video", type=str, help="Path to input video file")
    parser.add_argument("--live", action="store_true", help="Use webcam as input")
    parser.add_argument("--draw", action="store_true", help="Draw pose axes on the output")
    parser.add_argument("--output", type=str, required=True, help="Path to save output video file")
    args = parser.parse_args()

    if not args.video and not args.live:
        print("Error: Must specify either --video or --live")
        return

    detector = cv2.AKAZE_create()

    if args.live:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        print("Error: Could not open video source")
        return

    # Get video properties for output
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) if not args.live else 30
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame, rvec, tvec = process_image(frame, detector, args.draw)
        if rvec is not None and tvec is not None:
            print(f"Rotation: {rvec.ravel()}, Translation: {tvec.ravel()}")

        out.write(processed_frame)

    cap.release()
    out.release()
    print(f"Saved processed video to: {args.output}")

if __name__ == "__main__":
    main()
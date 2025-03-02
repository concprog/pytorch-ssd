from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import (
    create_mobilenetv1_ssd,
    create_mobilenetv1_ssd_predictor,
)
from vision.ssd.mobilenetv1_ssd_lite import (
    create_mobilenetv1_ssd_lite,
    create_mobilenetv1_ssd_lite_predictor,
)
from vision.ssd.squeezenet_ssd_lite import (
    create_squeezenet_ssd_lite,
    create_squeezenet_ssd_lite_predictor,
)
from vision.ssd.mobilenet_v2_ssd_lite import (
    create_mobilenetv2_ssd_lite,
    create_mobilenetv2_ssd_lite_predictor,
)
from vision.utils.misc import Timer
import cv2
import sys
import numpy as np
from numba import njit


@njit
def to_gray(img):
    h, w = img.shape[:2]
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

    return np.clip(dog, 0, 1)


def sort_points(pts):
    centroid = np.mean(pts, axis=0)
    diff = pts - centroid
    angles = np.arctan2(diff[:, 1], diff[:, 0])
    sorted_indices = np.argsort(-angles)
    return pts[sorted_indices]


def extract_keypoints(keypoints):
    if not keypoints:
        return None
    pts = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    if len(pts) <= 4:
        return pts
    hull = cv2.convexHull(pts)
    epsilon = 0.03 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)
    return approx.reshape(-1, 2) if len(approx) <= 4 else None


if len(sys.argv) < 5:
    print(
        "Usage: python run_ssd_video_example.py <net type> <model path> <label path> <video path>"
    )
    sys.exit(0)

net_type = sys.argv[1]
model_path = sys.argv[2]
label_path = sys.argv[3]
video_path = sys.argv[4]

class_names = [name.strip() for name in open(label_path).readlines()]
detector = cv2.AKAZE_create()

# Initialize SSD network
if net_type == "vgg16-ssd":
    net = create_vgg_ssd(len(class_names), is_test=True)
elif net_type == "mb1-ssd":
    net = create_mobilenetv1_ssd(len(class_names), is_test=True)
elif net_type == "mb1-ssd-lite":
    net = create_mobilenetv1_ssd_lite(len(class_names), is_test=True)
elif net_type == "mb2-ssd-lite":
    net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True, device="cpu")
elif net_type == "sq-ssd-lite":
    net = create_squeezenet_ssd_lite(len(class_names), is_test=True)
else:
    print("Invalid network type")
    sys.exit(1)

net.load(model_path)
net.to("cpu")

# Create predictor
predictor = {
    "vgg16-ssd": create_vgg_ssd_predictor,
    "mb1-ssd": create_mobilenetv1_ssd_predictor,
    "mb1-ssd-lite": create_mobilenetv1_ssd_lite_predictor,
    "mb2-ssd-lite": lambda net: create_mobilenetv2_ssd_lite_predictor(
        net, device="cpu"
    ),
    "sq-ssd-lite": create_squeezenet_ssd_lite_predictor,
}[net_type](net)

cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("output_video.avi", fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess for keypoint detection
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
    processed_frame = (mask_stuff(rgb_frame) * 255).astype(np.uint8)

    # SSD detection
    boxes, labels, probs = predictor.predict(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 10, 0.4
    )

    for i in range(boxes.size(0)):
        box = boxes[i, :].numpy().astype(int)
        label = f"{class_names[labels[i]]}: {probs[i]:.2f}"

        # Draw bounding box
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 2)
        cv2.putText(
            frame,
            label,
            (box[0] + 10, box[1] + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 0, 255),
            2,
        )

        # Keypoint processing with margin
        margin = 50
        x1, y1 = max(0, box[0] - margin), max(0, box[1] - margin)
        x2, y2 = min(frame_width, box[2] + margin), min(frame_height, box[3] + margin)
        roi = processed_frame[y1:y2, x1:x2]

        if roi.size == 0:
            continue

        # Detect and process keypoints
        keypoints = detector.detect(roi)
        four_pts = extract_keypoints(keypoints)

        if four_pts is not None and len(four_pts) == 4:
            four_pts += np.array([x1, y1])  # Adjust coordinates
            sorted_pts = sort_points(four_pts)

            # Draw keypoints
            for pt in sorted_pts:
                cv2.circle(frame, tuple(pt.astype(int)), 5, (0, 0, 255), -1)

            # Pose estimation
            obj_pts = np.array(
                [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32
            )
            camera_matrix = np.array(
                [
                    [frame_width, 0, frame_width / 2],
                    [0, frame_height, frame_height / 2],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            )

            ret, rvec, tvec = cv2.solvePnP(
                obj_pts, sorted_pts.astype(np.float32), camera_matrix, None
            )

            if ret:
                # Draw pose axes
                axis_pts = np.float32([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]]).reshape(
                    -1, 3
                )
                img_pts, _ = cv2.projectPoints(
                    axis_pts, rvec, tvec, camera_matrix, None
                )
                origin = tuple(sorted_pts[0].astype(int))
                cv2.line(
                    frame, origin, tuple(img_pts[0].ravel().astype(int)), (0, 0, 255), 3
                )
                cv2.line(
                    frame, origin, tuple(img_pts[1].ravel().astype(int)), (0, 255, 0), 3
                )
                cv2.line(
                    frame, origin, tuple(img_pts[2].ravel().astype(int)), (255, 0, 0), 3
                )

                # Print pose information
                print(f"Object: {class_names[labels[i]]}")
                print(f"Rotation: {rvec.ravel()}\nTranslation: {tvec.ravel()}\n")

    out.write(frame)

cap.release()
out.release()
print("Processing complete. Output saved to output_video.avi")

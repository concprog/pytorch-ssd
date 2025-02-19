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
import os
from time import time
import matplotlib.pyplot as plt

if len(sys.argv) < 5:
    print("Usage: python run_ssd_video_example.py <net type> <model path> <label path> <video path>")
    sys.exit(0)
net_type = sys.argv[1]
model_path = sys.argv[2]
label_path = sys.argv[3]
video_path = sys.argv[4]

class_names = [name.strip() for name in open(label_path).readlines()]

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
    print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
    sys.exit(1)
net.load(model_path)
net.to("cpu")

if net_type == "vgg16-ssd":
    predictor = create_vgg_ssd_predictor(net, candidate_size=200)
elif net_type == "mb1-ssd":
    predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)
elif net_type == "mb1-ssd-lite":
    predictor = create_mobilenetv1_ssd_lite_predictor(net, candidate_size=200)
elif net_type == "mb2-ssd-lite":
    predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200, device="cpu")
elif net_type == "sq-ssd-lite":
    predictor = create_squeezenet_ssd_lite_predictor(net, candidate_size=200)
else:
    predictor = create_vgg_ssd_predictor(net, candidate_size=200)

# Open the video file
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    sys.exit(1)

# Set up the VideoWriter to save output video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
out_path = "output_video.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(out_path, fourcc, fps, (frame_width, frame_height))

fcent_x, fcent_y = frame_width // 2, frame_height // 2

# Lists to record x and y errors
x_err_list = []
y_err_list = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Convert frame from BGR to RGB for prediction
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, labels, probs = predictor.predict(image, 10, 0.4)
    for i in range(boxes.size(0)):
        box = boxes[i, :]
        box = list(map(int, box))
        bcent_x, bcent_y = box[0] + box[2] // 2, box[1] + box[3] // 2
        x_err, y_err = bcent_x - fcent_x, bcent_y - fcent_y
        # Record the errors
        x_err_list.append(x_err)
        y_err_list.append(y_err)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
        label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
        cv2.putText(frame, label, (box[0] + 20, box[1] + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    cv2.imshow("Gate BBox", frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Processed video saved as {out_path}")

# Create the plots directory if it doesn't exist
if not os.path.exists("plots"):
    os.makedirs("plots")

# Plot x error
plt.figure()
plt.plot(x_err_list, label="X Error", color='blue')
plt.xlabel("Detection Index")
plt.ylabel("X Error (pixels)")
plt.title("X Error over Detections")
plt.legend()
plt.savefig(os.path.join("plots", f"x_error_plot_{time()}.png"))
plt.close()

# Plot y error
plt.figure()
plt.plot(y_err_list, label="Y Error", color='green')
plt.xlabel("Detection Index")
plt.ylabel("Y Error (pixels)")
plt.title("Y Error over Detections")
plt.legend()
plt.savefig(os.path.join("plots", f"y_error_plot{time()}.png"))
plt.close()


import cv2
import numpy as np
import onnxruntime as ort
import time

# ======================
# KONFIGURASI
# ======================
MODEL_PATH = "v8s_320.onnx"
IMG_SIZE = 320
CONF_THRESHOLD = 0.4
IOU_THRESHOLD = 0.5

# Ganti dengan daftar class sesuai training kamu
CLASS_NAMES = ["Longitudinal Crack", "Transverse Crack", "Alligator Crack", "Potholes"]

# ======================
# LOAD MODEL
# ======================
so = ort.SessionOptions()
so.intra_op_num_threads = 4
session = ort.InferenceSession(MODEL_PATH, so)
input_name = session.get_inputs()[0].name

# ======================
# VIDEO SOURCE
# ======================
cap = cv2.VideoCapture("vid1.mp4")
# cap = cv2.VideoCapture(0)  # untuk webcam

prev_time = 0

def preprocess(frame):
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def nms(boxes, scores, iou_threshold):
    indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESHOLD, iou_threshold)
    return indices

while True:
    ret, frame = cap.read()
    if not ret:
        break

    original = frame.copy()
    h, w = frame.shape[:2]

    img_input = preprocess(frame)

    outputs = session.run(None, {input_name: img_input})
    preds = np.squeeze(outputs[0]).T

    boxes = []
    scores = []
    class_ids = []

    for pred in preds:
        conf = np.max(pred[4:])
        if conf > CONF_THRESHOLD:
            class_id = np.argmax(pred[4:])
            cx, cy, bw, bh = pred[:4]

            x = int((cx - bw / 2) * w / IMG_SIZE)
            y = int((cy - bh / 2) * h / IMG_SIZE)
            width = int(bw * w / IMG_SIZE)
            height = int(bh * h / IMG_SIZE)

            boxes.append([x, y, width, height])
            scores.append(float(conf))
            class_ids.append(class_id)

    indices = nms(boxes, scores, IOU_THRESHOLD)

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, width, height = boxes[i]
            label = CLASS_NAMES[class_ids[i]]
            confidence = scores[i]

            cv2.rectangle(original, (x, y), (x + width, y + height), (0, 255, 0), 2)
            cv2.putText(
                original,
                f"{label} {confidence:.2f}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

    # ======================
    # FPS COUNTER
    # ======================
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time

    cv2.putText(
        original,
        f"FPS: {fps:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
    )

    cv2.imshow("YOLOv8 ONNX Detection", original)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

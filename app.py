import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
from keras import models, layers, optimizers, callbacks
import matplotlib.pyplot as plt
import threading
import queue

# Load YOLOv8 model from Ultralytics hub
model = YOLO('yolov8x')

# Open video capture
cap = cv2.VideoCapture('video.mp4')

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

object_counts = defaultdict(list)
frames = []
train_losses = []

# Data collection for training
X_train = []
y_train = []

# Define a simple neural network model for demonstration
nn_model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(4,)),  # 4 features: x, y, width, height
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
nn_model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Custom callback for real-time training loss visualization
class TrainingLossPlotter(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        train_losses.append(logs['loss'])

# Initialize the callback
loss_plotter = TrainingLossPlotter()

# Function to create line plot
def create_line_plot(x, y, title, xlabel, ylabel):
    plt.figure(figsize=(5, 3))
    for label, counts in y.items():
        plt.plot(x[:len(counts)], counts, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig('temp_plot.png')
    plt.close()
    return cv2.imread('temp_plot.png')

# Function to train the model in a separate thread
def train_model(X, y, queue):
    history = nn_model.fit(X, y, epochs=50, batch_size=32, callbacks=[loss_plotter], validation_split=0.2)
    queue.put(history)

# Data collection and visualization loop
frame_count = 0
training_started = False
training_queue = queue.Queue()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model(frame)

    # Count objects and collect training data
    current_counts = defaultdict(int)
    frame_data = []
    frame_labels = []
    for result in results:
        for box in result.boxes:
            label = result.names[int(box.cls)]
            current_counts[label] += 1
            
            # Collect training data
            x, y, w, h = box.xywh[0]  # center x, center y, width, height
            frame_data.append([x.item(), y.item(), w.item(), h.item()])
            frame_labels.append(1 if label != '' else 0)  # Binary classification: person vs non-person
    
    # Only add data if objects were detected
    if frame_data:
        X_train.extend(frame_data)
        y_train.extend(frame_labels)

    # Update object counts
    for label, count in current_counts.items():
        object_counts[label].append(count)
    
    frames.append(frame_count)

    # Start training if we have enough data and training hasn't started yet
    if len(X_train) >= 100 and not training_started:
        X_train_np = np.array(X_train)
        y_train_np = np.array(y_train)
        training_thread = threading.Thread(target=train_model, args=(X_train_np, y_train_np, training_queue))
        training_thread.start()
        training_started = True

    # Create object counts plot
    counts_plot = create_line_plot(frames, object_counts, 'Object Counts', 'Frame', 'Count')

    # Create training loss plot (if training has started)
    if train_losses:
        loss_plot = create_line_plot(range(len(train_losses)), {'Loss': train_losses}, 'Training Loss', 'Epoch', 'Loss')
    else:
        loss_plot = np.zeros((height // 2, width // 2, 3), dtype=np.uint8)

    # Overlay analytics on the frame
    frame_with_boxes = results[0].plot()
    
    # Resize plots to fit the frame
    counts_plot = cv2.resize(counts_plot, (width // 2, height // 2))
    loss_plot = cv2.resize(loss_plot, (width // 2, height // 2))

    # Place plots on the frame
    frame_with_boxes[0:height // 2, 0:width // 2] = counts_plot
    frame_with_boxes[0:height // 2, width // 2:width] = loss_plot

    cv2.namedWindow("YOLOv8 Object Detection", cv2.WINDOW_NORMAL)

    # Display the frame with bounding boxes and analytics
    cv2.imshow('YOLOv8 Object Detection', frame_with_boxes)

    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Wait for training to complete if it has started
if training_started:
    training_thread.join()
    history = training_queue.get()
    print("Training completed.")

# Release video capture
cap.release()
cv2.destroyAllWindows()

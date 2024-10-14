import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
from keras import models, layers, optimizers, callbacks
import matplotlib.pyplot as plt
import threading
import queue
import time
import os
from sklearn.metrics import precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay

# Load model from Ultralytics hub
model = YOLO('yolo11x')

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

# Load pre-trained weights if they exist
if os.path.exists('model.weights.h5'):
    print("Loading pre-trained weights...")
    nn_model.load_weights('model.weights.h5')

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

# Function to create precision-recall curve
def create_precision_recall_plot(y_true, y_pred):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    plt.figure(figsize=(5, 3))
    plt.plot(recall, precision, marker='.')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.tight_layout()
    plt.savefig('precision_recall_plot.png')
    plt.close()
    return cv2.imread('precision_recall_plot.png')

# Function to create confusion matrix plot
def create_confusion_matrix_plot(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.tight_layout()
    plt.savefig('confusion_matrix_plot.png')
    plt.close()
    return cv2.imread('confusion_matrix_plot.png')

# Function to train the model in a separate thread
def train_model(X, y, queue):
    history = nn_model.fit(X, y, epochs=50, batch_size=32, callbacks=[loss_plotter], validation_split=0.2)
    queue.put(history)

# Object tracking
track_history = defaultdict(lambda: [])

# Data collection and visualization loop
frame_count = 0
training_started = False
training_queue = queue.Queue()

while cap.isOpened():
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection and tracking
    results = model.track(frame, persist=True)

    # Count objects and collect training data
    current_counts = defaultdict(int)
    frame_data = []
    frame_labels = []
    
    boxes = results[0].boxes.xywh.cpu()
    track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else None

    for box, track_id in zip(boxes, track_ids):
        x, y, w, h = box
        label = results[0].names[int(results[0].boxes.cls[track_ids.index(track_id)])]
        current_counts[label] += 1
        
        # Collect training data
        frame_data.append([x.item(), y.item(), w.item(), h.item()])
        frame_labels.append(1 if label == 'person' else 0)  # Binary classification: person vs non-person

        # Update tracking history
        track = track_history[track_id]
        track.append((float(x), float(y)))  # x, y center point
        if len(track) > 30:  # retain 30 tracks for 30 frames
            track.pop(0)

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

    # Plot the tracks
    for track_id, track in track_history.items():
        points = np.array(track).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame_with_boxes, [points], isClosed=False, color=(230, 230, 230), thickness=2)
    
    # Resize plots to fit the frame
    counts_plot = cv2.resize(counts_plot, (width // 2, height // 2))
    loss_plot = cv2.resize(loss_plot, (width // 2, height // 2))

    # Place plots on the frame
    frame_with_boxes[0:height // 2, 0:width // 2] = counts_plot
    frame_with_boxes[0:height // 2, width // 2:width] = loss_plot

    # Calculate FPS and inference time
    end_time = time.time()
    inference_time = end_time - start_time
    fps = 1 / inference_time

    # Display FPS and inference time on the frame
    cv2.putText(frame_with_boxes, f'FPS: {fps:.2f}', (10, 620), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame_with_boxes, f'Inference Time: {inference_time:.2f}s', (10, 650), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.namedWindow("Object Detection with Neural Network Training", cv2.WINDOW_NORMAL)

    # Display the frame with bounding boxes and analytics
    cv2.imshow('Object Detection with Neural Network Training', frame_with_boxes)

    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Wait for training to complete if it has started
if training_started:
    training_thread.join()
    history = training_queue.get()
    print("Training completed.")

    # Save model weights
    nn_model.save_weights('model.weights.h5')

    # Create and display additional plots
    y_pred = nn_model.predict(np.array(X_train))
    precision_recall_plot = create_precision_recall_plot(y_train, y_pred)
    confusion_matrix_plot = create_confusion_matrix_plot(y_train, y_pred.round())

    cv2.imshow('Precision-Recall Curve', precision_recall_plot)
    cv2.imshow('Confusion Matrix', confusion_matrix_plot)
    cv2.waitKey(0)

# Release video capture
cap.release()
cv2.destroyAllWindows()

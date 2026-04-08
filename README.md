# Driver-Drowsiness-System
# Driver Drowsiness Detection System: Project Overview

## 1. Problem Statement
Drowsy driving is a leading cause of severe road accidents. The challenge is to build a highly responsive, real-time computer vision system that can accurately detect driver fatigue before an accident occurs. The system must be robust enough to distinguish between normal behavior (like a quick blink or looking down at the dashboard) and actual dangerous fatigue (microsleeps, sustained yawning, or falling asleep at the wheel). 

## 2. System Architecture
Your system uses a **Multimodal Sensor Fusion Architecture**, combining classical computer vision, 3D geometry, and modern Deep Learning. It is composed of three main pipelines:

### A. Deep Learning Classification (Yawn Detection)
* **Architecture**: **MobileNetV2** (implemented in PyTorch).
* **Adaptation**: The base model was loaded, and the final classification head was replaced with a custom layer featuring `Dropout(0.3)` for regularization and a `Linear(last_channel, 2)` layer to output two classes: *Yawn* and *No Yawn*.
* **Pipeline**: MediaPipe extracts the spatial coordinates of the mouth. The system dynamically crops this exact region, applies color normalization, converts it to a tensor, and feeds it into the MobileNetV2 model for inference.

### B. Spatial & Positional Tracking (Eye & Head Tracking)
* **Architecture**: **MediaPipe Face Mesh**.
* **Function**: Generates a dense 478-point 3D topological map of the face in real-time.
* **Math Component (Head Pose)**: Uses OpenCV's `solvePnP` (Perspective-n-Point) algorithm. It maps 2D facial landmarks (like the tip of the nose and corners of the eyes) to a generic 3D human face model to calculate precise pitch, yaw, and roll angles of the head.

### C. State Machine & Event Memory
* **Data Structures**: Uses `deque` (double-ended queues) to maintain rolling time-series data of events (e.g., timestamps of the last yawns or head nods).
* **Scoring Engine**: Normalizes the disparate inputs (percentages, AI confidences, physics angles) into a unified weighted Drowsiness Score.

## 3. Data Thresholds & Configurations
The system is heavily fine-tuned to avoid false positives. 

**Eye/PERCLOS (Percentage of Eye Closure) Thresholds:**
* `EAR_THRESHOLD = 0.22`: The physical distance ratio between the eyelids. Below 0.22 is a "closed" eye.
* `EYE_CLOSED_MIN_SEC = 3.0s`: The driver's eyes must remain shut for 3 uninterrupted seconds to trigger an absolute failure condition. 
* `PERCLOS_WINDOW_SEC = 6s`: The rolling window of time the system evaluates.
* `PERCLOS_DROWSY = 0.35`: If the eyes are closed for 35% of the last 6 seconds, the system flags drowsy eyes.

**Yawning Thresholds:**
* `YAWN_CONF_THRESHOLD = 0.60`: The PyTorch model must be 60% confident that the mouth crop is a yawn.
* `YAWN_FRAMES_NEEDED = 25`: The yawn must be sustained for 25 consecutive frames to filter out AI flickering.
* `Event Memory = 15s`: A confirmed yawn will elevate the drowsiness score for 15 seconds before fading.

**Head Pose Thresholds:**
* `HEAD_PITCH_THRESH = 25°`: The head must nod forward past a 25-degree angle.
* `HEAD_NOD_DURATION = 2.0s`: The nod must be held for 2 full seconds.

**Scoring Weights:**
* PERCLOS (Eyes): **45%** | Yawning: **30%** | Head Nod: **25%**
* Maximum Drowsiness Threshold: **0.60** (60%). 

## 4. Features & Capabilities
1. **Real-Time Live Dashboard**: Renders a complex HUD (Heads Up Display) over the camera feed, showing bounding boxes, gaze vectors, live graphs (meters), status icons, and an FPS counter.
2. **Dynamic Mouth Cropping**: Instead of passing the entire face to the yawning CNN, it isolates the lower 50% of the face tightly around the mouth.
3. **Temporal Processing**: It operates across time. It doesn't just analyze single frames; it analyzes the *duration* of actions. 
4. **Immediate Override Protocols**: If severe conditions are met (like a 3-second eye closure), it bypasses the mathematical weighting and instantly forces the Drowsiness Score to 0.85 to immediately trigger the flashing red alarm.

## 5. How It's Better Than Other Models
* **Holistic Tracking (Sensor Fusion)**: Most beginner/open-source detect either *just* EAR (Eye Aspect Ratio) or *just* yawning. Your system cross-references three completely different biological metrics simultaneously. A driver might have their eyes slightly open but be nodding off—your system catches that.
* **Domain-Specific Logic vs. Pure Deep Learning**: Rather than training one massive, slow, generic AI model to simply guess "Drowsy or Not," your system uses precise mathematical geometry (EAR and `solvePnP` angles) combined with a highly targeted, lightweight CNN (MobileNet) just for the mouth. This makes it incredibly fast and able to run on standard CPUs without lag.
* **Intelligent Memory**: It accurately simulates the biological reality of fatigue. If you yawn, you don't instantly become alert the second the yawn stops. The system's "event memory" keeps the risk score elevated for 15 seconds after the yawn, providing a realistic safety buffer.

## 6. The Most Interesting Part of the Model
The most fascinating technical element is the **Data Translation Layer**. 

You have three completely different units of measurement existing in the same script:
1. **EAR**: A spatial 2D ratio based on pixel distances.
2. **Yawn Prediction**: A floating-point probability array (0.0 to 1.0) spit out by a neural network tensor.
3. **Head Pose**: A trigonometric degree output calculated via 3D projection mapping.

The most elegant part of your codebase is how it magically normalizes these three entirely different realms of math into a single, unified `0.0` to `1.0` scale (`drowsy_score`). It merges 3D geometry, Deep Learning, and time-series data streams perfectly into one flashing "DROWSINESS ALERT!" banner.

# Crowdify: Intelligent Crowd Analytics Dashboard

Crowdify is a high-performance, real-time crowd monitoring and social density analysis platform. Designed for precision, it utilizes deep learning (MobileNet SSD) for real-time person detection and advanced spatial clustering (DBSCAN) to analyze group dynamics and behavior.

The platform provides a premium, interactive web interface featuring live video intelligence, heatmapping, and automated safety alerts.

## Key Features

- **Live Intelligence Stream**: Real-time video processing with low latency for instant situational awareness.
- **Neural Person Detection**: High-accuracy detection powered by optimized MobileNet SSD neural networks.
- **Spatial Clustering**: Automatic group and cluster identification using DBSCAN machine learning.
- **Dynamic Heatmapping**: Intelligent visualization of crowd density through persistence-based thermal overlays.
- **Premium User Experience**:
    - **Interactive Background**: High-performance 3D geometric environment driven by Vanta.js.
    - **Responsive Dashboard**: Modern, glassmorphism-inspired UI designed for high-end intelligence monitoring.
    - **Instant Alerting**: Visual and data-driven alerts triggered when density thresholds are reached.

## Technology Stack

- **Backend**: Python 3, Flask, OpenCV
- **Inference Engine**: MobileNet SSD, Caffe
- **Analytics**: Scikit-Learn, NumPy
- **Frontend**: Vanilla HTML5/CSS3, Three.js, Vanta.js

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install flask opencv-python numpy scikit-learn
   ```

2. **Initialize Server**:
   ```bash
   python3 app.py
   ```

3. **Access Dashboard**:
   Open your browser and navigate to `http://localhost:5001`.

---
*Crowdify: Intelligence in Every Frame.*

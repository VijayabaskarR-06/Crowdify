import cv2
import numpy as np
import os
from sklearn.cluster import DBSCAN

class CrowdMonitor:
    def __init__(self, model_proto: str = None, 
                 model_weights: str = None,
                 alert_threshold: int = 5, 
                 cluster_distance: int = 75, 
                 cluster_size_threshold: int = 3):
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if model_proto is None:
            model_proto = os.path.join(base_dir, 'mobilenet', 'MobileNetSSD_deploy.prototxt')
        if model_weights is None:
            model_weights = os.path.join(base_dir, 'mobilenet', 'MobileNetSSD_deploy.caffemodel')

        self.net = cv2.dnn.readNetFromCaffe(model_proto, model_weights)
        self.person_class_id = 15
        
        self.alert_threshold = alert_threshold
        self.cluster_distance = cluster_distance
        self.cluster_size_threshold = cluster_size_threshold
        
        self.current_people_count = 0
        self.current_cluster_count = 0
        self.heatmap = None
        self.h = 0
        self.w = 0

    def process_frame(self, frame):
        if frame is None:
            return None
        
        if self.heatmap is None:
            self.h, self.w = frame.shape[:2]
            self.heatmap = np.zeros((self.h, self.w), dtype=np.float32)

        h, w = self.h, self.w
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        self.net.setInput(blob)
        detections = self.net.forward()

        people_centroids = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            class_id = int(detections[0, 0, i, 1])

            if confidence > 0.5 and class_id == self.person_class_id:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype("int")
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                
                if 0 <= cx < w and 0 <= cy < h:
                    people_centroids.append([cx, cy])
                    self.heatmap[cy, cx] += 0.5 
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (57, 255, 20), 2)

        self.current_people_count = len(people_centroids)

        self.current_cluster_count = 0
        if len(people_centroids) > 0:
            people_np = np.array(people_centroids)
            clustering = DBSCAN(eps=self.cluster_distance, min_samples=self.cluster_size_threshold).fit(people_np)
            labels = clustering.labels_

            unique_clusters = set(labels)
            if -1 in unique_clusters:
                unique_clusters.remove(-1)
            self.current_cluster_count = len(unique_clusters)

        heatmap_blur = cv2.GaussianBlur(self.heatmap, (51, 51), 0)
        heatmap_norm = cv2.normalize(heatmap_blur, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_color = cv2.applyColorMap(heatmap_norm.astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_color = cv2.resize(heatmap_color, (w, h))
        
        overlay = cv2.addWeighted(heatmap_color, 0.6, frame, 0.4, 0)

        cv2.putText(overlay, f"People: {self.current_people_count}", (20, 40),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(overlay, f"Clusters: {self.current_cluster_count}", (w - 200, 40),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 0), 2)

        if self.current_people_count >= self.alert_threshold:
            cv2.putText(overlay, "ALERT: CROWD FORMING", (20, 80),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)

        return overlay

    @property
    def is_crowded(self) -> bool:
        return self.current_people_count >= self.alert_threshold

    def get_stats(self) -> dict:
        import datetime
        return {
            "people_count": self.current_people_count,
            "cluster_count": self.current_cluster_count,
            "alert": self.is_crowded,
            "timestamp": datetime.datetime.now().isoformat()
        }

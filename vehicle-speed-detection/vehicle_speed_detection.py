"""
Araç Hız Tespiti / Vehicle Speed Detection
------------------------------------------

Bu program YOLOv8 nesne tespit modeli ve Supervision kütüphanesi kullanarak
videodaki araçların hızını hesaplar ve ekranda gösterir.

This program uses YOLOv8 object detection and the Supervision library
to calculate and display vehicle speeds in a video.

Özellikler / Features:
- YOLOv8 ile araç tespiti (object detection with YOLOv8)
- ByteTrack ile nesne takibi (object tracking with ByteTrack)
- Perspektif dönüşümü ile hız tahmini (speed estimation via perspective transform)
- Araç hızını ekrana yazdırma (displaying vehicle speed on video)
"""

import supervision as sv
import cv2
from ultralytics import YOLO
import numpy as np
from collections import defaultdict, deque

# Dönüşüm için kaynak koordinatlar / Source coordinates for perspective transform
SOURCE = np.array([[850, 480], [880, 480], [760, 1080], [700, 1080]])

# Hedef alanın genişliği ve yüksekliği / Target width and height
TARGET_WIDTH = 20
TARGET_HEIGHT = 120

# Hedef koordinatlar / Target coordinates
TARGET = np.array(
    [
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ]
)


class ViewTransformer:
    """
    Görüntü Perspektif Dönüştürücü / View Perspective Transformer
    -------------------------------------------------------------

    Belirlenen kaynak (SOURCE) ve hedef (TARGET) noktaları kullanarak
    görüntü üzerindeki koordinatları dönüştürür. Bu sayede araçların
    kameraya göre konumu gerçek dünyaya yakın bir şekilde hesaplanabilir.

    Transforms image coordinates based on defined source (SOURCE) and
    target (TARGET) points. This enables estimating the real-world-like
    position of vehicles relative to the camera.

    Attributes:
        m (np.ndarray): Perspektif dönüşüm matrisi / Perspective transform matrix
    """

    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        """
        Args:
            source (np.ndarray): Kaynak noktalar / Source points
            target (np.ndarray): Hedef noktalar / Target points
        """
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        Verilen noktaları perspektif dönüşümü uygular. /
        Applies perspective transform to given points.

        Args:
            points (np.ndarray): Dönüştürülecek noktalar / Points to transform

        Returns:
            np.ndarray: Dönüştürülmüş noktalar / Transformed points
        """
        if len(points) != 0:
            reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
            transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
            return transformed_points.reshape(-1, 2)
        else:
            # Hata durumunda geçici değer döndür / Return temporary value if error
            return np.array([[1, 2], [5, 8]], dtype='float')


if __name__ == "__main__":
    # Video yolu / Video path
    video_path = "C:\\Users\\CYBORG\\Videos\\vehicles4.mp4"

    # Video bilgilerini al / Get video information
    video_info = sv.VideoInfo.from_video_path(video_path=video_path)

    # YOLOv8 modelini yükle / Load YOLOv8 model
    model = YOLO("yolov8n.pt")

    # Nesne takibi / Object tracking
    byte_track = sv.ByteTrack(frame_rate=video_info.fps)

    # Görselleştirme için parametreler / Visualization parameters
    thickness = 3
    text_scale = 1

    # Annotator'lar / Annotators
    bounding_box_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        text_scale=text_scale,
        text_thickness=thickness,
        text_position=sv.Position.BOTTOM_CENTER, color_lookup=sv.ColorLookup.TRACK)

    trace_annotator = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=video_info.fps * 2,
        position=sv.Position.BOTTOM_CENTER, color_lookup=sv.ColorLookup.TRACK)

    # Video karelerini oku / Get video frames
    frame_generator = sv.get_video_frames_generator(source_path=video_path)

    # Belirlenen bölge / Defined polygon zone
    polygon_zone = sv.PolygonZone(polygon=SOURCE)

    # Perspektif dönüşüm nesnesi / Perspective transformer
    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

    # Araç koordinatlarını depola / Store vehicle coordinates
    coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))

    # Ana döngü / Main loop
    for frame in frame_generator:
        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = detections[polygon_zone.trigger(detections)]
        detections = byte_track.update_with_detections(detections=detections)

        # Araç merkez noktaları / Vehicle center points
        points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        points = view_transformer.transform_points(points=points).astype(int)

        # Y koordinatlarını kaydet / Store Y-coordinates
        for tracker_id, [_, y] in zip(detections.tracker_id, points):
            coordinates[tracker_id].append(y)

        # Araç hız hesaplama / Calculate vehicle speed
        labels = []
        for tracker_id in detections.tracker_id:
            if len(coordinates[tracker_id]) < video_info.fps / 2:
                labels.append(f"#{tracker_id}")
            else:
                coordinate_start = coordinates[tracker_id][-1]
                coordinate_end = coordinates[tracker_id][0]
                distance = abs(coordinate_start - coordinate_end)
                time = len(coordinates[tracker_id]) / video_info.fps
                speed = distance / time * 3.6
                labels.append(f"#{tracker_id} {int(speed)} km/h")

        # Görsel çıktılar / Visual outputs
        annotated_frame = frame.copy()
        annotated_frame = sv.draw_polygon(annotated_frame, polygon=SOURCE, color=sv.Color.RED)
        annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = bounding_box_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

        # Çerçeveyi göster / Show frame
        cv2.imshow("frame", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()

# yolo-detection-to-gps-coordinates
A drone-based computer vision framework that projects YOLO detections from angled images onto GPS coordinates, enhancing situational awareness by leveraging camera intrinsics and extrinsics extracted from EXIF metadata.
# Drone Vision for Georeferenced Object Detection

## Drone Vision for Georeferenced Object Detection

### 1. Motivation & Problem Context
Modern disaster response and geospatial applications require rapid, reliable identification of features on the ground — whether collapsed buildings after earthquakes, flooded zones after storms, or damaged infrastructure during conflict. Drones are uniquely suited for this task because they can capture oblique (angled) imagery at scale.  

However, raw object detections (bounding boxes) from neural networks like **YOLO** only exist in **pixel space**. To be useful for responders, these detections must be translated into **real-world georeferenced coordinates (latitude/longitude)**.  

This toolkit bridges the gap by integrating:
- **Computer vision (YOLO)** for real-time object detection.  
- **Camera geometry** from intrinsics (sensor size, focal length, pixel pitch) and extrinsics (gimbal tilt, yaw, altitude).  
- **Photogrammetry principles** to project detections from the image plane onto the Earth’s surface.  
- **EXIF/XMP metadata** embedded in drone imagery, which provides camera and GPS information.  

---

### 2. Pipeline Overview

1. **Metadata Extraction (EXIF/XMP)**  
   - Extracts focal length, pixel pitch, sensor size, GPS, altitude, gimbal pitch, and yaw.  
   - Falls back to known specs if missing.  

2. **Field of View (FOV) Calculation**  
   - Computes Horizontal, Vertical, and Diagonal FOV.  
   - Defines angular extent of the camera’s view.  

3. **YOLO Detection**  
   - Custom-trained YOLO detects objects of interest (e.g., damaged buildings).  
   - Bounding boxes and confidences extracted.  

4. **Image → Angle Mapping**  
   - Each detection center `(u, v)` → angular offsets `(θx, θy)` relative to the optical axis:  
     ```
     θx = arctan((u - cx) / fx)
     θy = arctan((v - cy) / fy)
     ```

5. **Projection onto Ground Plane**  
   - Combines tilt + altitude to intersect detections with ground plane:  
     ```
     r = H / tan(δ)
     ```
   - Produces forward and lateral offsets from nadir.  

6. **Georeferencing**  
   - Offsets rotated into Earth’s **ENU frame** using yaw.  
   - Converted into GPS shifts (lat/lon).  

7. **Export & Visualization**  
   - **CSV** → tabular output of detections with GPS.  
   - **KML** → visualization in Google Earth/Maps.  
   - Overlayed images show detections, rulers, and angular annotations.  

---

### 3. The Science Behind It

- **Computer Vision (YOLO):** CNNs transform pixels into semantic detections.  
- **Camera Geometry:** Pinhole camera model with intrinsics/extrinsics.  
- **Projection Trigonometry:** Pixels → angles → ground intersections.  
- **Geodesy & GIS:** ENU offsets transformed into geographic coordinates.  
- **Remote Sensing:** EXIF altitude and SRTM ground models provide terrain-aware correction.  

---

### 4. Example Workflow (Conceptual)

- DJI drone captures an oblique disaster image.  
- **EXIF:** FocalLength = 10.26 mm, PixelPitch = 2.41 µm, GimbalPitch = -48.8°, Yaw = 135°, GPS = (lat, lon).  
- **YOLO:** Detects collapsed building at `(u=540, v=320)`.  
- **Intrinsics:** Converts pixel → angles `(θx, θy)`.  
- **Projection:** Angles + altitude → 85 m forward, 12 m right.  
- **Georeferencing:** Rotated via yaw → GPS coordinates.  
- **Outputs:** CSV + KML ready for Google Earth visualization.  

---

## :rocket: News
- [2025-09-01] Initial release of oblique vision georeferencing toolkit.  



import ultralytics
# check ultralytics version if needed
import cvzone
from ultralytics import YOLO
import cv2
import math
from typing import Dict, Tuple

# -------------------- helpers --------------------

def extract_fov_info(image_path: str, pixel_pitch_um: float = 2.41, fallback_focal_mm: float = 10.26) -> Dict:
    """
    Step-1: return HFOV/VFOV + image size using EXIF focal length or fallback.
    """
    from PIL import Image, ExifTags

    # image + size
    img = Image.open(image_path)
    width_px, height_px = img.size

    # focal length from EXIF (if found)
    exif = img._getexif()
    focal_length_mm = None
    if exif:
        for tag_id, value in exif.items():
            tag = ExifTags.TAGS.get(tag_id, tag_id)
            if tag == "FocalLength":
                focal_length_mm = (value[0] / value[1]) if isinstance(value, tuple) and len(value) == 2 else float(value)
                break
    if not focal_length_mm:
        focal_length_mm = fallback_focal_mm

    # sensor size from pixel pitch
    pixel_pitch_mm = pixel_pitch_um / 1000.0
    sensor_w_mm = width_px * pixel_pitch_mm
    sensor_h_mm = height_px * pixel_pitch_mm
    sensor_d_mm = math.hypot(sensor_w_mm, sensor_h_mm)

    # FOV
    hfov_deg = math.degrees(2 * math.atan(sensor_w_mm / (2 * focal_length_mm)))
    vfov_deg = math.degrees(2 * math.atan(sensor_h_mm / (2 * focal_length_mm)))
    dfov_deg = math.degrees(2 * math.atan(sensor_d_mm / (2 * focal_length_mm)))

    return dict(
        width_px=width_px, height_px=height_px,
        hfov_deg=hfov_deg, vfov_deg=vfov_deg, dfov_deg=dfov_deg,
        focal_length_mm=focal_length_mm, pixel_pitch_um=pixel_pitch_um
    )

def intrinsics_for_size(W:int, H:int, *, hfov_deg:float, vfov_deg:float,
                        cx:float=None, cy:float=None) -> Dict[str, float]:
    """
    Get fx, fy (px) from FOV + size; cx, cy default to image center.
    """
    fx_px = (W/2) / math.tan(math.radians(hfov_deg/2))
    fy_px = (H/2) / math.tan(math.radians(vfov_deg/2))
    if cx is None: cx = W/2
    if cy is None: cy = H/2
    return dict(fx=fx_px, fy=fy_px, cx=cx, cy=cy)

def pixel_to_angles(u:int, v:int, fx:float, fy:float, cx:float, cy:float) -> Dict[str, float]:
    """
    Get θx, θy, off-axis angle, and image-plane bearing for pixel (u,v).
    """
    dx = (u - cx) / fx
    dy = (v - cy) / fy
    theta_x = math.degrees(math.atan(dx))
    theta_y = math.degrees(math.atan(dy))
    theta_offaxis = math.degrees(math.atan(math.hypot(dx, dy)))
    bearing_img = math.degrees(math.atan2(dy, dx))
    return dict(theta_x=theta_x, theta_y=theta_y,
                theta_offaxis=theta_offaxis, bearing_in_image_deg=bearing_img)

# -------------------- main flow --------------------

# original + resized image
orig_path = 'yolo-detection-to-gps-coordinates/test images/Blantyre2.JPG'
frame = cv2.imread(orig_path)
imgR = cv2.resize(frame, (1080, 720))
cv2.waitKey(3000)

# step 1: FOV from original image
fov_info = extract_fov_info(orig_path, pixel_pitch_um=2.41, fallback_focal_mm=10.26)
# intrinsics for resized image
H_resized, W_resized = imgR.shape[:2]
intr = intrinsics_for_size(W_resized, H_resized,
                           hfov_deg=fov_info["hfov_deg"],
                           vfov_deg=fov_info["vfov_deg"])
fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]

# load YOLO model
model = YOLO(r'C:\Users\LENOVO\Documents\Computer Vision\situational awareness - YOLO\Yolo Best Model\best.pt')

# run YOLO
results = model.predict(imgR, save=True, conf=0.6)

# loop detections
for r in results:
    img = r.orig_img.copy()
    boxes = r.boxes
    centers = []

    # draw centers + labels
    for i, box in enumerate(boxes, start=1):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx_box, cy_box = (x1 + x2) // 2, (y1 + y2) // 2
        centers.append((cx_box, cy_box))

        cv2.circle(img, (cx_box, cy_box), 4, (0, 0, 255), -1)
        cv2.putText(img,  f"({cx_box},{cy_box})", (cx_box -45, cy_box + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (9, 10, 11), 2)

    # draw bboxes + labels
    for box in boxes:
        x, y, w, h = box.xywh[0]
        x1 = int(x - w/2)
        y1 = int(y - h/2)
        bbox = (x1, y1, int(w), int(h))
        cvzone.cornerRect(img, bbox, l=3, rt=1, colorC=(255, 0, 0), colorR=(0, 255, 0))

        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = f"{r.names[cls]} {conf:.2f}"
        cvzone.putTextRect(img, label, (x1, y1 - 10), scale=1.5, thickness=2,
                           colorR=(255, 0, 0), offset=2, border=1)

    # image center cross
    h, w = img.shape[:2]
    cx_img, cy_img = w // 2, h // 2
    cv2.line(img, (cx_img - 10, cy_img), (cx_img + 10, cy_img), (255, 0, 255), 2)
    cv2.line(img, (cx_img, cy_img - 10), (cx_img, cy_img + 10), (255, 0, 255), 2)

    # step 2: compute angles for each center
    angle_rows = []
    for (ux, vy) in centers:
        a = pixel_to_angles(ux, vy, fx=fx, fy=fy, cx=cx, cy=cy)
        angle_rows.append({"u": ux, "v": vy, **a})

        lbl = f"theta_x: {a['theta_x']:.1f} deg, theta_y: {a['theta_y']:.1f} deg"
        cvzone.putTextRect(img, lbl, (int(ux) - 60, int(vy) - 10),
                           scale=0.7, thickness=1,
                           colorT=(0, 0, 0), colorR=(255, 255, 255),
                           offset=3, border=1)

    # lines from center to detections
    for (ux, vy) in centers:
        cv2.line(img, (cx_img, cy_img), (ux, vy), (255, 225, 5), 2)

    # console summary
    print("\nDetections (angles relative to optical axis):")
    for row in angle_rows:
        print(f"pix=({row['u']},{row['v']}) "
              f"θx={row['theta_x']:.3f}°, θy={row['theta_y']:.3f}°, "
              f"θ_off={row['theta_offaxis']:.3f}°, φ_img={row['bearing_in_image_deg']:.1f}°")

    # save + show
    cv2.imwrite("Output_yolo/B13_angles.jpg", img)
    cv2.imshow("Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

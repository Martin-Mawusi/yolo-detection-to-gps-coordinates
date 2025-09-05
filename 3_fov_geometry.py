# 2_fov_extractor.py
from PIL import Image, ExifTags
import math
import re
import xml.etree.ElementTree as ET

__all__ = ["extract_fov_info"]

def extract_fov_info(
    image_path: str,
    pixel_pitch_um: float = 2.41,          # sensor pixel size (µm)
    fallback_focal_mm: float = 10.26,      # use if EXIF FocalLength is missing
    principal_point: tuple | None = None,  # set (cx, cy) in px if you have calibration
    print_assignments: bool = True,
) -> dict:
    """
    Extract camera geometry + tilt info we’ll reuse later.

    Returns a dict with:
      - image_path, width_px, height_px
      - focal_length_mm
      - pixel_pitch_um/mm, sensor_width_mm/height_mm/diagonal_mm
      - fx_px, fy_px, cx_px, cy_px
      - hfov_deg, vfov_deg, dfov_deg
      - gimbal_pitch_deg (DJI, 0=horizon, -90=nadir if present)
      - tilt_down_deg (positive down from horizon)
      - top_edge_depression_deg, bottom_edge_depression_deg  (from horizon, +down)
      - horizon_crosses_image (bool)
      - hfov_from_fx_deg, vfov_from_fy_deg (sanity checks)
    """

    # ---------- helpers: read XMP + parse DJI gimbal pitch ----------
    def _extract_xmp_packet(jpeg_path: str):
        with open(jpeg_path, "rb") as f:
            data = f.read()
        start = data.find(b"<?xpacket begin=")
        if start == -1:
            start = data.find(b"<x:xmpmeta")
            if start == -1:
                return None
        end = data.find(b"<?xpacket end=", start)
        if end == -1:
            close = data.find(b"</x:xmpmeta>", start)
            if close == -1:
                return None
            end = close + len(b"</x:xmpmeta>")
            return data[start:end].decode("utf-8", errors="ignore")
        end = data.find(b">", end)
        if end == -1:
            return None
        end += 1
        return data[start:end].decode("utf-8", errors="ignore")

    def _parse_gimbal_pitch_from_xmp(xmp_xml: str):
        if not xmp_xml:
            return None
        try:
            root = ET.fromstring(xmp_xml)
            for elem in root.iter():
                # look for attribute value
                for k, v in elem.attrib.items():
                    if "GimbalPitchDegree" in k:
                        return float(v)
                # or element text
                if "GimbalPitchDegree" in elem.tag and elem.text:
                    return float(elem.text.strip())
            # regex fallbacks
            m = re.search(r"GimbalPitchDegree\s*=\s*\"?(-?\d+(?:\.\d+)?)", xmp_xml)
            if m:
                return float(m.group(1))
            m2 = re.search(r"GimbalPitchDegree[^<]*>(-?\d+(?:\.\d+)?)<", xmp_xml)
            if m2:
                return float(m2.group(1))
        except Exception:
            pass
        return None

    # ---------- open image + read EXIF focal length ----------
    img = Image.open(image_path)
    width_px, height_px = img.size

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

    # ---------- sensor geometry from pixel pitch ----------
    pixel_pitch_mm = pixel_pitch_um / 1000.0
    sensor_w_mm = width_px * pixel_pitch_mm
    sensor_h_mm = height_px * pixel_pitch_mm
    sensor_d_mm = math.hypot(sensor_w_mm, sensor_h_mm)

    # ---------- FOVs (from mm) ----------
    hfov_deg = math.degrees(2 * math.atan(sensor_w_mm / (2 * focal_length_mm)))
    vfov_deg = math.degrees(2 * math.atan(sensor_h_mm / (2 * focal_length_mm)))
    dfov_deg = math.degrees(2 * math.atan(sensor_d_mm / (2 * focal_length_mm)))

    # ---------- intrinsics in pixels ----------
    fx_px = focal_length_mm / pixel_pitch_mm
    fy_px = focal_length_mm / pixel_pitch_mm
    # principal point
    if principal_point is None:
        cx_px, cy_px = (width_px / 2.0, height_px / 2.0)
    else:
        cx_px, cy_px = principal_point

    # quick FOV sanity check from fx/fy
    hfov_from_fx_deg = math.degrees(2 * math.atan((width_px / 2.0) / fx_px))
    vfov_from_fy_deg = math.degrees(2 * math.atan((height_px / 2.0) / fy_px))

    # ---------- gimbal pitch + tilt conventions ----------
    # DJI: 0° = level, -90° = nadir; convert to +down tilt
    xmp_xml = _extract_xmp_packet(image_path)
    gimbal_pitch_deg = _parse_gimbal_pitch_from_xmp(xmp_xml)  # may be None

    tilt_down_deg = None
    top_edge_depression_deg = None
    bottom_edge_depression_deg = None
    horizon_crosses_image = None

    if gimbal_pitch_deg is not None:
        tilt_down_deg = -gimbal_pitch_deg  # +down from horizon
        # VFOV bounds as depression angles (+down)
        top_edge_depression_deg = tilt_down_deg - vfov_deg / 2.0
        bottom_edge_depression_deg = tilt_down_deg + vfov_deg / 2.0
        horizon_crosses_image = (top_edge_depression_deg < 0.0 < bottom_edge_depression_deg)

    results = {
        "image_path": image_path,
        "width_px": width_px,
        "height_px": height_px,
        "focal_length_mm": focal_length_mm,
        "pixel_pitch_um": pixel_pitch_um,
        "pixel_pitch_mm": pixel_pitch_mm,
        "sensor_width_mm": sensor_w_mm,
        "sensor_height_mm": sensor_h_mm,
        "sensor_diagonal_mm": sensor_d_mm,
        "fx_px": fx_px,
        "fy_px": fy_px,
        "cx_px": cx_px,
        "cy_px": cy_px,
        "hfov_deg": hfov_deg,
        "vfov_deg": vfov_deg,
        "dfov_deg": dfov_deg,
        "hfov_from_fx_deg": hfov_from_fx_deg,
        "vfov_from_fy_deg": vfov_from_fy_deg,
        "gimbal_pitch_deg": gimbal_pitch_deg,         # DJI (0 level, -90 nadir)
        "tilt_down_deg": tilt_down_deg,               # +down from horizon
        "top_edge_depression_deg": top_edge_depression_deg,      # +down
        "bottom_edge_depression_deg": bottom_edge_depression_deg,# +down
        "horizon_crosses_image": horizon_crosses_image,
    }

    if print_assignments:
        print("\n=== Variables (copy/paste ready) ===")
        for k, v in results.items():
            if isinstance(v, float):
                print(f"{k} = {v:.6f}")
            else:
                print(f"{k} = {repr(v)}")

    return results

if __name__ == "__main__":
    # quick test when you hit Run/Play in your IDE
    img_path = r"test images/Blantyre1.JPG"
    extract_fov_info(img_path, print_assignments=True)

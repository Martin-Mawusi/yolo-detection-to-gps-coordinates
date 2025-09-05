# 2_fov_extractor.py
from PIL import Image, ExifTags
import math
import re
import xml.etree.ElementTree as ET

__all__ = ["extract_fov_info"]

def extract_fov_info(
    image_path: str,
    pixel_pitch_um: float = 2.41,
    fallback_focal_mm: float = 10.26,
    print_assignments: bool = True,
) -> dict:
    """
    Extract camera geometry & orientation info from an image.

    Returns a dict with:
      - image_path, width_px, height_px
      - focal_length_mm
      - pixel_pitch_um, sensor_width_mm, sensor_height_mm, sensor_diagonal_mm
      - hfov_deg, vfov_deg, dfov_deg
      - gimbal_pitch_deg
      - upper_fov_angle_deg, lower_fov_angle_deg  (pitch ± VFOV/2 from horizon)

    If print_assignments=True, also prints copy/paste-ready Python assignments.
    """

    # Helper: Read the embedded XMP metadata packet from the image
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

    # Helper: Parse the gimbal pitch angle value from the XMP metadata
    def _parse_gimbal_pitch_from_xmp(xmp_xml: str):
        if not xmp_xml:
            return None
        try:
            root = ET.fromstring(xmp_xml)
            for elem in root.iter():
                # Look for attributes that contain the gimbal pitch angle
                for k, v in elem.attrib.items():
                    if "GimbalPitchDegree" in k:
                        return float(v)
                # Look for gimbal pitch stored as text inside an element
                if "GimbalPitchDegree" in elem.tag and elem.text:
                    return float(elem.text.strip())
            # Fall back to regex if XML parsing misses it
            m = re.search(r"GimbalPitchDegree\s*=\s*\"?(-?\d+(?:\.\d+)?)", xmp_xml)
            if m:
                return float(m.group(1))
            m2 = re.search(r"GimbalPitchDegree[^<]*>(-?\d+(?:\.\d+)?)<", xmp_xml)
            if m2:
                return float(m2.group(1))
        except Exception:
            pass
        return None

    # Open the image and read its width, height, and EXIF focal length
    img = Image.open(image_path)
    width_px, height_px = img.size

    exif = img._getexif()
    focal_length = None
    if exif:
        for tag_id, value in exif.items():
            tag = ExifTags.TAGS.get(tag_id, tag_id)
            if tag == "FocalLength":
                # Convert rational or direct values into float focal length
                if isinstance(value, tuple) and len(value) == 2:
                    focal_length = value[0] / value[1]
                else:
                    focal_length = float(value)
                break
    if not focal_length:
        # Use fallback focal length if EXIF data is missing
        focal_length = fallback_focal_mm

    # Calculate sensor physical dimensions using pixel pitch
    pixel_pitch_mm = pixel_pitch_um / 1000.0
    sensor_w_mm = width_px * pixel_pitch_mm
    sensor_h_mm = height_px * pixel_pitch_mm
    sensor_d_mm = math.hypot(sensor_w_mm, sensor_h_mm)

    # Compute horizontal, vertical, and diagonal field of view in degrees
    hfov = math.degrees(2 * math.atan(sensor_w_mm / (2 * focal_length)))
    vfov = math.degrees(2 * math.atan(sensor_h_mm / (2 * focal_length)))
    dfov = math.degrees(2 * math.atan(sensor_d_mm / (2 * focal_length)))

    # Extract the gimbal pitch angle from DJI’s XMP metadata if available
    xmp_xml = _extract_xmp_packet(image_path)
    gimbal_pitch = _parse_gimbal_pitch_from_xmp(xmp_xml)

    # Calculate the upper and lower field of view angles relative to the horizon
    upper_angle = lower_angle = None
    if gimbal_pitch is not None:
        upper_angle = gimbal_pitch + (vfov / 2)
        lower_angle = gimbal_pitch - (vfov / 2)

    results = {
        "image_path": image_path,
        "width_px": width_px,
        "height_px": height_px,
        "focal_length_mm": focal_length,
        "pixel_pitch_um": pixel_pitch_um,
        "sensor_width_mm": sensor_w_mm,
        "sensor_height_mm": sensor_h_mm,
        "sensor_diagonal_mm": sensor_d_mm,
        "hfov_deg": hfov,
        "vfov_deg": vfov,
        "dfov_deg": dfov,
        "gimbal_pitch_deg": gimbal_pitch,
        "upper_fov_angle_deg": upper_angle,
        "lower_fov_angle_deg": lower_angle,
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
    # Run a quick test when executing this file directly in your IDE
    img_path = r"test images/Blantyre1.JPG"
    extract_fov_info(img_path, print_assignments=True)

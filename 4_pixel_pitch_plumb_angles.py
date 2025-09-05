from PIL import Image, ExifTags
import math
import re
import xml.etree.ElementTree as ET

def extract_fov_info(
    image_path: str,
    pixel_pitch_um: float = 2.41,
    fallback_focal_mm: float = 10.26
) -> dict:
    """
    Extracts camera geometry info from an image file.
    Returns a dictionary of variables and also prints them as Python assignments.
    """

    # --- grab XMP block from image ---
    def _extract_xmp_packet(jpeg_path: str) -> str | None:
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

    # --- pull gimbal pitch angle if present ---
    def _parse_gimbal_pitch_from_xmp(xmp_xml: str | None) -> float | None:
        if not xmp_xml:
            return None
        try:
            root = ET.fromstring(xmp_xml)
            for elem in root.iter():
                for k, v in elem.attrib.items():
                    if "GimbalPitchDegree" in k:
                        return float(v)
                if "GimbalPitchDegree" in elem.tag and elem.text:
                    return float(elem.text.strip())
            m = re.search(r"GimbalPitchDegree\s*=\s*\"?(-?\d+(?:\.\d+)?)", xmp_xml)
            if m:
                return float(m.group(1))
            m2 = re.search(r"GimbalPitchDegree[^<]*>(-?\d+(?:\.\d+)?)<", xmp_xml)
            if m2:
                return float(m2.group(1))
        except Exception:
            pass
        return None

    # --- open image + EXIF focal length ---
    img = Image.open(image_path)
    width_px, height_px = img.size

    exif = img._getexif()
    focal_length = None
    if exif:
        for tag_id, value in exif.items():
            tag = ExifTags.TAGS.get(tag_id, tag_id)
            if tag == "FocalLength":
                if isinstance(value, tuple) and len(value) == 2:
                    focal_length = value[0] / value[1]
                else:
                    focal_length = float(value)
                break

    if not focal_length:
        focal_length = fallback_focal_mm

    # --- sensor size from pixel pitch ---
    pixel_pitch_mm = pixel_pitch_um / 1000.0
    sensor_w_mm = width_px * pixel_pitch_mm
    sensor_h_mm = height_px * pixel_pitch_mm
    sensor_d_mm = math.hypot(sensor_w_mm, sensor_h_mm)

    # --- field of view ---
    hfov = math.degrees(2 * math.atan(sensor_w_mm / (2 * focal_length)))
    vfov = math.degrees(2 * math.atan(sensor_h_mm / (2 * focal_length)))
    dfov = math.degrees(2 * math.atan(sensor_d_mm / (2 * focal_length)))

    # --- gimbal tilt ---
    xmp_xml = _extract_xmp_packet(image_path)
    gimbal_pitch = _parse_gimbal_pitch_from_xmp(xmp_xml)

    # --- top/bottom FOV edges relative to horizon ---
    upper_angle = None
    lower_angle = None
    if gimbal_pitch is not None:
        upper_angle = gimbal_pitch + (vfov / 2)
        lower_angle = gimbal_pitch - (vfov / 2)

    # --- collect results ---
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

    # --- print in assignment style ---
    print("\n=== Variables (copy/paste ready) ===")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"{k} = {v:.6f}")
        else:
            print(f"{k} = {repr(v)}")

    return results


# quick run
if __name__ == "__main__":
    img_path = r"test images/Blantyre1.JPG"
    results = extract_fov_info(img_path)

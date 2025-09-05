# oblique_projector.py
import math
import re
import cv2
import cvzone
import piexif
from PIL import Image, ExifTags
from lxml import etree
from ultralytics import YOLO
import srtm

# ---------------------------- CONFIG ----------------------------
ORIG_IMG_PATH = 'yolo-detection-to-gps-coordinates/test images/Blantyre2.JPG'  # image path
RESIZE_W, RESIZE_H = 1080, 720                # resize dims
MODEL_PATH = 'Yolo Best Model/best.pt'  # model path

PIXEL_PITCH_UM = 2.41        # sensor pixel size (µm)
FALLBACK_FOCAL_MM = 10.26    # fallback focal (mm)
# ----------------------------------------------------------------

# ======================== STEP 1: FOV, TILT, YAW ========================

def _extract_xmp_blob(image_path: str) -> str | None:
    with open(image_path, 'rb') as f:
        data = f.read()
    m = re.search(br'<x:xmpmeta[^>]*>.*?</x:xmpmeta>', data, flags=re.DOTALL)
    if not m:
        return None
    return m.group(0).decode(errors='ignore')

def _parse_float_from_xmp_text(xmp_text: str, keys: list[str]) -> float | None:
    # try attributes, elements, then regex
    try:
        root = etree.fromstring(xmp_text.encode('utf-8', errors='ignore'))
        for elem in root.iter():
            for k, v in elem.attrib.items():
                if any(key in k for key in keys):
                    try:
                        return float(v)
                    except:
                        pass
            if elem.text and any(key in elem.tag for key in keys):
                try:
                    return float(elem.text.strip())
                except:
                    pass
    except Exception:
        pass
    pat = r'(' + '|'.join(map(re.escape, keys)) + r')\s*[:=]?\s*"?(-?\d+(?:\.\d+)?)'
    m = re.search(pat, xmp_text, flags=re.IGNORECASE)
    if m:
        try:
            return float(m.group(2))
        except:
            return None
    return None

def extract_fov_tilt_yaw(image_path: str,
                         pixel_pitch_um: float = PIXEL_PITCH_UM,
                         fallback_focal_mm: float = FALLBACK_FOCAL_MM):
    # image size
    img_pil = Image.open(image_path)
    W0, H0 = img_pil.size

    # focal length (EXIF) or fallback
    exif = img_pil._getexif()
    focal_mm = None
    if exif:
        for tag_id, value in exif.items():
            tag = ExifTags.TAGS.get(tag_id, tag_id)
            if tag == "FocalLength":
                focal_mm = (value[0]/value[1]) if isinstance(value, tuple) and len(value) == 2 else float(value)
                break
    if not focal_mm:
        focal_mm = fallback_focal_mm

    # sensor mm from pixel pitch
    px_mm = pixel_pitch_um / 1000.0
    sensor_w_mm = W0 * px_mm
    sensor_h_mm = H0 * px_mm
    sensor_d_mm = math.hypot(sensor_w_mm, sensor_h_mm)

    # FOVs
    hfov_deg = math.degrees(2 * math.atan(sensor_w_mm / (2 * focal_mm)))
    vfov_deg = math.degrees(2 * math.atan(sensor_h_mm / (2 * focal_mm)))
    dfov_deg = math.degrees(2 * math.atan(sensor_d_mm / (2 * focal_mm)))

    # XMP: pitch + yaw if present
    xmp_text = _extract_xmp_blob(image_path)
    gimbal_pitch_deg = None    # DJI: 0=horizon, -90=nadir
    yaw_deg = None             # DJI: FlightYawDegree (0=N, +CW)
    if xmp_text:
        gimbal_pitch_deg = _parse_float_from_xmp_text(xmp_text, ["GimbalPitchDegree"])
        yaw_deg = _parse_float_from_xmp_text(xmp_text, ["FlightYawDegree", "Yaw"])

    # our tilt: +down from horizon
    tilt_down_deg = -gimbal_pitch_deg if (gimbal_pitch_deg is not None) else None

    return {
        "W0": W0, "H0": H0,
        "focal_mm": focal_mm,
        "pixel_pitch_um": pixel_pitch_um,
        "hfov_deg": hfov_deg, "vfov_deg": vfov_deg, "dfov_deg": dfov_deg,
        "gimbal_pitch_deg": gimbal_pitch_deg,
        "tilt_down_deg": tilt_down_deg,
        "yaw_deg": yaw_deg
    }

def intrinsics_for_size(W:int, H:int, hfov_deg:float, vfov_deg:float, cx:float=None, cy:float=None):
    fx = (W/2) / math.tan(math.radians(hfov_deg/2))
    fy = (H/2) / math.tan(math.radians(vfov_deg/2))
    if cx is None: cx = W/2
    if cy is None: cy = H/2
    return fx, fy, cx, cy

def pixel_to_angles(u:int, v:int, fx:float, fy:float, cx:float, cy:float):
    dx = (u - cx) / fx
    dy = (v - cy) / fy
    theta_x = math.degrees(math.atan(dx))          # left/right
    theta_y = math.degrees(math.atan(dy))          # up/down
    theta_off = math.degrees(math.atan(math.hypot(dx, dy)))
    bearing_img = math.degrees(math.atan2(dy, dx))
    return theta_x, theta_y, theta_off, bearing_img

# ======================== STEP 2: AGL + DRONE GPS ========================

def _rational_to_float(val):
    try:
        if hasattr(val, 'numerator') and hasattr(val, 'denominator'):
            return float(val.numerator) / float(val.denominator or 1.0)
        if isinstance(val, tuple) and len(val) == 2:
            num, den = val
            return float(num) / float(den or 1.0)
        return float(val)
    except Exception:
        return None

def _dms_to_decimal(dms, ref):
    if dms is None:
        return None
    try:
        d, m, s = dms
        d = _rational_to_float(d)
        m = _rational_to_float(m)
        s = _rational_to_float(s)
        dec = d + (m/60.0) + (s/3600.0)
        if ref in ['S', 'W']:
            dec = -dec
        return dec
    except Exception:
        try:
            dec = float(dms)
            if ref in ['S', 'W']:
                dec = -dec
            return dec
        except Exception:
            return None

def _extract_exif_gps(image_path):
    exif_dict = piexif.load(image_path)
    gps_ifd = exif_dict.get("GPS", {}) or {}
    lat = lon = None
    if piexif.GPSIFD.GPSLatitude in gps_ifd and piexif.GPSIFD.GPSLatitudeRef in gps_ifd:
        lat = _dms_to_decimal(
            gps_ifd[piexif.GPSIFD.GPSLatitude],
            gps_ifd[piexif.GPSIFD.GPSLatitudeRef].decode(errors='ignore') if isinstance(gps_ifd[piexif.GPSIFD.GPSLatitudeRef], bytes) else gps_ifd[piexif.GPSIFD.GPSLatitudeRef]
        )
    if piexif.GPSIFD.GPSLongitude in gps_ifd and piexif.GPSIFD.GPSLongitudeRef in gps_ifd:
        lon = _dms_to_decimal(
            gps_ifd[piexif.GPSIFD.GPSLongitude],
            gps_ifd[piexif.GPSIFD.GPSLongitudeRef].decode(errors='ignore') if isinstance(gps_ifd[piexif.GPSIFD.GPSLongitudeRef], bytes) else gps_ifd[piexif.GPSIFD.GPSLongitudeRef]
        )

    alt_m_msl = None
    alt_ref = gps_ifd.get(piexif.GPSIFD.GPSAltitudeRef)
    if piexif.GPSIFD.GPSAltitude in gps_ifd:
        alt_m = _rational_to_float(gps_ifd[piexif.GPSIFD.GPSAltitude])
        if alt_m is not None:
            if alt_ref == 1:
                alt_m = -alt_m
            alt_m_msl = alt_m

    return {'lat': lat, 'lon': lon, 'alt_m_msl': alt_m_msl}

def _ground_elevation_msl(lat, lon):
    try:
        data = srtm.get_data()
        h = data.get_elevation(lat, lon)
        return float(h) if h is not None else None
    except Exception:
        return None

def compute_agl_and_drone_gps(image_path: str, verbose=True):
    # GPS from EXIF
    gps = _extract_exif_gps(image_path)
    lat = gps.get('lat')
    lon = gps.get('lon')
    alt_msl = gps.get('alt_m_msl')

    # try DJI XMP RelativeAltitude first
    xmp = _extract_xmp_blob(image_path)
    rel_alt = None
    if xmp:
        m = re.search(r'RelativeAltitude\s*[:=]\s*([-+]?\d+(?:\.\d+)?)', xmp)
        if m:
            rel_alt = float(m.group(1))
        else:
            try:
                root = etree.fromstring(xmp.encode('utf-8', errors='ignore'))
                for elem in root.iter():
                    for k, v in elem.attrib.items():
                        if 'RelativeAltitude' in k:
                            try:
                                rel_alt = float(v)
                                break
                            except:
                                pass
            except Exception:
                pass

    if rel_alt is not None:
        if verbose:
            print(f"[AGL] DJI XMP RelativeAltitude: {rel_alt:.2f} m")
        return rel_alt, lat, lon

    # else MSL - ground
    if (lat is None) or (lon is None) or (alt_msl is None):
        raise ValueError("Missing lat/lon or EXIF GPSAltitude (MSL); cannot compute AGL.")
    ground_msl = _ground_elevation_msl(lat, lon)
    if ground_msl is None:
        raise ValueError("SRTM ground elevation unavailable; cannot compute AGL.")
    agl = alt_msl - ground_msl
    if verbose:
        print(f"[AGL] EXIF altitude (MSL) - ground (MSL) = {alt_msl:.2f} - {ground_msl:.2f} = {agl:.2f} m")
    return agl, lat, lon

# ============== STEP 3: pixel -> ground (forward/right) -> GPS ==============

def forward_right_offsets(u:int, v:int, W:int, H:int,
                          hfov_deg:float, vfov_deg:float,
                          tilt_down_deg:float, AGL_m:float,
                          fx:float=None, fy:float=None,
                          cx:float=None, cy:float=None):
    if fx is None or fy is None or cx is None or cy is None:
        fx, fy, cx, cy = intrinsics_for_size(W, H, hfov_deg, vfov_deg)
    # per-pixel angles (radians)
    theta_h = math.atan((u - cx) / fx)
    theta_v = math.atan((v - cy) / fy)
    t = math.radians(tilt_down_deg)        # +down from horizon
    delta = t + theta_v                    # pixel depression
    if delta <= 0:
        return None  # above horizon
    r  = AGL_m / math.tan(delta)           # range to ground
    r0 = AGL_m / math.tan(t)               # center range
    forward = r - r0                       # along look direction
    right   = r * math.tan(theta_h)        # lateral
    return forward, right

def rotate_forward_right_to_ENU(forward_m:float, right_m:float, yaw_deg:float):
    # DJI yaw: 0=N, 90=E, CW positive
    psi = math.radians(yaw_deg)
    E = forward_m*math.sin(psi) + right_m*math.cos(psi)
    N = forward_m*math.cos(psi) - right_m*math.sin(psi)
    return E, N

def enu_offset_to_gps(lat0:float, lon0:float, dE:float, dN:float):
    deg_per_m_lat = 1.0/111_320.0
    deg_per_m_lon = 1.0/(111_320.0*math.cos(math.radians(lat0)))
    lat = lat0 + dN*deg_per_m_lat
    lon = lon0 + dE*deg_per_m_lon
    return lat, lon

def center_ground_point_from_drone(lat_drone:float, lon_drone:float,
                                   tilt_down_deg:float, yaw_deg:float, AGL_m:float):
    """ground hit for the image center ray"""
    t = math.radians(tilt_down_deg)
    if t <= 0:
        return None  # center at/above horizon
    r0 = AGL_m / math.tan(t)
    dE0, dN0 = rotate_forward_right_to_ENU(

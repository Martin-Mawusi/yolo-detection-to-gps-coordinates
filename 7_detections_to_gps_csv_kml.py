# oblique_projector_with_export.py
import math, re, os, csv
import cv2, cvzone, piexif, srtm
from datetime import datetime, timezone
from PIL import Image, ExifTags
from lxml import etree
from ultralytics import YOLO

# ---------------------------- CONFIG ----------------------------
ORIG_IMG_PATH = 'yolo-detection-to-gps-coordinates/test images/Blantyre2.JPG'  # image path
RESIZE_W, RESIZE_H = 1080, 720               # resize size
MODEL_PATH = 'Yolo Best Model/best.pt'  # model path

PIXEL_PITCH_UM = 2.41        # sensor pixel size (e.g., 2.41 µm)
FALLBACK_FOCAL_MM = 10.26    # if EXIF FocalLength missing

OUTPUT_DIR = "Output_yolo"  # output folder
os.makedirs(OUTPUT_DIR, exist_ok=True)
CSV_PATH = os.path.join(OUTPUT_DIR, "detections_gps2.csv")  # CSV output
KML_PATH = os.path.join(OUTPUT_DIR, "detections_gps2.kml")  # KML output
# ----------------------------------------------------------------

# ======================== STEP 1: FOV, TILT, YAW ========================

def _extract_xmp_blob(image_path: str) -> str | None:
    # read raw XMP block from file
    with open(image_path, 'rb') as f:
        data = f.read()
    m = re.search(br'<x:xmpmeta[^>]*>.*?</x:xmpmeta>', data, flags=re.DOTALL)
    if not m:
        return None
    return m.group(0).decode(errors='ignore')

def _parse_float_from_xmp_text(xmp_text: str, keys: list[str]) -> float | None:
    # pull numeric value from attributes/elements; fallback to regex
    try:
        root = etree.fromstring(xmp_text.encode('utf-8', errors='ignore'))
        for elem in root.iter():
            for k, v in elem.attrib.items():
                if any(key in k for key in keys):
                    try: return float(v)
                    except: pass
            if elem.text and any(key in elem.tag for key in keys):
                try: return float(elem.text.strip())
                except: pass
    except Exception: pass
    pat = r'(' + '|'.join(map(re.escape, keys)) + r')\s*[:=]?\s*"?(-?\d+(?:\.\d+)?)'
    m = re.search(pat, xmp_text, flags=re.IGNORECASE)
    if m:
        try: return float(m.group(2))
        except: return None
    return None

def extract_fov_tilt_yaw(image_path: str,
                         pixel_pitch_um: float = PIXEL_PITCH_UM,
                         fallback_focal_mm: float = FALLBACK_FOCAL_MM):
    # image size
    img_pil = Image.open(image_path)
    W0, H0 = img_pil.size
    # Focal length
    exif = img_pil._getexif()
    focal_mm = None
    if exif:
        for tag_id, value in exif.items():
            tag = ExifTags.TAGS.get(tag_id, tag_id)
            if tag == "FocalLength":
                focal_mm = (value[0]/value[1]) if isinstance(value, tuple) and len(value)==2 else float(value)
                break
    if not focal_mm: focal_mm = fallback_focal_mm
    # Sensor size (mm) via pixel pitch
    px_mm = pixel_pitch_um/1000.0
    sensor_w_mm = W0*px_mm; sensor_h_mm = H0*px_mm
    sensor_d_mm = math.hypot(sensor_w_mm, sensor_h_mm)
    # FOVs (deg)
    hfov = math.degrees(2*math.atan(sensor_w_mm/(2*focal_mm)))
    vfov = math.degrees(2*math.atan(sensor_h_mm/(2*focal_mm)))
    dfov = math.degrees(2*math.atan(sensor_d_mm/(2*focal_mm)))
    # XMP tilt + yaw
    xmp_text = _extract_xmp_blob(image_path)
    gimbal_pitch_deg = yaw_deg = None
    if xmp_text:
        gimbal_pitch_deg = _parse_float_from_xmp_text(xmp_text, ["GimbalPitchDegree"])
        yaw_deg = _parse_float_from_xmp_text(xmp_text, ["FlightYawDegree","Yaw"])
    # +down tilt convention
    tilt_down_deg = -gimbal_pitch_deg if gimbal_pitch_deg is not None else None
    return {"W0":W0,"H0":H0,"hfov_deg":hfov,"vfov_deg":vfov,"dfov_deg":dfov,
            "focal_mm":focal_mm,"gimbal_pitch_deg":gimbal_pitch_deg,
            "tilt_down_deg":tilt_down_deg,"yaw_deg":yaw_deg}

def intrinsics_for_size(W:int,H:int,hfov_deg:float,vfov_deg:float,cx=None,cy=None):
    # focal lengths (px) from FOV and size
    fx=(W/2)/math.tan(math.radians(hfov_deg/2))
    fy=(H/2)/math.tan(math.radians(vfov_deg/2))
    if cx is None: cx=W/2
    if cy is None: cy=H/2
    return fx,fy,cx,cy

def pixel_to_angles(u,v,fx,fy,cx,cy):
    # angles from pixel to optical axis
    dx=(u-cx)/fx; dy=(v-cy)/fy
    return (math.degrees(math.atan(dx)),
            math.degrees(math.atan(dy)),
            math.degrees(math.atan(math.hypot(dx,dy))),
            math.degrees(math.atan2(dy,dx)))

# ======================== STEP 2: AGL + DRONE GPS ========================

def _rational_to_float(val):
    # safe rational/tuple to float
    try:
        if hasattr(val,'numerator') and hasattr(val,'denominator'):
            return float(val.numerator)/float(val.denominator or 1.0)
        if isinstance(val,tuple) and len(val)==2:
            num,den=val; return float(num)/float(den or 1.0)
        return float(val)
    except Exception: return None

def _dms_to_decimal(dms,ref):
    # DMS → decimal degrees with N/E/S/W sign
    if dms is None: return None
    try:
        d,m,s=dms
        d=_rational_to_float(d); m=_rational_to_float(m); s=_rational_to_float(s)
        dec=d+(m/60.0)+(s/3600.0)
        if ref in ['S','W']: dec=-dec
        return dec
    except Exception:
        try:
            dec=float(dms);
            if ref in ['S','W']: dec=-dec
            return dec
        except: return None

def _extract_exif_gps(image_path):
    # read GPS from EXIF
    exif_dict=piexif.load(image_path)
    gps_ifd=exif_dict.get("GPS",{}) or {}
    lat=lon=None
    if piexif.GPSIFD.GPSLatitude in gps_ifd and piexif.GPSIFD.GPSLatitudeRef in gps_ifd:
        lat=_dms_to_decimal(gps_ifd[piexif.GPSIFD.GPSLatitude],
            gps_ifd[piexif.GPSIFD.GPSLatitudeRef].decode(errors='ignore') if isinstance(gps_ifd[piexif.GPSIFD.GPSLatitudeRef],bytes) else gps_ifd[piexif.GPSIFD.GPSLatitudeRef])
    if piexif.GPSIFD.GPSLongitude in gps_ifd and piexif.GPSIFD.GPSLongitudeRef in gps_ifd:
        lon=_dms_to_decimal(gps_ifd[piexif.GPSIFD.GPSLongitude],
            gps_ifd[piexif.GPSIFD.GPSLongitudeRef].decode(errors='ignore') if isinstance(gps_ifd[piexif.GPSIFD.GPSLongitudeRef],bytes) else gps_ifd[piexif.GPSIFD.GPSLongitudeRef])
    alt_msl=None; alt_ref=gps_ifd.get(piexif.GPSIFD.GPSAltitudeRef)
    if piexif.GPSIFD.GPSAltitude in gps_ifd:
        alt_m=_rational_to_float(gps_ifd[piexif.GPSIFD.GPSAltitude])
        if alt_m is not None:
            if alt_ref==1: alt_m=-alt_m
            alt_msl=alt_m
    return {'lat':lat,'lon':lon,'alt_m_msl':alt_msl}

def _ground_elevation_msl(lat,lon):
    # SRTM ground elevation (MSL)
    try:
        data=srtm.get_data(); h=data.get_elevation(lat,lon)
        return float(h) if h is not None else None
    except: return None

def compute_agl_and_drone_gps(image_path,verbose=True):
    # AGL from DJI XMP if present, else MSL - ground
    gps=_extract_exif_gps(image_path); lat=gps.get('lat'); lon=gps.get('lon'); alt_msl=gps.get('alt_m_msl')
    # Try DJI XMP relative altitude
    xmp=_extract_xmp_blob(image_path); rel_alt=None
    if xmp:
        m=re.search(r'RelativeAltitude\s*[:=]\s*([-+]?\d+(?:\.\d+)?)',xmp)
        if m: rel_alt=float(m.group(1))
    if rel_alt is not None:
        if verbose: print(f"[AGL] DJI RelativeAltitude: {rel_alt:.2f} m")
        return rel_alt,lat,lon
    if (lat is None) or (lon is None) or (alt_msl is None):
        raise ValueError("Missing lat/lon or altitude.")
    ground=_ground_elevation_msl(lat,lon)
    if ground is None: raise ValueError("No SRTM ground elevation.")
    agl=alt_msl-ground
    if verbose: print(f"[AGL] {alt_msl:.2f}-{ground:.2f}={agl:.2f} m")
    return agl,lat,lon

# ============== STEP 3+4: pixel -> offsets -> GPS ========================

def forward_right_offsets(u,v,W,H,hfov,vfov,tilt_down,AGL,fx=None,fy=None,cx=None,cy=None):
    # pixel → forward/right offsets (m) at ground
    if fx is None or fy is None: fx,fy,cx,cy=intrinsics_for_size(W,H,hfov,vfov)
    th=math.atan((u-cx)/fx); tv=math.atan((v-cy)/fy)
    t=math.radians(tilt_down); delta=t+tv
    if delta<=0: return None
    r=AGL/math.tan(delta); r0=AGL/math.tan(t)
    return r-r0, r*math.tan(th)  # forward, right

def rotate_forward_right_to_ENU(fwd,right,yaw_deg):
    # rotate offsets by yaw → E,N
    psi=math.radians(yaw_deg)
    E=fwd*math.sin(psi)+right*math.cos(psi)
    N=fwd*math.cos(psi)-right*math.sin(psi)
    return E,N

def enu_offset_to_gps(lat0,lon0,dE,dN):
    # ENU meters → lat/lon
    dlat=dN/111_320.0; dlon=dE/(111_320.0*math.cos(math.radians(lat0)))
    return lat0+dlat, lon0+dlon

def center_ground_point(lat_d,lon_d,tilt_down,yaw_deg,AGL):
    # ground hit at image center
    t=math.radians(tilt_down)
    if t<=0: return lat_d,lon_d
    r0=AGL/math.tan(t); dE,dN=rotate_forward_right_to_ENU(r0,0,yaw_deg)
    return enu_offset_to_gps(lat_d,lon_d,dE,dN)

# ====================== EXPORT HELPERS (CSV/KML) ======================

def write_csv(main_lat,main_lon,rows,path=CSV_PATH):
    # write detections + main point to CSV
    fieldnames=["id","name","u","v","lat","lon","dE_m","dN_m","forward_m","right_m",
                "agl_m","tilt_deg","yaw_deg","image","timestamp_iso"]
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fieldnames); w.writeheader()
        w.writerow({"id":0,"name":"main_coordinate","u":"","v":"",
                    "lat":f"{main_lat:.8f}","lon":f"{main_lon:.8f}",
                    "dE_m":"","dN_m":"","forward_m":"","right_m":"",
                    "agl_m":"","tilt_deg":"","yaw_deg":"","image":os.path.basename(ORIG_IMG_PATH),
                    "timestamp_iso":datetime.now(timezone.utc).isoformat()})
        for r in rows: w.writerow(r)

def write_kml(main_lat,main_lon,rows,path=KML_PATH,doc_name="Detections"):
    # write a simple KML with placemarks
    def pm(name,lat,lon,desc=""):
        return f"<Placemark><name>{name}</name><description><![CDATA[{desc}]]></description><Point><coordinates>{lon:.8f},{lat:.8f},0</coordinates></Point></Placemark>"
    parts=['<?xml version="1.0" encoding="UTF-8"?>','<kml xmlns="http://www.opengis.net/kml/2.2">',f"<Document><name>{doc_name}</name>",
           pm("main_coordinate",main_lat,main_lon,"Original image GPS")]
    for r in rows:
        desc=f"id={r['id']}, u={r['u']}, v={r['v']}, dE={r['dE_m']}m, dN={r['dN_m']}m"
        parts.append(pm(r['name'],float(r['lat']),float(r['lon']),desc))
    parts.append("</Document></kml>")
    with open(path,"w",encoding="utf-8") as f: f.write("\n".join(parts))

# ============================= MAIN =============================

def main():
    # FOV + tilt + yaw
    info=extract_fov_tilt_yaw(ORIG_IMG_PATH); hfov,vfov=info["hfov_deg"],info["vfov_deg"]
    tilt_down=info["tilt_down_deg"] if info["tilt_down_deg"] else 45.0
    yaw_deg=info["yaw_deg"] if info["yaw_deg"] else 0.0
    # AGL + drone GPS
    agl_m,lat_d,lon_d=compute_agl_and_drone_gps(ORIG_IMG_PATH)
    # center ground point
    center_lat,center_lon=center_ground_point(lat_d,lon_d,tilt_down,yaw_deg,agl_m)
    # intrinsics for resized image
    fx,fy,cx,cy=intrinsics_for_size(RESIZE_W,RESIZE_H,hfov,vfov)

    # image + model + predict
    frame=cv2.imread(ORIG_IMG_PATH); imgR=cv2.resize(frame,(RESIZE_W,RESIZE_H))
    model=YOLO(MODEL_PATH); results=model.predict(imgR,save=True,conf=0.6)

    det_rows=[]; det_counter=1
    for r in results:
        # let Ultralytics draw on a copy
        img = r.plot(line_width=2)
        boxes = r.boxes

        h,w=img.shape[:2]; cx_img,cy_img=w//2,h//2
        # center cross
        cv2.line(img,(cx_img-10,cy_img),(cx_img+10,cy_img),(255,0,255),2)
        cv2.line(img,(cx_img,cy_img-10),(cx_img,cy_img+10),(255,0,255),2)

        # per detection: center, angles, projection
        for box in boxes:
            x1,y1,x2,y2=map(int,box.xyxy[0])
            ux,vy=(x1+x2)//2,(y1+y2)//2

            # center dot + coords
            cv2.circle(img,(ux,vy),4,(0,0,255),-1)
            cv2.putText(img,f"({ux},{vy})",(ux-45,vy+25),cv2.FONT_HERSHEY_SIMPLEX,0.5,(9,10,11),2)

            # angles text
            thx,thy,thoff,bimg=pixel_to_angles(ux,vy,fx,fy,cx,cy)
            lbl=f"theta_x: {thx:.1f} deg, theta_y: {thy:.1f} deg"
            cvzone.putTextRect(img,lbl,(ux+8,vy-20),scale=0.7,thickness=1,colorT=(0,0,0),colorR=(255,255,255))

            # line center → detection (visual)
            cv2.line(img,(cx_img,cy_img),(ux,vy),(255,225,5),2)

            # ground projection → ENU → GPS
            fr=forward_right_offsets(ux,vy,w,h,hfov,vfov,tilt_down,agl_m,fx,fy,cx,cy)
            if fr is None: continue
            fwd,right=fr; dE,dN=rotate_forward_right_to_ENU(fwd,right,yaw_deg)
            lat_p,lon_p=enu_offset_to_gps(center_lat,center_lon,dE,dN)

            # collect row
            det_rows.append({"id":det_counter,"name":f"det_{det_counter}","u":ux,"v":vy,
                             "lat":f"{lat_p:.8f}","lon":f"{lon_p:.8f}",
                             "dE_m":f"{dE:.3f}","dN_m":f"{dN:.3f}",
                             "forward_m":f"{fwd:.3f}","right_m":f"{right:.3f}",
                             "agl_m":f"{agl_m:.3f}","tilt_deg":f"{tilt_down:.3f}","yaw_deg":f"{yaw_deg:.3f}",
                             "image":os.path.basename(ORIG_IMG_PATH),
                             "timestamp_iso":datetime.now(timezone.utc).isoformat()})
            det_counter+=1

        cv2.imwrite(os.path.join(OUTPUT_DIR,"C_angles_geo_1.jpg"),img)
        cv2.imshow("Result",img); cv2.waitKey(0); cv2.destroyAllWindows()

    # Write CSV & KML
    write_csv(lat_d,lon_d,det_rows); write_kml(lat_d,lon_d,det_rows,doc_name=os.path.basename(ORIG_IMG_PATH))
    print(f"Saved CSV: {CSV_PATH}\nSaved KML: {KML_PATH}")

if __name__=="__main__": main()

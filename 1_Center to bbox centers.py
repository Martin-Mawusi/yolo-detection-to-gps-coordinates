import ultralytics
# print("Ultralytics version:", ultralytics.__version__)
import cvzone
from ultralytics import YOLO
import cv2

frame = cv2.imread('test images/Blantyre2.JPG')
imgR = cv2.resize(frame, (1080, 720))
# img1=cv2.imread('test images/three.jpeg')
# cv2.imshow("Resized Image", imgR)
cv2.waitKey(3000)


# Trained model
model = YOLO(r'/oblique_vision_georeferencing1/Yolo Best Model/best.pt')


results = model.predict(imgR, save=True, conf=0.6)


# for r in results:
#     img = r.plot()

for r in results:
    img = r.orig_img.copy()
    boxes = r.boxes

    centers = []

    # for box in boxes:
    #     x1, y1, x2, y2 = map(int, box.xyxy[0])  # bounding box corners
    #     cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # center point
    #     centers.append((cx, cy))  # save in list
    #
    #     # draw a small red dot at the center
    #     cv2.circle(img, (cx, cy), 4, (0, 0, 255), -4)

    for i, box in enumerate(boxes, start=1):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        centers.append((cx, cy))

        # draw red dot
        cv2.circle(img, (cx, cy), 4, (0, 0, 255), -1)

        # label with ID number
        cv2.putText(img,  f"({cx},{cy})", (cx -45, cy + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (9, 10, 11), 2)

    for box in boxes:
        # xywh gives center-x, center-y, width, height
        x, y, w, h = box.xywh[0]

        # Convert to top-left x,y for cvzone.cornerRect
        x1 = int(x - w/2)
        y1 = int(y - h/2)
        bbox = (x1, y1, int(w), int(h))

        cvzone.cornerRect(img, bbox, l=3, rt=1, colorC=(255, 0, 0), colorR=(0, 255, 0))

        # Get confidence and class
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = f"{r.names[cls]} {conf:.2f}"

        # Put text above box
        cvzone.putTextRect(img, label, (x1, y1 - 10), scale=1.5, thickness=2, colorR=(255, 0, 0), offset=2, border=1)
        # cvzone.putTextRect(
        #     img,
        #     label,
        #     (x1, y1 - 10),
        #     scale=1.7,  # smaller text size
        #     thickness=1,  # thinner text
        #     colorR=(255, 0, 0),  # rectangle color
        #     colorT=(255, 255, 255),  # text color (white)
        #     offset=5,  # reduce padding around text
        #     border=1  # thin border for sharpness
        # )


    # --- draw center of image
    h, w = img.shape[:2]
    cx_img, cy_img = w // 2, h // 2

    # draw lines from image center to each detection center
    for (cx, cy) in centers:
        cv2.line(img, (cx_img, cy_img), (cx, cy), (255, 225, 5), 2)  # pink line

    # Draw vertical and horizontal lines for cross
    cv2.line(img, (cx_img - 10, cy_img), (cx_img + 10, cy_img), (255, 0, 255), 2)  # pink
    cv2.line(img, (cx_img, cy_img - 10), (cx_img, cy_img + 10), (255, 0, 255), 2)  # pink


    from math import hypot

    # sensor pixel pitch (mm per sensor pixel)
    PIXEL_PITCH_MM = 0.00241  # e.g., 2.41 µm

    # compute effective mm per pixel on the *resized* image
    h0, w0 = frame.shape[:2]  # original image (before resize)
    h1, w1 = img.shape[:2]  # resized image you ran through YOLO (r.orig_img)

    # if resize resize is uniformly, scale_x ≈ scale_y; pick x-axis
    scale_x = w1 / w0 if w0 else 1.0
    mm_per_resized_pixel = PIXEL_PITCH_MM / (scale_x if scale_x != 0 else 1.0)

    # center of image (pink cross)
    # h, w = img.shape[:2]
    # cx_img, cy_img = w // 2, h // 2
    # cv2.line(img, (cx_img - 10, cy_img), (cx_img + 10, cy_img), (255, 0, 255), 2)  # pink
    # cv2.line(img, (cx_img, cy_img - 10), (cx_img, cy_img + 10), (255, 0, 255), 2)  # pink

    # lines and distances
    YELLOW = (0, 255, 255)
    distances_px = []

    distances_mm = []

    for (cx, cy) in centers:
        # draw line center->detection center
        # cv2.line(img, (cx_img, cy_img), (cx, cy), YELLOW, 2)

        # distance in pixels on the *resized* image
        d_px = hypot(cx - cx_img, cy - cy_img)

        # convert to mm on the sensor, correcting for resize
        d_mm = d_px * mm_per_resized_pixel
        distances_mm.append(d_mm)

        # label at the midpoint of the line
        mx, my = (cx_img + cx) // 2, (cy_img + cy) // 2

        # use cvzone for readable text with a background
        cvzone.putTextRect(
            img,
            f"{d_mm:.2f} mm",
            (int(mx)-25, int(my) ),  # slight offset so it doesn't sit exactly on the line
            scale=1,
            thickness=1,
            colorT=(0, 0, 0),  # black text
            colorR=YELLOW,  # yellow rectangle (dominant, readable)
            offset=1,
            # border=1
        )

    # edge-to-edge rulers with mm labels
    ORANGE = (0, 165, 255)  # BGR, new dominant color
    thickness = 2
    #
    h, w = img.shape[:2]

     # 1) WIDTH ruler — full bottom edge
    cv2.line(img, (0, h - 1), (w - 1, h - 1), ORANGE, thickness)
    width_mm = w * mm_per_resized_pixel
    # midpoint label
    mx_w, my_w = (w - 1) // 2, h - 1
    cvzone.putTextRect(
        img,
        f"{width_mm:.2f} mm",
        (int(mx_w) - 80, int(my_w) - 10),
        scale=1,
        thickness=2,
        colorT=(0, 0, 0),
        colorR=ORANGE,
        offset=1,
        # border=1
    )

    # 2) HEIGHT ruler — full right edge
    cv2.line(img, (w - 1, 0), (w - 1, h - 1), ORANGE, thickness)
    height_mm = h * mm_per_resized_pixel
    # midpoint label
    mx_h, my_h = w - 1, h // 2
    cvzone.putTextRect(
        img,
        f"{height_mm:.2f} mm",
        (int(mx_h) - 80, int(my_h) - 10),
        scale=1,
        thickness=2,
        colorT=(0, 0, 0),
        colorR=ORANGE,
        offset=1,
        # border=1
    )

    cv2.imwrite("Output_yolo/B11.jpg", img)

    cv2.imshow("Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


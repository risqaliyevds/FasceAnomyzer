import cv2

def process_img(img, face_detection):
    # Define image size
    H, W, _ = img.shape
    # Convert color to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)

    try: # Try to find human face
        for detection in out.detections:
            # Define coordinates of face
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box

            # Coordinates of blurring zone
            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

            x1 = int(x1 * W)
            y1 = int(y1 * H)
            w = int(w * W)
            h = int(h * H)

            # Blurring only face zone
            img[y1:y1 + h, x1:x1 + w, :] = cv2.blur(img[y1:y1 + h, x1:x1 + w, :], (100, 100))
    except:
        pass

    # Return final image
    return img
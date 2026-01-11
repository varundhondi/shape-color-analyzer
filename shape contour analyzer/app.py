import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("🔷 Shape & Contour Analyzer")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    img = np.array(image)

    st.image(img, caption="Original Image", width=700)

    # ---------- PREPROCESSING ----------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)

    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # ---------- CONTOUR DETECTION ----------
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    output = img.copy()
    count = 0

    st.subheader("Detected Shapes & Measurements")

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 800:  # filter small noise
            count += 1
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)

            sides = len(approx)

            if sides == 3:
                shape = "Triangle"
            elif sides == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / float(h)
                shape = "Square" if 0.95 <= aspect_ratio <= 1.05 else "Rectangle"
            elif sides > 4:
                shape = "Circle"
            else:
                shape = "Unknown"

            cv2.drawContours(output, [cnt], -1, (0, 255, 0), 2)
            x, y = approx[0][0]
            cv2.putText(
                output,
                shape,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )

            st.write(
                f"🔹 **{shape}** | Area: `{int(area)}` | Perimeter: `{int(perimeter)}`"
            )

    st.success(f"Total Objects Detected: {count}")
    st.image(output, caption="Analyzed Image", width=700)
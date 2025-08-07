import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import io
import datetime
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet

# Page config
st.set_page_config(
    page_title="Concrete Crack Detection Web App",
    page_icon="🧱",
    layout="wide"
)

@st.cache_resource
def load_model():
    # Replace with path to your actual model
    return tf.keras.models.load_model("Crack Detection/crack_model.h5")

model = load_model()

def predict_crack_pil(img_pil):
    img = img_pil.resize((120, 120)).convert("RGB")
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    prediction = model.predict(img_array)[0][0]
    label = "Crack Detected" if prediction > 0.5 else "No Crack"
    return label, prediction

def mask_and_area(img_pil, alpha=0.5):
    img_rgb = np.array(img_pil.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(gray)
    if contours:
        cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
        crack_area_px = cv2.countNonZero(mask)
    else:
        crack_area_px = 0

    overlay = img_bgr.copy()
    red_mask = np.zeros_like(overlay)
    red_mask[:] = (0, 0, 255)  # Red overlay in BGR
    mask_3ch = cv2.merge([mask, mask, mask]) // 255
    overlay = np.where(mask_3ch == 0,
                       overlay,
                       cv2.addWeighted(overlay, 1 - alpha, red_mask, alpha, 0))
    highlight_img = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    pil_overlay = Image.fromarray(highlight_img)

    return pil_overlay, crack_area_px, mask, img_pil.size

def convert_area(crack_area_px, img_size, unit, real_width):
    img_w, img_h = img_size
    pixel_area = img_w * img_h
    px_to_unit2 = None
    if real_width and real_width > 0:
        px_to_cm = real_width / img_w
        if unit == "cm²":
            px_to_unit2 = px_to_cm ** 2
        elif unit == "mm²":
            px_to_unit2 = (px_to_cm * 10) ** 2
        elif unit == "in²":
            px_to_unit2 = (real_width / 2.54 / img_w) ** 2
    crack_area_unit = crack_area_px * px_to_unit2 if px_to_unit2 else None
    area_frac = (crack_area_px / float(pixel_area)) * 100 if pixel_area > 0 else 0.0
    return crack_area_unit, area_frac

def extract_metadata(img_pil):
    width, height = img_pil.size
    dpi = img_pil.info.get("dpi", (72, 72))  # default DPI if missing
    aspect_ratio = width / height if height != 0 else None
    image_format = img_pil.format
    return {
        "Width": width,
        "Height": height,
        "Aspect Ratio": f"{aspect_ratio:.2f}" if aspect_ratio else "N/A",
        "DPI": f"{dpi[0]}x{dpi[1]}",
        "Format": image_format
    }

def generate_pdf_report(report_data_list):
    pdf_buffer = io.BytesIO()
    pdf = SimpleDocTemplate(pdf_buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("🧱 Concrete Crack Detection Report", styles["Title"]))
    elements.append(Spacer(1, 12))
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elements.append(Paragraph(f"Generated on: {timestamp}", styles["Normal"]))
    elements.append(Spacer(1, 12))

    for data in report_data_list:
        elements.append(Paragraph(f"Image: {data['filename']}", styles["Heading2"]))
        # Add thumbnail image from in-memory buffer
        thumb_io = io.BytesIO()
        data["thumbnail"].save(thumb_io, format="PNG")
        thumb_io.seek(0)
        elements.append(RLImage(thumb_io, width=150, height=150))
        elements.append(Spacer(1, 10))

        meta = data["metadata"]
        elements.append(Paragraph("Image Metadata:", styles["Heading3"]))
        for key in ["Width", "Height", "Aspect Ratio", "DPI", "Format"]:
            elements.append(Paragraph(f"{key}: {meta.get(key, 'N/A')}", styles["Normal"]))

        ca_px = data["crack_area_px"]
        ca_unit = data.get("crack_area_unit")
        unit = data.get("unit")
        area_str = f"{ca_px} px²"
        if ca_unit is not None:
            area_str += f" / {ca_unit:.2f} {unit}"
        elements.append(Paragraph(f"Total Crack Area: {area_str}", styles["Normal"]))
        elements.append(Paragraph(f"Crack Area Fraction: {data['area_fraction']:.2f} %", styles["Normal"]))
        elements.append(Spacer(1, 20))

    pdf.build(elements)
    pdf_buffer.seek(0)
    return pdf_buffer.read()

# -------- App starts here --------

if "history" not in st.session_state:
    st.session_state["history"] = []

st.title("Concrete Crack Detection Web App 🧱")
st.markdown("""
Upload one or more images of concrete surfaces to detect cracks using a trained AI model.  
Use the slider to adjust red overlay transparency. View original and highlighted images side-by-side.  
Download a PDF report summarizing crack areas and image metadata for all uploads.
""")

unit = st.selectbox("Show area in:", ["px²", "cm²", "mm²", "in²"])
real_world_width_cm = st.number_input(
    "Enter the real-world width (cm) represented by uploaded images (leave 0 if unknown):",
    min_value=0.0,
    value=0.0,
    step=0.1,
)
use_conversion = unit != "px²" and real_world_width_cm > 0

alpha = st.slider("Overlay transparency (red crack highlight)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

uploaded_files = st.file_uploader(
    "Choose image files",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

summary_data = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.header(f"File: {uploaded_file.name}")

        img_pil = Image.open(uploaded_file).convert("RGB")

        # Metadata display
        meta = extract_metadata(img_pil)
        st.markdown(f"""
        **Image Metadata:**  
        - Resolution: {meta['Width']} x {meta['Height']} px  
        - Aspect Ratio: {meta['Aspect Ratio']}  
        - DPI: {meta['DPI']}  
        - Format: {meta['Format']}
        """)

        col_orig, col_highlight = st.columns(2)
        with col_orig:
            st.image(img_pil, caption="Original Image", use_container_width=True)

        label, prob = predict_crack_pil(img_pil)

        if label == "Crack Detected":
            st.success(f"Prediction: {label} (probability: {prob:.2f})")
        else:
            st.info(f"Prediction: {label} (probability: {prob:.2f})")

        st.session_state.history.append({
            "Filename": uploaded_file.name,
            "Result": label,
            "Probability": f"{prob:.2f}"
        })

        if label == "Crack Detected":
            highlighted_img, crack_area_px, mask, size = mask_and_area(img_pil, alpha=alpha)
            area_unit, frac = convert_area(crack_area_px, size, unit, real_world_width_cm if use_conversion else None)

            with col_highlight:
                st.image(highlighted_img, caption="Crack Highlighted Image", use_container_width=True)

                # Download highlighted image
                img_buffer = io.BytesIO()
                highlighted_img.save(img_buffer, format="PNG")
                st.download_button(
                    label="Download Highlighted Crack Image",
                    data=img_buffer.getvalue(),
                    file_name=f"crack_highlighted_{uploaded_file.name}.png",
                    mime="image/png"
                )

            st.markdown(f"""
            - **Total crack area:** {crack_area_px} px²{f' / {area_unit:.2f} {unit}' if area_unit is not None else ''}
            - **Fraction of image covered:** {frac:.2f} %
            """)

            # For PDF report summary
            summary_data.append({
                "filename": uploaded_file.name,
                "thumbnail": img_pil.copy().resize((128, 128)),
                "crack_area_px": crack_area_px,
                "crack_area_unit": area_unit,
                "area_fraction": frac,
                "unit": unit,
                "metadata": meta
            })
        else:
            st.write("No crack detected; skipping crack highlight and area calculation.")

    # Summary Table
    if summary_data:
        st.subheader("Summary of Crack Areas")
        df_summary = pd.DataFrame([{
            "Filename": d["filename"],
            "Crack Area (px²)": d["crack_area_px"],
            f"Crack Area ({d['unit']})": f"{d['crack_area_unit']:.2f}" if d["crack_area_unit"] else "-",
            "Coverage (%)": f"{d['area_fraction']:.2f}"
        } for d in summary_data])
        st.dataframe(df_summary, use_container_width=True)

        # PDF report download
        pdf_bytes = generate_pdf_report(summary_data)
        st.download_button(
            label="Download Crack Detection Report (PDF)",
            data=pdf_bytes,
            file_name="crack_detection_report.pdf",
            mime="application/pdf"
        )

with st.expander("🕑 View Image Analysis History"):
    if st.session_state["history"]:
        st.dataframe(pd.DataFrame(st.session_state["history"]), use_container_width=True)
    else:
        st.write("No images tested yet.")

# Footer
footer_html = """
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #f0f2f6;
    color: #666666;
    text-align: center;
    font-size: 12px;
    padding: 8px 0;
    border-top: 1px solid #e6e6e6;
    z-index: 1000;
}
</style>
<div class="footer">
    © 2025 Prince Osei Boateng. All rights reserved.
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)

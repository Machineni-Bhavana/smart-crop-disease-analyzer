import streamlit as st  # type: ignore
from PIL import Image  # type: ignore
import numpy as np  # type: ignore

# Import custom modules
from predict import predict_disease  # type: ignore
from utils.disease_info import get_disease_info  # type: ignore
from database import init_db, log_prediction, get_analytics_data  # type: ignore
from fpdf import FPDF  # type: ignore
import tempfile

# Ensure DB is initialized
init_db()


def create_pdf_report(disease_name, probability, info):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=15, style="B")
    pdf.cell(200, 10, text="Crop Disease Diagnostic Report", ln=True, align="C")
    pdf.ln(10)

    pdf.set_font("Arial", size=12, style="B")
    pdf.cell(200, 10, text=f"Detected Disease: {disease_name}", ln=True)
    pdf.cell(200, 10, text=f"Confidence Score: {probability:.2%}", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", size=11, style="B")
    pdf.cell(200, 10, text="Symptoms:", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 8, text=info["symptoms"])
    pdf.ln(5)

    pdf.set_font("Arial", size=11, style="B")
    pdf.cell(200, 10, text="Treatment Suggestion:", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 8, text=info["treatment_suggestion"])
    pdf.ln(5)

    pdf.set_font("Arial", size=11, style="B")
    pdf.cell(200, 10, text="Prevention Tips:", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 8, text=info["prevention_tips"])

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(tmp_file.name)
    return tmp_file.name


def main():
    st.set_page_config(
        page_title="Smart Crop Disease Analyzer", layout="wide", page_icon="🌿"
    )
    st.title("🌿 Smart Crop Disease Analyzer")
    st.markdown(
        "Upload a crop leaf image to detect diseases using our AI-powered Convolutional Neural Network (CNN)."
    )

    tab_app, tab_analytics = st.tabs(["Disease Diagnosis", "Analytics Dashboard"])

    with tab_app:
        st.header("1. Image Upload")
        uploaded_file = st.file_uploader(
            "Upload crop leaf image (JPG/PNG)", type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Leaf Image", use_container_width=True)

            if st.button("Run CNN Prediction", type="primary"):
                with st.spinner("Analyzing the image..."):
                    img_array = np.array(image.convert("RGB"))

                    # Unpack the new response dict
                    result = predict_disease(img_array)

                    disease_name = result["disease_name"]
                    probability = result["probability"]
                    is_blurry = result["is_blurry"]
                    confidence_met = result["confidence_threshold_met"]
                    latency = result["latency"]

                    # Log to DB
                    crop_type = (
                        disease_name.split(" ")[0]
                        if " " in disease_name
                        else disease_name
                    )
                    log_prediction(
                        crop_type, disease_name, probability, is_blurry, latency
                    )

                    info = get_disease_info(disease_name)

                    st.header("2. Disease Prediction Result")

                    # Warning if blurry
                    if is_blurry:
                        st.warning(
                            "⚠️ Image appears blurry. The prediction might be less accurate."
                        )

                    # Confidence threshold check
                    if not confidence_met:
                        st.error(
                            "📉 Low Confidence Prediction (<70%). Please retake the image for better results."
                        )
                    else:
                        st.success(f"High Confidence Match! (Latency: {latency:.2f}s)")

                    st.subheader(f"**Predicted Disease:** {disease_name}")

                    # Real-time probability visualization (Gauge/Progress)
                    st.write(f"Confidence: {probability:.2%}")
                    st.progress(float(probability))

                    st.header("3. Treatment Advice")
                    st.write(f"**Symptoms:** {info['symptoms']}")
                    st.write(
                        f"**Treatment Suggestion:** {info['treatment_suggestion']}"
                    )
                    st.write(f"**Prevention Tips:** {info['prevention_tips']}")

                    # Export PDF
                    pdf_path = create_pdf_report(disease_name, probability, info)
                    with open(pdf_path, "rb") as pdf_file:
                        PDFbyte = pdf_file.read()
                    st.download_button(
                        label="📄 Download Diagnostic Report (PDF)",
                        data=PDFbyte,
                        file_name="crop_diagnostic_report.pdf",
                        mime="application/octet-stream",
                    )

    with tab_analytics:
        st.header("Analytics Dashboard")
        st.markdown("Insights are based on actual usage history.")

        df = get_analytics_data()

        if df.empty:
            st.info("No prediction data available yet. Please run some scans first.")
        else:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Disease Occurrence Summary")
                st.dataframe(df["disease_name"].value_counts().reset_index())

                st.subheader("Average Confidence per Disease")
                conf_df = df.groupby("disease_name")["confidence"].mean().reset_index()
                conf_df["confidence"] = conf_df["confidence"].apply(
                    lambda x: f"{x:.2%}"
                )
                st.dataframe(conf_df)

            with col2:
                # Let's see some basic charts if matplotlib/plotly is available
                st.subheader("Confidence Score Distribution")
                try:
                    import plotly.express as px

                    fig = px.histogram(
                        df,
                        x="confidence",
                        color="disease_name",
                        nbins=10,
                        title="Overall Confidence Scores",
                    )
                    st.plotly_chart(fig)
                except ImportError:
                    st.bar_chart(df["confidence"])

                st.metric("Total Scans Performed", len(df))
                st.metric("Average System Latency (s)", f"{df['latency'].mean():.2f}")


if __name__ == "__main__":
    main()

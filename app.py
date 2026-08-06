import time
from pathlib import Path
import streamlit as st
from PIL import Image

from config import Config
from src.inference import DiseasePredictor
from src.utils.exceptions import CropAnalyzerException, ModelLoadError

# Configure Streamlit Page
st.set_page_config(
    page_title="Smart Crop Disease Analyzer",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2E7D32;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F1F8E9;
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 5px solid #2E7D32;
        margin-bottom: 1rem;
    }
    .prediction-title {
        font-size: 1.4rem;
        font-weight: 600;
        color: #1B5E20;
    }
    .prediction-sub {
        font-size: 1.0rem;
        color: #333333;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_predictor(model_path: str):
    """Cache DiseasePredictor model instance across user sessions."""
    return DiseasePredictor(checkpoint_path=model_path)

def main():
    st.markdown('<div class="main-header">🌿 Smart Crop Disease Analyzer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Upload a plant leaf image to detect potential diseases using a deep-learning image classifier.</div>',
        unsafe_allow_html=True
    )
    
    # Sidebar Info & Settings
    st.sidebar.image("https://img.icons8.com/color/96/plant-under-sun.png", width=80)
    st.sidebar.title("Model & Settings")
    
    checkpoint_file = Config.BEST_MODEL_PATH
    
    if not checkpoint_file.exists():
        st.sidebar.warning(f"No checkpoint found at '{checkpoint_file}'. Train a model first.")
        st.error(
            f"❌ Trained model checkpoint not found at `{checkpoint_file}`.\n\n"
            "Please train the model first by executing:\n"
            "```bash\n"
            "python scripts/train.py\n"
            "```"
        )
        return

    # Attempt predictor loading
    try:
        predictor = load_predictor(str(checkpoint_file))
        st.sidebar.success(f"Loaded Architecture: **{predictor.model_name.upper()}**")
        st.sidebar.info(f"Target Device: **{predictor.device}**")
        st.sidebar.write(f"Supported Classes: **{predictor.num_classes}**")
    except ModelLoadError as e:
        st.error(f"Failed to load model checkpoint: {str(e)}")
        return
    except Exception as e:
        st.error(f"Unexpected error loading model: {str(e)}")
        return

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("1. Upload Leaf Photograph")
        uploaded_file = st.file_uploader(
            "Choose a JPEG, JPG, or PNG image of a crop leaf:",
            type=["jpg", "jpeg", "png"],
            help="Select a clear photograph of a plant leaf."
        )

        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Leaf Image", use_column_width=True)
            except Exception as e:
                st.error(f"Unable to decode uploaded file as an image: {str(e)}")
                return

    with col2:
        st.subheader("2. Diagnostic Results")
        
        if uploaded_file is not None:
            analyze_btn = st.button("🔍 Analyze Leaf", type="primary", use_container_width=True)
            
            if analyze_btn:
                with st.spinner("Running PyTorch Deep Learning Inference..."):
                    try:
                        # Reset file pointer and read bytes
                        uploaded_file.seek(0)
                        image_bytes = uploaded_file.read()
                        
                        # Measure full end-to-end timing
                        t_start = time.perf_counter()
                        prediction = predictor.predict(image_bytes)
                        t_total = (time.perf_counter() - t_start)
                        
                        plant = prediction["plant"]
                        disease = prediction["disease"]
                        conf_pct = prediction["confidence_percent"]
                        model_latency_ms = prediction["inference_time_ms"]
                        model_latency_sec = model_latency_ms / 1000.0
                        
                        # Display primary result card
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="prediction-title">🌱 Detected Category: {plant} — {disease}</div>
                            <div class="prediction-sub">Confidence: <strong>{conf_pct}</strong></div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Metrics columns
                        m_col1, m_col2, m_col3 = st.columns(3)
                        m_col1.metric("Crop / Plant", plant)
                        m_col2.metric("Disease Condition", disease)
                        m_col3.metric("Model Latency", f"{model_latency_sec:.2f} s", f"{model_latency_ms:.0f} ms")
                        
                        st.divider()
                        
                        # Top-3 predictions progress bars
                        st.subheader("Top 3 Predictions")
                        top_preds = prediction["top_predictions"]
                        
                        for i, pred in enumerate(top_preds, 1):
                            lbl = f"{i}. {pred['plant']} — {pred['disease']}"
                            c_val = pred["confidence"]
                            st.write(f"**{lbl}** ({pred['confidence_percent']})")
                            st.progress(c_val)
                            
                    except CropAnalyzerException as e:
                        st.error(f"Inference Error: {str(e)}")
                    except Exception as e:
                        st.error(f"An unexpected error occurred during analysis: {str(e)}")
        else:
            st.info("👈 Upload a leaf image on the left panel and click **Analyze Leaf** to perform disease diagnosis.")

if __name__ == "__main__":
    main()

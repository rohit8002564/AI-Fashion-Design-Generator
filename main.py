import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import io

# --- Page Config ---
st.set_page_config(page_title="AI Fashion Design Generator", layout="wide")

# --- Title ---
st.title("👗 AI Fashion Design Generator")
st.markdown("Generate stunning fashion designs using AI. Powered by Stable Diffusion + Streamlit.")

# --- Sidebar ---
st.sidebar.header("🛠️ Design Controls")
prompt = st.sidebar.text_input("Enter your fashion prompt", value="A futuristic evening gown with metallic textures")
guidance_scale = st.sidebar.slider("Creativity (Guidance Scale)", 1.0, 20.0, 7.5)
num_inference_steps = st.sidebar.slider("Inference Steps", 10, 100, 50)

# --- Load Model ---
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float32
    ).to(device)
    return pipe

with st.spinner("Loading Stable Diffusion model..."):
    try:
        pipe = load_model()
    except Exception as e:
        st.error("❌ Failed to load model. Check your requirements or Streamlit logs.")
        st.stop()

# --- Generate Image ---
if st.button("🎨 Generate Fashion Design"):
    with st.spinner("Generating image..."):
        image = pipe(prompt, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps).images[0]
        st.image(image, caption="Generated Design", use_column_width=True)

        # --- Download Button ---
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        byte_im = buf.getvalue()
        st.download_button(
            label="📥 Download Image",
            data=byte_im,
            file_name="fashion_design.png",
            mime="image/png"
        )

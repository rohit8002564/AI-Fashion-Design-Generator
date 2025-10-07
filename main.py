import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import io
from huggingface_hub import login

# --- Login to HuggingFace (replace with your token) ---
# login("hf_your_token_here")

# --- Page Config ---
st.set_page_config(page_title="AI Fashion Design Generator", layout="wide")

st.title("👗 AI Fashion Design Generator")
st.markdown("Generate stunning fashion designs using AI. Powered by Stable Diffusion + Streamlit.")

# --- Sidebar Controls ---
st.sidebar.header("🛠️ Design Controls")
prompt = st.sidebar.text_input("Enter your fashion prompt", value="A futuristic evening gown with metallic textures")
guidance_scale = st.sidebar.slider("Creativity (Guidance Scale)", 1.0, 20.0, 7.5)
num_inference_steps = st.sidebar.slider("Inference Steps", 10, 100, 50)

# --- Load Model ---
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=dtype,
        use_auth_token=True   # required for this model
    ).to(device)
    return pipe

with st.spinner("Loading Stable Diffusion model..."):
    try:
        pipe = load_model()
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

# --- Generate Image ---
if st.button("🎨 Generate Fashion Design"):
    with st.spinner("Generating image..."):
        image = pipe(
            prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps
        ).images[0]

        st.image(image, caption="Generated Design", use_column_width=True)

        # Download button
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        st.download_button(
            label="📥 Download Image",
            data=buf.getvalue(),
            file_name="fashion_design.png",
            mime="image/png"
        )

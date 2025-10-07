import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
import requests
from PIL import Image
from io import BytesIO

# --- Load Stable Diffusion Model ---
@st.cache_resource
def load_model():
    model = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(device)

pipe = load_model()

# --- Generate Image from Prompt ---
def generate_image(prompt):
    image = pipe(prompt).images[0]
    return image

# --- Search Similar Products via SerpAPI ---
def search_products(query):
    api_key = "1aadff5156d5c16d10cd8d25e948d22945afcab63583eb9ce2e0db3d030dd906"  # Replace with your actual key
    url = f"https://serpapi.com/search.json?q={query}&tbm=shop&api_key={api_key}"
    response = requests.get(url)
    results = response.json().get("shopping_results", [])
    return results

# --- Streamlit UI ---
st.set_page_config(page_title="AI Fashion Design Generator", layout="centered")
st.title("👗 AI Fashion Design Generator")
st.markdown("Describe your fashion idea and let AI bring it to life!")

prompt = st.text_input("📝 Enter your fashion prompt:", placeholder="e.g. A futuristic streetwear hoodie")

if st.button("🎨 Generate Design"):
    if prompt:
        with st.spinner("Generating your fashion design..."):
            image = generate_image(prompt)
            st.image(image, caption="🖼️ Your AI-Generated Design", use_column_width=True)

            st.subheader("🛍️ Similar Products Online")
            products = search_products(prompt)
            if products:
                for product in products[:5]:
                    st.markdown(f"**{product.get('title', 'No Title')}**")
                    if product.get("thumbnail"):
                        st.image(product["thumbnail"], width=150)
                    st.write(product.get("link", "No Link"))
            else:
                st.info("No similar products found. Try refining your prompt.")
    else:
        st.warning("Please enter a fashion prompt to begin.")

# --- Footer ---
st.markdown("---")
st.caption("Built with ❤️ by Rohit Kumar | Powered by Stable Diffusion + Streamlit")
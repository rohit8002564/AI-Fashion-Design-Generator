import streamlit as st
import requests
import base64
import json
import time
from io import BytesIO
from PIL import Image
import os

# --- Configuration ---
API_KEY = os.getenv("IMAGEN_API_KEY") or "AIzaSyD-8R9WxuyFijbnCqnHkXEMIELPTlZqZFo"  # Replace with your key or use env variable
MODEL_NAME = "imagen-3.0"
API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models/"
API_URL = f"{API_BASE_URL}{MODEL_NAME}:generateImage?key={API_KEY}"
MAX_RETRIES = 5

ASPECT_MAP = {
    "1:1": "SQUARE",
    "3:4": "PORTRAIT",
    "16:9": "LANDSCAPE"
}

# --- Utility Functions ---
def generate_image(prompt, aspect_ratio="1:1", num_images=1, max_retries=MAX_RETRIES):
    full_prompt = (
        f"A hyper-realistic, high-fashion runway photograph of a garment: '{prompt}'. "
        "Focus on texture, stitching, dramatic lighting, and a minimalist background."
    )

    aspect_api = ASPECT_MAP.get(aspect_ratio, "SQUARE")

    payload = {
        "instances": [{"prompt": {"text": full_prompt}}],
        "parameters": {
            "sampleCount": num_images,
            "aspectRatio": aspect_api,
            "outputMimeType": "image/jpeg"
        }
    }

    headers = {'Content-Type': 'application/json'}

    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            result = response.json()

            # Extract base64 image data
            images = []
            if "predictions" in result:
                for pred in result["predictions"]:
                    if "bytesBase64Encoded" in pred:
                        images.append(pred["bytesBase64Encoded"])
            elif "images" in result:
                for img_data in result["images"]:
                    if "bytesBase64Encoded" in img_data:
                        images.append(img_data["bytesBase64Encoded"])

            if images:
                return images, None

            return None, "API returned no images. Try adjusting your prompt."

        except requests.exceptions.HTTPError as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return None, f"HTTP error after {max_retries} attempts: {e}"
        except requests.exceptions.RequestException as e:
            return None, f"Network error: {e}"
        except Exception as e:
            return None, f"Unexpected error: {e}"

    return None, "Image generation failed after retries."

# --- Simulated E-commerce Section ---
def render_similar_products_simulation(design_prompt):
    st.markdown("---")
    st.subheader("🛍️ Find Similar Affordable Products")
    keywords = design_prompt.lower().split()
    if 'dress' in keywords or 'gown' in keywords:
        product_type = "Cocktail Dress"
    elif 'jacket' in keywords or 'coat' in keywords:
        product_type = "Trench Coat"
    elif 'shirt' in keywords or 'top' in keywords:
        product_type = "Silk Blouse"
    else:
        product_type = "Modern Apparel"

    st.info(f"**Simulated Search Result:** Your design matches **{product_type}**.")

    col1, col2, col3 = st.columns(3)
    products = [
        {"name": f"Affordable {product_type}", "price": "$49.99", "placeholder": "https://placehold.co/150x200/4c4c4c/ffffff?text=Product+A"},
        {"name": "Designer Lookalike", "price": "$65.00", "placeholder": "https://placehold.co/150x200/6b6b6b/ffffff?text=Product+B"},
        {"name": "Clearance Item", "price": "$29.95", "placeholder": "https://placehold.co/150x200/8d8d8d/ffffff?text=Product+C"},
    ]

    for i, (col, product) in enumerate(zip([col1, col2, col3], products)):
        with col:
            st.image(product['placeholder'], caption=product['name'], use_column_width=True)
            st.markdown(f"**{product['price']}**")
            st.button(f"Buy Now (P{i+1})", key=f"buy_{i}", use_container_width=True)

# --- Streamlit App ---
def app():
    st.set_page_config(page_title="AI Fashion Studio", layout="wide")
    st.title("👗 AI Fashion Design Studio")
    st.caption("Generate unique garment designs using Google's Imagen 3.0 model.")

    with st.form(key='design_form'):
        prompt = st.text_area(
            "Describe your dream garment or collection piece:",
            placeholder="A futuristic, asymmetrical silk gown in deep emerald green...",
            height=150
        )

        col_ar, col_num = st.columns(2)
        with col_ar:
            aspect_ratio = st.selectbox(
                "Select Design Aspect Ratio:",
                options=["1:1 (Square)", "3:4 (Portrait)", "16:9 (Landscape)"],
                index=1
            ).split(" ")[0]

        with col_num:
            num_images = st.slider("Number of Images to Generate (Max 4):", 1, 4, 1)

        submit_button = st.form_submit_button(label='✨ Generate Design', type="primary")

    if submit_button and prompt:
        with st.spinner(f'🎨 Drafting {num_images} designs...'):
            b64_images, error = generate_image(prompt, aspect_ratio, num_images)
            st.session_state['generated_images_b64'] = b64_images
            st.session_state['generation_error'] = error
            st.session_state['last_prompt'] = prompt
            st.session_state['last_aspect_ratio'] = aspect_ratio
            st.session_state['last_num_images'] = num_images

    st.session_state.setdefault('generated_images_b64', None)
    st.session_state.setdefault('generation_error', None)
    st.session_state.setdefault('last_prompt', None)
    st.session_state.setdefault('last_aspect_ratio', "1:1")
    st.session_state.setdefault('last_num_images', 1)

    if st.session_state['generation_error']:
        st.error(f"Design Generation Error: {st.session_state['generation_error']}")

    if st.session_state['generated_images_b64']:
        b64_images = st.session_state['generated_images_b64']
        st.subheader(f"Your AI-Generated Designs ({len(b64_images)} Options)")

        cols = st.columns(len(b64_images))
        for i, b64_data in enumerate(b64_images):
            try:
                image_bytes = base64.b64decode(b64_data)
                img = Image.open(BytesIO(image_bytes))
                with cols[i]:
                    st.image(img, caption=f"Option {i+1}", use_column_width=True)
                    st.download_button(
                        label=f"📥 Download {i+1}",
                        data=image_bytes,
                        file_name=f"ai_fashion_design_option_{i+1}.jpg",
                        mime="image/jpeg",
                        key=f"download_{i}"
                    )
            except Exception as e:
                with cols[i]:
                    st.error(f"Error decoding image {i+1}: {e}")

        render_similar_products_simulation(st.session_state['last_prompt'])

if __name__ == '__main__':
    app()

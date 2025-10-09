import streamlit as st
import requests
import base64
import json
import time
from io import BytesIO
from PIL import Image

# --- Firebase Placeholder (Mandatory Context setup) ---
# Since this is a standalone Python script for Streamlit, we define these 
# variables as placeholders. In a web environment, they would be injected.
__app_id = "ai-fashion-design-app"
__firebase_config = "{}"
__initial_auth_token = ""

# --- Configuration ---
# API Key is expected to be provided by the runtime environment 
# (e.g., set via environment variable or provided by the canvas).
# We leave it empty here as per instructions for Canvas/Immersive environments.
API_KEY = "" # The Canvas environment will inject the key

# Model and Endpoint Configuration
MODEL_NAME = "imagen-3.0-generate-002"
API_BASE_URL = "AIzaSyDODlUSG7KKovWIO9MpNIXIcHp_ZQ0oj30"
API_URL = f"{API_BASE_URL}{MODEL_NAME}:predict?key={API_KEY}"
MAX_RETRIES = 5

# --- Utility Functions ---

def generate_image(prompt, aspect_ratio="1:1", num_images=1, max_retries=MAX_RETRIES):
    """
    Calls the Imagen 3.0 API to generate a fashion image.
    Implements exponential backoff for request retries.
    
    Args:
        prompt (str): The text description of the fashion design.
        aspect_ratio (str): The aspect ratio (e.g., "1:1", "3:4").
        num_images (int): The number of images to generate (1 to 4).
    
    Returns:
        tuple: (list of base64 image strings, error message string or None)
    """
    
    # 1. Enhance the prompt for fashion design
    full_prompt = (
        f"A hyper-realistic, high-fashion runway photograph of a garment based on this description: "
        f"'{prompt}'. Focus on texture, stitching, and dramatic lighting. The background should be minimalist."
    )

    # 2. Construct the API Payload
    payload = {
        "instances": {
            "prompt": full_prompt,
        },
        "parameters": {
            "sampleCount": num_images, # Accepts the new parameter
            "aspectRatio": aspect_ratio,
            "outputMimeType": "image/jpeg"
        }
    }
    
    headers = {'Content-Type': 'application/json'}
    
    # 3. Request with Exponential Backoff
    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
            response.raise_for_status() # Raise exception for bad status codes (4xx or 5xx)
            
            result = response.json()
            
            # Check for generated image data - return the list of base64 images
            predictions = result.get('predictions', [])
            b64_images = [p['bytesBase64Encoded'] for p in predictions if p.get('bytesBase64Encoded')]
            
            if b64_images:
                return b64_images, None # Returns a list of base64 strings
            
            return None, "API returned successfully but no image data was found in the response structure. This may be due to safety filters."

        except requests.exceptions.HTTPError as e:
            # Handle specific HTTP errors (like 429 rate limit or 5xx server errors)
            st.warning(f"HTTP Error on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                delay = 2 ** attempt
                time.sleep(delay)
            else:
                return None, f"Failed after {max_retries} attempts due to HTTP error: {e}"
        except requests.exceptions.RequestException as e:
            # Handle connection errors, timeouts, etc.
            return None, f"A network error occurred: {e}"
        except Exception as e:
            # Handle JSON parsing errors or unexpected structure issues
            return None, f"An unexpected error occurred during API processing: {e}"

    return None, "Image generation request failed."


# --- Streamlit App ---

def render_similar_products_simulation(design_prompt):
    """Simulates a search for similar products based on the generated design."""
    st.markdown("---")
    st.subheader("🛍️ Find Similar Affordable Products")
    
    # Simple prompt analysis simulation
    keywords = design_prompt.lower().split()
    if 'dress' in keywords or 'gown' in keywords:
        product_type = "Cocktail Dress"
        link_text = "View Elegant Dresses"
    elif 'jacket' in keywords or 'coat' in keywords:
        product_type = "Trench Coat"
        link_text = "Browse Outerwear"
    elif 'shirt' in keywords or 'top' in keywords:
        product_type = "Silk Blouse"
        link_text = "Shop Tops"
    else:
        product_type = "Modern Apparel"
        link_text = "Explore New Arrivals"

    st.info(
        f"**Simulated Search Result:** We analyzed your design for a **{product_type}**. "
        f"In a real app, this would query e-commerce platforms using image features."
    )

    col1, col2, col3 = st.columns(3)
    
    # Placeholder product cards
    products = [
        {"id": 1, "name": f"Affordable {product_type}", "price": "$49.99", "link": "#", "placeholder": "https://placehold.co/150x200/4c4c4c/ffffff?text=Product+A"},
        {"id": 2, "name": "Designer Lookalike", "price": "$65.00", "link": "#", "placeholder": "https://placehold.co/150x200/6b6b6b/ffffff?text=Product+B"},
        {"id": 3, "name": "Clearance Item", "price": "$29.95", "link": "#", "placeholder": "https://placehold.co/150x200/8d8d8d/ffffff?text=Product+C"},
    ]
    
    for i, (col, product) in enumerate(zip([col1, col2, col3], products)):
        with col:
            st.image(product['placeholder'], caption=product['name'], use_column_width="always")
            st.markdown(f"**{product['price']}**")
            # Added a unique key for the button to prevent Streamlit warning
            st.button(f"Buy Now (P{i+1})", key=f"buy_{product['id']}_{i}", use_container_width=True)
            

def app():
    """Main Streamlit application function."""
    
    # 1. Page Configuration and Styling
    st.set_page_config(
        page_title="AI Fashion Studio",
        layout="wide",
        initial_sidebar_state="auto"
    )

    st.title("👗 AI Fashion Design Studio")
    st.caption("Generate unique garment designs using Google's Imagen 3.0 model.")

    # 2. Input Section
    with st.form(key='design_form'):
        prompt = st.text_area(
            "Describe your dream garment or collection piece:",
            placeholder="A futuristic, asymmetrical silk gown in deep emerald green, with metallic silver stitching and a geometric neckline.",
            height=150,
            key='user_prompt'
        )
        
        col_ar, col_num = st.columns(2) # New columns for Aspect Ratio and Count
        
        with col_ar:
            aspect_ratio = st.selectbox(
                "Select Design Aspect Ratio:",
                options=["1:1 (Square)", "3:4 (Portrait)", "16:9 (Landscape)"],
                index=1 # Default to portrait for fashion
            ).split(" ")[0]
            
        with col_num:
            num_images = st.slider(
                "Number of Images to Generate (Max 4):",
                min_value=1,
                max_value=4,
                value=1,
                step=1
            )

        submit_button = st.form_submit_button(
            label='✨ Generate Design', 
            type="primary"
        )

    # 3. Generation Logic
    if submit_button and prompt:
        with st.spinner(f'🎨 Drafting {num_images} designs... This may take up to 30 seconds per request.'):
            # Clear previous results
            st.session_state['generated_images_b64'] = None # Updated key name for clarity (list of images)
            st.session_state['generation_error'] = None
            
            b64_images, error = generate_image(prompt, aspect_ratio, num_images) # Pass num_images
            
            st.session_state['generated_images_b64'] = b64_images
            st.session_state['generation_error'] = error
            st.session_state['last_prompt'] = prompt
            st.session_state['last_aspect_ratio'] = aspect_ratio
            st.session_state['last_num_images'] = num_images

    # 4. Display Results
    
    # Initialize session state keys if they don't exist
    if 'generated_images_b64' not in st.session_state:
        st.session_state['generated_images_b64'] = None
    if 'generation_error' not in st.session_state:
        st.session_state['generation_error'] = None
    if 'last_prompt' not in st.session_state:
        st.session_state['last_prompt'] = None
    if 'last_aspect_ratio' not in st.session_state: 
        st.session_state['last_aspect_ratio'] = "1:1"
    if 'last_num_images' not in st.session_state:
        st.session_state['last_num_images'] = 1


    if st.session_state['generation_error']:
        st.error(f"Design Generation Error: {st.session_state['generation_error']}")
    
    if st.session_state['generated_images_b64']: 
        
        b64_images = st.session_state['generated_images_b64']
        
        st.subheader(f"Your AI-Generated Designs ({len(b64_images)} Options)")
        
        # Display meta-information once
        col_meta1, col_meta2, col_meta3 = st.columns(3)
        with col_meta1:
            st.info(f"**Prompt:** {st.session_state['last_prompt']}")
        with col_meta2:
            st.info(f"**Model:** `{MODEL_NAME}`")
        with col_meta3:
            st.info(f"**Aspect Ratio:** `{st.session_state['last_aspect_ratio']}`")

        # Create columns for the generated images (up to 4)
        cols = st.columns(len(b64_images))
        
        for i, b64_data in enumerate(b64_images):
            try:
                # Decode base64 string to bytes
                image_bytes = base64.b64decode(b64_data)
                
                # Use BytesIO and PIL to open the image
                image_file = BytesIO(image_bytes)
                img = Image.open(image_file)
                
                with cols[i]:
                    st.image(img, caption=f"Option {i+1}", use_column_width=True)
                    
                    # Add a download button for each image
                    st.download_button(
                        label=f"📥 Download {i+1}",
                        data=image_bytes,
                        file_name=f"ai_fashion_design_option_{i+1}.jpg",
                        mime="image/jpeg",
                        key=f"download_{i}", # Unique key for each button
                        use_container_width=True
                    )
            
            except Exception as e:
                with cols[i]:
                    st.error(f"Error decoding image {i+1}: {e}")
                    st.info("Check the console for API response details.")

        # 5. Simulated E-commerce Search (Use the last prompt)
        st.markdown("---")
        render_similar_products_simulation(st.session_state['last_prompt'])


if __name__ == '__main__':
    # Streamlit app startup
    app()

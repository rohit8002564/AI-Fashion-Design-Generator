import streamlit as st
from PIL import Image
from io import BytesIO
import torch
from diffusers import StableDiffusionPipeline
import os
import random

# -------------------------
# --- Load Stable Diffusion Model ---
# -------------------------
@st.cache_resource(show_spinner=True)
def load_model():
    try:
        st.write("🔄 Loading Stable Diffusion model... This may take a minute...")
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None # Disable NSFW filter blocking images silently
        )
        # Safety override
        pipe.safety_checker = lambda images, **kwargs: (images, False)
        
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
            st.success("✅ Model loaded on GPU!")
        else:
            st.info("⚙️ Running on CPU mode (slower but works fine).")
        return pipe
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

pipe = load_model()

# -------------------------
# --- AI Image Generation ---
# -------------------------
def generate_fashion_images(prompt, negative_prompt="", num_images=5, steps=50, scale=7.5):
    """
    Generates fashion images using the Stable Diffusion pipeline with custom parameters.
    :param prompt: The positive text prompt.
    :param negative_prompt: The prompt describing elements to avoid.
    :param num_images: Number of images to generate (must be <= 5 for columns).
    :param steps: Number of inference steps (quality/time control).
    :param scale: Classifier-free guidance scale (prompt adherence control).
    """
    images = []
    for _ in range(num_images):
        try:
            # Prepare arguments for the pipeline call
            pipeline_kwargs = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "num_inference_steps": steps,
                "guidance_scale": scale,
            }

            if torch.cuda.is_available():
                with torch.autocast("cuda"):
                    image = pipe(**pipeline_kwargs).images[0]
            else:
                image = pipe(**pipeline_kwargs).images[0]
            images.append(image)
        except Exception as e:
            st.error(f"Image generation failed: {e}")
            break
    return images

# -------------------------
# --- Similar Products Mock ---
# -------------------------
def get_similar_products(prompt):
    """
    Mocks similar products based on keywords found in the user's prompt.
    """
    prompt_lower = prompt.lower()
    
    # Define keywords for customization
    colors = ["red", "blue", "green", "black", "white", "yellow", "pink"]
    types = ["dress", "shirt", "pants", "jacket", "skirt", "suit", "sneakers"]
    styles = ["floral", "sequins", "denim", "wool", "silk", "leather", "cotton"]
    
    found_color = next((c for c in colors if c in prompt_lower), "navy")
    found_type = next((t for t in types if t in prompt_lower), "item")
    found_style = next((s for s in styles if s in prompt_lower), "classic")
    
    # Generate dynamic mock products
    products = []
    
    # Product 1: Directly matches main keywords
    title_1 = f"{found_color.capitalize()} {found_style.capitalize()} {found_type.capitalize()}"
    products.append({"title": title_1, "link": "https://example.com/match1", "price": f"${random.randint(40, 150)}"})

    # Product 2: Uses a related color and different style
    related_color = random.choice([c for c in colors if c != found_color] + [found_color])
    related_style = random.choice([s for s in styles if s != found_style] + [found_style])
    title_2 = f"{related_color.capitalize()} {related_style.capitalize()} {found_type.capitalize()}"
    products.append({"title": title_2, "link": "https://example.com/match2", "price": f"${random.randint(50, 160)}"})

    # Product 3: Different item type and style
    different_type = random.choice([t for t in types if t != found_type] + [found_type])
    title_3 = f"Affordable {different_type.capitalize()} Collection"
    products.append({"title": title_3, "link": "https://example.com/match3", "price": f"${random.randint(30, 99)}"})
    
    return products

# -------------------------
# --- Streamlit UI ---
# -------------------------
st.set_page_config(page_title="AI Fashion Design Generator", layout="wide")

st.title("🎨 AI Fashion Design Generator")
st.write("Type a fashion design idea below and generate **AI-based clothing designs instantly!**")

# --- UI CONTROLS: Sidebar for Advanced Settings ---
with st.sidebar:
    st.header("⚙️ Advanced Generation Settings")
    
    # 1. Negative Prompt 
    negative_prompt = st.text_area(
        "🚫 Negative Prompt (What to Exclude)",
        value="blurry, low quality, deformed, extra limbs, bad anatomy, ugly, tiling, cropped",
        help="Enter details you want the model to avoid in the image."
    )

    st.markdown("---")

    # 2. Number of Images (Max 5 for current column layout)
    num_images = st.slider(
        "Number of Designs to Generate",
        min_value=1, max_value=5, value=3, step=1,
        help="Controls how many designs are created in one batch (max 5)."
    )

    # 3. Inference Steps
    inference_steps = st.slider(
        "Quality (Inference Steps)",
        min_value=20, max_value=100, value=50, step=5,
        help="Higher values increase quality but significantly increase generation time."
    )

    # 4. Guidance Scale
    guidance_scale = st.slider(
        "Prompt Adherence (Guidance Scale)",
        min_value=1.0, max_value=20.0, value=7.5, step=0.5,
        help="Higher values make the image follow the prompt more strictly, but can sometimes look less natural."
    )
    st.markdown("---")
    st.info("Adjust these settings to refine your AI designs!")

# --- Main Prompt Input ---
prompt = st.text_input("💬 Enter your fashion design prompt (e.g., 'A red floral evening gown with sequins')")

if st.button("✨ Generate Designs"):
    if not pipe:
        st.error("Model not loaded properly. Please restart the app.")
    elif prompt.strip() == "":
        st.warning("Please enter a valid prompt!")
    else:
        with st.spinner(f"🧵 Generating {num_images} designs... Please wait (time depends on steps)..."):
            # Pass custom parameters, including the new negative prompt, to the generation function
            images = generate_fashion_images(
                prompt,
                negative_prompt=negative_prompt, 
                num_images=num_images,
                steps=inference_steps,
                scale=guidance_scale
            )

            if not images:
                st.error("No images generated! Try another prompt or check your setup.")
            else:
                save_folder = "generated_designs"
                os.makedirs(save_folder, exist_ok=True)

                st.subheader("👗 Generated Fashion Designs:")
                # Determine columns based on the number of images generated
                cols = st.columns(num_images if num_images <= 3 else 3)
                
                for idx, image in enumerate(images):
                    with cols[idx % len(cols)]:
                        st.image(image, caption=f"Design {idx+1}", use_column_width=True)

                        # Save locally (Note: This is mock save functionality in Streamlit environments)
                        image_path = os.path.join(save_folder, f"design_{idx+1}.png")
                        image.save(image_path)

                        # Download button
                        buf = BytesIO()
                        image.save(buf, format="PNG")
                        st.download_button(
                            label=f"Download Design {idx+1}",
                            data=buf.getvalue(),
                            file_name=f"fashion_design_{idx+1}.png",
                            mime="image/png"
                        )

                # Show similar products
                st.subheader("🛍️ Similar Affordable Products:")
                products = get_similar_products(prompt)
                for prod in products:
                    st.markdown(f"[{prod['title']}]({prod['link']}) — **{prod['price']}**")

st.markdown("---")
st.caption("Made with ❤️ using Streamlit and Stable Diffusion")

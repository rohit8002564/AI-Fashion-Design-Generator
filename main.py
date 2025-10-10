   import streamlit as st
   import base64
   import json
   import time
   from io import BytesIO
   from PIL import Image
   import os
   from google.cloud import aiplatform  # New import
   from google.oauth2 import service_account  # For auth if needed

   # --- Configuration ---
   PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID")  # Set in env (e.g., "your-project-123")
   LOCATION = "us-central1"  # Or "europe-west4" etc.
   MODEL_ID = "imagegeneration@006"  # Imagen 3 model ID

   # Initialize Vertex AI (runs once)
   def init_vertex_ai():
       if not PROJECT_ID:
           raise ValueError("Set GOOGLE_PROJECT_ID env var")
       aiplatform.init(project=PROJECT_ID, location=LOCATION)

   # --- Updated Utility Function ---
   def generate_image(prompt, aspect_ratio="1:1", num_images=1, max_retries=5):
       init_vertex_ai()  # Ensure initialized

       full_prompt = (
           f"A hyper-realistic, high-fashion runway photograph of a garment: '{prompt}'. "
           "Focus on texture, stitching, dramatic lighting, and a minimalist background."
       )

       # Map aspect ratio (Imagen uses different enums)
       aspect_map = {
           "1:1": aiplatform.gapic.schema_pb2.AspectRatio.ASPECT_RATIO_1_1,
           "3:4": aiplatform.gapic.schema_pb2.AspectRatio.ASPECT_RATIO_3_4,
           "16:9": aiplatform.gapic.schema_pb2.AspectRatio.ASPECT_RATIO_16_9,
       }
       aspect = aspect_map.get(aspect_ratio, aiplatform.gapic.schema_pb2.AspectRatio.ASPECT_RATIO_1_1)

       parameters = {
           "prompt": full_prompt,
           "number_of_images": num_images,
           "aspect_ratio": aspect,
           "safety_filter_level": "block_some",  # Adjust for safety
           "add_watermark": False,
       }

       endpoint = aiplatform.Endpoint.external(f"projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{MODEL_ID}")

       for attempt in range(max_retries):
           try:
               response = endpoint.predict(instances=[parameters])
               predictions = response.predictions

               if predictions:
                   b64_images = []
                   for pred in predictions:
                       # Imagen returns bytes; encode to base64
                       image_bytes = pred['bytesBase64Encoded']  # Adjust key based on actual response
                       b64_images.append(image_bytes)
                   return b64_images, None

               return None, "API returned no images. Check prompt or safety filters."

           except Exception as e:
               if attempt < max_retries - 1:
                   time.sleep(2 ** attempt)
               else:
                   return None, f"Error after {max_retries} attempts: {e}"

       return None, "Image generation failed after retries."

   # --- Rest of your app code remains the same ---
   # (Keep render_simulated_products_simulation, app(), etc.)
   # In app(), call generate_image as before.

   if __name__ == '__main__':
       app()
   

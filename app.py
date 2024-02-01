import streamlit as st
import os
from vertexai.preview.generative_models import GenerativeModel, Part
import vertexai

# Initialize Vertex AI with your project and location
PROJECT_ID = os.environ.get("GCP_PROJECT")  # Your Google Cloud Project ID
LOCATION = os.environ.get("GCP_REGION")  # Your Google Cloud Project Region
vertexai.init(project=PROJECT_ID, location=LOCATION)

# Function to load the Gemini Pro Vision model
@st.cache_resource
def load_model():
    return GenerativeModel("gemini-pro-vision")

# Function to display a video from a GCS URI and create a Part object
def display_video_and_create_part(gcs_uri):
    if gcs_uri:
        video_url = "https://storage.googleapis.com/" + gcs_uri.split("gs://")[1]
        st.video(video_url)
        return Part.from_uri(gcs_uri, mime_type="video/mp4")
    return None

# Function to generate and display content
def generate_and_display_content(model, prompt, video_part, key):
    if video_part and st.button("Generate", key=key):
        with st.spinner("Generating..."):
            response = model.generate_content([prompt, video_part], generation_config={"temperature": 0.1, "max_output_tokens": 2048}, stream=True)
            final_response = " ".join([resp.text for resp in response if resp.text])
            st.write(final_response)

def main():
    st.header("Paigeon AI Video Analyzer")
    multimodal_model = load_model()

    # User input for GCS URI
    user_input_uri = st.text_input("Enter the Google Cloud Storage URI for your video:", "gs://your-bucket/path/to/your_video.mp4")

    # Display video and create Part object if URI is provided
    video_part = None
    if user_input_uri:
        video_part = display_video_and_create_part(user_input_uri)

    # Tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs(["Video Description", "Video Tags", "Video Highlights", "Video Shopping Objects"])

    # Video Description
    with tab1:
        st.subheader("Generate Video Description")
        prompt = "Describe what is happening in this video."
        generate_and_display_content(multimodal_model, prompt, video_part, key="video_description")

    # Video Tags
    with tab2:
        st.subheader("Generate Video Tags")
        prompt = "Generate tags for this video followed by '#'"
        generate_and_display_content(multimodal_model, prompt, video_part, key="video_tags")

    # Video Highlights
    with tab3:
        st.subheader("Generate Video Highlights")
        prompt = "Summarize the key highlights of this video."
        generate_and_display_content(multimodal_model, prompt, video_part, key="video_highlights")

    # Video Geolocation
    with tab4:
        st.subheader("Generate Video Shopping Object")
        prompt = "Identify the objects present in this video which can be used for online shopping"
        generate_and_display_content(multimodal_model, prompt, video_part, key="video_geolocation")

if __name__ == "__main__":
    main()

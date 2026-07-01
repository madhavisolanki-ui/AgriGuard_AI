"""Streamlit user interface for the AgriGuard prediction API."""

from __future__ import annotations

import base64
import os
from typing import Any

import requests
import streamlit as st
from dotenv import load_dotenv


DEFAULT_API_URL = "http://localhost:8000"

load_dotenv()


def encode_image_to_base64(file_bytes: bytes) -> str:
    """Encode raw image bytes into a base64 string."""
    return base64.b64encode(file_bytes).decode("utf-8")


def submit_prediction(
    api_url: str,
    image_bytes: bytes,
    filename: str,
    content_type: str,
) -> dict[str, Any]:
    """Submit an image to the FastAPI prediction endpoint."""
    payload = {
        "image_base64": encode_image_to_base64(image_bytes),
        "filename": filename,
        "content_type": content_type,
    }
    response = requests.post(
        f"{api_url.rstrip('/')}/predict",
        json=payload,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def main() -> None:
    """Render the Streamlit application."""
    st.set_page_config(page_title="AgriGuard", page_icon="AG", layout="centered")

    st.title("AgriGuard")
    st.write("Upload a crop image to get a prediction from the FastAPI backend.")

    api_url = st.text_input(
        "API URL",
        value=os.getenv("STREAMLIT_API_URL", DEFAULT_API_URL),
    )
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp"],
    )

    if uploaded_file is not None:
        st.image(uploaded_file, caption=uploaded_file.name, use_container_width=True)

        if st.button("Get Prediction"):
            try:
                with st.spinner("Sending image to the API..."):
                    result = submit_prediction(
                        api_url=api_url,
                        image_bytes=uploaded_file.getvalue(),
                        filename=uploaded_file.name,
                        content_type=uploaded_file.type or "image/jpeg",
                    )

                st.success("Prediction received successfully.")
                st.subheader("Result")
                st.json(result)
            except requests.RequestException as exc:
                st.error(f"Failed to contact the prediction API: {exc}")
            except ValueError as exc:
                st.error(f"Invalid response received from the API: {exc}")


if __name__ == "__main__":
    main()

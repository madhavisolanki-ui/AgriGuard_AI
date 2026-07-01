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


def _render_probability_bar(label: str, score: float, is_top: bool) -> None:
    """Render a probability bar with a subtle highlight for the top class."""
    display_label = label.replace("_", " ").title()
    prefix = "TOP " if is_top else ""
    st.write(f"{prefix}{display_label} - {score:.2%}")
    st.progress(min(max(score, 0.0), 1.0))


def main() -> None:
    """Render the Streamlit application."""
    st.set_page_config(page_title="AgriGuard", page_icon="AG", layout="wide")

    st.markdown(
        """
        <style>
            .agri-hero {
                padding: 1.5rem 1.8rem;
                border-radius: 1.25rem;
                background: linear-gradient(135deg, #0f172a 0%, #14532d 100%);
                color: white;
                box-shadow: 0 12px 30px rgba(15, 23, 42, 0.2);
                margin-bottom: 1.25rem;
            }
            .agri-card {
                padding: 1rem;
                border: 1px solid rgba(148, 163, 184, 0.25);
                border-radius: 1rem;
                background: white;
                box-shadow: 0 8px 20px rgba(15, 23, 42, 0.06);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="agri-hero">
            <h1 style="margin-bottom: 0.25rem;">AgriGuard</h1>
            <p style="margin: 0; opacity: 0.9;">
                Upload a crop image and get a fast, model-backed prediction from the API.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    api_url = st.sidebar.text_input(
        "API URL",
        value=os.getenv("STREAMLIT_API_URL", DEFAULT_API_URL),
    )
    st.sidebar.caption("The backend should be running before you submit an image.")

    left_col, right_col = st.columns([1.15, 0.85], gap="large")

    with left_col:
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png", "webp"],
        )

        if uploaded_file is not None:
            st.markdown('<div class="agri-card">', unsafe_allow_html=True)
            st.image(
                uploaded_file,
                caption=uploaded_file.name,
                use_container_width=True,
            )
            st.caption(f"Detected file type: {uploaded_file.type or 'image/jpeg'}")

            run_prediction = st.button("Get Prediction", type="primary")

            if run_prediction:
                try:
                    with st.spinner("Sending image to the API..."):
                        result = submit_prediction(
                            api_url=api_url,
                            image_bytes=uploaded_file.getvalue(),
                            filename=uploaded_file.name,
                            content_type=uploaded_file.type or "image/jpeg",
                        )

                    st.success("Prediction received successfully.")
                    st.session_state["prediction_result"] = result
                except requests.RequestException as exc:
                    st.error(f"Failed to contact the prediction API: {exc}")
                except ValueError as exc:
                    st.error(f"Invalid response received from the API: {exc}")

            st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        result = st.session_state.get("prediction_result")
        st.subheader("Prediction Details")

        if result:
            predicted_class = result.get("predicted_class", "unknown")
            confidence = float(result.get("confidence", 0.0))
            probabilities = result.get("probabilities", {})

            metric_left, metric_right = st.columns(2)
            metric_left.metric(
                "Predicted Class",
                predicted_class.replace("_", " ").title(),
            )
            metric_right.metric("Confidence", f"{confidence:.2%}")

            st.progress(confidence)

            st.markdown("#### Class Probabilities")
            sorted_probabilities = sorted(
                probabilities.items(),
                key=lambda item: item[1],
                reverse=True,
            )
            for index, (label, score) in enumerate(sorted_probabilities):
                _render_probability_bar(label, float(score), is_top=index == 0)

            st.markdown("#### Raw API Response")
            st.json(result)
        else:
            st.info("Submit an image to see prediction confidence and class breakdown here.")


if __name__ == "__main__":
    main()

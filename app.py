import streamlit as st
import numpy as np
import io

from src.preprocessing import (
    load_image,
    image_to_pixels,
    pixels_to_image
)

from src.compression import (
    compress_image,
    save_model,
    calculate_elbow
)

from src.metrics import (
    calculate_mse,
    calculate_psnr,
    calculate_ssim
)


# --------------------------------
# Page Configuration
# --------------------------------

st.set_page_config(
    page_title="AI Image Color Compression",
    page_icon="🖼️",
    layout="wide"
)


# --------------------------------
# Title
# --------------------------------

st.title("🖼️ AI-Based Image Color Compression")

st.write(
    "Reduce image colors using K-Means clustering "
    "while preserving visual quality."
)


# --------------------------------
# Upload Image
# --------------------------------

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    # --------------------------------
    # Load Image
    # --------------------------------

    image = load_image(uploaded_file)

    # Original file size
    original_size = len(
        uploaded_file.getvalue()
    )


    # --------------------------------
    # Compression Settings
    # --------------------------------

    st.subheader("⚙️ Compression Settings")

    n_colors = st.slider(
        "Number of Colors",
        min_value=2,
        max_value=256,
        value=16,
        step=2
    )

    st.info(
        f"Image will be compressed to approximately "
        f"{n_colors} representative colors."
    )


    # --------------------------------
    # Original Image
    # --------------------------------

    st.subheader("🖼️ Original Image")

    st.image(
        image,
        use_container_width=True
    )


    # --------------------------------
    # Convert Image to Pixels
    # --------------------------------

    pixels, height, width = image_to_pixels(
        image
    )


    # ================================================
    # COMPRESS IMAGE
    # ================================================

    if st.button("🚀 Compress Image"):

        with st.spinner(
            "Running K-Means clustering..."
        ):

            # Train K-Means
            compressed_pixels, model = compress_image(
                pixels,
                n_colors
            )

            # Convert pixels back to image
            compressed_image = pixels_to_image(
                compressed_pixels,
                height,
                width
            )


        # --------------------------------
        # Save Trained Model
        # --------------------------------

        save_model(
            model,
            "models/kmeans_model.pkl"
        )

        st.success(
            "K-Means model trained and saved successfully!"
        )


        # --------------------------------
        # Calculate Metrics
        # --------------------------------

        mse = calculate_mse(
            image,
            compressed_image
        )

        psnr = calculate_psnr(
            image,
            compressed_image
        )

        ssim_score = calculate_ssim(
            image,
            compressed_image
        )


        # --------------------------------
        # Save Compressed Image in Memory
        # --------------------------------

        buffer = io.BytesIO()

        compressed_image.save(
            buffer,
            format="PNG",
            optimize=True
        )

        compressed_data = buffer.getvalue()

        compressed_size = len(
            compressed_data
        )


        # --------------------------------
        # Size Reduction
        # --------------------------------

        size_reduction = (
            (
                original_size - compressed_size
            )
            / original_size
        ) * 100


        # ================================================
        # IMAGE COMPARISON
        # ================================================

        st.subheader("🔍 Image Comparison")

        col1, col2 = st.columns(2)


        with col1:

            st.write("### Original")

            st.image(
                image,
                use_container_width=True
            )


        with col2:

            st.write("### Compressed")

            st.image(
                compressed_image,
                use_container_width=True
            )


        # ================================================
        # METRICS
        # ================================================

        st.subheader(
            "📊 Compression Results"
        )

        col1, col2, col3, col4 = st.columns(4)


        with col1:

            st.metric(
                "Colors",
                n_colors
            )


        with col2:

            st.metric(
                "MSE",
                f"{mse:.2f}"
            )


        with col3:

            st.metric(
                "PSNR",
                f"{psnr:.2f} dB"
            )


        with col4:

            st.metric(
                "SSIM",
                f"{ssim_score:.4f}"
            )


        # ================================================
        # FILE SIZE
        # ================================================

        st.subheader("💾 File Size")

        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Original",
                f"{original_size / 1024:.2f} KB"
            )


        with col2:

            st.metric(
                "Compressed",
                f"{compressed_size / 1024:.2f} KB"
            )


        with col3:

            st.metric(
                "Size Reduction",
                f"{size_reduction:.2f}%"
            )


        # ================================================
        # COLOR PALETTE
        # ================================================

        st.subheader(
            "🎨 Generated Color Palette"
        )

        centers = model.cluster_centers_


        palette = np.uint8(
            np.clip(
                centers,
                0,
                255
            )
        )


        palette_width = 800
        palette_height = 100

        color_width = (
            palette_width // len(palette)
        )


        palette_image = np.zeros(
            (
                palette_height,
                color_width * len(palette),
                3
            ),
            dtype=np.uint8
        )


        for i, color in enumerate(palette):

            start = i * color_width
            end = start + color_width

            palette_image[
                :,
                start:end
            ] = color


        st.image(
            palette_image,
            caption=(
                f"{n_colors} Representative Colors"
            ),
            use_container_width=True
        )


        # ================================================
        # DOWNLOAD COMPRESSED IMAGE
        # ================================================

        st.download_button(
            label="⬇️ Download Compressed Image",
            data=compressed_data,
            file_name="compressed_image.png",
            mime="image/png"
        )


    # ================================================
    # ELBOW METHOD
    # ================================================

    st.subheader("📈 Elbow Method Analysis")


    if st.checkbox(
        "Show Elbow Analysis"
    ):

        with st.spinner(
            "Calculating optimal K values..."
        ):

            k_values, inertias = calculate_elbow(
                pixels
            )


        # Create chart data
        chart_data = {
            "K": k_values,
            "Inertia": inertias
        }


        # Display chart
        st.line_chart(
            chart_data,
            x="K",
            y="Inertia"
        )


        st.write(
            "As the number of clusters (K) increases, "
            "inertia generally decreases. The elbow region "
            "can help identify a reasonable number of "
            "color clusters."
        )


        # Display values
        st.write("### Inertia Values")

        for k, inertia in zip(
            k_values,
            inertias
        ):

            st.write(
                f"K = {k} → "
                f"Inertia = {inertia:.2f}"
            )
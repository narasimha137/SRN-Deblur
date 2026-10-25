
import os
import sys
import tempfile
import subprocess
from pathlib import Path

import streamlit as st
from PIL import Image

# Repository paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_SCRIPT = BASE_DIR / "run_model.py"
DEFAULT_DATALIST = BASE_DIR / "datalist_gopro.txt"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

st.set_page_config(
    page_title="SRN Image Deblurring",
    page_icon="🖼️",
    layout="wide",
)

st.title("SRN-Deblur: Image Deblurring")
st.write(
    "Upload a blurry image and use the pretrained "
    "Scale-Recurrent Network to restore it."
)

# The original repository uses legacy TensorFlow.
# Set SRN_PYTHON if its compatible Python interpreter
# is not available under the name python2.
python_executable = os.environ.get("SRN_PYTHON", "python2")

with st.sidebar:
    st.header("Model settings")

    model_type = st.selectbox(
        "Model type",
        ["color", "gray", "lstm"],
        index=0,
        help="Choose the model matching your downloaded checkpoint.",
    )

    gpu_id = st.selectbox(
        "Device",
        ["-1", "0"],
        format_func=lambda x: "CPU" if x == "-1" else "GPU 0",
    )

    height = st.selectbox("Model height", [256, 480, 720, 960], index=2)
    width = st.selectbox("Model width", [256, 640, 1280, 1920], index=2)

    st.caption("Height and width must be multiples of 16.")

uploaded_file = st.file_uploader(
    "Choose a blurry image",
    type=["png", "jpg", "jpeg", "bmp"],
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        left, right = st.columns(2)

        with left:
            st.subheader("Original image")
            st.image(image, use_container_width=True)

        if st.button("Remove blur", type="primary"):
            if not MODEL_SCRIPT.is_file():
                st.error("Cannot find run_model.py in the repository.")
                st.stop()

            if not DEFAULT_DATALIST.is_file():
                st.error(
                    "Cannot find datalist_gopro.txt. "
                    "The original model may require this file even during testing."
                )
                st.stop()

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                input_dir = temp_path / "input"
                output_dir = temp_path / "output"

                input_dir.mkdir()
                output_dir.mkdir()

                input_image = input_dir / "input.png"
                image.save(input_image)

                command = [
                    python_executable,
                    str(MODEL_SCRIPT),
                    "--phase=test",
                    f"--model={model_type}",
                    f"--gpu={gpu_id}",
                    f"--height={height}",
                    f"--width={width}",
                    f"--input_path={input_dir}",
                    f"--output_path={output_dir}",
                    f"--datalist={DEFAULT_DATALIST}",
                ]

                with st.spinner(
                    "Running SRN-Deblur. First-time model loading may take a while..."
                ):
                    try:
                        result = subprocess.run(
                            command,
                            cwd=str(BASE_DIR),
                            capture_output=True,
                            text=True,
                            timeout=900,
                            check=False,
                        )
                    except FileNotFoundError:
                        st.error(
                            f"Python interpreter '{python_executable}' was not found. "
                            "Install the compatible legacy environment or set "
                            "SRN_PYTHON to its interpreter path."
                        )
                        st.stop()
                    except subprocess.TimeoutExpired:
                        st.error(
                            "Inference timed out. Try a smaller model height and width."
                        )
                        st.stop()

                if result.returncode != 0:
                    st.error("The SRN-Deblur model failed.")
                    with st.expander("Show error details"):
                        st.code(result.stderr or result.stdout)
                    st.stop()

                # Locate the generated image. The original model
                # may save using a filename derived from the input.
                output_images = [
                    p for p in output_dir.iterdir()
                    if p.suffix.lower() in
                    [".png", ".jpg", ".jpeg", ".bmp"]
                ]

                if not output_images:
                    st.error(
                        "No restored image was generated. Check that pretrained "
                        "checkpoints are installed and compatible with the selected "
                        "model. See the terminal output below."
                    )
                    with st.expander("Show model output"):
                        st.code(result.stdout or "No standard output.")
                        st.code(result.stderr or "No error output.")
                    st.stop()

                restored_path = output_images[0]
                restored_image = Image.open(restored_path).convert("RGB")

                with right:
                    st.subheader("Deblurred image")
                    st.image(restored_image, use_container_width=True)

                # Keep the image in memory before the temporary folder closes.
                from io import BytesIO

                buffer = BytesIO()
                restored_image.save(buffer, format="PNG")

                st.download_button(
                    label="Download restored image",
                    data=buffer.getvalue(),
                    file_name="deblurred.png",
                    mime="image/png",
                )

    except Exception as exc:
        st.error(f"Could not process the image: {exc}")

st.divider()
st.caption("Powered by SRN-Deblur | Scale-Recurrent Network")

# Installs dependencies in the order that avoids compiling dlib from source.
# See README.md for why this order matters.
pip install dlib-bin
pip install face_recognition --no-deps
pip install git+https://github.com/ageitgey/face_recognition_models
pip install "setuptools<81"
pip install click Pillow numpy opencv-python-headless streamlit

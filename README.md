<img width="1919" height="995" alt="Captură de ecran 2026-09-25 174823" src="https://github.com/user-attachments/assets/25012ca8-389a-4a45-98bb-59076a68966b" />
# Pediatric Pneumonia Detection Using AI 🫁🤖

This repository contains a deep learning-based web application designed to assist in diagnosing pediatric pneumonia from chest X-ray images. This system was developed as part of a Bachelor's degree thesis in Automation and Applied Informatics.

## 🌟 Key Features

*   **Advanced Image Classification**: Utilizes state-of-the-art Convolutional Neural Network (CNN) architectures, specifically **ResNet50** and **DenseNet121**, to accurately classify pediatric chest X-rays.
*   **Explainable AI (XAI)**: Integrates **Grad-CAM** (Gradient-weighted Class Activation Mapping) to generate heatmaps. This provides visual explanations of the model's diagnostic decisions by highlighting the critical regions in the X-rays that influenced the prediction.
*   **User-Friendly Web Interface**: Features a web front-end built with **Flask** (as seen in the preview above), allowing for secure access, clinical triage support, and real-time visualization of results.
*   **Reliable Data Storage**: Uses an **SQLite** database to manage and store diagnostic records efficiently.

## 🛠️ Technology Stack

*   **Backend & Machine Learning**: Python, Flask, Deep Learning Frameworks
*   **Computer Vision**: OpenCV (for image processing and Grad-CAM generation)
*   **Database**: SQLite
*   **Frontend**: HTML/CSS

## 📂 Repository Structure

*   `app.py`: The main Flask application script handling routing, image processing, and model inference.
*   `utils.py`: Utility functions and helper scripts (e.g., Grad-CAM generation algorithms, image preprocessing).
*   `templates/`: Directory containing HTML templates for the web interface.

## 🚀 Installation and Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/laura-ion2/Pediatric-Pneumonia-Detection-AI.git](https://github.com/laura-ion2/Pediatric-Pneumonia-Detection-AI.git)
    cd Pediatric-Pneumonia-Detection-AI
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    *(Note: Ensure a `requirements.txt` file is present in the repository)*
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the application:**
    ```bash
    python app.py
    ```
    Access the web interface at `http://127.0.0.1:5000` in your web browser.

## 👨‍💻 Author

**Laura Ion** ([@laura-ion2](https://github.com/laura-ion2))

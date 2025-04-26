# DRIP – Dynamic Root Locus Integration Platform

**DRIP** is an interactive web-based application built using Streamlit that allows users to perform dynamic Root Locus analysis on control systems.

## 🚀 Features
- Interactive Root Locus Visualization
- Dynamic System Behavior Examination
- Immediate Stability Assessment
- Gain Adjustment and Impact Analysis
- Step and Ramp Response Analysis
- Detailed Pole-Zero Information

## 📸 Preview
*(Insert screenshots or gifs if you want, inside `/assets` folder)*

## 🛠️ How to Run Locally

1. Clone this repository:
    ```bash
    git clone https://github.com/your-username/drip-root-locus-app.git
    cd drip-root-locus-app
    ```

2. Create a virtual environment (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the Streamlit app:
    ```bash
    streamlit run ff.py
    ```

## 🧰 Requirements
- Python 3.8+
- Streamlit
- NumPy
- Matplotlib
- Control
- Pandas

(Full list in `requirements.txt`.)

## 📁 Folder Structure
```bash
.
├── ff.py              # Main Streamlit App
├── requirements.txt   # Dependencies
├── README.md          # Project description
├── .gitignore         # Common ignores
└── assets/            # Optional images/icons

# DRIP – Dynamic Root Locus Integration Platform

**DRIP** is a web application built using Streamlit that enables users to interactively perform Root Locus analysis of control systems.  
It allows visualization, stability assessment, and response evaluation of dynamic systems based on user-defined or example transfer functions.

---

## Features

- **Interactive Root Locus Visualization**  
  Display the full root locus plot of the open-loop system and observe pole movements as gain varies.

- **Custom and Predefined Systems**  
  Choose from example systems (First Order, Second Order, PID-controlled plant, Underdamped system, etc.) or input your own numerator and denominator coefficients.

- **Root Locus Rule Analysis**  
  Analyze the system using classical root locus rules such as starting points, stopping points, asymptotes, angle of departure, breakaway points, real-axis segments, and imaginary-axis crossings.

- **Dynamic Gain Control**  
  Adjust the system gain using a slider or manual input, including infinite gain (K → ∞) cases.

- **Closed-loop Pole Analysis**  
  Visualize closed-loop poles for selected gain values and assess system stability in real-time.

- **Step and Ramp Response Analysis**  
  Simulate and visualize system responses (Unity, Step, Ramp inputs) for different gain settings.

- **Performance Metrics Calculation**  
  Automatically compute metrics such as peak value, steady-state value, overshoot, rise time, settling time, delay time, and more.

- **Detailed System Information**  
  Inspect the open-loop transfer function, pole-zero plots, and system type.

---

## How to Run Locally

1. Clone this repository:
    ```bash
    git clone https://github.com/your-username/drip-root-locus-app.git
    cd drip-root-locus-app
    ```

2. Create and activate a virtual environment (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Launch the Streamlit app:
    ```bash
    streamlit run ff.py
    ```

---

## Requirements

- Python 3.8+
- Streamlit
- NumPy
- Matplotlib
- Control Systems Library (`control`)
- Pandas

(All dependencies are listed in `requirements.txt`.)

---

## License

This project is licensed under the MIT License.  
See the [LICENSE](LICENSE) file for more information.

---

## Authors

- **D. Rushikesh**  
  Lead Developer and System Designer  
  ECE Department

- **Chandan Sai Pavan**  
  Backend Integration and Stability Analysis Specialist  
  ECE Department

- **Deepak Yadav K**  
  Output Validation and Mathematical Accuracy Verifier  
  ECE Department


import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from control import tf, rlocus, feedback, step_response
import pandas as pd
from matplotlib.patches import Circle

# Configure the page
st.set_page_config(
    page_title="DRIP – Dynamic Root Locus Integration Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional header and description
st.markdown("""
    <div style='text-align: center; padding: 1.5rem; background-color: #f9f9f9; border-radius: 10px; color: #333333; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);'>
        <h1 style='color: #005f73; margin-bottom: 1rem;'>DRIP – Dynamic Root Locus Integration Platform</h1>
        <p style='font-size: 1.1rem;'>
            Experience precise <b>Root Locus</b> methodologies with features such as:
        </p>
        <ul style='list-style-position: inside; text-align: left; margin-top: 1rem; line-height: 1.8;'>
            <li>Interactive Root Locus Visualization</li>
            <li>Dynamic System Behavior Examination</li>
            <li>Immediate Stability Assessment</li>
            <li>Impact Analysis of Gain Variations</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Example Systems
EXAMPLE_SYSTEMS = {
    "Custom": {"num": [1], "den": [1, 1]},
    "First Order": {"num": [1], "den": [1, 1]},
    "Second Order": {"num": [1], "den": [1, 0.5, 1]},
    "PID-controlled Plant": {"num": [1, 0.5, 0.25], "den": [1, 2, 1, 0]},
    "Underdamped System": {"num": [1], "den": [1, 0.2, 1]},
    "Unstable System": {"num": [1], "den": [1, -2, 1]},
    "Complex Zeros": {"num": [1, 0, 1], "den": [1, 0, 0, 1]},
}

# Sidebar configuration
with st.sidebar:
    st.markdown("""
        <h3 style='text-align: center; color: #005f73;'>System Configuration</h3>
        <hr style='border:1px solid #e0e0e0;'>
        """, unsafe_allow_html=True)
    selected_example = st.selectbox(
        "Choose an example system:",
        list(EXAMPLE_SYSTEMS.keys()),
        help="Select a predefined system or choose 'Custom' to enter your own."
    )
    if selected_example == "Custom":
        num_str = st.text_input("Numerator Coefficients (comma-separated):", "1", help="Enter coefficients for the numerator polynomial, highest order first.")
        den_str = st.text_input("Denominator Coefficients (comma-separated):", "1, 1", help="Enter coefficients for the denominator polynomial, highest order first.")
        try:
            num = [float(x.strip()) for x in num_str.split(",")]
            den = [float(x.strip()) for x in den_str.split(",")]
        except ValueError:
            st.error("❌ Invalid coefficients. Using default values.")
            num = [1]
            den = [1, 1]
    else:
        num = EXAMPLE_SYSTEMS[selected_example]["num"]
        den = EXAMPLE_SYSTEMS[selected_example]["den"]
        st.info(f"Numerator: {num}\nDenominator: {den}")

# Section divider
st.markdown("""
    <hr style='border:1px solid #e0e0e0; margin: 2rem 0;'>
    """, unsafe_allow_html=True)

# Tabs for main analysis
root_tab, resp_tab, info_tab = st.tabs([
    "Root Locus Analysis",
    "Response Analysis",
    "System Information"
])

# Create transfer function and calculate root locus
system = tf(num, den)
rlist, klist = rlocus(system, plot=False)

# Create polynomials for display
num_poly = np.poly1d(num)
den_poly = np.poly1d(den)

# Calculate reasonable gain range
max_k = min(100.0, np.max(klist) * 1.2) if len(klist) > 0 else 100.0

# Get poles and zeros for plot range calculation
poles = system.poles()
zeros = system.zeros() if len(num) > 1 else []

# Display System Transfer Function Analysis in main window
st.markdown("---")  # Add separator

# Root Locus Analysis Tab
with root_tab:
    st.markdown("""
        <div style='background-color: #f9f9f9; border-radius: 10px; padding: 1rem; margin-bottom: 1.5rem; box-shadow: 0 2px 8px rgba(0,0,0,0.07);'>
            <h2 style='color: #005f73; margin-bottom: 0.5rem;'>Root Locus Analysis</h2>
            <p style='color: #333; font-size: 1.05rem;'>Explore the root locus of your system and interactively analyze stability and pole locations.</p>
        </div>
    """, unsafe_allow_html=True)

    # System Transfer Function Analysis Section
    st.header("System Transfer Function Analysis")
    st.markdown("We start with the loop gain transfer function:")

    # Format polynomials for LaTeX display
    def format_poly_latex(coeffs, var='s'):
        terms = []
        degree = len(coeffs) - 1
        
        for i, coef in enumerate(coeffs):
            if coef == 0:
                continue
            power = degree - i
            
            if power == 0:
                term = f"{coef:g}"
            elif power == 1:
                if coef == 1:
                    term = var
                elif coef == -1:
                    term = f"-{var}"
                else:
                    term = f"{coef:g}{var}"
            else:
                if coef == 1:
                    term = f"{var}^{power}"
                elif coef == -1:
                    term = f"-{var}^{power}"
                else:
                    term = f"{coef:g}{var}^{power}"
            
            if coef > 0 and len(terms) > 0:
                terms.append("+")
            terms.append(term)
        
        return " ".join(terms) if terms else "0"

    # Format the transfer function components
    num_latex = format_poly_latex(num)
    den_latex = format_poly_latex(den)

    # Display the transfer function
    st.latex(f"G(s) \cdot H(s) = \\frac{{N(s)}}{{D(s)}} = \\frac{{{num_latex}}}{{{den_latex}}}")

    st.markdown("The characteristic equation (i.e., the denominator of the closed loop transfer function) is")
    
    st.latex(r"1 + K \cdot G(s) \cdot H(s) = 0, \text{ or } 1 + K \cdot \frac{N(s)}{D(s)} = 0,")

    st.markdown("which we can rewrite as:")

    # Format the expanded characteristic equation
    st.latex(f"D(s) + K \cdot N(s) = {den_latex} + K({num_latex}) = 0")

    # Add summary section for zeros, poles, and branches
    n = len(poles)
    m = len(zeros)
    q = n - m
    zero_str = ', '.join([f"$s = {np.round(z.real, 4)}" + (f" {'+' if z.imag >= 0 else '-'} {abs(np.round(z.imag, 4))}j$" if abs(z.imag) > 1e-8 else '$') for z in zeros]) if len(zeros) > 0 else 'None'
    pole_str = ', '.join([f"$s = {np.round(p.real, 4)}" + (f" {'+' if p.imag >= 0 else '-'} {abs(np.round(p.imag, 4))}j$" if abs(p.imag) > 1e-8 else '$') for p in poles])
    st.markdown("""
    <br>
    If we plot the roots of this equation as K varies, we obtain the root locus. To sketch or understand the root locus, note:
    <ul>
        <li>The numerator polynomial has <b>{m}</b> zero(s) at {zero_str}.</li>
        <li>The denominator polynomial yields <b>{n}</b> pole(s) at {pole_str}. Therefore there are {n} branches to the locus.</li>
        <li>There exist <b>q = n - m = {q}</b> closed-loop pole(s) as $K \to \infty$, $|s| \to \infty$.</li>
    </ul>
    """, unsafe_allow_html=True)

    st.info("""
    This is the basic form of the characteristic equation. 
    You can now:
    1. Use the 'Set K' feature to see how different K values affect the roots
    2. Analyze the root locus properties using the rules below
    3. Study the system stability for different K values
    """)

    st.markdown("---")  # Add separator before root locus analysis

    st.header("Root Locus Analysis")
    
    try:
        # Root Locus Rules Analysis
        st.subheader("Select rule to be explained")
        st.markdown("*(The explanation of the rule applied to this loop gain is below the graph.)*")

        # Helper functions for root locus analysis
        def highlight_points_on_plot(points, color, marker, label, size=100):
            """Helper function to highlight points on the root locus plot"""
            if len(points) > 0:
                plt.scatter(points.real, points.imag, c=color, marker=marker, s=size, label=label, zorder=5)

        def plot_angle_marker(center, radius, start_angle, end_angle, color='blue'):
            """Helper function to draw angle markers on the plot"""
            angle = np.linspace(start_angle, end_angle, 100)
            x = center.real + radius * np.cos(angle)
            y = center.imag + radius * np.sin(angle)
            plt.plot(x, y, color=color, linestyle='--', alpha=0.5)

        def analyze_root_locus(system, selected_rule):
            """Analyze root locus based on selected rule"""
            poles = system.poles()
            zeros = system.zeros() if len(num) > 1 else []
            n = len(poles)  # number of poles
            m = len(zeros)  # number of zeros

            if selected_rule == "Show only the locus":
                return "The root locus shows all possible closed-loop pole locations as the gain K varies from 0 to ∞."

            elif selected_rule == "Starting point(s) of locus (K→0)":
                points = [f"s_{{{i+1}}} = {p:.3f}" for i, p in enumerate(poles)]
                highlight_points_on_plot(poles, 'red', 'x', 'Starting Points (K=0)')
                return "\\begin{align*}" + "\\\\\n".join(points) + "\\end{align*}"

            elif selected_rule == "Stopping point(s) of locus (K→∞)":
                if m == 0:
                    alpha = sum(poles).real / n  # centroid
                    phi = [(2*k + 1)*180/n for k in range(n)]  # asymptote angles
                    return f"With {n} poles and no zeros, the asymptotes have angles: {', '.join([f'{p:.1f}°' for p in phi])}"
                else:
                    points = [f"s_{{{i+1}}} = {z:.3f}" for i, z in enumerate(zeros)]
                    highlight_points_on_plot(np.array(zeros), 'green', 'o', 'Stopping Points (K=∞)')
                    return "\\begin{align*}" + "\\\\\n".join(points) + "\\end{align*}"

            elif selected_rule == "Locus on Real Axis":
                real_poles = poles[np.abs(poles.imag) < 1e-10].real
                real_zeros = zeros[np.abs(np.array(zeros).imag) < 1e-10].real if len(zeros) > 0 else np.array([])
                points = np.sort(np.concatenate([real_poles, real_zeros]))
                segments = []
                for i in range(len(points)-1):
                    poles_right = sum(1 for p in real_poles if p > points[i])
                    zeros_right = sum(1 for z in real_zeros if z > points[i])
                    if (poles_right + zeros_right) % 2 == 1:
                        segments.append(f"({points[i]:.3f}, {points[i+1]:.3f})")
                        plt.plot([points[i], points[i+1]], [0, 0], 'r-', linewidth=3, alpha=0.5)
                return "Root locus exists on real axis in segments:\n" + "\n".join(segments)

            elif selected_rule == "Asymptotes as |s|→∞":
                if n > m:
                    alpha = (sum(poles) - sum(zeros)).real / (n - m)  # centroid
                    phi = [(2*k + 1)*180/(n - m) for k in range(n - m)]  # asymptote angles
                    # Draw asymptotes
                    for angle in phi:
                        rad_angle = np.deg2rad(angle)
                        dx = np.cos(rad_angle)
                        dy = np.sin(rad_angle)
                        plt.arrow(alpha, 0, dx*5, dy*5, head_width=0.2, 
                                head_length=0.3, fc='purple', ec='purple', alpha=0.5)
                    plt.axvline(x=alpha, color='green', linestyle='--', alpha=0.5, 
                              label=f'Centroid (x={alpha:.3f})')
                    return f"Asymptote angles: {', '.join([f'{p:.1f}°' for p in phi])}"
                else:
                    return "No asymptotes (number of zeros ≥ number of poles)"

            elif selected_rule == "Break-Out (and In) locations":
                real_poles = poles[np.abs(poles.imag) < 1e-10].real
                if len(real_poles) >= 2:
                    potential_break = (real_poles[:-1] + real_poles[1:]) / 2
                    highlight_points_on_plot(potential_break, 'purple', '*', 'Potential Break Points')
                return "Break points occur where dK/ds = 0"

            elif selected_rule == "Angle of Departure from complex poles":
                complex_poles = poles[np.abs(poles.imag) > 1e-10]
                if len(complex_poles) == 0:
                    return "No complex poles in the system"
                angles = []
                for p in complex_poles:
                    angle = 180
                    for other_p in poles:
                        if not np.allclose(p, other_p):
                            angle += np.angle(p - other_p, deg=True)
                    for z in zeros:
                        angle -= np.angle(p - z, deg=True)
                    angles.append(f"From pole at {p:.3f}: {angle % 360:.1f}°")
                    plot_angle_marker(p, 0.5, np.deg2rad(angle-30), np.deg2rad(angle+30))
                highlight_points_on_plot(complex_poles, 'red', 'x', 'Complex Poles')
                return "Angles of Departure:\n" + "\n".join(angles)

            elif selected_rule == "Angle of Arrival to complex zeros":
                complex_zeros = np.array(zeros)[np.abs(np.array(zeros).imag) > 1e-10] if len(zeros) > 0 else []
                if len(complex_zeros) == 0:
                    return "No complex zeros in the system"
                angles = []
                for z in complex_zeros:
                    angle = 0
                    for p in poles:
                        angle += np.angle(z - p, deg=True)
                    for other_z in zeros:
                        if not np.allclose(z, other_z):
                            angle -= np.angle(z - other_z, deg=True)
                    angles.append(f"To zero at {z:.3f}: {angle % 360:.1f}°")
                    plot_angle_marker(z, 0.5, np.deg2rad(angle-30), np.deg2rad(angle+30))
                highlight_points_on_plot(complex_zeros, 'green', 'o', 'Complex Zeros')
                return "Angles of Arrival:\n" + "\n".join(angles)

            elif selected_rule == "Locus Crosses Imaginary Axis":
                plt.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Imaginary Axis')
                return "Crossing points can be found using the Routh-Hurwitz criterion"

            elif selected_rule == "Get K from Pole Location":
                return "Click on the plot to select a point and calculate the corresponding K value"

            return ""  # Default empty return

        # Create two columns for controls and plot
        col1, col2 = st.columns([1, 2])

        with col1:
            # Rule selection (remove last option)
            selected_rule = st.radio(
                "",  # Empty label since we have the subheader
                [
                    "Show only the locus",
                    "Starting point(s) of locus (K→0)",
                    "Stopping point(s) of locus (K→∞)",
                    "Locus on Real Axis",
                    "Asymptotes as |s|→∞",
                    "Break-Out (and In) locations",
                    "Angle of Departure from complex poles",
                    "Angle of Arrival to complex zeros",
                    "Locus Crosses Imaginary Axis",
                    "Set K (slider is below), and show pole locations"
                ],
                index=0
            )

        with col2:
            # Create figure with title
            fig = plt.figure(figsize=(10, 8))
            plt.title("Root Locus of G(s)H(s)")
            plt.xlabel("σ (real part of s)")
            plt.ylabel("jω")
            plt.grid(True, which='both', linestyle='--', alpha=0.6)
            rlist, klist = rlocus(system, plot=True)
            plt.scatter(np.real(poles), np.imag(poles), marker='x', color='red', s=100, label='Open-loop Poles')
            if len(zeros) > 0:
                plt.scatter(np.real(zeros), np.imag(zeros), marker='o', color='green', s=100, label='Open-loop Zeros')

            # Highlight features for each rule
            if selected_rule == "Starting point(s) of locus (K→0)":
                plt.scatter(np.real(poles), np.imag(poles), marker='D', color='magenta', s=120, label='Starting Points (K=0)')
            elif selected_rule == "Stopping point(s) of locus (K→∞)":
                if len(zeros) > 0:
                    plt.scatter(np.real(zeros), np.imag(zeros), marker='s', color='cyan', s=120, label='Stopping Points (K=∞)')
            elif selected_rule == "Locus on Real Axis":
                # Highlight real axis segments
                all_points = np.concatenate([poles, zeros])
                real_points = np.sort(np.real(all_points[np.abs(np.imag(all_points)) < 1e-8]))
                for i in range(len(real_points)-1):
                    left = real_points[i]
                    right = real_points[i+1]
                    # Count poles and zeros to the right
                    poles_right = sum(1 for p in np.real(poles) if p > left)
                    zeros_right = sum(1 for z in np.real(zeros) if z > left)
                    if (poles_right + zeros_right) % 2 == 1:
                        plt.plot([left, right], [0, 0], color='orange', linewidth=4, alpha=0.7, label='Real Axis Segment' if i == 0 else None)
            elif selected_rule == "Asymptotes as |s|→∞":
                n = len(poles)
                m = len(zeros)
                if n > m:
                    centroid = (np.sum(poles) - np.sum(zeros)) / (n - m)
                    angles = [(2*k + 1)*np.pi/(n - m) for k in range(n - m)]
                    for angle in angles:
                        x = np.real(centroid)
                        y = 0
                        dx = np.cos(angle)
                        dy = np.sin(angle)
                        plt.arrow(x, y, dx*3, dy*3, head_width=0.2, head_length=0.3, fc='purple', ec='purple', alpha=0.7)
                    plt.axvline(x=np.real(centroid), color='purple', linestyle='--', alpha=0.5, label='Centroid')
            elif selected_rule == "Break-Out (and In) locations":
                # Calculate break points: roots of dK/ds = 0
                # K(s) = -D(s)/N(s) => dK/ds = 0 => N(s)D'(s) - D(s)N'(s) = 0
                from numpy.polynomial import Polynomial as P
                D = np.poly1d(den)
                N = np.poly1d(num)
                Dp = D.deriv()
                Np = N.deriv()
                # Form the break equation: N(s)D'(s) - D(s)N'(s) = 0
                break_poly = np.polysub(np.polymul(num, Dp.coeffs), np.polymul(den, Np.coeffs))
                break_roots = np.roots(break_poly)
                # Only real break points on real axis and between real poles
                real_breaks = [br.real for br in break_roots if np.abs(br.imag) < 1e-8]
                for br in real_breaks:
                    plt.scatter(br, 0, marker='*', color='purple', s=200, label='Break Point')
            elif selected_rule == "Angle of Departure from complex poles":
                # Highlight complex poles
                complex_poles = [p for p in poles if np.abs(p.imag) > 1e-8]
                plt.scatter(np.real(complex_poles), np.imag(complex_poles), marker='x', color='blue', s=150, label='Complex Poles')
            elif selected_rule == "Angle of Arrival to complex zeros":
                # Highlight complex zeros
                complex_zeros = [z for z in zeros if np.abs(z.imag) > 1e-8]
                plt.scatter(np.real(complex_zeros), np.imag(complex_zeros), marker='o', color='blue', s=150, label='Complex Zeros')
            elif selected_rule == "Locus Crosses Imaginary Axis":
                plt.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Imaginary Axis')
            # For Set K, don't plot here (handled below)
            plt.legend()
            if selected_rule != "Set K (slider is below), and show pole locations":
                st.pyplot(fig)

        # Add K slider (with ∞ option) below the plot (always visible)
        if 'k_value' not in st.session_state:
            st.session_state['k_value'] = 0.0
        if 'k_inf' not in st.session_state:
            st.session_state['k_inf'] = False

        def update_k_from_text():
            try:
                val = float(st.session_state['k_text'])
                if 0 <= val <= 100:
                    st.session_state['k_value'] = round(val, 3)
                    st.session_state['k_inf'] = False
            except Exception:
                pass

        def update_k_from_slider():
            st.session_state['k_text'] = f"{st.session_state['k_value']:.3f}"
            st.session_state['k_inf'] = False

        def update_k_from_inf():
            if st.session_state['k_inf']:
                st.session_state['k_text'] = '∞'
            else:
                st.session_state['k_text'] = f"{st.session_state['k_value']:.3f}"

        col_k1, col_k2 = st.columns([1, 2])
        with col_k1:
            st.text_input(
                "Set Gain (K):",
                value=f"{st.session_state['k_value']:.3f}" if not st.session_state['k_inf'] else '∞',
                key='k_text',
                on_change=update_k_from_text,
                disabled=st.session_state['k_inf']
            )
            st.checkbox('∞ (infinity)', key='k_inf', on_change=update_k_from_inf)
        with col_k2:
            st.slider(
                label="K value",
                min_value=0.0,
                max_value=100.0,
                value=st.session_state['k_value'],
                step=0.001,
                key='k_value',
                on_change=update_k_from_slider,
                format="%.3f",
                disabled=st.session_state['k_inf']
            )
            st.markdown(f"0.000 {' ' * 90} 100.000 ∞")

        k_value = 1e6 if st.session_state['k_inf'] else st.session_state['k_value']

        # Only show the second graph (with closed-loop poles) for 'Set K...' rule
        if selected_rule == "Set K (slider is below), and show pole locations":
            fig2 = plt.figure(figsize=(10, 8))
            plt.title("Root Locus of G(s)H(s) with Closed-loop Poles")
            plt.xlabel("σ (real part of s)")
            plt.ylabel("jω")
            plt.grid(True, which='both', linestyle='--', alpha=0.6)
            _, _ = rlocus(system, plot=True)
            plt.scatter(np.real(poles), np.imag(poles), marker='x', color='red', s=100, label='Open-loop Poles')
            if len(zeros) > 0:
                plt.scatter(np.real(zeros), np.imag(zeros), marker='o', color='green', s=100, label='Open-loop Zeros')
            closed_loop_sys = feedback(k_value * system, 1)
            closed_poles = closed_loop_sys.poles()
            plt.scatter(np.real(closed_poles), np.imag(closed_poles), marker='D', color='magenta', s=120, label='Closed-loop Poles (K)')
            plt.legend()
            st.pyplot(fig2)

        # Use k_value for all closed-loop calculations below

        # If "Set K (slider is below), and show pole locations" is selected, just show the info, don't add another slider
        if selected_rule == "Set K (slider is below), and show pole locations":
            st.markdown("### Set K and calculate root locations")
            st.markdown("""
            If you set K=0 (below), the starting points are displayed (i.e., the poles of the closed loop transfer function when K=0) as pink diamonds. 
            As you increase K the closed loop poles (i.e., pink diamonds) move towards the stopping points as K→∞.
            """)
            n = len(poles)
            m = len(zeros)
            q = n - m
            st.markdown(f"Since q = n - m = {q}, there are {q} roots with stopping points where |s|→∞.")

            # Detailed step-by-step calculation in LaTeX
            st.markdown("#### The characteristic equation is")
            st.latex(r"1 + K \cdot G(s) \cdot H(s) = 0, \quad \text{or} \quad 1 + K \cdot \frac{N(s)}{D(s)} = 0,")

            st.latex(r"D(s) + K \cdot N(s) = 0")

            # Show with actual polynomials
            num_latex = format_poly_latex(num)
            den_latex = format_poly_latex(den)
            st.latex(f"{den_latex} + K({num_latex}) = 0")

            # Show with current K value
            k_fmt = f"{k_value:.3g}" if k_value < 1e5 else "\infty"
            st.markdown(f"with $K = {k_fmt}$ this is")
            # Compute the expanded polynomial for this K
            den_poly = np.poly1d(den)
            num_poly = np.poly1d(num)
            char_poly = den_poly + k_value * num_poly
            # Format the expanded polynomial for LaTeX
            def poly1d_to_latex(poly, var='s'):
                coeffs = poly.coeffs
                degree = len(coeffs) - 1
                terms = []
                for i, coef in enumerate(coeffs):
                    if abs(coef) < 1e-10:
                        continue
                    power = degree - i
                    coef_str = f"{coef:.3g}" if abs(coef) < 1e4 else f"{coef:.2e}"
                    if power == 0:
                        term = f"{coef_str}"
                    elif power == 1:
                        if coef == 1:
                            term = var
                        elif coef == -1:
                            term = f"-{var}"
                        else:
                            term = f"{coef_str}{var}"
                    else:
                        if coef == 1:
                            term = f"{var}^{{{power}}}"
                        elif coef == -1:
                            term = f"-{var}^{{{power}}}"
                        else:
                            term = f"{coef_str}{var}^{{{power}}}"
                    if coef > 0 and len(terms) > 0:
                        terms.append("+")
                    terms.append(term)
                return " ".join(terms) if terms else "0"
            char_poly_latex = poly1d_to_latex(char_poly)
            st.latex(f"{char_poly_latex} = 0")

            # Show the roots for this K
            roots = np.roots(char_poly)
            if len(roots) > 0:
                root_strs = []
                for r in roots:
                    real = np.round(r.real, 3)
                    imag = np.round(r.imag, 3)
                    if abs(imag) < 1e-8:
                        root_strs.append(f"s = {real}")
                    else:
                        root_strs.append(f"s = {real} \pm {abs(imag)}j")
                st.markdown(f"This has roots at $" + ", \\; ".join(root_strs) + "$.")
                st.markdown("These are shown as pink diamonds on the graph.")

        # Display rule explanation based on selection
        st.markdown("### Rule Explanation")
        explanation = analyze_root_locus(system, selected_rule)
        if selected_rule == "Angle of Departure from complex poles":
            st.markdown("""
            <h3 style='color:#b00;'>Angle of Departure</h3>
            <span style='color:#b00; font-size:0.95em;'><i>Link to in depth description of rule for finding angle of departure from complex poles of G(s)H(s).</i></span>
            <br><br>
            We will find the angle of departure of the locus from each complex pole (marked by a pink diamond).<br>
            Angles between this pole and other poles are shown as blue/purple arcs.<br>
            Angles between this pole and zeros are shown as green/cyan arcs.<br>
            The angle of departure is shown as a pink arc.
            """, unsafe_allow_html=True)
            # For each complex pole, show the calculation
            for idx, p in enumerate(poles):
                if abs(p.imag) > 1e-8:
                    st.markdown(f"<b>For pole at $s = {np.round(p.real,2)} {'+' if p.imag >= 0 else '-'} {abs(np.round(p.imag,2))}j$:</b>", unsafe_allow_html=True)
                    # Angles to zeros
                    for j, z in enumerate(zeros):
                        vec = p - z
                        angle = np.angle(vec, deg=True)
                        st.markdown(f"<ul><li><b>zero at {np.round(z.real,2)}{'+' if z.imag >= 0 else '-'}{abs(np.round(z.imag,2))}j</b>: vector = $({np.round(vec.real,2)} {'+' if vec.imag >= 0 else '-'} {abs(np.round(vec.imag,2))}j)$, angle $= {angle:.2f}^\circ$</li></ul>", unsafe_allow_html=True)
                    # Angles to other poles
                    for j, p2 in enumerate(poles):
                        if not np.allclose(p, p2):
                            vec = p - p2
                            angle = np.angle(vec, deg=True)
                            st.markdown(f"<ul><li><b>pole at {np.round(p2.real,2)}{'+' if p2.imag >= 0 else '-'}{abs(np.round(p2.imag,2))}j</b>: vector = $({np.round(vec.real,2)} {'+' if vec.imag >= 0 else '-'} {abs(np.round(vec.imag,2))}j)$, angle $= {angle:.2f}^\circ$</li></ul>", unsafe_allow_html=True)
            st.markdown("The angle of departure is shown as a pink arc on the plot.")
        elif explanation.startswith("\\begin{align*}"):
            st.latex(explanation)
        else:
            st.write(explanation)
        
        # Calculate closed-loop poles for current gain value
        closed_loop_sys = feedback(k_value * system, 1)
        closed_poles = closed_loop_sys.poles()
        
        # Display closed-loop poles
        st.subheader("Closed-loop Poles")
        poles_df = pd.DataFrame({
            "Pole": [f"Pole {i+1}" for i in range(len(closed_poles))],
            "Value": [f"{np.round(cp, 4)}" for cp in closed_poles],
            "Magnitude": [f"{np.round(np.abs(cp), 4)}" for cp in closed_poles],
            "Angle (deg)": [f"{np.round(np.angle(cp, deg=True), 2)}°" 
                          for cp in closed_poles],
            "Stability": ["Stable" if cp.real < 0 else "Unstable" 
                        for cp in closed_poles]
        })
        st.dataframe(poles_df)
        
        # System stability indicator
        system_stable = all(cp.real < 0 for cp in closed_poles)
        if system_stable:
            st.success("✅ System is STABLE at the current gain")
        else:
            st.error("⚠️ System is UNSTABLE at the current gain")
            
    except Exception as e:
        st.error(f"Error in root locus analysis: {str(e)}")

# Response Analysis Tab
with resp_tab:
    st.header("Response Analysis")
    try:
        # K slider for this tab
        if 'resp_k_value' not in st.session_state:
            st.session_state['resp_k_value'] = 1.0
        if 'resp_k_inf' not in st.session_state:
            st.session_state['resp_k_inf'] = False

        def update_resp_k_from_text():
            try:
                val = float(st.session_state['resp_k_text'])
                if 0 <= val <= 100:
                    st.session_state['resp_k_value'] = round(val, 3)
                    st.session_state['resp_k_inf'] = False
            except Exception:
                pass

        def update_resp_k_from_slider():
            st.session_state['resp_k_text'] = f"{st.session_state['resp_k_value']:.3f}"
            st.session_state['resp_k_inf'] = False

        def update_resp_k_from_inf():
            if st.session_state['resp_k_inf']:
                st.session_state['resp_k_text'] = '∞'
            else:
                st.session_state['resp_k_text'] = f"{st.session_state['resp_k_value']:.3f}"

        col1, col2 = st.columns([1, 2])
        with col1:
            st.text_input(
                "Set Gain (K):",
                value=f"{st.session_state['resp_k_value']:.3f}" if not st.session_state['resp_k_inf'] else '∞',
                key='resp_k_text',
                on_change=update_resp_k_from_text,
                disabled=st.session_state['resp_k_inf']
            )
            st.checkbox('∞ (infinity)', key='resp_k_inf', on_change=update_resp_k_from_inf)
        with col2:
            st.slider(
                label="K value",
                min_value=0.0,
                max_value=100.0,
                value=st.session_state['resp_k_value'],
                step=0.001,
                key='resp_k_value',
                on_change=update_resp_k_from_slider,
                format="%.3f",
                disabled=st.session_state['resp_k_inf']
            )
            st.markdown(f"0.000 {' ' * 90} 100.000 ∞")

        resp_k = 1e6 if st.session_state['resp_k_inf'] else st.session_state['resp_k_value']

        # Response type selector
        response_type = st.radio(
            "Select Response Type:",
            ["Unity", "Step", "Ramp"],
            index=1
        )

        # Closed-loop transfer function for selected K
        closed_loop = feedback(resp_k * system, 1)
        t = np.linspace(0, 20, 1000)

        # Show transfer function in LaTeX
        st.subheader("Response Analysis Transfer Function")
        num_latex = format_poly_latex(num)
        den_latex = format_poly_latex(den)
        st.latex(r"T(s) = \frac{G(s)H(s)}{1 + G(s)H(s)} = \frac{%s}{%s}" % (num_latex, den_latex + "+" + num_latex))

        if response_type == "Unity":
            st.latex(r"Y(s) = T(s) ")
            t, y = step_response(closed_loop, t)
        elif response_type == "Step":
            st.latex(r"Y(s) = T(s) \cdot \frac{1}{s}")
            t, y = step_response(closed_loop, t)
        elif response_type == "Ramp":
            st.latex(r"Y(s) = T(s) \cdot \frac{1}{s^2}")
            from control import forced_response
            try:
                t, y, _ = forced_response(closed_loop, T=t, U=t)
            except ValueError:
                t, y = forced_response(closed_loop, T=t, U=t)

        # Plot response
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(t, y)
        ax.set_title(f'{response_type} Response (K = {resp_k})')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.grid(True)
        ax.legend([f'{response_type} Response'])
        st.pyplot(fig)

        # Calculate metrics
        peak = np.max(y)
        steady_state = y[-1]
        overshoot = ((peak - steady_state) / steady_state * 100 if steady_state > 0 else None)
        settling_idx = np.where(np.abs(y - steady_state) <= 0.02 * steady_state)[0]
        settling_time = t[settling_idx[0]] if len(settling_idx) > 0 else None
        if steady_state > 0:
            rise_low_idx = np.where(y >= 0.1 * steady_state)[0]
            rise_high_idx = np.where(y >= 0.9 * steady_state)[0]
            rise_time = (t[rise_high_idx[0]] - t[rise_low_idx[0]] if len(rise_low_idx) > 0 and len(rise_high_idx) > 0 else None)
        else:
            rise_time = None
        peak_time = t[np.argmax(y)] if len(y) > 0 else None
        delay_idx = np.where(y >= 0.5 * steady_state)[0]
        delay_time = t[delay_idx[0]] if len(delay_idx) > 0 else None
        initial_value = y[0] if len(y) > 0 else None
        final_value = y[-1] if len(y) > 0 else None

        # Display metrics
        metrics_df = pd.DataFrame({
            "Metric": [
                "Peak Value", "Steady State", "Overshoot", "Rise Time (10-90%)", "Settling Time (2%)", "Peak Time", "Delay Time (50%)", "Initial Value", "Final Value"
            ],
            "Value": [
                f"{peak:.4f}" if peak is not None else "N/A",
                f"{steady_state:.4f}" if steady_state is not None else "N/A",
                f"{overshoot:.2f}%" if overshoot is not None else "N/A",
                f"{rise_time:.4f}s" if rise_time is not None else "N/A",
                f"{settling_time:.4f}s" if settling_time is not None else "N/A",
                f"{peak_time:.4f}s" if peak_time is not None else "N/A",
                f"{delay_time:.4f}s" if delay_time is not None else "N/A",
                f"{initial_value:.4f}" if initial_value is not None else "N/A",
                f"{final_value:.4f}" if final_value is not None else "N/A"
            ]
        })
        st.dataframe(metrics_df)

    except Exception as e:
        st.error(f"Error in response analysis: {str(e)}")

# System Information Tab
with info_tab:
    st.header("System Information")
    
    try:
        # Display transfer function
        st.subheader("Open-Loop Transfer Function")
        num_poly = np.poly1d(num)
        den_poly = np.poly1d(den)

        # Use LaTeX for transfer function
        def format_poly_latex(coeffs, var='s'):
            terms = []
            degree = len(coeffs) - 1
            for i, coef in enumerate(coeffs):
                if coef == 0:
                    continue
                power = degree - i
                if power == 0:
                    term = f"{coef:g}"
                elif power == 1:
                    if coef == 1:
                        term = var
                    elif coef == -1:
                        term = f"-{var}"
                    else:
                        term = f"{coef:g}{var}"
                else:
                    if coef == 1:
                        term = f"{var}^{power}"
                    elif coef == -1:
                        term = f"-{var}^{power}"
                    else:
                        term = f"{coef:g}{var}^{power}"
                if coef > 0 and len(terms) > 0:
                    terms.append("+")
                terms.append(term)
            return " ".join(terms) if terms else "0"

        num_latex = format_poly_latex(num)
        den_latex = format_poly_latex(den)
        st.latex(f"G(s) = \\frac{{{num_latex}}}{{{den_latex}}}")

        # System type
        system_type = sum(1 for d in den if abs(d) < 1e-10)
        st.markdown(f"**System Type:** ${{{system_type}}}$")

        # Display poles and zeros in two columns, LaTeX formatted
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Open-Loop Poles")
            for i, p in enumerate(poles):
                real = np.round(p.real, 4)
                imag = np.round(p.imag, 4)
                if abs(imag) < 1e-8:
                    st.latex(f"s_{{{i+1}}} = {real}")
                else:
                    sign = '+' if imag >= 0 else '-'
                    st.latex(f"s_{{{i+1}}} = {real} {sign} {abs(imag)}j")
                st.markdown(f"Stability: {'Stable' if p.real < 0 else 'Unstable'}")
        with col2:
            st.subheader("Open-Loop Zeros")
            if len(zeros) > 0:
                for i, z in enumerate(zeros):
                    real = np.round(z.real, 4)
                    imag = np.round(z.imag, 4)
                    if abs(imag) < 1e-8:
                        st.latex(f"z_{{{i+1}}} = {real}")
                    else:
                        sign = '+' if imag >= 0 else '-'
                        st.latex(f"z_{{{i+1}}} = {real} {sign} {abs(imag)}j")
            else:
                st.info("No finite zeros in the system.")

        # Pole-Zero map
        st.subheader("Pole-Zero Map")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(poles.real, poles.imag, 'rx', markersize=10, label='Poles')
        if len(zeros) > 0:
            ax.plot(np.array(zeros).real, np.array(zeros).imag, 'go', 
                   markersize=10, label='Zeros')
        ax.set_title('Pole-Zero Map')
        ax.set_xlabel('Real Axis')
        ax.set_ylabel('Imaginary Axis')
        ax.grid(True)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.legend()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error in system information display: {str(e)}")
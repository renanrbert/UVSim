import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.integrate import simpson
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

# ==== FUNCTIONS FROM ORIGINAL SCRIPT ====

def create_spectrum(stk_data, emin=3.8, emax=5.6, delta=0.01, shift=0.0, width=0.05):
    """Create theoretical spectrum with Gaussian broadening"""
    x_ev = np.arange(emin, emax, delta)
    y_ev = np.zeros_like(x_ev)

    for i in range(len(x_ev)):
        for j in range(stk_data.shape[0]):
            E = stk_data[j, 0] / 8065.54429  # cm⁻¹ → eV
            y_ev[i] += stk_data[j, 1] * np.exp(-((x_ev[i] - E - shift) / (np.sqrt(2) * width)) ** 2)

    if np.max(y_ev) > 0:
        y_ev = y_ev / np.max(y_ev)

    return x_ev, y_ev

def calculate_similarity_in_range(exp_data, theory_x, theory_y, exp_emin, exp_emax):
    """Calculate similarity only in experimental range"""
    mask = (theory_x >= exp_emin) & (theory_x <= exp_emax)
    theory_x_filtered = theory_x[mask]
    theory_y_filtered = theory_y[mask]

    cs = CubicSpline(exp_data[:, 0], exp_data[:, 1])
    y_exp = cs(theory_x_filtered)

    fg = simpson(y_exp * theory_y_filtered, x=theory_x_filtered)
    f2 = simpson(y_exp ** 2, x=theory_x_filtered)
    g2 = simpson(theory_y_filtered ** 2, x=theory_x_filtered)

    return fg / np.sqrt(f2 * g2)

def optimize_parameters_original(exp_data, stk_data, exp_emin, exp_emax):
    """EXACTLY as in original code: two-step Nelder-Mead optimization"""
    def objective(params):
        shift, width = params
        x, y = create_spectrum(stk_data, exp_emin, emax, shift=shift, width=width)
        return -calculate_similarity_in_range(exp_data, x, y, exp_emin, exp_emax)

    initial_guess = [0.0, 0.052651242]

    # Step 1: optimize shift (width fixed)
    result = minimize(
        objective,
        initial_guess,
        bounds=((-1.5, 1.5), (0.052651242, 0.052651242)),
        method='Nelder-Mead'
    )

    # Step 2: optimize width (shift fixed)
    result = minimize(
        objective,
        result.x,
        bounds=((result.x[0], result.x[0]), (0.0, 0.5)),
        method='Nelder-Mead'
    )

    return result.x[0], result.x[1]  # shift_opt, width_opt

def load_and_preprocess_experimental_data(file_obj):
    """Load and preprocess experimental data"""
    data = np.genfromtxt(file_obj)

    # Remove NaN rows
    data = data[~np.isnan(data).any(axis=1)]

    # Sort by energy (first column)
    data = data[np.argsort(data[:, 0])]

    # Remove duplicate energies
    unique_energies, unique_indices = np.unique(data[:, 0], return_index=True)
    data = data[unique_indices]

    return data

def normalize_transitions_in_plot_range(stk_data, plot_emin, plot_emax):
    """Normalize theoretical transitions only within plot range"""
    energy_eV = stk_data[:, 0] / 8065.54429

    mask = (energy_eV >= plot_emin) & (energy_eV <= plot_emax)

    if np.any(mask):
        intensities_in_range = stk_data[mask, 1]
        if len(intensities_in_range) > 0:
            max_intensity_in_range = np.max(intensities_in_range)
            if max_intensity_in_range > 0:
                normalized_intensities = stk_data[:, 1] / max_intensity_in_range
            else:
                normalized_intensities = stk_data[:, 1]
        else:
            normalized_intensities = stk_data[:, 1]
    else:
        max_intensity = np.max(stk_data[:, 1])
        if max_intensity > 0:
            normalized_intensities = stk_data[:, 1] / max_intensity
        else:
            normalized_intensities = stk_data[:, 1]

    return energy_eV, normalized_intensities

# ==== STREAMLIT INTERFACE ====

st.title("🔬 UV-Vis TD-DFT Spectrum Analyzer")

st.markdown("""
**Upload your experimental UV-Vis spectrum (.csv/.dat) and theoretical TD-DFT transitions (.stk files)**  
*Automatically optimizes Gaussian broadening and computes similarity scores*
""")

# Plot emin/emax inputs MOVED TO MAIN AREA (visible like original version)
col1, col2 = st.columns(2)
plot_emin = col1.number_input("plot-emin (eV)", value=1.5, step=0.1, min_value=0.1)
plot_emax = col2.number_input("plot-emax (eV)", value=5.0, step=0.1, min_value=0.1)

col1, col2 = st.columns([2, 1])
with col1:
    st.header("📁 File Upload")
    exp_file = st.file_uploader(
        "Experimental spectrum (.csv/.dat - 2 columns: Energy eV, Intensity)",
        type=["csv", "dat", "txt"],
        help="Format: energy(eV),intensity per line"
    )
with col2:
    st.info("**.stk format:**\nenergy(cm⁻¹),oscillator_strength")

stk_files = st.file_uploader(
    "Theoretical TD-DFT files (.stk - multiple OK)",
    type=["stk"],
    accept_multiple_files=True,
    help="One file per functional/method. Energy in cm⁻¹, 2nd column = f(oscillator strength)"
)

if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
    if exp_file is None or len(stk_files) == 0:
        st.error("❌ Please upload experimental data AND at least one .stk file")
        st.stop()

    with st.spinner("Processing spectra..."):
        # 1. Load experimental data
        try:
            exp_data = load_and_preprocess_experimental_data(exp_file)
            exp_emin, exp_emax = np.min(exp_data[:, 0]), np.max(exp_data[:, 0])
            st.success(f"✅ Experimental: {exp_data.shape[0]} points ({exp_emin:.2f}-{exp_emax:.2f} eV)")
        except Exception as e:
            st.error(f"❌ Experimental file error: {e}")
            st.stop()

        results = []
        n_funcionais = len(stk_files)

        # Create subplot layout (3 cols)
        n_cols = 3
        n_rows = (n_funcionais + n_cols - 1) // n_cols
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, 3 * n_rows), dpi=120)
        if n_funcionais == 1:
            axs = np.array([axs])
        axs = axs.flatten()

        for idx, stk_file in enumerate(stk_files):
            name = stk_file.name.replace('.stk', '')
            try:
                stk_data = np.loadtxt(stk_file)
                st.info(f"Processing {name}: {stk_data.shape[0]} transitions")

                # Optimize parameters
                shift_opt, width_opt = optimize_parameters_original(exp_data, stk_data, exp_emin, exp_emax)

                # Optimized spectrum
                x_opt, y_opt = create_spectrum(stk_data, exp_emin, exp_emax, shift=shift_opt, width=width_opt)
                similarity = calculate_similarity_in_range(exp_data, x_opt, y_opt, exp_emin, exp_emax)
                similarity_percent = similarity * 100

                results.append({
                    "Method/File": name,
                    "Shift (eV)": f"{shift_opt:.3f}",
                    "Width (eV)": f"{width_opt:.3f}",
                    "Similarity (%)": f"{similarity_percent:.1f}"
                })

                # Plot
                ax = axs[idx]
                energy_eV, intensity_norm = normalize_transitions_in_plot_range(stk_data, plot_emin, plot_emax)
                plot_mask = (energy_eV >= plot_emin) & (energy_eV <= plot_emax)
                plot_energy = energy_eV[plot_mask]
                plot_intensity = intensity_norm[plot_mask]

                # Theoretical sticks + envelope
                ax.vlines(plot_energy, 0, plot_intensity, color='red', linewidth=1, alpha=0.7)
                ax.plot(x_opt, y_opt, 'r-', linewidth=2, label=f'{name}\nS={similarity_percent:.1f}%')

                # Experimental
                ax.plot(exp_data[:, 0], exp_data[:, 1], 'k-', linewidth=2, label='Experimental')

                ax.set_xlabel('Energy (eV)', fontsize=11)
                ax.set_ylabel('Normalized Intensity', fontsize=11)
                ax.set_xlim(plot_emin, plot_emax)
                ax.set_ylim(0, 1.05)
                ax.grid(alpha=0.3, linestyle='--')
                ax.legend(fontsize=9)
                ax.set_title(name, fontsize=12, fontweight='bold')

            except Exception as e:
                st.error(f"❌ Error processing {name}: {str(e)[:100]}")

        # Hide empty subplots
        for idx in range(n_funcionais, len(axs)):
            axs[idx].set_visible(False)

        plt.tight_layout()
        st.pyplot(fig)

    # Results table
    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('Similarity (%)', ascending=False)
        
        st.markdown("## 📈 Results Table (Ranked by Similarity)")
        st.dataframe(results_df, use_container_width=True, hide_index=True)
        
        # Download
        csv_data = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="💾 Download Results CSV",
            data=csv_data,
            file_name=f"uvvis_analysis_{len(results)}_methods.csv",
            mime="text/csv",
            use_container_width=True
        )

st.markdown("""
---
**UV-Vis TD-DFT Analyzer** • *Built for computational chemists*  
[GitHub](https://github.com/YOUR_USERNAME/uvvis-tddft-analyzer) • Optimized for ORCA/Gaussian .stk outputs
""")

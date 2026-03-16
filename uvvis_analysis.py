#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
import pandas as pd
import argparse
import os

def create_spectrum(stk_data, emin=3.8, emax=5.6, delta=0.01, shift=0.0, width=0.05):
    x_ev = np.arange(emin, emax, delta)
    y_ev = np.zeros_like(x_ev)
    
    for i in range(len(x_ev)):
        for j in range(stk_data.shape[0]):
            E = stk_data[j, 0] / 8065.54429  # cm⁻¹ to eV
            y_ev[i] += stk_data[j,1] * np.exp(-((x_ev[i]-E-shift)/(np.sqrt(2)*width))**2)
    
    return x_ev, y_ev/np.max(y_ev)

def calculate_similarity_in_range(exp_data, theory_x, theory_y, exp_emin, exp_emax):
    mask = (theory_x >= exp_emin) & (theory_x <= exp_emax)
    theory_x_filtered = theory_x[mask]
    theory_y_filtered = theory_y[mask]
    
    cs = CubicSpline(exp_data[:,0], exp_data[:,1])
    y_exp = cs(theory_x_filtered)
    
    fg = simpson(y_exp * theory_y_filtered, x=theory_x_filtered)
    f2 = simpson(y_exp**2, x=theory_x_filtered)
    g2 = simpson(theory_y_filtered**2, x=theory_x_filtered)
    
    return fg/np.sqrt(f2*g2)

def optimize_parameters_original(exp_data, stk_data, exp_emin, exp_emax):
    def objective(params):
        shift, width = params
        x, y = create_spectrum(stk_data, exp_emin, exp_emax, shift=shift, width=width)
        return -calculate_similarity_in_range(exp_data, x, y, exp_emin, exp_emax)
    
    # Initial parameters and bounds IDENTICAL to original
    initial_guess = [0.0, 0.052651242]
    
    # First step: optimize shift with fixed width
    result = minimize(objective, initial_guess,
                     bounds=((-1.5, 1.5), (0.052651242, 0.052651242)),
                     method='Nelder-Mead')
    
    # Second step: optimize width with fixed shift
    result = minimize(objective, result.x,
                     bounds=((result.x[0], result.x[0]), (0.0, 0.5)),
                     method='Nelder-Mead')
    
    # Third step: optimize both shift and width together, starting from optimized values
    result = minimize(objective, result.x,
                     bounds=((-1.5, 1.5), (0.0, 0.5)),
                     method='Nelder-Mead')
    
    return result.x[0], result.x[1]  # shift_opt, width_opt

def load_and_preprocess_experimental_data(filepath):
    data = np.genfromtxt(filepath)
    
    # Remove rows with NaN
    data = data[~np.isnan(data).any(axis=1)]
    
    # Sort by energy (first column)
    data = data[np.argsort(data[:,0])]
    
    # Remove duplicate energy values
    unique_energies, unique_indices = np.unique(data[:,0], return_index=True)
    data = data[unique_indices]
    
    return data

def load_functionals_from_file(filepath):
    functionals = {}
    try:
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line and not line.startswith('#'):  # Ignore empty lines and comments
                    try:
                        name, path = line.split(':', 1)
                        name = name.strip()
                        path = path.strip()
                        if name and path:
                            functionals[name] = path
                        else:
                            print(f"Line {line_num}: Invalid format - '{line}'")
                    except ValueError:
                        print(f"Line {line_num}: Invalid format - '{line}'")
    except FileNotFoundError:
        print(f"Functionals file not found: {filepath}")
    except Exception as e:
        print(f"Error reading functionals file: {e}")
    
    return functionals

def normalize_transitions_in_plot_range(stk_data, plot_emin, plot_emax):
    # Convert energies to eV
    energy_eV = stk_data[:,0] / 8065.54429
    
    # Filter transitions within plot range
    mask = (energy_eV >= plot_emin) & (energy_eV <= plot_emax)
    
    if np.any(mask):
        # Normalize only transitions within range
        intensities_in_range = stk_data[mask, 1]
        if len(intensities_in_range) > 0:
            max_intensity_in_range = np.max(intensities_in_range)
            if max_intensity_in_range > 0:
                normalized_intensities = stk_data[:,1] / max_intensity_in_range
            else:
                normalized_intensities = stk_data[:,1]
        else:
            normalized_intensities = stk_data[:,1]
    else:
        # If no transitions in range, normalize by all
        max_intensity = np.max(stk_data[:,1])
        if max_intensity > 0:
            normalized_intensities = stk_data[:,1] / max_intensity
        else:
            normalized_intensities = stk_data[:,1]
    
    return energy_eV, normalized_intensities

def main():
    parser = argparse.ArgumentParser(description='Comparative analysis of UV-Vis spectra for different TD-DFT functionals')
    parser.add_argument('-e', '--experimental', required=True, help='Path to experimental file (.dat)')
    parser.add_argument('-o', '--output', default='results', help='Prefix for output files')
    parser.add_argument('-s', '--single', help='Single .stk file in the same directory as the python program')
    parser.add_argument('--plot-emin', type=float, default=1.5, help='Plot lower limit (eV)')
    parser.add_argument('--plot-emax', type=float, default=5.0, help='Plot upper limit (eV)')
    args = parser.parse_args()

    try:
        uvvis_ev = load_and_preprocess_experimental_data(args.experimental)
        exp_emin, exp_emax = np.min(uvvis_ev[:,0]), np.max(uvvis_ev[:,0])
        print(f"Experimental data: {uvvis_ev.shape[0]} points")
        print(f"Experimental range: {exp_emin:.2f} - {exp_emax:.2f} eV")
    except Exception as e:
        print(f"Error loading experimental file: {e}")
        exit(1)

    functionals = {}
    
    if args.single:
        # Handle single .stk file
        stk_filename = args.single
        if not stk_filename.endswith('.stk'):
            stk_filename += '.stk'
        
        if os.path.exists(stk_filename):
            # Use filename without extension as functional name
            functional_name = os.path.splitext(stk_filename)[0]
            functionals[functional_name] = stk_filename
            print(f"Single functional loaded: {functional_name} -> {stk_filename}")
        else:
            print(f"Error: File {stk_filename} not found in current directory")
            exit(1)
    elif args.functionals:
        functionals = load_functionals_from_file(args.functionals)
    elif args.list:
        for item in args.list:
            try:
                name, path = item.split(':', 1)
                if not os.path.exists(path):
                    print(f"File not found for {name}: {path}")
                    continue
                functionals[name] = path
            except ValueError:
                print(f"Invalid format for functional: {item}. Use 'NAME:path/to/file.stk'")
                continue
    else:
        print("Error: You must inform functionals via file (-f), list (-l), or single file (-s)")
        exit(1)

    if not functionals:
        print("No valid functional provided.")
        exit(1)

    print(f"\nFunctionals to be processed: {len(functionals)}")
    for name, path in functionals.items():
        print(f"  {name}: {path}")

    results = []
    
    n_functionals = len(functionals)
    n_cols = 3
    n_rows = (n_functionals + n_cols - 1) // n_cols  # Round up
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, 3*n_rows), dpi=100)
    axs = axs.flatten()  # Convert to 1D array for easier indexing

    for idx, (functional, path) in enumerate(functionals.items()):
        try:
            print(f"\nProcessing functional {idx+1}/{n_functionals}: {functional}...")
            stk_data = np.loadtxt(path)
            print(f"  Theoretical transitions: {stk_data.shape[0]}")
            
            # Parameter optimization within experimental range
            shift_opt, width_opt = optimize_parameters_original(uvvis_ev, stk_data, exp_emin, exp_emax)
            print(f"  Optimized shift: {shift_opt:.4f} eV")
            print(f"  Optimized width: {width_opt:.4f} eV")
            
            # Create spectrum with optimized parameters within experimental range
            x_opt, y_opt = create_spectrum(stk_data, exp_emin, exp_emax, shift=shift_opt, width=width_opt)
            
            # Calculate similarity within experimental range
            similarity = calculate_similarity_in_range(uvvis_ev, x_opt, y_opt, exp_emin, exp_emax)
            similarity_percent = similarity * 100  # Convert to percentage
            print(f"  Similarity: {similarity:.4f} ({similarity_percent:.1f}%)")
            
            # Store results (keep original values for calculations)
            results.append({
                'Functional': functional,
                'Shift (eV)': shift_opt,
                'Width (eV)': width_opt,
                'Similarity': similarity,
                'Similarity (%)': similarity_percent
            })
            
            # Plot configuration
            ax = axs[idx]
            
            # Normalize transitions only within plot range
            energy_eV, intensity_norm = normalize_transitions_in_plot_range(stk_data, args.plot_emin, args.plot_emax)
            
            # Filter transitions within plot range
            plot_mask = (energy_eV >= args.plot_emin) & (energy_eV <= args.plot_emax)
            plot_energy = energy_eV[plot_mask]
            plot_intensity = intensity_norm[plot_mask]
            
            # Theoretical lines (only within plot range)
            ax.vlines(plot_energy, 0, plot_intensity, color='red', linewidth=1)
            
            # Optimized theoretical spectrum
            ax.plot(x_opt, y_opt, 'r-', linewidth=1.5, label=f'{functional} (S={similarity_percent:.1f}%)')
            
            # Experimental spectrum
            ax.plot(uvvis_ev[:,0], uvvis_ev[:,1], 'k-', linewidth=1.2, label='Experimental')
            
            # Plot settings
            ax.set_xlabel('Energy (eV)', fontsize=9)
            ax.set_ylabel('Normalized Intensity', fontsize=9)
            ax.set_xlim(args.plot_emin, args.plot_emax)
            ax.set_ylim(0, 1.1)
            ax.grid(alpha=0.2)
            ax.legend(fontsize=7)
            ax.set_title(f'{functional}', fontsize=10)
            
        except Exception as e:
            print(f"Error in processing {functional}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Hide empty axes if there are fewer than the maximum
    for idx in range(len(functionals), len(axs)):
        axs[idx].set_visible(False)

    # Output results
    if results:
        plt.tight_layout()
        output_image = f'{args.output}_comparison.pdf'
        plt.savefig(output_image, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved as: {output_image}")
        
        # Prepare data for CSV with specific formatting
        results_df = pd.DataFrame(results)
        
        # Format columns for CSV
        csv_data = []
        for _, row in results_df.iterrows():
            csv_data.append({
                'Functional': row['Functional'],
                'Shift (eV)': f"{row['Shift (eV)']:.2f}",
                'Width (eV)': f"{row['Width (eV)']:.2f}",
                'Similarity (%)': f"{row['Similarity (%)']:.1f}"
            })
        
        results_df_csv = pd.DataFrame(csv_data)
        output_table = f'{args.output}_table.csv'
        results_df_csv.to_csv(output_table, index=False)
        print(f"Results table saved as: {output_table}")
        
        print("\nCOMPARATIVE RESULTS:")
        print(results_df_csv.to_string(index=False))
        
        # Sort by similarity (highest first) for ranking
        results_df_sorted = results_df.sort_values('Similarity', ascending=False)
        ranking_data = []
        for _, row in results_df_sorted.iterrows():
            ranking_data.append({
                'Functional': row['Functional'],
                'Similarity (%)': f"{row['Similarity (%)']:.1f}",
                'Shift (eV)': f"{row['Shift (eV)']:.2f}",
                'Width (eV)': f"{row['Width (eV)']:.2f}"
            })
        
        ranking_df = pd.DataFrame(ranking_data)
        print("\nRANKING BY SIMILARITY:")
        print(ranking_df.to_string(index=False))
        
    else:
        print("\nNo results generated. Verify errors.")

if __name__ == "__main__":
    main()

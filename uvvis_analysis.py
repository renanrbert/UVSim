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
            E = stk_data[j, 0] / 8065.54429  # cm⁻¹ para eV
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
    
    # Parâmetros iniciais e bounds IDÊNTICOS ao original
    initial_guess = [0.0, 0.052651242]
    
    # Primeiro passo: otimiza shift com width fixo
    result = minimize(objective, initial_guess,
                     bounds=((-1.5, 1.5), (0.052651242, 0.052651242)),
                     method='Nelder-Mead')
    
    # Segundo passo: otimiza width com shift fixo
    result = minimize(objective, result.x,
                     bounds=((result.x[0], result.x[0]), (0.0, 0.5)),
                     method='Nelder-Mead')
    
    return result.x[0], result.x[1]  # shift_opt, width_opt

def load_and_preprocess_experimental_data(filepath):
    data = np.genfromtxt(filepath)
    
    # Remove linhas com NaN
    data = data[~np.isnan(data).any(axis=1)]
    
    # Ordena por energia (primeira coluna)
    data = data[np.argsort(data[:,0])]
    
    # Remove valores duplicados de energia
    unique_energies, unique_indices = np.unique(data[:,0], return_index=True)
    data = data[unique_indices]
    
    return data

def load_functionals_from_file(filepath):
    funcionais = {}
    try:
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line and not line.startswith('#'):  # Ignora linhas vazias e comentários
                    try:
                        name, path = line.split(':', 1)
                        name = name.strip()
                        path = path.strip()
                        if name and path:
                            funcionais[name] = path
                        else:
                            print(f"Linha {line_num}: Formato inválido - '{line}'")
                    except ValueError:
                        print(f"Linha {line_num}: Formato inválido - '{line}'")
    except FileNotFoundError:
        print(f"Arquivo de funcionais não encontrado: {filepath}")
    except Exception as e:
        print(f"Erro ao ler arquivo de funcionais: {e}")
    
    return funcionais

def normalize_transitions_in_plot_range(stk_data, plot_emin, plot_emax):
    # Converte energias para eV
    energy_eV = stk_data[:,0] / 8065.54429
    
    # Filtra transições dentro do intervalo do plot
    mask = (energy_eV >= plot_emin) & (energy_eV <= plot_emax)
    
    if np.any(mask):
        # Normaliza apenas as transições dentro do intervalo
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
        # Se não há transições no intervalo, normaliza por todas
        max_intensity = np.max(stk_data[:,1])
        if max_intensity > 0:
            normalized_intensities = stk_data[:,1] / max_intensity
        else:
            normalized_intensities = stk_data[:,1]
    
    return energy_eV, normalized_intensities

def main():
    parser = argparse.ArgumentParser(description='Análise comparativa de espectros UV-Vis para diferentes funcionais TD-DFT')
    parser.add_argument('-e', '--experimental', required=True, help='Caminho para o arquivo experimental (.dat)')
    parser.add_argument('-o', '--output', default='results', help='Prefixo para os arquivos de saída')
    parser.add_argument('-f', '--functionals', help='Caminho para arquivo com lista de funcionais (.txt)')
    parser.add_argument('-l', '--list', nargs='+', help='Lista direta de funcionais no formato "NOME:caminho/arquivo.stk"')
    parser.add_argument('--plot-emin', type=float, default=1.5, help='Limite inferior do plot (eV)')
    parser.add_argument('--plot-emax', type=float, default=5.0, help='Limite superior do plot (eV)')
    args = parser.parse_args()

    try:
        uvvis_ev = load_and_preprocess_experimental_data(args.experimental)
        exp_emin, exp_emax = np.min(uvvis_ev[:,0]), np.max(uvvis_ev[:,0])
        print(f"Experimental data: {uvvis_ev.shape[0]} points")
        print(f"Experimental range: {exp_emin:.2f} - {exp_emax:.2f} eV")
    except Exception as e:
        print(f"Error loading experimental file: {e}")
        exit(1)

    funcionais = {}
    
    if args.functionals:
        funcionais = load_functionals_from_file(args.functionals)
    elif args.list:
        for item in args.list:
            try:
                name, path = item.split(':', 1)
                if not os.path.exists(path):
                    print(f"File not found for {name}: {path}")
                    continue
                funcionais[name] = path
            except ValueError:
                print(f"Invalid format for functional: {item}. Use 'NOME:caminho/arquivo.stk'")
                continue
    else:
        print("Error: You must inform functionals via file (-f) or list (-l)")
        exit(1)

    if not funcionais:
        print("No valid functional provided.")
        exit(1)

    print(f"\nFunctionals to be processed: {len(funcionais)}")
    for name, path in funcionais.items():
        print(f"  {name}: {path}")

    results = []
    
    n_funcionais = len(funcionais)
    n_cols = 3
    n_rows = (n_funcionais + n_cols - 1) // n_cols  # Arredonda para cima
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, 3*n_rows), dpi=100)
    axs = axs.flatten()  # Converte para array 1D para facilitar indexação

    for idx, (functional, path) in enumerate(funcionais.items()):
        try:
            print(f"\nProcessing functional {idx+1}/{n_funcionais}: {functional}...")
            stk_data = np.loadtxt(path)
            print(f"  Theoretical transitions: {stk_data.shape[0]}")
            
            # Otimização dos parâmetros no intervalo experimental
            shift_opt, width_opt = optimize_parameters_original(uvvis_ev, stk_data, exp_emin, exp_emax)
            print(f"  Optimized shift: {shift_opt:.4f} eV")
            print(f"  Optimized width: {width_opt:.4f} eV")
            
            # Cria espectro com parâmetros otimizados no intervalo experimental
            x_opt, y_opt = create_spectrum(stk_data, exp_emin, exp_emax, shift=shift_opt, width=width_opt)
            
            # Calcula similaridade no intervalo experimental
            similarity = calculate_similarity_in_range(uvvis_ev, x_opt, y_opt, exp_emin, exp_emax)
            similarity_percent = similarity * 100  # Converte para porcentagem
            print(f"  Similarity: {similarity:.4f} ({similarity_percent:.1f}%)")
            
            # Armazena resultados (mantém valores originais para cálculos)
            results.append({
                'Functional': functional,
                'Shift (eV)': shift_opt,
                'Width (eV)': width_opt,
                'Similarity': similarity,
                'Similarity (%)': similarity_percent
            })
            
            # Configuração do gráfico
            ax = axs[idx]
            
            # Normaliza as transições apenas dentro do intervalo do plot
            energy_eV, intensity_norm = normalize_transitions_in_plot_range(stk_data, args.plot_emin, args.plot_emax)
            
            # Filtra transições dentro do intervalo do plot
            plot_mask = (energy_eV >= args.plot_emin) & (energy_eV <= args.plot_emax)
            plot_energy = energy_eV[plot_mask]
            plot_intensity = intensity_norm[plot_mask]
            
            # Linhas teóricas (apenas no intervalo do plot)
            ax.vlines(plot_energy, 0, plot_intensity, color='red', linewidth=1)
            
            # Espectro teórico otimizado
            ax.plot(x_opt, y_opt, 'r-', linewidth=1.5, label=f'{functional} (S={similarity_percent:.1f}%)')
            
            # Espectro experimental
            ax.plot(uvvis_ev[:,0], uvvis_ev[:,1], 'k-', linewidth=1.2, label='Experimental')
            
            # Configurações do gráfico
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

    # Esconde eixos vazios se tiver menos de 18 funcionais
    for idx in range(len(funcionais), len(axs)):
        axs[idx].set_visible(False)

    # 5. Saída dos resultados
    if results:
        plt.tight_layout()
        output_image = f'{args.output}_comparison.pdf'
        plt.savefig(output_image, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved as: {output_image}")
        
        # Prepara dados para CSV com formatação específica
        results_df = pd.DataFrame(results)
        
        # Formata as colunas para CSV
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
        
        # Ordena por similaridade (maior primeiro) para o ranking
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

import networkx as nx               
import matplotlib.pyplot as plt     
import numpy as np                  
from scipy.integrate import solve_ivp 
import scipy.sparse as sp 
from scipy.stats import qmc 
import pandas as pd          
import os                    
import time

# ==========================================
# 1. PARAMETERS (BASE VALUES)
# ==========================================
BASE_PARAMS = {
    'D_COEFF': 2.0, 'D_JA': 2.0, 'D_SA': 2.0, 'D_JAZ9': 2.0,
    'JAZ9startingvalue': 630000, 'P0startingvalue': 50,
    'AREA': 1039.0, 'WEIGHT': 20.0,
    'GAMMA': 2e-6,
    'K_JA_DECAY': 2e-6, 'K_SA_DECAY': 2e-6, 'K_JAZ9_DECAY': 2e-6,
    'LAMBDAGJA': 0.184, 'LAMBDAGSA': 0.184, 'LAMBDAGJAZ9': 0.184,
    'LAMBDAEJA': 2e-6, 'LAMBDAESA': 2e-6,
    'ALPHAJA': 66.0, 'ALPHASA': 66.0, 'ALPHAJAZ9': 66.0,
    'K_TRANS': 13.3, 'K_CAT': 1.5, 'JAZ9synthesis': 20.0,
    'K': 0.00026, 'dcJA_JAZ9': 0.00000001,
    'WJA_JA': 1.0, 'WSA_JA': -1.0, 'WP0_SA': 1.0
}

HOLE_DIST_THRESHOLD = 8.0 
A_DIST_THRESHOLD = 23.0
B_DIST_THRESHOLD = 38.0
C_DIST_THRESHOLD = 48.0
T_MAX = 11700      
DT = 20

# Validation Data
VAL_0 = 63
VAL_A15, VAL_A30, VAL_A60, VAL_A90, VAL_A195 = 48, 29, 27, 26, 27
VAL_A = 36.7
VAL_B15, VAL_B30, VAL_B60, VAL_B90, VAL_B195 = 55, 32, 25, 25, 26
VAL_B = 37.7
VAL_C15, VAL_C30, VAL_C60, VAL_C90, VAL_C195 = 65, 50, 40, 37, 33
VAL_C = 48.0
VAL = 40.8

class LeafSystem:
    def __init__(self, params=BASE_PARAMS, rows=35, cols=35, radius=1.0):
        self.p = params.copy()
        self.rows = rows; self.cols = cols; self.radius = radius
        self.G = nx.Graph()
        self.build_grid()
        self.nodes_list = list(self.G.nodes)
        self.node_to_idx = {node: i for i, node in enumerate(self.nodes_list)}
        self.num_cells = len(self.nodes_list)
        
        self.adjacency = []
        for i, node in enumerate(self.nodes_list):
            neighbor_indices = []
            if self.G.nodes[node]['state'] != 'dead':
                for n in self.G.neighbors(node):
                    if self.G.nodes[n]['state'] != 'dead':
                        neighbor_indices.append(self.node_to_idx[n])
            self.adjacency.append(neighbor_indices)

        self.dead_indices = [i for i, n in enumerate(self.nodes_list) if self.G.nodes[n]['state'] == 'dead']
        self.zoneA_indices = []
        self.zoneB_indices = []
        self.zoneC_indices = []
        
        for node in self.nodes_list:
            state = self.G.nodes[node]['state']
            idx = self.node_to_idx[node]
            if state == 'zoneA': self.zoneA_indices.append(idx)
            if state == 'zoneB': self.zoneB_indices.append(idx)
            if state == 'zoneC': self.zoneC_indices.append(idx)

        # Sparse Matrices
        row_A, col_A, data_A = [], [], []
        row_L, col_L, data_L = [], [], []
        for i, neighbors in enumerate(self.adjacency):
            if not neighbors: continue 
            degree = len(neighbors)
            row_L.append(i); col_L.append(i); data_L.append(-float(degree))
            for n in neighbors:
                row_A.append(i); col_A.append(n); data_A.append(1.0)
                row_L.append(i); col_L.append(n); data_L.append(1.0)
        N = self.num_cells
        self.A_matrix = sp.csr_matrix((data_A, (row_A, col_A)), shape=(N, N))
        self.L_matrix = sp.csr_matrix((data_L, (row_L, col_L)), shape=(N, N))

    def calculate_hex_corners(self, cx, cy):
        corners = []
        for i in range(6):
            rad = np.deg2rad(60 * i)
            px = cx + self.radius * np.cos(rad)
            py = cy + self.radius * np.sin(rad)
            corners.append([px, py])
        return np.array(corners)

    def build_grid(self):
        x_spacing = 1.5 * self.radius
        y_spacing = np.sqrt(3) * self.radius
        for c in range(self.cols):
            for r in range(self.rows):
                cx = c * x_spacing
                cy = r * y_spacing + (y_spacing/2 if c % 2 else 0)
                dist = np.sqrt((cx - self.cols*x_spacing//2)**2 + (cy - self.rows*y_spacing//2)**2)
                
                if dist < HOLE_DIST_THRESHOLD: state = 'dead'
                elif dist < A_DIST_THRESHOLD: state = 'zoneA'
                elif dist < B_DIST_THRESHOLD: state = 'zoneB'
                elif dist < C_DIST_THRESHOLD: state = 'zoneC'
                else: state = 'zoneD'
                
                self.G.add_node((c, r), center=(cx, cy), dist=dist,
                                vertices=self.calculate_hex_corners(cx, cy), state=state)

        for c in range(self.cols):
            for r in range(self.rows):
                offsets = [(0,-1), (0,1), (-1,-1), (-1,0), (1,-1), (1,0)] if c%2==0 else [(0,-1), (0,1), (-1,0), (-1,1), (1,0), (1,1)]
                for dc, dr in offsets:
                    nb = (c + dc, r + dr)
                    if self.G.has_node(nb): self.G.add_edge((c, r), nb)

    def ode_system(self, t, y):
        p = self.p 
        N = self.num_cells
        
        y = np.clip(y, -1e100, 1e100)
        
        P0 = y[0:N]; GJA = y[N:2*N]; EJA = y[2*N:3*N]; JA = y[3*N:4*N]
        GSA = y[4*N:5*N]; ESA = y[5*N:6*N]; SA = y[6*N:7*N]; JAZ9 = y[7*N:8*N]
        
        sum_P0 = self.A_matrix.dot(P0); sum_JA = self.A_matrix.dot(JA); sum_SA = self.A_matrix.dot(SA)
        diff_P0 = self.L_matrix.dot(P0); diff_JA = self.L_matrix.dot(JA); diff_SA = self.L_matrix.dot(SA); diff_JAZ9 = self.L_matrix.dot(JAZ9)
        
        dP0 = (p['D_COEFF']*p['WEIGHT'] / p['AREA']) * diff_P0 - (p['GAMMA'] * P0)
        
        nJA_sq = np.square(sum_JA); nSA_sq = np.square(sum_SA); K_sq = p['K']**2
        hill_GJA_JA = nJA_sq / (K_sq + nJA_sq)
        hill_GJA_SA = nSA_sq / (K_sq + nSA_sq)
        
        oJA = (p['WJA_JA'] * hill_GJA_JA) + (p['WSA_JA'] * hill_GJA_SA)
        TJA = np.maximum(0, oJA) 
        trans_JA = p['ALPHAJA'] * (TJA**2 / (1 + TJA**2 ))
        dGJA = trans_JA - (p['LAMBDAGJA'] * GJA)
        
        dEJA = (p['K_TRANS'] * GJA) - (p['LAMBDAEJA'] * EJA)
        dJA = (p['K_CAT'] * EJA) + (p['K_CAT'] * P0) - (p['K_JA_DECAY'] * JA) + ((p['D_JA'] * p['WEIGHT'] / p['AREA']) * diff_JA)
        
        nP0_sq = np.square(sum_P0)
        hill_GSA_P0 = nP0_sq / (K_sq + nP0_sq)
        oSA = (p['WP0_SA'] * hill_GSA_P0)
        TSA = np.maximum(0, oSA)
        trans_SA = p['ALPHASA'] * (TSA**2 / (1 + TSA**2))
        dGSA = trans_SA - (p['LAMBDAGSA'] * GSA)
        
        dESA = (p['K_TRANS'] * GSA) - (p['LAMBDAESA'] * ESA)
        dSA = (p['K_CAT'] * ESA) - (p['K_SA_DECAY'] * SA) + ((p['D_SA'] * p['WEIGHT'] / p['AREA']) * diff_SA)
        dJAZ9 = ((p['D_JAZ9'] * p['WEIGHT'] / p['AREA']) * diff_JAZ9) - (JA * JAZ9 * p['dcJA_JAZ9']) - (p['K_JAZ9_DECAY'] * JAZ9) + p['JAZ9synthesis']
        
        dydt = np.concatenate((dP0, dGJA, dEJA, dJA, dGSA, dESA, dSA, dJAZ9))
        if self.dead_indices:
            for k in range(8): dydt[k*N : (k+1)*N][self.dead_indices] = 0
        return dydt

    def run_simulation(self):
        N = self.num_cells
        y0 = np.zeros(self.num_cells * 8)
        y0[7*N : 8*N] = self.p['JAZ9startingvalue']
        for node in self.nodes_list:
            if self.G.nodes[node]['dist'] < 12.0 and self.G.nodes[node]['state'] != 'dead':
                y0[self.node_to_idx[node]] = self.p['P0startingvalue'] 
        
        t_eval = np.linspace(0, T_MAX, int(T_MAX/DT))
        
        sol = solve_ivp(self.ode_system, (0, T_MAX), y0, t_eval=t_eval, method='RK45', dense_output=True)
        return sol

if __name__ == "__main__":
    import optuna  # <-- The new, modern AI optimizer!

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = f"results_optimized_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    keys_to_vary = [
        'D_COEFF', 'D_JA', 'D_SA', 'D_JAZ9', 'GAMMA', 'K_JA_DECAY', 'K_SA_DECAY', 'K_JAZ9_DECAY',
        'LAMBDAGJA', 'LAMBDAGSA', 'LAMBDAGJAZ9', 'LAMBDAEJA', 'LAMBDAESA',
        'ALPHAJA', 'ALPHASA', 'ALPHAJAZ9', 'K_TRANS', 'K_CAT', 'K', 'dcJA_JAZ9', 'JAZ9synthesis', 'JAZ9startingvalue', 'P0startingvalue'
    ]
    
    # ==========================================
    # 1. OPTUNA OBJECTIVE FUNCTION
    # ==========================================
    def objective(trial):
        start_time = time.time()
        current_params = BASE_PARAMS.copy()
        
        # Optuna "suggests" a value for every parameter.
        # log=True is the secret weapon here. It searches across magnitudes efficiently!
        for key in keys_to_vary:
            low_bound = BASE_PARAMS[key] * 0.1
            high_bound = BASE_PARAMS[key] * 10.0
            current_params[key] = trial.suggest_float(key, low_bound, high_bound, log=True)
            
        try:
            leaf = LeafSystem(params=current_params, rows=35, cols=35)
            solution = leaf.run_simulation()
            
            t_specific = [0.0, 900.0, 1800.0, 3600.0, 5400.0, 11700.0, 95400]
            sol_states = solution.sol(t_specific)
            N = leaf.num_cells
            
            zoneA_vals, zoneB_vals, zoneC_vals = {}, {}, {}
            for i, t_val in enumerate(t_specific):
                vals = sol_states[7*N : 8*N, i]
                zoneA_vals[t_val] = np.mean(vals[leaf.zoneA_indices]) if len(leaf.zoneA_indices) > 0 else 0.0
                zoneB_vals[t_val] = np.mean(vals[leaf.zoneB_indices]) if len(leaf.zoneB_indices) > 0 else 0.0
                zoneC_vals[t_val] = np.mean(vals[leaf.zoneC_indices]) if len(leaf.zoneC_indices) > 0 else 0.0    
                
            povp = (zoneA_vals[900]+zoneA_vals[1800]+zoneA_vals[3600]+zoneA_vals[5400]+zoneA_vals[11700]+zoneA_vals[95400]+current_params['JAZ9startingvalue']
            +zoneB_vals[900]+zoneB_vals[1800]+zoneB_vals[3600]+zoneB_vals[5400]+zoneB_vals[11700]+zoneB_vals[95400]+current_params['JAZ9startingvalue']
            +zoneC_vals[900]+zoneC_vals[1800]+zoneC_vals[3600]+zoneC_vals[5400]+zoneC_vals[11700]+zoneC_vals[95400]+current_params['JAZ9startingvalue'])/18
            
            povp = (zoneA_vals[900]+zoneA_vals[1800]+zoneA_vals[3600]+zoneA_vals[5400]+zoneA_vals[11700]+current_params['JAZ9startingvalue']
            +zoneB_vals[900]+zoneB_vals[1800]+zoneB_vals[3600]+zoneB_vals[5400]+zoneB_vals[11700]+current_params['JAZ9startingvalue']
            +zoneC_vals[900]+zoneC_vals[1800]+zoneC_vals[3600]+zoneC_vals[5400]+zoneC_vals[11700]+current_params['JAZ9startingvalue'])/18
            if povp == 0: povp = 1e-9
            norm = povp / VAL
            
            err9A = zoneA_vals[900]/norm - VAL_A15
            err18A = zoneA_vals[1800]/norm - VAL_A30
            err36A = zoneA_vals[3600]/norm - VAL_A60
            err54A = zoneA_vals[5400]/norm - VAL_A90
            err117A = zoneA_vals[11700]/norm - VAL_A195
            err0A = current_params['JAZ9startingvalue']/norm - VAL_0
            total_errA = abs(err9A) + abs(err18A) + abs(err36A) + abs(err54A) + abs(err117A) + abs(err0A)
            err9B = zoneB_vals[900]/norm - VAL_B15
            err18B = zoneB_vals[1800]/norm - VAL_B30
            err36B = zoneB_vals[3600]/norm - VAL_B60
            err54B = zoneB_vals[5400]/norm - VAL_B90
            err117B = zoneB_vals[11700]/norm - VAL_B195
            err0B = current_params['JAZ9startingvalue']/norm - VAL_0
            total_errB = abs(err9B) + abs(err18B) + abs(err36B) + abs(err54B) + abs(err117B) + abs(err0B)
            err9C = zoneC_vals[900]/norm - VAL_C15
            err18C = zoneC_vals[1800]/norm - VAL_C30
            err36C = zoneC_vals[3600]/norm - VAL_C60
            err54C = zoneC_vals[5400]/norm - VAL_C90
            err117C = zoneC_vals[11700]/norm - VAL_C195
            err0C = current_params['JAZ9startingvalue']/norm - VAL_0
            total_errC = abs(err9C) + abs(err18C) + abs(err36C) + abs(err54C) + abs(err117C) + abs(err0C)
            total_err = total_errA + total_errB + total_errC
            
            return total_err
            
        except Exception as e:
            # Tell Optuna this branch failed so it avoids this mathematical region
            raise optuna.TrialPruned()

    # ==========================================
    # 2. RUNNING OPTUNA
    # ==========================================
    print("Starting Optuna AI Optimization...")
    optuna.logging.set_verbosity(optuna.logging.INFO) # Prints clean, easy-to-read updates
    
    # Create a "Study" (Optuna's name for an optimization session)
    study = optuna.create_study(direction="minimize")
   # --- ADD THIS LINE: Force Optuna to test your original parameters FIRST ---
    study.enqueue_trial(BASE_PARAMS)
    # Run it for exactly 150 guesses. (You can change this number)
    study.optimize(objective, n_trials=1000)
    # ==========================================
    # 2.5 GRAPHING OPTIMIZATION HISTORY
    # ==========================================
    print("\nGenerating optimization history graph...")
    try:
        # Extract the error values from all completed trials
        completed_trials = [t for t in study.trials if t.value is not None]
        trial_nums = [t.number for t in completed_trials]
        errors = [t.value for t in completed_trials]
        
        # Calculate the "best error so far" to draw the descending red line
        best_errors = [min(errors[:i+1]) for i in range(len(errors))]
        
        fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
        
        # Plot all guesses as faint blue dots
        ax_hist.scatter(trial_nums, errors, alpha=0.5, color='blue', label='Individual Guess (Trial)')
        
        # Plot the "Best Score" as a thick red line
        ax_hist.plot(trial_nums, best_errors, color='red', linewidth=2, label='Best Error So Far')
        
        ax_hist.set_title("Optuna AI Learning Progress")
        ax_hist.set_xlabel("Trial Number")
        ax_hist.set_ylabel("Total Error Score")
        
        # Put a limit on the Y-axis so massive errors don't squash the graph flat
        max_y = max(best_errors) * 3 if best_errors else 1000
        ax_hist.set_ylim(0, max_y)
        
        ax_hist.grid(True, alpha=0.3)
        ax_hist.legend()
        
        # Save the picture to the folder
        plt.savefig(f"{output_dir}/optimization_history_graph.png", bbox_inches='tight', dpi=150)
        plt.close(fig_hist)
        print(f"Learning progress graph saved to {output_dir}/optimization_history_graph.png")
        
    except Exception as e:
        print(f"Failed to generate optimization history graph: {e}")

    # ==========================================
    # 3. EXTRACTING AND GRAPHING THE BEST RESULT
    # ==========================================
    print("\n" + "="*50)
    print("Optimization Finished!")
    print(f"The absolute best error found was: {study.best_value:.4f}")
    
    # Optuna perfectly remembers the best parameters in study.best_params
    best_params = BASE_PARAMS.copy()
    best_params.update(study.best_params)

    # Save to CSV
    results_data = [{'Error': study.best_value, 'Note': 'OPTUNA_BEST'}]
    results_data[0].update(best_params)
    pd.DataFrame(results_data).to_csv(f"{output_dir}/optuna_best_parameters.csv", index=False)
    print(f"Best parameters saved to {output_dir}/optuna_best_parameters.csv")

    print("\nRunning one final simulation to generate the graph of the best result...")
    try:
        leaf = LeafSystem(params=best_params, rows=35, cols=35)
        solution = leaf.run_simulation()
        N = leaf.num_cells
        all_JAZ9 = solution.y[7*N:8*N, :]
        
        fig, ax = plt.subplots(figsize=(11, 5.5))
        plt.subplots_adjust(right=0.6)
        
        idx_A = leaf.zoneA_indices
        idx_B = leaf.zoneB_indices
        idx_C = leaf.zoneC_indices
        
        if len(idx_A) > 0:
            ax.plot(solution.t, np.mean(all_JAZ9[idx_A], axis=0), color='purple', linestyle='-', linewidth=2, label='Zone A (JAZ9)')
        if len(idx_B) > 0:
            ax.plot(solution.t, np.mean(all_JAZ9[idx_B], axis=0), color='blue', linestyle='--', linewidth=2, label='Zone B (JAZ9)')
        if len(idx_C) > 0:
            ax.plot(solution.t, np.mean(all_JAZ9[idx_C], axis=0), color='green', linestyle=':', linewidth=2, label='Zone C (JAZ9)')
        
        ax.set_title(f"OPTUNA OPTIMIZED JAZ9 Concentration | Total Error: {study.best_value:.2f}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("JAZ9 Concentration")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        text_str =  f"--- OPTUNA BEST RESULT ---\n\n"
        text_str += f"TOTAL ERROR : {study.best_value:.4f}\n\n"
        text_str += f"--- OPTIMIZED PARAMETERS ---\n"
        
        for key in keys_to_vary:
            val = best_params[key]
            if val < 0.01 or val > 10000:
                text_str += f"{key:<15}: {val:.3e}\n"
            else:
                text_str += f"{key:<15}: {val:.4f}\n"
        
        fig.text(0.63, 0.5, text_str, fontsize=9, va='center', ha='left', 
                 family='monospace', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.savefig(f"{output_dir}/optuna_best_graph.png", bbox_inches='tight', dpi=150)
        plt.close()
        print("Graph generated successfully!")
        
    except Exception as e:
        print(f"Failed to generate final graph: {e}")
    # Save the CSV inside the same unique folder
    pd.DataFrame(results_data).to_csv(f"{output_dir}/sensitivity_results.csv", index=False)
    print("Done.")
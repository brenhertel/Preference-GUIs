import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

from itertools import permutations

from cvx_ELM import *
from utils import *
from downsampling import *

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter


class meta_opt(ElementwiseProblem):

    def __init__(self, demo):
        super().__init__(n_var=5,
                         n_obj=3,
                         n_constr=0,
                         xl=np.array([1e-6] * 5),
                         xu=np.array([1e6] * 5))
        self.thresh = 1e-3
        
        self.demo = demo
        (self.n_pts, self.n_dims) = np.shape(self.demo)
        
    def set_constraints(self, inds, csts):
        self.cst_inds = inds
        self.cst_pts = csts
        #print(self.cst_inds)
        #print(self.cst_pts)
        
    def _evaluate(self, X, out, *args, **kwargs):
        PA = ElMap_Perturbation_Analysis(self.demo, spatial=X[0], shape=X[1], tangent=X[2], stretch=X[3], bend=X[4])
        x_prob = PA.setup_problem()
        constraints = []
        for i in range(len(self.cst_inds)):
            for j in range(self.n_dims):
                #print([i, j])
                #print(self.cst_inds[i] + (PA.n_pts * j))
                #print(self.cst_pts[i, j])
                #print(PA.traj_stacked[self.cst_inds[i] + (PA.n_pts * j)])
                constraints.append( cp.abs(x_prob[self.cst_inds[i] + (PA.n_pts * j)] - self.cst_pts[i, j]) <= 0 )
            
        sol = PA.solve_problem(constraints, disp=False)
        
        f1 = sum_of_squared_error(self.demo, sol)
        f2 = angular_similarity(self.demo, sol)
        f3 = calc_jerk(sol)

        out["F"] = [f1, f2, f3]

def get_rand_sol(demo, X, inds, csts):
    (n_pts, n_dims) = np.shape(demo)
    (num_sols, num_vars) = np.shape(X)
    print(np.shape(X))
    choice1 = np.random.choice(num_sols)
    vars1 = X[choice1, :]
    print(vars1)
    PA = ElMap_Perturbation_Analysis(demo, spatial=vars1[0], shape=vars1[1], tangent=vars1[2], stretch=vars1[3], bend=vars1[4])
    x_prob = PA.setup_problem()
    constraints = []
    for i in range(len(inds)):
        for j in range(n_dims):
            constraints.append( cp.abs(x_prob[inds[i] + (PA.n_pts * j)] - csts[i, j]) <= 0 )
    sol = PA.solve_problem(constraints, disp=False)
    return sol
    
def get_sol(demo, params, inds, csts):
    (n_pts, n_dims) = np.shape(demo)
    PA = ElMap_Perturbation_Analysis(demo, spatial=params[0], shape=params[1], tangent=params[2], stretch=params[3], bend=params[4])
    x_prob = PA.setup_problem()
    constraints = []
    for i in range(len(inds)):
        for j in range(n_dims):
            constraints.append( cp.abs(x_prob[inds[i] + (PA.n_pts * j)] - csts[i, j]) <= 0 )
    sol = PA.solve_problem(constraints, disp=False)
    return sol

## IMPLEMENT VOTING
# overall min
# min in each objective
# biggest "traingle" (three points with most distance) (three votes)

# X: candidates (outputs to be voted on)
# F: voting criteria (metrics for optimization)
# num_out: number of candidates returned
# ppr: participants per round (to split voting & make it faster)
# num_rounds: number of voting rounds
def voting_rounds(X, F, num_out=3, ppr=10, num_rounds=20):
    num_can, f_dim = np.shape(F)
    F_normalized = np.zeros(np.shape(F))
    for i in range(f_dim):
        F_normalized[:, i] = F[:, i] / max(F[:, i])
    F = F_normalized
    votes = np.zeros((num_can, ))
    normed_F = np.linalg.norm(F, axis=1)
    for rnd in range(num_rounds):
        sel_inds = np.random.choice(num_can, size=ppr, replace=False)
        print(sel_inds)
        print(F[sel_inds])
        print(normed_F[sel_inds])
        
        #overall min
        overall_min_index = sel_inds[np.argmin(normed_F[sel_inds])]
        votes[overall_min_index] = votes[overall_min_index] + 1
        print("overall min ind: " + str(overall_min_index))
        
        #min in each dim
        for i in range(f_dim):
            min_dim_ind = sel_inds[np.argmin(F[sel_inds, i])]
            votes[min_dim_ind] = votes[min_dim_ind] + 1
            print("min dim " + str(i) + " ind: " + str(min_dim_ind))
            
        #biggest triangle
        all_triples = list(permutations(sel_inds, 3))
        biggest_triple = None
        biggest_perimeter = 0.
        for triple in all_triples:
            perimeter = np.linalg.norm(F[triple[0]] - F[triple[1]]) + np.linalg.norm(F[triple[1]] - F[triple[2]]) + np.linalg.norm(F[triple[2]] - F[triple[0]])
            if perimeter > biggest_perimeter:
                biggest_triple = triple
                biggest_perimeter = perimeter
        votes[biggest_triple[0]] = votes[biggest_triple[0]] + 1
        votes[biggest_triple[1]] = votes[biggest_triple[1]] + 1
        votes[biggest_triple[2]] = votes[biggest_triple[2]] + 1
        print("triple inds: " + str(biggest_triple))
        
    print(F)
    print(votes)
    best_candidates = []
    for i in range(num_out):
        best_candidates.append(np.argmax(votes))
        votes[np.argmax(votes)] = 0
    return best_candidates
        
def naive_voting(X, F):
    num_can, f_dim = np.shape(F)
    best_candidates = []
    for i in range(f_dim):
        best_candidates.append(np.argmin(F[:, i]))
    print(best_candidates)
    return best_candidates

def biggest_tri_voting(X, F):
    all_triples = list(permutations(np.arange(len(F)), 3))
    biggest_triple = None
    biggest_area = 0.
    for triple in all_triples:
        area = herons_formula(np.linalg.norm(F[triple[0]] - F[triple[1]]), np.linalg.norm(F[triple[1]] - F[triple[2]]), np.linalg.norm(F[triple[2]] - F[triple[0]]))
        if area > biggest_area:
            biggest_triple = triple
            biggest_area = area
    return biggest_triple
    
def smallest_tri_round(f_sel, F):
    all_doubles = list(permutations(np.arange(len(F)), 2))
    smallest_double = None
    smallest_area = float('inf')
    for double in all_doubles:
        if f_sel != double[0] and f_sel != double[1]:
            area = herons_formula(np.linalg.norm(F[double[0]] - F[double[1]]), np.linalg.norm(F[double[1]] - F[f_sel]), np.linalg.norm(F[f_sel] - F[double[0]]))
            if area < smallest_area:
                smallest_double = double
                smallest_area = area
    return [smallest_double[0], smallest_double[1], f_sel]
    
    

def build_string(F, idx, obj_names):
    num_can, f_dim = np.shape(F)
    F_normalized = np.zeros(np.shape(F))
    for i in range(f_dim):
        F_normalized[:, i] = F[:, i] / max(F[:, i])
    F = F_normalized
    obj_results = F[idx]
    out_str = ""
    for i in range(len(obj_results)):
        out_str = out_str + obj_names[i] + ": "
        if obj_results[i] > 0.7:
            out_str = out_str + "bad (" + str(round((1-obj_results[i]) * 100)) + "%)\n"
        elif obj_results[i] < 0.3:
            out_str = out_str + "good (" + str(round((1-obj_results[i]) * 100)) + "%)\n"
        else:
            out_str = out_str + "OK (" + str(round((1-obj_results[i]) * 100)) + "%)\n"
    return out_str

def main():
    # demonstration
    num_points = 50
    t = np.linspace(0, 10, num_points).reshape((num_points, 1))
    x_demo = np.sin(t) + 0.01 * t**2 - 0.05 * (t-5)**2
    
    full_demo = np.hstack((t, x_demo))
    
    problem = meta_opt(x_demo)
    inds = [0, 49]
    csts = np.array([[x_demo[0] - 1], [x_demo[-1] - 1]])
    #inds = []
    #csts = []
    problem.set_constraints(inds, csts)
    algorithm = NSGA2()
    termination = get_termination("n_eval", 3000)
    
    res = minimize(problem,
               algorithm,
               termination,
               seed=1,
               verbose=True)
    print("Optimal Parameters: ")
    print(res.X)
    print("Optimal Objectives: ")
    print(res.F)
    
    #plot = Scatter()
    #plot.add(problem.pareto_front(), color="black", alpha=0.7)
    #plot.add(res.F, color="red")
    #plot.show()
    
    cands = voting_rounds(res.X, res.F, num_out=10)
    
    num_total=3
    obj_names = ["Spatial Similarity", "Angular Similarity", "Smoothness"]
    dif_cands = [cands[0]]
    strs = [build_string(res.F, cands[0], obj_names)]
    for i in range(1, len(cands)):
        new_str = build_string(res.F, cands[i], obj_names)
        dif_str = True
        for str in strs:
            if new_str == str:
                dif_str = False
        if dif_str == True and len(dif_cands) < num_total:
            dif_cands.append(cands[i])
            strs.append(new_str)
    print(cands)
    print(dif_cands)
    
    sol1 = get_sol(x_demo, res.X[dif_cands[0]], inds, csts)
    sol1 = np.hstack((t, sol1))
    str1 = build_string(res.F, dif_cands[0], obj_names)
    sol2 = get_sol(x_demo, res.X[dif_cands[1]], inds, csts)
    sol2 = np.hstack((t, sol2))
    str2 = build_string(res.F, dif_cands[1], obj_names)
    sol3 = get_sol(x_demo, res.X[dif_cands[2]], inds, csts)
    sol3 = np.hstack((t, sol3))
    str3 = build_string(res.F, dif_cands[2], obj_names)
    
    full_csts = np.array([sol3[inds[0]], sol3[inds[1]]])
    
    traj_lists = [[sol1, str1, select_plot_1], [sol2, str2, select_plot_2], [sol3, str3, select_plot_3]]
    create_traj_ui(traj_lists, full_demo, full_csts)
    
    #plt.figure()
    #plt.plot(x_demo, 'k', lw=5, label="Demo")
    #plt.plot(sol1, 'r', lw=5, label="sol1")
    #plt.plot(sol2, 'g', lw=5, label="sol2")
    #plt.plot(sol3, 'b', lw=5, label="sol3")
    #if len(inds) > 0:
    #    plt.plot(inds[0], csts[0], 'k.', ms=15, label="Constraints")
    #for i in range(len(inds)):
    #    plt.plot(inds[i], csts[i], 'k.', ms=15)
    #plt.legend()
    #plt.title("Multi-Objective Optimization Results")
    #plt.show()
    
def main_3d():
    # demonstration
    traj = read_RAIL_demo('PUSHING', 1, 1)
    x_demo, x_inds = DouglasPeuckerPoints2(traj, 50)
    
    print(np.shape(x_demo))
    n_pts, n_dims = np.shape(x_demo)
    
    problem = meta_opt(x_demo)
    inds = [0, n_pts-1]
    csts = np.array([x_demo[0], x_demo[-1]])
    #inds = []
    #csts = []
    problem.set_constraints(inds, csts)
    algorithm = NSGA2()
    termination = get_termination("n_eval", 3000)
    
    res = minimize(problem,
               algorithm,
               termination,
               seed=1,
               verbose=True)
    print("Optimal Parameters: ")
    print(res.X)
    print("Optimal Objectives: ")
    print(res.F)
    
    
    cands = voting_rounds(res.X, res.F, num_out=10)
    
    num_total=3
    obj_names = ["Spatial Similarity", "Angular Similarity", "Smoothness"]
    dif_cands = [cands[0]]
    strs = [build_string(res.F, cands[0], obj_names)]
    for i in range(1, len(cands)):
        new_str = build_string(res.F, cands[i], obj_names)
        print(new_str)
        dif_str = True
        for str in strs:
            if new_str == str:
                dif_str = False
        if dif_str == True and len(dif_cands) < num_total:
            dif_cands.append(cands[i])
            strs.append(new_str)
    if len(dif_cands) < num_total:
        cands = naive_voting(res.X, res.F)
        dif_cands = []
        strs = []
        for i in range(num_total):
            dif_cands.append(cands[i])
            strs.append(build_string(res.F, cands[i], obj_names))
        
    print(cands)
    print(dif_cands)
    
    sol1 = get_sol(x_demo, res.X[dif_cands[0]], inds, csts)
    str1 = build_string(res.F, dif_cands[0], obj_names)
    sol2 = get_sol(x_demo, res.X[dif_cands[1]], inds, csts)
    str2 = build_string(res.F, dif_cands[1], obj_names)
    sol3 = get_sol(x_demo, res.X[dif_cands[2]], inds, csts)
    str3 = build_string(res.F, dif_cands[2], obj_names)
    
    #full_csts = np.array([sol3[inds[0]], sol3[inds[1]]])
    
    traj_lists = [[sol1, str1, select_plot_1], [sol2, str2, select_plot_2], [sol3, str3, select_plot_3]]
    create_traj_ui(traj_lists, x_demo, csts)
    
def main_3d_v2():
    # demonstration
    traj = read_RAIL_demo('PUSHING', 1, 1)
    x_demo, x_inds = DouglasPeuckerPoints2(traj, 50)
    
    print(np.shape(x_demo))
    n_pts, n_dims = np.shape(x_demo)
    
    problem = meta_opt(x_demo)
    inds = [0, n_pts-1]
    csts = np.array([x_demo[0], x_demo[-1]])
    #inds = []
    #csts = []
    problem.set_constraints(inds, csts)
    algorithm = NSGA2()
    termination = get_termination("n_eval", 2000)
    
    res = minimize(problem,
               algorithm,
               termination,
               seed=1,
               verbose=True)
    print("Optimal Parameters: ")
    print(res.X)
    print("Optimal Objectives: ")
    print(res.F)
    
    
    cands = biggest_tri_voting(res.X, res.F)
    print(cands)
    
    num_total=3
    obj_names = ["Spatial Similarity", "Angular Similarity", "Smoothness"]
    strs = [build_string(res.F, cands[0], obj_names), build_string(res.F, cands[1], obj_names), build_string(res.F, cands[2], obj_names)]
    
    sol1 = get_sol(x_demo, res.X[cands[0]], inds, csts)
    str1 = build_string(res.F, cands[0], obj_names)
    sol2 = get_sol(x_demo, res.X[cands[1]], inds, csts)
    str2 = build_string(res.F, cands[1], obj_names)
    sol3 = get_sol(x_demo, res.X[cands[2]], inds, csts)
    str3 = build_string(res.F, cands[2], obj_names)
    
    
    traj_lists = [[sol1, strs[0], select_plot_1], [sol2, strs[1], select_plot_2], [sol3, strs[2], select_plot_3]]
    create_traj_ui(traj_lists, x_demo, csts)
    
if __name__ == "__main__":
    main_3d_v2()

"""
File: differential_evolution.py
Author: Mauro Bandeira
Date: April 29, 2024
Description: This script takes into account the differential evolution model into searching parameters for the global minima value of a given error function.
Institution: Instituto de Física, Universidade de São Paulo
Email: mbandeira@usp.br

References:
- Characterization of structures from X-ray scattering data using genetic algorithms: https://doi.org/10.1098/rsta.1999.0469
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import floor, log
import time
from scipy.stats import pearsonr
import inspect

def read_data(file):
    df = pd.read_csv(file, sep=',')
    x = df.iloc[:,0]
    y = df.iloc[:,1]
    sx = df.iloc[:,2]
    sy = df.iloc[:,3]
    header = df.columns.values

    return np.array(x), np.array(y), np.asarray(sx), np.asarray(sy), header

def significant_figures(value, error):
    decimals_error = 1 - floor(log(error, 10))
    new_value = "{:.{precision}f}".format(value, precision=max(0, decimals_error))
    new_error = "{:.{precision}f}".format(error, precision=max(0, decimals_error))
    
    return new_value, new_error

def chi_square_test(y_exp, y_model):
    chi_square = np.abs(np.sum((y_exp - y_model)**2/y_model))

    return chi_square

def results(best_fit_params, NGL, path, x, y, sx, sy, header, data_path, graph_path, eq):
    with open(path, 'w') as file:
        print("\nBest fit parameters:")
        file.write("Best fit parameters:\n")
        for i in range(len(best_fit_params)):
            value, error = significant_figures(best_fit_params[i], best_fit_errors[i])
            string = f"{parameters[i]} = ({value} ± {error}) {unit[i]}\n"
            print(string, end='')
            file.write(string)
        file.write(f'\nFitted model:\n{eq}')
        file.write(f'\nNGL: {NGL}')
        file.write(f"\nChi-square: {chi_square_test(y_data, y_fitted)}")
        file.write(f"\nPearson's R: {R}")
        file.write(f"\np-value: {p_value}")
        file.write(f'\n\nData file: {data_path}\nGraph file: {graph_path}\n')
        file.write(f'\nData used:\n')
        for i in range(len(header)):
            file.write(f'{header[i]}\t')
        for i in range(len(x)):
            file.write(f'\n{x[i]}\t{y[i]}\t{sx[i]}\t{sy[i]}')


        print(f'\nNGL: {NGL}')
        print(f"Chi-square: {chi_square_test(y_data, y_fitted)}")
        print(f"Pearson's R: {R}")
        print(f"p-value: {p_value}")
        print(f"\nTime taken: {execution_time:.2f} ms")
        if sigma_alert == True:
            print("There was an error in the uncertainty values (0 or NaN is present). The uncertainty wasn't considered in the data fitting.")
        print(f'\nResults saved to: {path}')
    file.close()

def numerical_derivative(x, params, h=1e-5):
    y = (model(x + h, params) - model(x - h, params)) / (2 * h)
    return y

def error_function(p, x_exp, y_exp, sx_exp, sy_exp):
    y_model = model(x_exp, p)
    N = len(x_exp)
    log_exp = np.log(np.abs(y_exp))
    log_model = np.log(np.abs(y_model))
    diff = numerical_derivative(x_exp, p)
    global sigma_alert
    
    if (0 in sx_exp) or (np.isnan(sx_exp).any()) or (0 in sy_exp) or np.isnan(sy_exp).any():
        error_contribution = 1
        sigma_alert=True
    else:
        error_contribution = np.sqrt((diff * sx_exp)**2 + sy_exp**2) # σ_total² = (σ_x*d/dx f(x; P))² + σy²

    # mean-absolute error of the log transformed data: eq.(2.7) with corrections for dealing with data uncertainty
    # other error functions can be seen in the paper linked in this script's header
    sum = np.sum(np.abs(log_exp - log_model)/error_contribution)
    error = sum/(N-1)
    return error

def pointwise_error(best_fit_params, x_exp, y_exp, sx_exp, sy_exp):
    delta_factor = 1e-5
    tol = 1e-6  # tolerance factor
    base_error = error_function(best_fit_params, x_exp, y_exp, sx_exp, sy_exp)
    target_error = base_error * 1.05
    perturbations = np.zeros_like(best_fit_params)
    
    # determining the pointwise error for each parameter
    for i in range(len(best_fit_params)):
        perturbation = 0.0
        step = delta_factor * max(abs(best_fit_params[i]), 1.0)  # scale step based on parameter magnitude
        p_perturbed = best_fit_params.copy()

        while True:
            p_perturbed[i] = best_fit_params[i] + perturbation
            new_error = error_function(p_perturbed, x_exp, y_exp, sx_exp, sy_exp)
            
            if new_error > target_error:
                if step < tol:
                    break
                perturbation -= step  # go back
                step /= 2  # reduce step size
            else:
                perturbation += step
        
        perturbations[i] = perturbation

    return perturbations

    

def model(x, param):
    s, p, d, R = param
    y = np.where(x<10, R*(1/((3-p)**2) - 1/((x-s)**2)), np.where(x<20, R*(1/((3-s))**2 - 1/(x-10-p)**2), R*(1/((3-p)**2) - 1/((x-20-d)**2))))  
    # for multiple equation with shared parameters use nested np.where:
    # y = np.where(condition_1, equation_1, np.where(condition_2, equation_2, np.where(condition_3, equation_3, ...)))

    return y

def get_function_source(func):
    source_lines = inspect.getsource(func).split('\n')
    # extract the lines of the function body
    code_lines = source_lines[1:]
    # remove comments and strip whitespace
    code_lines = [line.split('#')[0].strip() for line in code_lines if line.strip()]
    # Join lines into a single string
    code_body = ' '.join(code_lines)
    code_body = code_body.replace('param ', 'param\n')
    code_body = code_body.replace('return y', '\n')
    return code_body

def differential_evolution(x, y, sx, sy, initial_guess, bounds, max_generations, mutation_constant, recombination_constant):
    n = len(initial_guess) # number of parameters to be fitted
    m = 10 * n # population size
    population = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(m, n)) # sets the initial population based on the parameter boundaries
    population[0] = initial_guess  # sets the initial guess as the first vector
    best_fit = population[0]
    best_error = error_function(population[0], x, y, sx, sy)

    generation_list = [] # for convergence graph
    error_list = [] # for convergence graph

    for generation in range(max_generations):
        for i in range(1, m):

            # Mutation
            a, b = np.random.choice(range(m), size=2, replace=False)
            mutant = best_fit + mutation_constant * (population[a] - population[b]) # mutates the best-so-far vector through eq. (2.1)

            # Recombination
            trial = np.copy(population[i])
            for j in range(n):
                if np.random.rand() <= recombination_constant:
                    trial[j] = mutant[j % n]
                else:
                    trial[j] = population[i][j % n]
                # check if t_j (trial[j]) is outside of constraint bounds and applies eq. (2.2)
                if trial[j] < bounds[j, 0] or trial[j] > bounds[j, 1]:
                    trial[j] = np.random.uniform(bounds[j, 0], bounds[j, 1])

            # Selection
            error_trial = error_function(trial, x, y, sx, sy)
            mask = error_trial <= error_function(population[i], x, y, sx, sy) # performs the inequality check from eq. (2.3)
            population[i] = np.where(mask, trial, population[i])
            if error_trial < best_error:
                best_fit = trial
                best_error = error_trial

        # when the background noise data is known (E_min), it is possible to stop the converging test through eq. (2.9):
        '''if error_function(best_fit, x, y, sx, sy) < E_min + 3 * delta_E_min:
            break
        '''
        # plot these to see the convergence 'speed'
        # generation_list.append(generation+1)
        # error_list.append(best_error) 

    return best_fit, generation_list, error_list


# user input:
data_path = r'data.dat'
graph_path = r'graph.png'
result_path =r'result.txt'
x_read, y_read, sx_read, sy_read, header = read_data(data_path)
x_data = x_read
y_data = y_read
sx_data = sx_read
sy_data = sy_read

sigma_alert = False
eq = get_function_source(model)

parameters = ['s', 'p', 'd', 'R']
unit = ['', '', '', '', 'nm^-1']
initial_guess = np.array([1.3, 0.8, 0.01, 0.0109])  # initial guess for parameters
bounds = np.array([[1.2, 1.4], [0.7, 0.9], [0, 0.02], [0.01, 0.02]])  # bounds for each parameter

# for fitting 
'''mask = (x_read > 5200) & (x_read < 5900) # limit the x-data range
x_data = x_read[mask]
sx_data = np.zeros_like(x_data)
y_data = y_read[mask]/np.sum(y_read[mask]) # normalization for P.D.F. of Lorentz-Cauchy distribution
sy_data = np.zeros_like(x_data)'''

cov_xy = np.cov((x_data, y_data))[0][1]
R, p_value = pearsonr(x_data, y_data)

k_m = 0.7 # mutation constant
k_r = 0.5 # recombination constant
max_generations = 1000

# runs DE
start_time = time.time()
best_fit_params, generation_list, error_list = differential_evolution(x_data, y_data, sx_data, sy_data, initial_guess, bounds, max_generations, k_m, k_r)
best_fit_errors = pointwise_error(best_fit_params, x_data, y_data, sx_data, sy_data)
end_time = time.time()
execution_time = (end_time - start_time)*1000

# result output:
y_fitted = model(x_data, best_fit_params)
NGL = len(x_data) - len(best_fit_params)
results(best_fit_params, NGL, result_path, x_data, y_data, sx_data, sy_data, header, data_path, graph_path, eq)

# graphs
# automatically find the x and y limits for the graph
padding = 0.1
x_min = np.min(x_data) - padding * (np.max(x_data) - np.min(x_data))
x_max = np.max(x_data) + padding * (np.max(x_data) - np.min(x_data))
y_min = np.min(y_data) - padding * (np.max(y_data) - np.min(y_data))
y_max = np.max(y_data) + padding * (np.max(y_data) - np.min(y_data))

fig = plt.figure(figsize=(15,15))
#grid = fig.add_gridspec(nrows = 2, ncols = 1, height_ratios=[3,1])
ax1 = fig.add_subplot()

# data fitting graph
x_scale = 1 # for scale correction and unit changing
y_scale = 1 # for scale correction and unit changing
x1_fit = np.linspace(-2*np.abs(np.min(x_data)), 2*np.max(x_data), 10000, endpoint=False)

ax1.plot(x_data*x_scale, y_data*y_scale, 'o', markersize=4, color = 'black')
ax1.plot(x1_fit*x_scale, model(x1_fit, best_fit_params)*y_scale, color='red', linewidth = 1)
#ax1.plot(x1_fit*x_scale, model(x1_fit, [5462, 60])*y_scale, color='red', linewidth = 1)
ax1.errorbar(x_data*x_scale, y_data*y_scale, xerr = sx_data*x_scale, yerr = sy_data*y_scale, fmt = ' ', ecolor = 'black', capsize = 2, elinewidth = 1, alpha=1)
ax1.set_title(r"Example of multiple fit with coupled equations using Differential Evolution algorithm", fontsize=24)
ax1.set_xlabel(r'Corrected quantum number $n$', fontsize=24)
ax1.set_ylabel(r'$\lambda^{-1}$ $\left[nm\right]$', fontsize=24)
ax1.set_xlim(x_min, x_max)
ax1.set_ylim(y_min, 0.0042)
ax1.grid(True)
#ax1.tick_params(axis='x', labelbottom = False, bottom = False) # removes the x-axis ticks so they won't overflow the residual plot

# residual plot
'''residual = y_data - model(np.array(x_data), best_fit_params)
ax2 = fig.add_subplot(grid[1], sharex=ax1)
ax2.axhline(y = 0, color = 'red', linewidth = 1)
ax2.plot(x_data*x_scale, residual*y_scale, 'o', markersize = 2, color = 'black', alpha=1)
ax2.errorbar(x_data*x_scale, residual*y_scale, xerr = sx_data*x_scale, yerr = sy_data*y_scale, fmt=' ', ecolor = 'black', capsize = 1, elinewidth = 1, alpha=1)
ax2.grid(True)
ax2.set_xlabel(r'Corrected quantum number $n$', fontsize=24)
ax2.set_ylabel('Residual', fontsize=14)'''

plt.subplots_adjust(hspace = 0) # joins the two graphs
plt.rcParams['figure.figsize'] = [24, 24]
plt.savefig(graph_path, bbox_inches='tight', pad_inches=0.2, format='png', dpi = 400)
plt.show()
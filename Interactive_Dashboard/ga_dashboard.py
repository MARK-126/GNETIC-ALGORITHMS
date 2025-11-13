"""
Interactive GA Dashboard with Plotly Dash
Real-time visualization of Genetic Algorithm optimization

Run with: python ga_dashboard.py
Then open http://localhost:8050 in your browser
"""

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from datetime import datetime
import json

# Import GA utilities (assuming they're in parent directory)
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'GA_Tutorial_1_Basics'))
from ga_utils_basics import *

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "GA Optimization Dashboard"

# Global variables to store optimization state
optimization_state = {
    'running': False,
    'generation': 0,
    'history': {
        'best_fitness': [],
        'mean_fitness': [],
        'worst_fitness': [],
        'diversity': [],
        'best_individual': []
    },
    'population': None,
    'best_individual': None,
    'best_fitness': -np.inf,
    'problem_type': 'sphere',
    'config': {}
}


# ==================== Fitness Functions ====================

def sphere_function(x):
    """Sphere function (minimize)."""
    return -np.sum(x**2)  # Negative for maximization


def rastrigin_function(x):
    """Rastrigin function (minimize)."""
    n = len(x)
    return -(10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))


def rosenbrock_function(x):
    """Rosenbrock function (minimize)."""
    return -np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


def ackley_function(x):
    """Ackley function (minimize)."""
    n = len(x)
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(2 * np.pi * x))
    return -(-20 * np.exp(-0.2 * np.sqrt(sum_sq / n)) - np.exp(sum_cos / n) + 20 + np.e)


PROBLEM_FUNCTIONS = {
    'sphere': sphere_function,
    'rastrigin': rastrigin_function,
    'rosenbrock': rosenbrock_function,
    'ackley': ackley_function
}


# ==================== GA Core Functions ====================

def calculate_diversity(population):
    """Calculate population diversity (std dev across all genes)."""
    return np.mean(np.std(population, axis=0))


def run_ga_generation(state):
    """Run one generation of GA and update state."""
    config = state['config']
    problem_func = PROBLEM_FUNCTIONS[state['problem_type']]

    # First generation: initialize
    if state['population'] is None:
        state['population'] = np.random.uniform(
            config['bounds'][0],
            config['bounds'][1],
            (config['pop_size'], config['dim'])
        )
        state['generation'] = 0

    # Evaluate fitness
    fitness = np.array([problem_func(ind) for ind in state['population']])

    # Track best
    best_idx = np.argmax(fitness)
    if fitness[best_idx] > state['best_fitness']:
        state['best_fitness'] = fitness[best_idx]
        state['best_individual'] = state['population'][best_idx].copy()

    # Record history
    state['history']['best_fitness'].append(fitness.max())
    state['history']['mean_fitness'].append(fitness.mean())
    state['history']['worst_fitness'].append(fitness.min())
    state['history']['diversity'].append(calculate_diversity(state['population']))
    state['history']['best_individual'].append(state['best_individual'].copy())

    # Selection (Tournament)
    selected = []
    for _ in range(config['pop_size'] - config['elite_size']):
        tournament_indices = np.random.choice(config['pop_size'], 3, replace=False)
        winner_idx = tournament_indices[np.argmax(fitness[tournament_indices])]
        selected.append(state['population'][winner_idx].copy())

    # Elitism
    elite_indices = np.argsort(fitness)[-config['elite_size']:]
    elite = [state['population'][i].copy() for i in elite_indices]

    # Crossover
    offspring = []
    for i in range(0, len(selected) - 1, 2):
        if np.random.rand() < config['crossover_rate']:
            point = np.random.randint(1, config['dim'])
            child1 = np.concatenate([selected[i][:point], selected[i+1][point:]])
            child2 = np.concatenate([selected[i+1][:point], selected[i][point:]])
            offspring.extend([child1, child2])
        else:
            offspring.extend([selected[i].copy(), selected[i+1].copy()])

    # Mutation
    for individual in offspring:
        for j in range(config['dim']):
            if np.random.rand() < config['mutation_rate']:
                individual[j] += np.random.normal(0, config['mutation_std'])
                individual[j] = np.clip(individual[j], config['bounds'][0], config['bounds'][1])

    # New population
    state['population'] = np.array(elite + offspring[:config['pop_size'] - config['elite_size']])
    state['generation'] += 1

    return state


# ==================== Dashboard Layout ====================

app.layout = html.Div([
    # Header
    html.Div([
        html.H1("🧬 Genetic Algorithm Optimization Dashboard",
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '10px'}),
        html.P("Real-time visualization of evolutionary optimization",
               style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '16px'})
    ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'marginBottom': '20px'}),

    # Controls
    html.Div([
        html.Div([
            html.H3("Optimization Problem"),
            dcc.Dropdown(
                id='problem-selector',
                options=[
                    {'label': 'Sphere Function', 'value': 'sphere'},
                    {'label': 'Rastrigin Function', 'value': 'rastrigin'},
                    {'label': 'Rosenbrock Function', 'value': 'rosenbrock'},
                    {'label': 'Ackley Function', 'value': 'ackley'}
                ],
                value='sphere',
                style={'width': '100%'}
            ),
        ], style={'width': '23%', 'display': 'inline-block', 'padding': '10px'}),

        html.Div([
            html.H3("Population Size"),
            dcc.Input(id='pop-size', type='number', value=50, min=10, max=200, step=10,
                     style={'width': '100%', 'padding': '5px'}),
        ], style={'width': '23%', 'display': 'inline-block', 'padding': '10px'}),

        html.Div([
            html.H3("Mutation Rate"),
            dcc.Slider(id='mutation-rate', min=0, max=0.5, step=0.05, value=0.1,
                      marks={i/10: str(i/10) for i in range(0, 6)},
                      tooltip={"placement": "bottom", "always_visible": True}),
        ], style={'width': '23%', 'display': 'inline-block', 'padding': '10px'}),

        html.Div([
            html.H3("Crossover Rate"),
            dcc.Slider(id='crossover-rate', min=0.5, max=1.0, step=0.05, value=0.8,
                      marks={i/10: str(i/10) for i in range(5, 11)},
                      tooltip={"placement": "bottom", "always_visible": True}),
        ], style={'width': '23%', 'display': 'inline-block', 'padding': '10px'}),
    ], style={'backgroundColor': '#f8f9fa', 'padding': '20px', 'marginBottom': '20px'}),

    # Action Buttons
    html.Div([
        html.Button('Start Optimization', id='start-button', n_clicks=0,
                   style={'backgroundColor': '#27ae60', 'color': 'white', 'padding': '10px 30px',
                         'fontSize': '16px', 'border': 'none', 'borderRadius': '5px',
                         'marginRight': '10px', 'cursor': 'pointer'}),
        html.Button('Reset', id='reset-button', n_clicks=0,
                   style={'backgroundColor': '#e74c3c', 'color': 'white', 'padding': '10px 30px',
                         'fontSize': '16px', 'border': 'none', 'borderRadius': '5px',
                         'cursor': 'pointer'}),
    ], style={'textAlign': 'center', 'marginBottom': '20px'}),

    # Status Display
    html.Div([
        html.Div(id='status-display', style={
            'textAlign': 'center', 'fontSize': '18px', 'padding': '15px',
            'backgroundColor': '#3498db', 'color': 'white', 'borderRadius': '5px'
        })
    ], style={'marginBottom': '20px'}),

    # Graphs - Row 1
    html.Div([
        html.Div([
            dcc.Graph(id='fitness-evolution-graph')
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),

        html.Div([
            dcc.Graph(id='diversity-graph')
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),
    ]),

    # Graphs - Row 2
    html.Div([
        html.Div([
            dcc.Graph(id='population-distribution')
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),

        html.Div([
            dcc.Graph(id='convergence-rate')
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),
    ]),

    # Best Solution Display
    html.Div([
        html.H3("Best Solution Found", style={'textAlign': 'center'}),
        html.Div(id='best-solution-display', style={
            'textAlign': 'center', 'fontSize': '14px', 'padding': '15px',
            'backgroundColor': '#f8f9fa', 'borderRadius': '5px'
        })
    ], style={'marginTop': '20px', 'marginBottom': '20px'}),

    # Interval component for updates
    dcc.Interval(
        id='interval-component',
        interval=500,  # milliseconds
        n_intervals=0,
        disabled=True
    ),

    # Hidden div to store state
    html.Div(id='hidden-state', style={'display': 'none'})
])


# ==================== Callbacks ====================

@app.callback(
    [Output('interval-component', 'disabled'),
     Output('status-display', 'children')],
    [Input('start-button', 'n_clicks'),
     Input('reset-button', 'n_clicks')],
    [State('problem-selector', 'value'),
     State('pop-size', 'value'),
     State('mutation-rate', 'value'),
     State('crossover-rate', 'value')]
)
def control_optimization(start_clicks, reset_clicks, problem, pop_size, mut_rate, cross_rate):
    """Control start/stop of optimization."""
    ctx = dash.callback_context

    if not ctx.triggered:
        return True, "Ready to start. Configure parameters and click 'Start Optimization'."

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'start-button' and start_clicks > 0:
        # Initialize optimization
        optimization_state['running'] = True
        optimization_state['problem_type'] = problem
        optimization_state['population'] = None
        optimization_state['generation'] = 0
        optimization_state['best_fitness'] = -np.inf
        optimization_state['history'] = {
            'best_fitness': [],
            'mean_fitness': [],
            'worst_fitness': [],
            'diversity': [],
            'best_individual': []
        }
        optimization_state['config'] = {
            'pop_size': int(pop_size),
            'dim': 5,  # Problem dimensionality
            'mutation_rate': float(mut_rate),
            'crossover_rate': float(cross_rate),
            'mutation_std': 0.5,
            'elite_size': 2,
            'bounds': [-5.12, 5.12]
        }
        return False, f"🚀 Optimization running... Problem: {problem.upper()}"

    elif button_id == 'reset-button':
        # Reset optimization
        optimization_state['running'] = False
        optimization_state['population'] = None
        optimization_state['generation'] = 0
        optimization_state['best_fitness'] = -np.inf
        optimization_state['history'] = {
            'best_fitness': [],
            'mean_fitness': [],
            'worst_fitness': [],
            'diversity': [],
            'best_individual': []
        }
        return True, "⚠️ Optimization reset. Ready to start new run."

    return True, "Ready to start."


@app.callback(
    [Output('fitness-evolution-graph', 'figure'),
     Output('diversity-graph', 'figure'),
     Output('population-distribution', 'figure'),
     Output('convergence-rate', 'figure'),
     Output('best-solution-display', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_graphs(n):
    """Update all graphs with latest optimization data."""

    # Run one generation if active
    if optimization_state['running'] and optimization_state['generation'] < 100:
        run_ga_generation(optimization_state)

    history = optimization_state['history']

    # Fitness Evolution Graph
    fitness_fig = go.Figure()
    if len(history['best_fitness']) > 0:
        fitness_fig.add_trace(go.Scatter(
            y=history['best_fitness'],
            mode='lines',
            name='Best Fitness',
            line=dict(color='green', width=3)
        ))
        fitness_fig.add_trace(go.Scatter(
            y=history['mean_fitness'],
            mode='lines',
            name='Mean Fitness',
            line=dict(color='blue', width=2)
        ))
        fitness_fig.add_trace(go.Scatter(
            y=history['worst_fitness'],
            mode='lines',
            name='Worst Fitness',
            line=dict(color='red', width=1, dash='dash')
        ))
    fitness_fig.update_layout(
        title='Fitness Evolution',
        xaxis_title='Generation',
        yaxis_title='Fitness',
        hovermode='x unified',
        height=350
    )

    # Diversity Graph
    diversity_fig = go.Figure()
    if len(history['diversity']) > 0:
        diversity_fig.add_trace(go.Scatter(
            y=history['diversity'],
            mode='lines+markers',
            name='Diversity',
            line=dict(color='purple', width=2),
            marker=dict(size=5)
        ))
    diversity_fig.update_layout(
        title='Population Diversity',
        xaxis_title='Generation',
        yaxis_title='Diversity (Std Dev)',
        height=350
    )

    # Population Distribution (2D projection)
    pop_fig = go.Figure()
    if optimization_state['population'] is not None:
        pop = optimization_state['population']
        fitness = np.array([PROBLEM_FUNCTIONS[optimization_state['problem_type']](ind)
                           for ind in pop])

        pop_fig.add_trace(go.Scatter(
            x=pop[:, 0],
            y=pop[:, 1],
            mode='markers',
            marker=dict(
                size=8,
                color=fitness,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Fitness")
            ),
            text=[f"Fitness: {f:.4f}" for f in fitness],
            hoverinfo='text'
        ))

        # Mark best individual
        if optimization_state['best_individual'] is not None:
            best = optimization_state['best_individual']
            pop_fig.add_trace(go.Scatter(
                x=[best[0]],
                y=[best[1]],
                mode='markers',
                marker=dict(size=15, color='red', symbol='star'),
                name='Best'
            ))

    pop_fig.update_layout(
        title='Population Distribution (Dimensions 0-1)',
        xaxis_title='Dimension 0',
        yaxis_title='Dimension 1',
        height=350
    )

    # Convergence Rate
    conv_fig = go.Figure()
    if len(history['best_fitness']) > 1:
        improvements = np.diff(history['best_fitness'])
        conv_fig.add_trace(go.Bar(
            y=improvements,
            marker_color=['green' if x > 0 else 'red' for x in improvements]
        ))
    conv_fig.update_layout(
        title='Fitness Improvement per Generation',
        xaxis_title='Generation',
        yaxis_title='Fitness Change',
        height=350
    )

    # Best Solution Display
    best_solution_text = html.Div([
        html.P(f"Generation: {optimization_state['generation']}",
               style={'fontSize': '18px', 'fontWeight': 'bold'}),
        html.P(f"Best Fitness: {optimization_state['best_fitness']:.6f}",
               style={'fontSize': '16px', 'color': '#27ae60'}),
        html.P(f"Best Individual: {optimization_state['best_individual'][:3] if optimization_state['best_individual'] is not None else 'N/A'}...",
               style={'fontSize': '14px', 'color': '#7f8c8d'})
    ])

    return fitness_fig, diversity_fig, pop_fig, conv_fig, best_solution_text


# ==================== Run Server ====================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🧬 GA OPTIMIZATION DASHBOARD")
    print("="*70)
    print("\nStarting server...")
    print("Open your browser and navigate to: http://localhost:8050")
    print("\nPress Ctrl+C to stop the server.\n")

    app.run_server(debug=True, host='0.0.0.0', port=8050)

import numpy as np
import plotly.graph_objects as go
from scipy.stats import beta, binom

# Function to generate synthetic observations X given success rate p
def generate_observations(p, N, seed):
    """
    Generate synthetic binary observations (hits/misses) given success rate p.
    :param p: True success rate (probability of hitting)
    :param N: Total number of trials
    :return: List of binary outcomes [0, 1, 1, ...]
    """
    np.random.seed(seed)
    return np.random.binomial(1, p, N)

# Function to compute the posterior Beta distribution
def compute_posterior(k, N, mean_prior, strength_prior):
    # Prior distribution parameters
    alpha_prior = mean_prior * strength_prior
    beta_prior = (1 - mean_prior) * strength_prior
    
    # Update parameters based on observed data
    alpha_posterior = alpha_prior + k
    beta_posterior = beta_prior + N - k
    return alpha_posterior, beta_posterior, alpha_prior, beta_prior

# Create the figure
def create_prior_post_fig(alpha_posterior, beta_posterior, alpha_prior, beta_prior, true_p, k, N):
    # Plot prior and posterior distributions
    x = np.linspace(0, 1, 500)


    # Compute PDFs
    prior_pdf = beta.pdf(x, alpha_prior, beta_prior)
    posterior_pdf = beta.pdf(x, alpha_posterior, beta_posterior)

    # Create Plotly figure
    fig = go.Figure()

    # Add prior distribution
    fig.add_trace(go.Scatter(
        x=x,
        y=prior_pdf,
        mode="lines",
        name="Prior",
        line=dict(color="blue", width=2)
    ))

    # Add posterior distribution
    fig.add_trace(go.Scatter(
        x=x,
        y=posterior_pdf,
        mode="lines",
        name="Posterior",
        line=dict(color="red", width=2)
    ))
    # Compute 95% credible interval for the posterior distribution
    lower_bound = beta.ppf(0.025, alpha_posterior, beta_posterior)
    upper_bound = beta.ppf(0.975, alpha_posterior, beta_posterior)

    # Add shaded area for 95% credible interval
    x_conf_post = np.linspace(lower_bound, upper_bound, 500)
    fig.add_trace(go.Scatter(
        x=x_conf_post,
        y=beta.pdf(x_conf_post, alpha_posterior, beta_posterior),
        fill='tozeroy',
        mode='none',
        name='95% Credible Interval',
        fillcolor='rgba(255, 0, 0, 0.2)'
    ))

    # Add vertical line for true success rate
    fig.add_vline(
        x=true_p, 
        line_dash="dash", 
        line_color="green", 
        annotation_text="True p", 
        annotation_position="top right", 
        annotation_y=1,
        annotation_font=dict(color="green")
    )

    # add vertical line for observed success rate
    fig.add_vline(
        x=k/N, 
        line_dash="dash", 
        line_color="orange", 
        annotation_text="Observed p", 
        annotation_position="top right",
        annotation_y=0.9,
        annotation_font=dict(color="orange")
    )

    # add vertical line for posterior mode
    fig.add_vline(
        x=(alpha_posterior - 1) / (alpha_posterior + beta_posterior - 2), 
        line_dash="dash", 
        line_color="red", 
        annotation_text="Posterior Mode",
        annotation_position="top right",
        annotation_y=0.8,
        annotation_font=dict(color="red")
    )

    # Update layout
    fig.update_layout(
        title="Prior and Posterior Distributions",
        xaxis_title="Success Rate (p)",
        yaxis_title="Probability Density",
        # legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        template="plotly_white",
        legend=dict(yanchor="top", y=1.1, xanchor="left", x=0.01, orientation="h")
    )

    return fig, lower_bound, upper_bound

# Plot the true success rate
def plot_success_rate(success_rate):
    # plot a horizontal bar chart of accuracy score that also shows 100% line
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[success_rate],
        y=['Accuracy'],
        orientation='h',
        marker=dict(color='rgba(50, 171, 96, 0.6)'),
        text=[f'{success_rate:.1%}'],
        # text color
        textfont=dict(color='white'),
        textposition = 'inside',
        showlegend=False,
    ))
    fig.add_trace(go.Bar(
        x=[1],
        y=['Accuracy'],
        orientation='h',
        marker=dict(color='rgba(50, 171, 96, 0.3)'),
        showlegend=False,
    ))
    fig.update_layout(
        xaxis_title='True Success Rate',
        yaxis_title='',
        yaxis=dict(visible=False),
        # overlay bars on top of each other
        barmode='overlay',
        # width and height of the figure
        # width=600,
        height=210,
    )
    return fig

# Plot the observations
def plot_observations(X):
    N = len(X)
    fig = go.Figure()
    # show each trial and its outcome in a figure
    fig.add_trace(go.Bar(
        x=np.arange(N)[X == 1],
        y= np.array([1] * N)[X == 1],
        marker=dict(color='green'),
        marker_line_color='black',
        # marker_line_width=1,
        showlegend=True,
        name='Success'
    ))
    fig.add_trace(go.Bar(
        x=np.arange(N)[X == 0],
        y= np.array([1] * N)[X == 0],
        marker=dict(color='red'),
        marker_line_color='black',
        # marker_line_width=1,
        showlegend=True,
        name='Failure'
    ))
    fig.update_layout(
        title='Simulated Observations',
        xaxis_title='Trial Number',
        # no gaps between bars
        bargap=0,
        height=200,
        yaxis=dict(visible=False),
        # custom legend with two items green success and red failure
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
    )
    return fig

# Create the prior distribution figure
def create_prior_fig(alpha_prior, beta_prior):
    # Plot prior and posterior distributions
    x = np.linspace(0, 1, 500)

    # Compute PDFs
    prior_pdf = beta.pdf(x, alpha_prior, beta_prior)

    # create 95% credible interval for the prior distribution
    lower_bound = beta.ppf(0.025, alpha_prior, beta_prior)
    upper_bound = beta.ppf(0.975, alpha_prior, beta_prior)

    # find x values for 95% credible interval
    x_conf_prior = np.linspace(lower_bound, upper_bound, 500)
    y_conf_prior = beta.pdf(x_conf_prior, alpha_prior, beta_prior)


    # Create Plotly figure
    fig = go.Figure()

    # Add prior distribution
    fig.add_trace(go.Scatter(
        x=x,
        y=prior_pdf,
        mode="lines",
        name="Prior Distribution",
        line=dict(color="blue", width=2),
        # fill='tozeroy',
        # show legend
        showlegend=True
    ))

    # Add shaded area for 95% credible interval
    fig.add_trace(go.Scatter(
        x=x_conf_prior,
        y=y_conf_prior,
        fill='tozeroy',
        mode='none',
        name='95% Credible Interval',
        fillcolor='rgba(0, 0, 255, 0.2)'
    ))

    # Update layout
    fig.update_layout(
        title="Prior Distribution",
        xaxis_title="Success Rate (p)",
        yaxis_title="Probability Density",
        legend=dict(yanchor="top", y=1.1, xanchor="left", x=0.01, orientation="h"),
        template="plotly_white"
    )

    return fig

# Monte Carlo simulation for different number of observations
def Monte_Carlo_simulation(max_obs, true_p, mean_prior, strength_prior, seed):
    posterior_modes = []
    lower_bounds = []
    upper_bounds = []
    observed_p = []

    # Monte Carlo simulation
    for n in range(10, max_obs+1, 10):
        # Generate synthetic observations
        X = generate_observations(true_p, n, seed)
        k = np.sum(X)  # Number of successes

        # observed success rate
        observed_p.append(k/n)

        # Compute posterior distribution
        alpha_posterior, beta_posterior, _, _ = compute_posterior(k, n, mean_prior, strength_prior)

        # Compute posterior mode
        mode = (alpha_posterior - 1) / (alpha_posterior + beta_posterior - 2)
        lower_bound = beta.ppf(0.025, alpha_posterior, beta_posterior)
        upper_bound = beta.ppf(0.975, alpha_posterior, beta_posterior)

        posterior_modes.append(mode)
        lower_bounds.append(lower_bound)
        upper_bounds.append(upper_bound)

    return posterior_modes, lower_bounds, upper_bounds, observed_p

# Create Monte Carlo simulation figure
def create_MC_fig(max_obs, true_p, mean_prior, strength_prior, alpha_prior, beta_prior, seed):
    posterior_modes, lower_bounds, upper_bounds, observed_p = Monte_Carlo_simulation(max_obs, true_p, mean_prior, strength_prior, seed)

    # Create a line plot showing the convergence of the posterior mode to the true value
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.arange(10, max_obs+1, 10),
        y=upper_bounds,
        mode='lines',
        line=dict(color='rgba(255, 0, 0, 0.01)', width=0),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=np.arange(10, max_obs+1, 10),
        y=posterior_modes,
        mode='lines',
        line=dict(color='red', width=3),
        name='Posterior Mode (95% Credible Interval)', 
        fill = 'tonexty'
    ))
    fig.add_trace(go.Scatter(
        x=np.arange(10, max_obs+1, 10),
        y=lower_bounds,
        mode='lines',
        line=dict(color='rgba(255, 0, 0, 0.01)', width=0),
        fill = 'tonexty',
        showlegend=False
    ))


    fig.add_trace(go.Scatter(
        x=np.arange(10, max_obs+1, 10),
        y=observed_p,
        mode='lines',
        line=dict(color='orange', width=2),
        name='Observed p'
    ))

    fig.add_hline(
        y=true_p, 
        line_dash="dash", 
        line_color="green", 
        annotation_text="True p", 
        annotation_position="top right",
        annotation_font=dict(color="green")
    )

    # add hline for mean and 95% credible interval of prior distribution
    prior_mean = mean_prior
    prior_lower_bound = beta.ppf(0.025, alpha_prior, beta_prior)
    prior_upper_bound = beta.ppf(0.975, alpha_prior, beta_prior)

    fig.add_shape(
        type="rect",
        x0=0,
        y0=prior_lower_bound,
        x1=max_obs,
        y1=prior_upper_bound,
        fillcolor="blue",
        opacity=0.2,
        line=dict(width=0),
        layer="below"
    )
    fig.add_hline(
        y=prior_mean, 
        line_dash="dash", 
        line_color="blue", 
        annotation_text="Mean of Prior", 
        annotation_position="bottom right",
        annotation_font=dict(color="blue")
    )

    # Update layout
    fig.update_layout(
        title="Convergence of Posterior Mode to True Value",
        xaxis_title="Number of Observations (N)",
        yaxis_title="Posterior Mode",
        template="plotly_white",
        showlegend=True,
        # legend dict
        legend=dict(
            yanchor="top",
            y=1.1,
            xanchor="left",
            x=0.01,
            orientation="h"
        )
    )

    return fig

# Create the likelihood figure
def create_likelihood_fig(p, N):
    # Create a bar chart showing the likelihood of different observed success rates given a true success rate
    x = np.arange(1, N+1)
    y = np.array([binom.pmf(k, N, p) for k in x])
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x,
        y=y,
        marker=dict(color='blue'),
        # line color
        marker_line_color='blue',
        showlegend=False
    ))
    fig.add_vline(
        x=N*p, 
        line_dash="dash", 
        line_color="orange", 
        annotation_text="Expected Successes", 
        annotation_position="bottom right",
        annotation_font=dict(color="orange")
    )
    fig.update_layout(
        title='Likelihood of Observed Success Rate',
        xaxis_title='Number of Successes',
        yaxis_title='Probability',
        # height=200,
        # yaxis=dict(visible=False),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
    )
    return fig
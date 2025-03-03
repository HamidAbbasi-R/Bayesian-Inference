import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import beta, binom, norm, gamma, uniform
import plotly.express as px

# Function to generate synthetic binary observations X given success rate p
def generate_observations(p, N, seed):
    """
    Generate synthetic binary observations (hits/misses) given success rate p.
    :param p: True success rate (probability of hitting)
    :param N: Total number of trials
    :return: List of binary outcomes [0, 1, 1, ...]
    """
    np.random.seed(seed)
    return np.random.binomial(1, p, N)

# Function to plot the likelihood of different success rates given the observed data
def plot_likelihood(k, N, p_true):
    """
    Plot the likelihood of different success rates given the observed data.
    
    Parameters:
        k (int): Number of successes.
        N (int): Total number of trials.
        
    Returns:
        fig (go.Figure): Plotly figure object.
    """

    p_values = np.linspace(0, 1, 1000)
    likelihood = binom.pmf(k, N, p_values)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=p_values, 
        y=likelihood, 
        mode='lines', 
        name='Likelihood',
        line=dict(color='orange', width=2),
    ))
    fig.add_vline(
        x=p_true,
        line_dash="dash",
        line_color="green",
        annotation_text="True p",
        # annotation_position="bottom right",
        annotation_y=0.1,
        annotation_font=dict(color="green")
    )
    fig.add_vline(
        x=k/N,
        line_dash="dash",
        line_color="orange",
        annotation_text="MLE",
        # annotation_position="bottom right",
        annotation_y=0.2,
        annotation_font=dict(color="orange")
    )
    fig.update_layout(
        title='Likelihood of Different Success Rates',
        xaxis_title='Success Rate (p)',
        yaxis_title='Likelihood',
        template="plotly_white"
    )
    return fig

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
    fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])

    likelihood = binom.pmf(k, N, x)
    fig.add_trace(go.Scatter(
        x=x,
        y=likelihood,
        mode='lines',
        name='Likelihood',
        line=dict(color='orange', width=2),
        showlegend=True
    ), secondary_y=True)

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
        annotation_text="MLE", 
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
        legend=dict(yanchor="top", y=1.1, xanchor="left", x=0.01, orientation="h"),
        yaxis2=dict(showticklabels=False, title='', showgrid=False)
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
        name='MLE'
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

# Function to generate synthetic continuous observations given a normal distribution
def generate_random_normal_sequence(mu, sigma, N):
    """
    Generate a finite sequence of numbers drawn from a normal distribution with 
    a random mean and standard deviation.

    Parameters:
        mu_true (float): True mean of the normal distribution.
        sigma_true (float): True standard deviation of the normal distribution.
        N (int): Number of samples to generate.

    Returns:
        tuple: A tuple containing:
            - sequence (ndarray): Array of generated numbers.
    """
    
    # Generate N samples from a normal distribution with the chosen mu and sigma
    sequence = np.random.normal(loc=mu, scale=sigma, size=N)
    
    return sequence,

# Function to plot a sequence of continuous observations
def plot_sequence(sequence):
    """
    Plot a sequence of numbers using a bar chart.

    Parameters:
        sequence (ndarray): Array of numbers to plot.

    Returns:
        fig (go.Figure): Plotly figure object.
    """
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=np.arange(len(sequence)),
        y=sequence,
        marker=dict(color='blue'),
        marker_line_color='black',
        showlegend=False
    ))
    fig.update_layout(
        title='Sequence Plot',
        xaxis_title='Index',
        yaxis_title='Value',
        template="plotly_white"
    )
    return fig

def plot_histogram(sequence, mu_true, sigma_true):
    """
    Plot a histogram of a sequence of numbers.

    Parameters:
        sequence (ndarray): Array of numbers to plot.
        bins (int): Number of bins for the histogram.

    Returns:
        fig (go.Figure): Plotly figure object.
    """
    mu_MLE = np.mean(sequence)
    sigma_MLE = np.std(sequence)

    fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Histogram(
        x=sequence,
        marker=dict(color='blue'),
        opacity=0.75,
        name='Observations',
    ))
    x = np.linspace(np.min(sequence), np.max(sequence), 1000)
    y_true = norm.pdf(x, loc=mu_true, scale=sigma_true)
    y_MLE = norm.pdf(x, loc=mu_MLE, scale=sigma_MLE)
    fig.add_trace(go.Scatter(
        x=x,
        y=y_MLE,
        mode='lines',
        line=dict(color='red', width=2, dash='dash'),
        name='MLE Distribution',
        showlegend=True
    ), secondary_y=True)
    fig.add_trace(go.Scatter(
        x=x,
        y=y_true,
        mode='lines',
        line=dict(color='green', width=2),
        name='True Distribution',
        showlegend=True
    ), secondary_y=True)

    fig.update_layout(
        title='Histogram of Observations',
        xaxis_title='Value',
        yaxis_title='Count',
        yaxis2=dict(range=[0, max(y_true) * 1.1], showticklabels=False, title='', showgrid=False),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.05,
            xanchor='left',
            x=0.0
        ),
    )
    return fig

def get_prior_distributions_normal(
        mu_type="normal", sigma_type="gamma",
        mu_params=None, sigma_params=None,
        ):
    """
    Define prior distributions for mu and sigma.

    Parameters:
        mu_type (str): Type of prior for mu ("uniform" or "normal").
        sigma_type (str): Type of prior for sigma ("uniform" or "gamma").
        mu_params (dict): Parameters for mu prior:
                          - For "uniform": {"mu_min": float, "mu_max": float}.
                          - For "normal": {"mu_mean": float, "mu_std": float}.
        sigma_params (dict): Parameters for sigma prior:
                             - For "uniform": {"sigma_min": float, "sigma_max": float}.
                             - For "gamma": {"shape": float, "rate": float}.

    Returns:
        tuple: Prior distributions for mu and sigma.
    """
    # Prior for mu
    if mu_type == "normal":
        if mu_params is None or "mu_mean" not in mu_params or "mu_std" not in mu_params:
            raise ValueError("For normal mu prior, provide 'mu_mean' and 'mu_std'.")
        mu_mean = mu_params["mu_mean"]
        mu_std = mu_params["mu_std"]
        mu_prior = norm(loc=mu_mean, scale=mu_std)
    
    if sigma_type == "gamma":
        if sigma_params is None or "shape" not in sigma_params or "rate" not in sigma_params:
            raise ValueError("For gamma sigma prior, provide 'shape' and 'rate'.")
        shape = sigma_params["shape"]
        rate = sigma_params["rate"]
        sigma_prior = gamma(a=shape, scale=1/rate)  # Gamma uses scale = 1/rate

    return mu_prior, sigma_prior

def visualize_joint_prior(mu_prior, sigma_prior, num_samples=10000, bins=50):
    """
    Visualize the joint prior distribution of mu and sigma using a 2D histogram with Plotly.

    Parameters:
        mu_prior: The prior distribution for mu (e.g., scipy.stats distribution).
        sigma_prior: The prior distribution for sigma (e.g., scipy.stats distribution).
        num_samples (int): Number of samples to draw from the priors.
        bins (int): Number of bins for the 2D histogram.

    Returns:
        None: Displays an interactive 2D histogram using Plotly.
    """
    # Sample from the priors
    mu_samples = mu_prior.rvs(size=num_samples)  # Samples from mu_prior
    sigma_samples = sigma_prior.rvs(size=num_samples)  # Samples from sigma_prior

    # Create a Plotly figure with a 2D histogram
    fig = px.density_heatmap(
        x=mu_samples,
        y=sigma_samples,
        nbinsx=bins,
        nbinsy=bins,
        # color_continuous_scale="Blues",
        labels={'x': 'μ', 'y': 'σ'},
        title="Joint Prior Distribution of μ and σ",
        marginal_x="histogram",     # other options 
        marginal_y="histogram"
    )

    fig.update_layout(
        title="Joint Prior Distribution of μ and σ",
        xaxis_title="μ",
        yaxis_title="σ"
    )

    # Show the figure
    return fig

# visualize the likelihood of observing the data given mu and sigma
def visualize_posterior_continuous(data, mu_true, sigma_true, mu_prior, sigma_prior, num_bins_mu=50, num_bins_sigma=50):
    """
    Compute and visualize the log-likelihood of observing the data given mu and sigma.

    Parameters:
        data (array-like): Observed data points.
        num_bins_mu (int): Number of bins for mu.
        num_bins_sigma (int): Number of bins for sigma.

    Returns:
        None: Displays a 2D heatmap of the log-likelihood.
    """
    # Step 1: Estimate bounds for mu and sigma based on the data
    mu_min, mu_max = np.mean(data) - 0.6 * np.std(data), np.mean(data) + 0.6 * np.std(data)
    sigma_min, sigma_max = 0.8 * np.std(data), 1.2 * np.std(data)  # Avoid sigma=0; start from a small positive value

    # Step 2: Create a grid of mu and sigma values
    mu_values = np.linspace(mu_min, mu_max, num_bins_mu)
    sigma_values = np.linspace(sigma_min, sigma_max, num_bins_sigma)
    mu_grid, sigma_grid = np.meshgrid(mu_values, sigma_values)

    # Step x: Evaluate the priors on the grid
    log_prior_mu = mu_prior.logpdf(mu_grid)  # Evaluate P(μ) on the grid
    log_prior_sigma = sigma_prior.logpdf(sigma_grid)  # Evaluate P(σ) on the grid

    # Step 3: Compute the log-likelihood for each mu-sigma pair
    log_likelihood = np.zeros((num_bins_mu, num_bins_sigma), dtype=float)

    for i in range(num_bins_mu):
        for j in range(num_bins_sigma):
            mu = mu_values[i]
            sigma = sigma_values[j]
            log_likelihood[j, i] = np.sum(norm.logpdf(data, loc=mu, scale=sigma))


    # Step 5: Compute the unnormalized posterior
    log_unnormalized_posterior = log_likelihood + log_prior_mu + log_prior_sigma

    # Subtract the maximum value to avoid overflow when exponentiating
    log_unnormalized_posterior -= np.max(log_unnormalized_posterior)  # Numerical stability
    unnormalized_posterior = np.exp(log_unnormalized_posterior)
    posterior = unnormalized_posterior / np.sum(unnormalized_posterior)

    # Step 6: Find the mode of the posterior distribution
    max_idx = np.unravel_index(np.argmax(posterior), posterior.shape)
    mu_mode = mu_values[max_idx[1]]
    sigma_mode = sigma_values[max_idx[0]]


    # Step 4: Visualize the log-likelihood as a heatmap
    fig_likelihood = go.Figure(data=go.Heatmap(
        z=log_likelihood,
        x=mu_values,
        y=sigma_values,
        colorscale='blues_r',  # Inverted colorscale
    ))
    
    # Add a marker for the MLE
    fig_likelihood.add_trace(go.Scatter(
        x=[np.mean(data)],
        y=[np.std(data)],
        mode='markers+text',
        marker=dict(color='orange', size=10),
        marker_line=dict(color='black', width=1),
        name = 'MLE',
    ))

    # Add a marker for the true mu and sigma
    fig_likelihood.add_trace(go.Scatter(
        x=[mu_true],
        y=[sigma_true],
        mode='markers+text',
        marker=dict(color='green', size=10),
        marker_line=dict(color='black', width=1),
        name = 'True Value',
    ))

    fig_likelihood.update_layout(
        title="Log-Likelihood of Data Given μ and σ",
        xaxis_title="μ",
        yaxis_title="σ",
        legend=dict(
            orientation='h',
            x = 0, y = 1.15
            ),
    )

    fig_posterior = go.Figure(data=go.Heatmap(
        z=posterior,
        x=mu_values,
        y=sigma_values,
        colorscale='blues_r',  # Inverted colorscale
    ))

    # Add a marker for the MLE
    fig_posterior.add_trace(go.Scatter(
        x=[np.mean(data)],
        y=[np.std(data)],
        mode='markers+text',
        marker=dict(color='orange', size=10),
        marker_line=dict(color='black', width=1),
        name = 'MLE',
    ))

    # Add a marker for the true mu and sigma
    fig_posterior.add_trace(go.Scatter(
        x=[mu_true],
        y=[sigma_true],
        mode='markers+text',
        marker=dict(color='green', size=10),
        marker_line=dict(color='black', width=1),
        name='True Value',
    ))

    # Add a marker for the posterior mode
    fig_posterior.add_trace(go.Scatter(
        x=[mu_mode],
        y=[sigma_mode],
        mode='markers+text',
        marker=dict(color='red', size=10),
        marker_line=dict(color='black', width=1),
        name='Posterior Mode',
    ))

    # Compute the cumulative sum of the posterior probabilities
    sorted_posterior = np.sort(posterior.flatten())[::-1]
    cumulative_sum = np.cumsum(sorted_posterior)
    
    # Find the threshold value for the 95% credible interval
    threshold_index = np.searchsorted(cumulative_sum, 0.95)
    threshold_value = sorted_posterior[threshold_index]

    # Create a mask for the 95% credible interval
    credible_interval_mask = posterior >= threshold_value

    # Highlight the 95% credible interval on the posterior heatmap
    fig_posterior.add_trace(go.Contour(
        z=credible_interval_mask.astype(int),
        x=mu_values,
        y=sigma_values,
        showscale=False,
        contours=dict(
            start=0.5,
            end=0.5,
            size=0.5,
            coloring='lines'        # other options: 
        ),
        line=dict(
            color='red',
            width=2,
            ),
        name='95% Credible Interval'
    ))

    fig_posterior.update_layout(
        title="Posterior Distribution of μ and σ",
        xaxis_title="μ",
        yaxis_title="σ",
        legend=dict(
            orientation='h',
            x = 0, y = 1.15
            ),
    )

    return fig_likelihood, fig_posterior
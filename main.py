import streamlit as st
import numpy as np
import utils


# User inputs
st.sidebar.title("Binary Bayesian Inference")
true_p = st.sidebar.slider("True Success Rate ($p_{\\text{true}}$)", 0.0, 1.0, 0.5, step=0.01)
N = st.sidebar.slider("Number of Binary Trials ($N_b$)", 10, 1000, 100, step=10)
seed = st.sidebar.number_input("Random Seed", value=42)
mean_prior = st.sidebar.slider("Mean of Prior Distribution", 0.001, 0.999, 0.35, step=0.01)
strength_prior = st.sidebar.slider("Strength of Prior Distribution", 2, 500, 100, step=1)
st.sidebar.markdown("---")
st.sidebar.title("Continuous Bayesian Inference")
N_seq = st.sidebar.slider("Number of Non-binary Trials ($N_{nb}$)", 10, 1000, 100, step=10)
mu_true = st.sidebar.slider("True Mean of Trials ($\\mu_{\\text{true}}$)", -2.0, 2.0, 0.0, step=0.1)
sigma_true = st.sidebar.slider("True Standard Deviation of Trials ($\\sigma_{\\text{true}}$)", 0.1, 2.0, 0.5, step=0.1)
mu_mean = st.sidebar.slider("Prior Distribution: $\mu_{\\text{mean}}$", -2.0, 2.0, 0.0, step=0.1)
mu_std = st.sidebar.slider("Prior Distribution: $\mu_{\\text{std}}$", 0.1, 2.0, 0.5, step=0.1)
shape = st.sidebar.slider("Prior Distribution: $\\text{shape}$", 0.1, 2.0, 1.0, step=0.1)
rate = st.sidebar.slider("Prior Distribution: $\\text{rate}$", 0.1, 2.0, 1.0, step=0.1)


# Streamlit app
st.title("Bayesian Inference: Estimating Probabilities with Limited Data")

st.header("Problem Definition")
st.write("""
In many real-world scenarios, we aim to estimate an unknown probability or success rate based on limited observations. 
This is a classic problem in Bayesian inference, where we combine prior beliefs about the parameter of interest with observed data 
to compute a posterior distribution that reflects our updated understanding.
""")

st.markdown("""
For example:
- **Baseball Player's Hit Rate**: Inferring a player's innate ability to hit the ball based on a finite number of at-bats.
- **Amazon Dealer's Success Rate**: Estimating the true success rate of a dealer based on a limited number of customer reviews.
- **Car Factory's Failure Rate**: Determining the true failure rate of a manufacturing process based on a finite number of tests.

In each case, the goal is to estimate the underlying success rate $ p $ (e.g., hit rate, success rate, failure rate) given the observed data and any prior knowledge.
""")

st.header("Mathematical Framework")
st.write("""
The problem can be framed mathematically using Bayes' theorem:
$$
P(p \mid X) = \\frac{P(X \mid p) P(p)}{P(X)},
$$
where:
- $ P(p \mid X) $: The posterior distribution, representing our updated belief about $ p $ after observing the data $ X $.
- $ P(X \mid p) $: The likelihood, which describes the probability of observing the data $ X $ given $ p $.
- $ P(p) $: The prior distribution, encoding our initial belief about $ p $ before seeing the data.
- $ P(X) $: The marginal likelihood, ensuring the posterior integrates to 1 (normalization constant).

For binary outcomes (success/failure), the likelihood follows a **binomial distribution**:
$$
P(X \mid p) = \\binom{N}{k} p^k (1-p)^{N-k},
$$
where:
- $ N $: Total number of trials.
- $ k $: Number of successes.

If we use a **Beta distribution** as the prior, $ P(p) = \\text{Beta}(p; \\alpha, \\beta) $, the posterior also becomes a Beta distribution:
$$
P(p \mid X) = \\text{Beta}(p; \\alpha + k, \\beta + N - k).
$$
Here:
- $ \\alpha, \\beta $: Parameters of the prior, representing pseudo-counts of successes and failures.
- $ \\alpha + k, \\beta + N - k $: Updated parameters of the posterior, combining prior knowledge and observed data.
""")

st.header("Examples of Bayesian Inference")
st.subheader("1. Baseball Player's Hit Rate")
st.write("""
Suppose a baseball player has achieved 30 hits out of 100 at-bats. If we assume a uniform prior ($ \\text{Beta}(1, 1) $), 
the posterior distribution for the player's hit rate $ p $ is:
$$
P(p \mid X) = \\text{Beta}(p; 1 + 30, 1 + 70) = \\text{Beta}(p; 31, 71).
$$
This reflects our updated belief about the player's innate hit rate after observing the data.
""")

st.subheader("2. Amazon Dealer's Success Rate")
st.write("""
An Amazon dealer has received 45 positive reviews out of 50 total reviews. Using an informative prior ($ \\text{Beta}(10, 2) $) 
to reflect a strong belief in high-quality service, the posterior distribution becomes:
$$
P(p \mid X) = \\text{Beta}(p; 10 + 45, 2 + 5) = \\text{Beta}(p; 55, 7).
$$
This posterior combines the prior belief with the observed review data.
""")

st.subheader("3. Car Factory's Failure Rate")
st.write("""
A car factory conducts 200 quality tests and finds 5 failures. With a weakly informative prior ($ \\text{Beta}(2, 2) $), 
the posterior distribution for the failure rate $ p $ is:
$$
P(p \mid X) = \\text{Beta}(p; 2 + 5, 2 + 195) = \\text{Beta}(p; 7, 197).
$$
This posterior reflects the updated understanding of the factory's true failure rate.
""")


st.header("Interactive Exploration with the Dashboard")

st.write("""
This dashboard allows you to interactively explore Bayesian inference by adjusting various inputs. 
You can simulate data, define prior distributions, and observe how the posterior distribution updates based on your choices. 
Here's how you can interact with the app:
""")

st.subheader("1. Setting the True Success Rate")
st.markdown("""
Use the slider **"True Success Rate (p)"** to define the underlying probability of success $ p $ in the simulation. 
This represents the true parameter you are trying to estimate. For example:
- Set $ p = 0.7 $ to simulate a scenario where the success rate is 70%.
- Adjust this value to see how different true success rates affect the posterior distribution.
""")
# Display a simple bar chart showing true_p out of 1
st.plotly_chart(utils.plot_success_rate(true_p))


st.subheader("2. Defining the Number of Trials")
st.markdown("""
Use the slider **"Number of Trials (N)"** to specify the total number of observations or trials in the experiment. 
For example:
- Set $ N = 100 $ to simulate 100 trials.
- Increase $ N $ to see how more data reduces uncertainty in the posterior distribution.
- Decrease $ N $ to observe the impact of limited data on the estimation process.
            
Also you can see the likelihood function for the given true success rate and number of trials.
It is computed using binomial distribution.
            
$$
P(X \mid p) = \\binom{N}{k} p^k (1-p)^{N-k},
$$
where:
- $ N $: Total number of trials.
- $ k $: Number of successes.
- $ p $: True success rate.
""")

# Generate synthetic observations
X = utils.generate_observations(true_p, N, seed)
k = np.sum(X)  # Number of successes

st.plotly_chart(utils.plot_observations(X))

# st.plotly_chart(utils.create_likelihood_fig(true_p, N))
st.write("$k = " + str(k) + "$ successes out of $N = " + str(N) + " $ trials (observed success rate, $ \\hat{p} = " + f"{k/N:.2f}" + " $). The true success rate is $ p_{\\text{true}} = " + str(true_p) + " $.")
st.write("Likelihood function: $P(X \mid p) = \\binom{" + str(N) + "}{" + str(k) + "} p^{" + str(k) + "} (1-p)^{" + str(N - k) + "}$")
st.plotly_chart(utils.plot_likelihood(k, N, true_p))


st.subheader("3. Controlling Randomness with the Seed")
st.markdown("""
The input **"Random Seed"** ensures reproducibility of the synthetic data generation process. 
By fixing the seed, you can generate the same sequence of random outcomes every time you run the simulation. 
For example:
- Use same seed number for consistent results.
- Change the seed to explore different random realizations of the data while keeping other parameters fixed.
""")

st.subheader("4. Specifying the Prior Distribution")
st.markdown("""
The prior distribution reflects your initial belief about the success rate before observing the data. 
Beta distribution is a flexible choice for prior distribution because it is conjugate to the binomial likelihood.
It is bounded between 0 and 1, making it suitable for modeling probabilities.
It is defined as:
            
$$
P(p) = \\frac{p^{\\alpha - 1} (1 - p)^{\\beta - 1}}{B(\\alpha, \\beta)}
$$
where:
- $ B(\\alpha, \\beta) $: Beta function.
- $ \\alpha, \\beta $: Shape parameters.

You can define beta distribution using two inputs:

- **Mean of Prior Distribution**: Use the slider to set the central tendency of the prior. 
  For example:
  - Set the mean to $ 0.5 $ for a neutral prior centered around 50% success.
  - Set the mean to $ 0.8 $ if you believe the success rate is likely high.

- **Strength of Prior Distribution**: Use the slider to control the confidence or weight of your prior belief. 
  For example:
  - A low strength (e.g., 10) corresponds to a weak prior with high uncertainty.
  - A high strength (e.g., 500) corresponds to a strong prior that dominates the observed data unless $ N $ is large.

These inputs determine the parameters $ \\alpha $ and $ \\beta $ of the Beta prior:
$$
\\alpha = \\text{mean} \\cdot \\text{strength}, \quad \\beta = (1 - \\text{mean}) \\cdot \\text{strength}.
$$
            
There is a special case of uniform prior distribution that is non-informative and can be set by choosing a mean of 0.5 and a strength of 2.
""")

# Compute posterior distribution
alpha_posterior, beta_posterior, alpha_prior, beta_prior = utils.compute_posterior(k, N, mean_prior, strength_prior)
st.plotly_chart(utils.create_prior_fig(alpha_prior, beta_prior))

st.subheader("5. Observing the Results")
st.markdown("""
Once you've set the inputs, the app will:
- Generate synthetic data based on the true success rate $ p $ and number of trials $ N $.
- Compute the posterior distribution by combining the prior and the observed data.
- Display the prior, likelihood, and posterior distributions using interactive visualizations.

You can analyze:
- How the posterior shifts as you adjust the true success rate, number of trials, or prior parameters.
- The influence of the prior strength on the posterior when data is scarce.
- The convergence of the posterior to the true success rate as $ N $ increases.
""")

# Display the plot
fig, lower_bound, upper_bound = utils.create_prior_post_fig(alpha_posterior, beta_posterior, alpha_prior, beta_prior, true_p, k, N)
st.plotly_chart(fig)

st.subheader("6. Experimenting with Scenarios")
st.markdown("""
Try these scenarios to deepen your understanding:
- **Weak Prior, Small Data**: Set a weak prior (low strength) and a small number of trials ($ N = 20 $). 
  Observe how the posterior is dominated by the data but still reflects some uncertainty.
  
- **Strong Prior, Large Data**: Set a strong prior (high strength) and a large number of trials ($ N = 500 $). 
  Notice how the posterior balances prior beliefs with overwhelming evidence from the data.

- **Multimodal Prior**: Simulate conflicting prior beliefs by setting a bimodal prior (e.g., mean = 0.2 and mean = 0.8). 
  Observe how the posterior resolves ambiguity as more data is collected.

Feel free to experiment with different combinations of inputs to build intuition about Bayesian inference!
""")

st.header("Key Takeaways")
st.markdown("""
- Bayesian inference allows us to systematically update our beliefs about an unknown parameter $ p $ by combining prior knowledge with observed data.
- The Beta distribution is particularly well-suited for modeling probabilities because it is bounded between 0 and 1 and conjugate to the binomial likelihood.
- The posterior distribution provides a complete probabilistic description of $ p $, enabling us to compute summary statistics like the mean, mode, and credible intervals.
- Real-world applications include estimating success rates, failure rates, and other probabilities in domains like sports, e-commerce, and manufacturing.
""")

st.header("The Role of Observations in Converging to the True Success Rate")

st.write("""
One of the most powerful aspects of Bayesian inference is its ability to converge to the true parameter value as more data becomes available. 
This convergence can be observed by comparing the **mode of the posterior distribution** with the true success rate $ p $. 
Let’s explore this phenomenon mathematically and conceptually.
""")

st.subheader("Posterior Mode and Convergence")
st.markdown("""
The mode of the posterior distribution, which represents the most probable value of $ p $ given the data and prior, is given by:
$$
\\text{Mode}[p \mid X] = \\frac{\\alpha_{\\text{posterior}} - 1}{\\alpha_{\\text{posterior}} + \\beta_{\\text{posterior}} - 2},
$$
where:
$$
\\alpha_{\\text{posterior}} = \\alpha_{\\text{prior}} + k, \quad \\beta_{\\text{posterior}} = \\beta_{\\text{prior}} + N - k.
$$

Here:
- $ k $: Number of successes in the observed data.
- $ N $: Total number of trials.
- $ \\alpha_{\\text{prior}}, \\beta_{\\text{prior}} $: Parameters of the prior distribution.

As $ N $ increases, the influence of the prior diminishes because $ k $ and $ N - k $ dominate $ \\alpha_{\\text{prior}} $ and $ \\beta_{\\text{prior}} $. 
In the limit of large $ N $, the posterior mode converges to the maximum likelihood estimate (MLE) of $ p $:
$$
\hat{p}_{\\text{MLE}} = \\frac{k}{N}.
$$

Since $ k \sim \\text{Binomial}(N, p) $, the law of large numbers ensures that $ \\frac{k}{N} \\to p $ as $ N \\to \infty $. 
Thus, regardless of the prior belief, the posterior mode approaches the true success rate $ p $ with sufficient observations.
""")

st.subheader("Monte Carlo Simulation for Validation")
st.markdown("""
To empirically validate this convergence, we can use **Monte Carlo simulation**. 
By repeatedly generating synthetic datasets with a fixed true success rate $ p $ and computing the posterior mode for increasing values of $ N $, 
we observe that the posterior mode gets progressively closer to $ p $. 

Mathematically, the expected deviation between the posterior mode and the true $ p $ decreases as $ N $ grows. 
For large $ N $, the posterior distribution becomes tightly concentrated around $ p $, reflecting high confidence in the estimate.
""")

max_obs = st.slider("Maximum Observations for Monte Carlo Simulation (N)", 50, 10_000, 3_000, 10)
st.plotly_chart(utils.create_MC_fig(max_obs, true_p, mean_prior, strength_prior, alpha_prior, beta_prior, seed))


st.subheader("Key Insight")
st.markdown("""
Regardless of the prior belief, the mode of the posterior distribution converges to the true success rate $ p $ as the number of observations $ N $ increases. 
This robustness to prior assumptions is a hallmark of Bayesian inference and underscores the importance of collecting sufficient data for accurate estimation.
""")

st.write("---")
st.header("Continuous Bayesian Inference")
st.write("""
In the previous section, we focused on binary outcomes (success/failure).
Now, let's explore continuous Bayesian inference, where the parameter of interest follows a continuous distribution.
Imagine you are observing a sequence of continuous-valued data points and trying to estimate the underlying mean and standard deviation.

An interesting example is estimating the mean return and volatility of a financial asset based on historical price data.
In this example, the observations are stock returns that are continuous random variables.
The goal is to estimate the true mean return and volatility of the asset based on the observed data.
The assumption is that the returns follow a normal distribution with unknown mean $ \\mu $ and standard deviation $ \\sigma $.
In a Bayesian framework:
  - We specify prior distributions for $ \\mu $ and $ \\sigma $ based on our beliefs.
  - We compute the likelihood of the observed data given $ \\mu $ and $ \\sigma $.
  - We update our beliefs about $ \\mu $ and $ \\sigma $ using Bayes' theorem to obtain the posterior distribution.

The Bayes theorem for continuous parameters is:

$$
P(\\mu, \\sigma \mid X) = \\frac{P(X \mid \\mu, \\sigma) P(\\mu, \\sigma)}{P(X)},
$$

where:
- $ P(\\mu, \\sigma \mid X) $: The posterior distribution for the mean $ \\mu $ and standard deviation $ \\sigma $.
- $ P(X \mid \\mu, \\sigma) $: The likelihood function for the normal distribution.
- $ P(\\mu, \\sigma) $: The joint prior distribution for $ \\mu $ and $ \\sigma $.
- $ P(X) $: The marginal likelihood, ensuring the posterior integrates to 1.

To demonstrate continuous Bayesian inference, we will simulate a sequence of continuous data points $X$.

$$
X \sim \mathcal{N}(\mu, \sigma^2)
$$

where:
- $ \\mu $: True mean of the normal distribution.
- $ \\sigma $: True standard deviation of the normal distribution.

We observe $X$, but we do not know the true values of $ \\mu $ and $ \\sigma $.
Our goal is to estimate these parameters using Bayesian inference.

Here you see the synthetic sequence of data points generated using $ \\mu_{\\text{true}}$ and $ \\sigma_{\\text{true}}$ set in the sidebar.
""")

seq = np.random.normal(loc=mu_true, scale=sigma_true, size=N_seq)
st.plotly_chart(utils.plot_sequence(seq[:100] if len(seq) > 100 else seq))

st.write("""
Figure below shows the histogram of the observed data points.
You can also see normal distribution based on the Maximum Likelihood Estimation (MLE) of the mean and standard deviation.
For a given sequence, the MLE of the mean is simply the sample mean and the MLE of the standard deviation is the sample standard deviation.

$$
\\hat{\\mu} = \\frac{1}{N} \\sum_{i=1}^{N} X_i, \\quad \\hat{\\sigma} = \\sqrt{\\frac{1}{N} \\sum_{i=1}^{N} (X_i - \\hat{\\mu})^2}
$$
          
""")
st.plotly_chart(utils.plot_histogram(seq, mu_true, sigma_true))

st.subheader("Prior Distributions for $ \\mu $ and $ \\sigma $, $P(\\mu, \\sigma)$")
st.write("""
Next, we will define the prior distributions for $ \\mu $ and $ \\sigma $.
For simplicity, we will use normal prior distributions for \\mu and gamma prior distributions for $ \\sigma $.
The prior distributions are defined as follows:
- $ \\mu \\sim \\mathcal{N}(\\mu_{\\text{mean}}, \\mu_{\\text{std}}^2) $: Normal distribution for the mean.
- $ \\sigma \\sim \\text{Gamma}(\\text{shape}, \\text{rate}) $: Gamma distribution for the standard deviation.
         
You can adjust the parameters $ \\mu_{\\text{mean}}, \\mu_{\\text{std}}, \\text{shape}, \\text{rate} $ in the sidebar to explore different prior distributions.
The joint prior distribution for $ \\mu $ and $ \\sigma $ is visualized below.
""")

mu_prior, sigma_prior = utils.get_prior_distributions_normal(
    mu_params = {
        "mu_mean": mu_mean,
        "mu_std": mu_std
    },
    sigma_params={
        "shape": shape,
        "rate": rate
    })
    
st.plotly_chart(utils.visualize_joint_prior(mu_prior, sigma_prior, num_samples=30_000, bins=100))


st.subheader("Likelihood Function, $P(X \mid \\mu, \\sigma)$")
st.write("""
Likelihood function for the normal distribution is:

$$ P(X \mid \mu, \sigma) = \\prod_{i=1}^{N} \\frac{1}{\\sqrt{2\\pi\\sigma^2}} \\exp\\left(-\\frac{(X_i - \\mu)^2}{2\\sigma^2}\\right) $$

where:
- $ X_i $: Observation $ i $.
- $ N $: Total number of observations.
- $ \\mu $: Mean of the normal distribution.
- $ \\sigma $: Standard deviation of the normal distribution.

The likelihood function is visualized below.
""")
fig_likelihood, fig_posterior = utils.visualize_posterior_continuous(seq, mu_true, sigma_true, mu_prior, sigma_prior, num_bins_mu = 100, num_bins_sigma=100)
st.plotly_chart(fig_likelihood)

st.subheader("Posterior Distribution for $ \\mu $ and $ \\sigma $, $P(\\mu, \\sigma \mid X)$")
st.write("""
Finally, the posterior distribution for $ \\mu $ and $ \\sigma $ is computed using Bayes' theorem.

$$
P(\\mu, \\sigma \mid X) \propto P(X \mid \\mu, \\sigma) P(\\mu, \\sigma)       
$$
         
The area with 95% highest posterior density is also shown with the white contour. 
""")
st.plotly_chart(fig_posterior)
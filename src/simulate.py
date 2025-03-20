import numpy as np

# Forward Equations
def forward_equations(a, v, t):
    """ Compute predicted summary statistics (R_pred, M_pred, V_pred) from true parameters (a, v, t). """
    y = np.exp(-a * v)
    R_pred = 1 / (y + 1)
    M_pred = t + (a / (2 * v)) * ((1 - y) / (1 + y))
    V_pred = (a / (2 * v**3)) * ((1 - 2 * a * v * y - y**2) / (y + 1)**2)
    return R_pred, M_pred, V_pred

# Sampling Distributions
def sample_observed_stats(R_pred, M_pred, V_pred, N):
    """ Generate noisy observed summary statistics (R_obs, M_obs, V_obs) given predicted values. """
    T_obs = np.random.binomial(N, R_pred)  # Eq. 7
    R_obs = T_obs / N
    M_obs = np.random.normal(M_pred, np.sqrt(V_pred / N))  # Eq. 8
    V_obs = np.random.gamma((N - 1) / 2, (2 * V_pred) / (N - 1))  # Eq. 9
    return R_obs, M_obs, V_obs

# Inverse Equations
def inverse_equations(R_obs, M_obs, V_obs):
    """ Compute estimated parameters (a_est, v_est, t_est) from observed summary statistics. """
    L = np.log(R_obs / (1 - R_obs))
    v_est = np.sign(R_obs - 0.5) * 4 * np.sqrt(L * (R_obs**2 * L - R_obs * L + R_obs - 0.5) / V_obs)  # Eq. 4
    a_est = L / v_est  # Eq. 5
    t_est = M_obs - (a_est / (2 * v_est)) * ((1 - np.exp(-v_est * a_est)) / (1 + np.exp(-v_est * a_est)))  # Eq. 6
    return a_est, v_est, t_est

# Simulation and Recovery
def simulate_and_recover(N, iterations=1000):
    """ Run simulate-and-recover for a given N. """
    results = []
    np.random.seed(42)

    for _ in range(iterations):
        # Generate true parameters
        a_true = np.random.uniform(0.5, 2)
        v_true = np.random.uniform(0.5, 2)
        t_true = np.random.uniform(0.1, 0.5)

        # Generate predicted statistics
        R_pred, M_pred, V_pred = forward_equations(a_true, v_true, t_true)

        # Generate observed statistics
        R_obs, M_obs, V_obs = sample_observed_stats(R_pred, M_pred, V_pred, N)

        # Recover parameters
        a_est, v_est, t_est = inverse_equations(R_obs, M_obs, V_obs)

        # Compute bias and squared error
        bias = [a_est - a_true, v_est - v_true, t_est - t_true]
        squared_error = [b**2 for b in bias]

        results.append([N, a_true, v_true, t_true, a_est, v_est, t_est, *bias, *squared_error])

    return np.array(results)

if __name__ == "__main__":
    Ns = [10, 40, 4000]
    all_results = np.vstack([simulate_and_recover(N) for N in Ns])
    np.savetxt("results.csv", all_results, delimiter=",",
               header="N,a_true,v_true,t_true,a_est,v_est,t_est,bias_a,bias_v,bias_t,se_a,se_v,se_t", comments="")
    print("Simulation complete. Results saved to results.csv.")

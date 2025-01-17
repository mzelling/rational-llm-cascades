import itertools
from tqdm import tqdm
import numpy as np
from scipy.stats import beta, triang, uniform, FitError
from niagara.probabilistic_modeling.marginals import betamix
from statsmodels.distributions.copula.api import (
    CopulaDistribution, GumbelCopula, IndependenceCopula
)
from scipy.optimize import minimize

### FUNCTIONS FOR MODELING PAIRWISE PROBABILITY:

def get_01_cases(pair_data, min_max=None):
    """
    Get indices for the relevant joint cases of 0 and 1 being smooth vs discrete.
    """
    if min_max is None:
        minima_01 = np.min(pair_data,axis=0)
        maxima_01 = np.max(pair_data,axis=0)
    else:
        minima_01, maxima_01 = min_max

    idx_0_equals_min = pair_data[:,0] == minima_01[0]
    idx_1_equals_min = pair_data[:,1] == minima_01[1]
    idx_0_equals_max = pair_data[:,0] == maxima_01[0]
    idx_1_equals_max = pair_data[:,1] == maxima_01[1]
    idx_0_exceeds_min = pair_data[:,0] > minima_01[0]
    idx_1_exceeds_min = pair_data[:,1] > minima_01[1]
    idx_0_less_than_max = pair_data[:,0] < maxima_01[0]
    idx_1_less_than_max = pair_data[:,1] < maxima_01[1]

    idx_01min = idx_0_equals_min & idx_1_equals_min
    idx_0min_1smooth = idx_0_equals_min & idx_1_exceeds_min & idx_1_less_than_max
    idx_0smooth_1min = idx_0_exceeds_min & idx_0_less_than_max & idx_1_equals_min
    idx_01smooth = idx_0_exceeds_min & idx_0_less_than_max & idx_1_exceeds_min & idx_1_less_than_max
    idx_0max_or_1max = idx_0_equals_max | idx_1_equals_max
    idx_0max_1smooth = idx_0_equals_max & idx_1_exceeds_min & idx_1_less_than_max
    idx_0smooth_1max = idx_0_exceeds_min & idx_0_less_than_max & idx_1_equals_max
    idx_0min_1max = idx_0_equals_min & idx_1_equals_max

    return {
        "0smooth": idx_0_less_than_max & idx_0_exceeds_min,
        "1smooth": idx_1_less_than_max & idx_1_exceeds_min,
        "0min": idx_0_equals_min,
        "1min": idx_1_equals_min,
        "0max": idx_0_equals_max,
        "1max": idx_1_equals_max,
        "01min": idx_01min,
        "0min_1max": idx_0min_1max,
        "0min_1smooth": idx_0min_1smooth,
        "0smooth_1min": idx_0smooth_1min,
        "01smooth": idx_01smooth,
        "0max_or_1max": idx_0max_or_1max,
        "0max_1smooth": idx_0max_1smooth,
        "0smooth_1max": idx_0smooth_1max,
    }


def get_joint_base_probs(pair_data, cases_01=None):
    """ 
    Get the joint base probabilities relating data from the smooth parts
    of the distributions and the discrete lumps at the min/max probs. 
    """
    if cases_01 is None:
        cases_01 = get_01_cases(pair_data)

    p_01min = np.mean(cases_01['01min'])
    p_0min_1max = np.mean(cases_01['0min_1max'])
    p_0min_1smooth = np.mean(cases_01['0min_1smooth'])
    p_0smooth_1min = np.mean(cases_01['0smooth_1min'])
    p_0smooth_1max = np.mean(cases_01['0smooth_1max'])
    p_01smooth = np.mean(cases_01['01smooth'])
    p_0max_or_1max = np.mean(cases_01['0max_or_1max'])

    assert np.allclose(p_01min + p_0min_1smooth + p_0smooth_1min + p_01smooth, 1.0 - p_0max_or_1max)

    return {
        'p_01min': p_01min,
        'p_0min_1max': p_0min_1max,
        'p_0min_1smooth': p_0min_1smooth,
        'p_0smooth_1min': p_0smooth_1min,
        'p_0smooth_1max': p_0smooth_1max,
        'p_01smooth': p_01smooth
    }


def get_marginal_base_probs(pair_data, cases_01=None):
    """ Get the marginal base probabilities for a variable's discrete and smooth parts. """
    if cases_01 is None:
        cases_01 = get_01_cases(pair_data)

    p_0min = np.mean(cases_01['0min'])
    p_1min = np.mean(cases_01['1min'])
    p_0smooth = np.mean(cases_01['0smooth'])
    p_1smooth = np.mean(cases_01['1smooth'])

    # sanity check
    p_0max = np.mean(cases_01['0max'])
    p_1max = np.mean(cases_01['1max'])
    assert np.allclose(p_0min + p_0max + p_0smooth, 1.0)
    assert np.allclose(p_1min + p_1max + p_1smooth, 1.0)

    return {
        'p_0min': p_0min,
        'p_0max': p_0max,
        'p_1min': p_1min,
        'p_1max': p_1max,
        'p_0smooth': p_0smooth,
        'p_1smooth': p_1smooth,
    }


def fit_offdiagonal_marginals(pair_data, cases_01=None, min_max=None):
    """
    Return the fitted distributions for the conditional marginals in the cases
    where 0 is min or max and 1 is smooth, and vice versa.

    Specifically, the returned probability distributions are
        P( X1 > min(X1), X1 <= t1 | X0 = min(X0), X1 > min(X1) )
        P( X0 > min(X0), X0 <= t0 | X1 = min(X1), X0 > min(X0) )
        P( X1 > min(X1), X1 <= t1 | X0 = max(X0), X1 > min(X1) )
        P( X0 > min(X0), X0 <= t0 | X1 = max(X1), X0 > min(X0) ),
    hence having the conditioning include that the variable of
    interest shall be smooth (neither min nor max). The discrete mass
    at the minimum is NOT accounted for in this conditional probability
    distribution because we condition on smoothness.
    """
    if cases_01 is None:
        cases_01 = get_01_cases(pair_data)
    
    # Compute the minima and maxima on the data
    if min_max is None:
        minima_01 = np.min(pair_data,axis=0)
        maxima_01 = np.max(pair_data,axis=0)
    else:
        minima_01, maxima_01 = min_max

    # Get the data
    data_0min_1smooth = pair_data[cases_01['0min_1smooth'], 1]
    data_0smooth_1min = pair_data[cases_01['0smooth_1min'], 0]
    data_0max_1smooth = pair_data[cases_01['0max_1smooth'], 1]
    data_0smooth_1max = pair_data[cases_01['0smooth_1max'], 0]

    distributions = {}

    for data, name, idx in [
            (data_0min_1smooth, 'dist_0min_1smooth', 1),
            (data_0smooth_1min, 'dist_0smooth_1min', 0), 
            (data_0max_1smooth, 'dist_0max_1smooth', 1), 
            (data_0smooth_1max, 'dist_0smooth_1max', 0)
        ]:

        if len(data) == 0:
            # the line below should never make a difference because the distribution
            # always gets multiplied by zero in this case
            dist = uniform(loc=minima_01[idx], scale=maxima_01[idx] - minima_01[idx])
        else:
            try:
                params = beta.fit(data, floc=minima_01[idx], fscale=maxima_01[idx]-minima_01[idx])
                assert params[-2] == minima_01[idx]
                assert params[-1] == maxima_01[idx] - minima_01[idx]
                dist = beta(*params)
            except FitError as e:
                params = triang.fit(data, floc=minima_01[idx], fscale=maxima_01[idx]-minima_01[idx])
                dist = triang(*params)
        
        distributions[name] = dist

    return distributions


def fit_smooth_marginals(pair_data, cases_01=None, min_max=None, reg=0.0):
    """ Fit smooth marginals (beta mixture without discrete masses) for both variables. """
    
    if cases_01 is None:
        cases_01 = get_01_cases(pair_data)

    if min_max is None:
        minima_01 = np.min(pair_data,axis=0)
        maxima_01 = np.max(pair_data,axis=0)
    else:
        minima_01, maxima_01 = min_max

    data_0smooth = pair_data[cases_01['0smooth'], 0]
    data_1smooth = pair_data[cases_01['1smooth'], 1]

    # Fit the beta mixture distributions
    params_0smooth = betamix.fit(data_0smooth, minima_01[0], maxima_01[0], reg=reg)
    params_1smooth = betamix.fit(data_1smooth, minima_01[1], maxima_01[1], reg=reg)
    dist_0smooth = betamix(*params_0smooth)
    dist_1smooth = betamix(*params_1smooth)

    return {
        "dist_0smooth": dist_0smooth,
        "dist_1smooth": dist_1smooth
    }

def fit_smooth_copula_model(pair_data, smooth_marginals, cases_01=None):
    """
    Fit smooth copula model given the data and the smooth marginal distributions.
    """
    if cases_01 is None:
        cases_01 = get_01_cases(pair_data)

    # Fit the copula!
    data_01smooth = pair_data[cases_01['01smooth'], :]
    theta = GumbelCopula().fit_corr_param(data_01smooth)
    if theta > 1:
        copula = GumbelCopula(theta)
    elif theta <= 1:
        copula = IndependenceCopula()
    marginals = [smooth_marginals['dist_0smooth'], smooth_marginals['dist_1smooth']]
    smooth_copula_model = CopulaDistribution(
        copula=copula, marginals=marginals
    )

    return smooth_copula_model


def compute_joint_thresholded_prob(t0, t1, joint_base_probs, offdiagonal_marginals, smooth_copula_model):
    """ 
    Compute the probability that x0 is less than or equal to t0 AND
    x1 is less than or equal to t1, for thresholds t0 and t1 that INCLUDE
    the respective minimum values of x0 and x1 and EXCLUDE the respective
    maximum values.

    Specifically, the returned probability is the unconditional probability
        P(X0 <= t0, X1 <= t1),
    with t0, t1 less than the respective maximum values.
    """
    joint_prob = (
        joint_base_probs['p_01min']
        + joint_base_probs['p_0smooth_1min'] * offdiagonal_marginals['dist_0smooth_1min'].cdf(t0)
        + joint_base_probs['p_0min_1smooth'] * offdiagonal_marginals['dist_0min_1smooth'].cdf(t1)
        + joint_base_probs['p_01smooth'] * smooth_copula_model.cdf(y=[t0,t1])
    )

    return joint_prob


def compute_marginal_thresholded_prob(t0, marginal_base_probs, smooth_marginals, **kwargs):
    """ 
    Compute the marginal probabilities that x0 is less than or equal to t0.
    The threshold t0 must be less than the maximum value of x0 and greater than
    or equal to the minimum value.

    The returned probability is
        P(X0 <= t0),
    without any conditioning.
    """
    marginal_prob = (
        marginal_base_probs['p_0min']
        + marginal_base_probs['p_0smooth'] * smooth_marginals['dist_0smooth'].cdf(t0)
    )
    return marginal_prob


def compute_conditional_deferral_prob(
        t0, t1, 
        marginal_base_probs, joint_base_probs, offdiagonal_marginals, smooth_marginals, smooth_copula_model,
        **kwargs
    ):
    """ 
    Compute the probability that x1 <= t1 given that x0 <= t0, only for t0 and t1 such
    that the discrete masses at the minimum x values are INCLUDED and the masses at the
    maximum values are EXCLUDED. 
    
    This is the actual probability P( X1 <= t1 | X0 <= t0 ), given
    that t1 < max(X1). No conditioning on x1 <= max(X1) is applied.
    However, since we exclude the maximum at X1, using this function
    to compute P( X1 <= max(X1) | X0 <= to ) yields a number less
    than 1, since the probability that X1 = max(X1) is not included.
    """
    num = compute_joint_thresholded_prob(t0, t1, joint_base_probs, offdiagonal_marginals, smooth_copula_model)
    denom = compute_marginal_thresholded_prob(t0, marginal_base_probs, smooth_marginals)

    if np.isnan(num/denom):
        print(f"NUM: {num}, DENOM: {denom}")
        print(f"\tt0: {t0}, t1: {t1}")
    return num/denom


def compute_conditional_min_probability(
        t0, joint_base_probs, marginal_base_probs, offdiagonal_marginals, smooth_marginals
    ):
    """ 
    Compute the conditional probability that x1 equals its minimum, given that
    x0 is less than or equal to threshold t0 (which EXCLUDES the upper mass and
    INCLUDES the lower mass).
    """
    # Take into account that offdiagonal marginals are fitted on smooth data without min/max
    conditional_0smooth_1min = offdiagonal_marginals['dist_0smooth_1min'].cdf(t0)
    joint_0smooth_1min = conditional_0smooth_1min * joint_base_probs['p_0smooth_1min']
    joint_prob_0lessthan_1min = joint_0smooth_1min + joint_base_probs['p_01min']
    # Calculate using Bayes' rule:
    marginal_prob_0lessthan = compute_marginal_thresholded_prob(t0, marginal_base_probs, smooth_marginals)
    return (joint_prob_0lessthan_1min)/marginal_prob_0lessthan


def compute_conditional_max_probability(
        t0, joint_base_probs, marginal_base_probs, offdiagonal_marginals, smooth_marginals
    ):
    """ 
    Compute the conditional probability that x1 equals its minimum, given that
    x0 is less than or equal to threshold t0 (which EXCLUDES the upper mass and
    INCLUDES the lower mass).
    """
    # First, compute conditional prob P(X0 <= t0 | X1 = max(X1)).
    # We need to consider that this prob is conditioned on X0 being smooth.
    # Consequently, we must add back the mass at min(X0).
    conditional_0smooth_1max = offdiagonal_marginals['dist_0smooth_1max'].cdf(t0)
    joint_0smooth_1max = conditional_0smooth_1max * joint_base_probs['p_0smooth_1max']
    joint_prob_0lessthan_1max = joint_0smooth_1max + joint_base_probs['p_0min_1max']
    # Calculate using Bayes rule:
    marginal_prob_0lessthan = compute_marginal_thresholded_prob(t0, marginal_base_probs, smooth_marginals)
    return (joint_prob_0lessthan_1max)/marginal_prob_0lessthan


def compute_expected_correctness(
        t0, marginal_base_probs, smooth_marginals,
        n_grid=250, cases_01=None, min_max=None
    ):
    """ 
    Compute the partial expected value of X0, integrated over all outcomes such
    that X0 > t0. Thus, the returned probability is roughly the expected correctness,
    multiplied by the probability that X0 > t0.

    Computation proceeds by numerically integrating over the distribution function.

    NOTE: to allow the case of computing the FULL expectation, when t0 = min(X0),
    the lower discrete mass at min(X0) is included in the integral.

    NOTE: settings eps does not influence the ability to include the minimum probability
    mass when t1 = min(X1). Indeed, eps only affects the integration interval but the
    check whether t1 equals min(X1) still leads to full inclusion of the minimum 
    probability mass.
    """
    if min_max is None:
        raise ValueError("please provide the argument min_max")
    else:
        minima_01, maxima_01 = min_max

    assert t0 >= minima_01[0], f"t0 must be at least {minima_01[0]}"
    assert t0 <= maxima_01[0], f"t0 must not exceed {maxima_01[0]}"

    # Check that the domain of integration is not empty
    x0_gridmin, x0_gridmax = (t0, maxima_01[0])

    if np.allclose(x0_gridmin, x0_gridmax):
        integral = 0.0
    else:
        # Set the integration grid from x0=t0 to x0=max(X0)
        x0_grid = np.linspace(x0_gridmin, x0_gridmax, n_grid)
        x0_grid_h = (x0_gridmax - x0_gridmin)/n_grid

        # Approximate the conditional density function
        probs = np.array(
            [ 
                compute_marginal_thresholded_prob(x0, marginal_base_probs, smooth_marginals)
                    for x0 in x0_grid
            ]
        )
        integrand = (probs[1:] - probs[:-1])/x0_grid_h
        # Compute the integral with the trapezoidal rule
        integral = np.sum(integrand * (x0_grid[1:] + x0_grid[:-1]) * x0_grid_h)/2

    # We are integrating from x0_min=t0 to x0_max=max(X0), so our expectation
    # ALWAYS captures the maximum discrete mass but does NOT include the 
    # minimum mass unless we want to compute a full expectation.

    # Compute the contributions to the expectation from the discrete terms:
    prob_of_max = marginal_base_probs['p_0max']
    max_expectation = maxima_01[0] * prob_of_max

    if np.allclose(t0, minima_01[0]):
        prob_of_min = marginal_base_probs['p_0min']
        min_expectation = minima_01[0] * prob_of_min
    else:
        min_expectation = 0.0

    return integral + min_expectation + max_expectation


def compute_expected_correctness_upon_deferral(
        t0, t1, 
        marginal_base_probs, joint_base_probs, offdiagonal_marginals, smooth_marginals, smooth_copula_model,
        n_grid=100, cases_01=None, min_max=None
    ):
    """ 
    Compute the expected value of X1 given that X0 <= t0, which EXCLUDES max(X0).
    Only integrate over the outcomes where X1 > t1. Thus, the returned is
    roughly the expected correctness given X0 <= t0, multiplied by the probability
    that X1 > t1.

    Computation proceeds by numerically integrating over distribution function.

    NOTE: to allow the case of computing the FULL expectation, when t1 = min(X1),
    the lower discrete mass at min(X1) is included in the integral.

    NOTE: settings eps does not influence the ability to include the minimum probability
    mass when t1 = min(X1). Indeed, eps only affects the integration interval but the
    check whether t1 equals min(X1) still leads to full inclusion of the minimum 
    probability mass.
    """

    if min_max is None:
        raise ValueError("please provide the argument min_max")
    else:
        minima_01, maxima_01 = min_max

    # Gather the probabilistic arguments for simplicity
    prob_args = (
        marginal_base_probs, joint_base_probs, offdiagonal_marginals, smooth_marginals, smooth_copula_model
    )

    # Check that our domain of integration is not empty
    x1_gridmin, x1_gridmax = (t1, maxima_01[1])

    if np.allclose(x1_gridmin, x1_gridmax):
        integral = 0.0
    else:
        # Set up the integration grid from x1=t1 to x1=max(X1)
        x1_grid = np.linspace(x1_gridmin, x1_gridmax, n_grid)
        x1_grid_h = (x1_gridmax - x1_gridmin)/n_grid
        # Approximate the conditional density function
        conditional_probs = np.array(
            [ compute_conditional_deferral_prob(t0, x1, *prob_args) for x1 in x1_grid ]
        )
        integrand = (conditional_probs[1:] - conditional_probs[:-1])/x1_grid_h

        if np.any(np.isnan(integrand)):
            print(f"COND_PROBS: {conditional_probs}")
            print(f"\tx1_grid_h: {x1_grid_h}")
            print(f"\tINTEGRAND: {integrand}, t0={t0}, t1={t1}")

        # Compute the integral with the trapezoidal rule
        integral = np.sum(integrand * (x1_grid[1:] + x1_grid[:-1]) * x1_grid_h)/2

    # We are integrating from x1_min=t1 to x1_max=max(X1), so our expectation
    # It always captures the maximum discrete mass but does NOT include the 
    # minimum mass unless we want to compute a full expectation.

    # Compute the contributions to the expectation from the discrete terms:
    cond_args = (joint_base_probs, marginal_base_probs, offdiagonal_marginals, smooth_marginals)
    conditional_max_prob = compute_conditional_max_probability(t0, *cond_args)
    max_expectation = maxima_01[1] * conditional_max_prob

    if np.allclose(t1, minima_01[1]):
        conditional_min_prob = compute_conditional_min_probability(t0, *cond_args)
        min_expectation = minima_01[1] * conditional_min_prob
    else:
        min_expectation = 0.0

    if np.isnan(min_expectation):
        print(f"HEY MIN! t0={t0} t1={t1}")
    if np.isnan(max_expectation):
        print(f"HEY MAX! t0={t0} t1={t1}")

    return integral + min_expectation + max_expectation


def map_to_grid_idx(t, t_min, t_max, t_grid, float=False):
    """ 
    Map t to a discrete EQUISPACED grid ranging from t_min to t_max. 
    Return the index on the grid, or the continuous residual relative
    to the grid indices.
    """
    t_clamped = max(min(t,t_max), t_min)

    residual = (t_clamped - t_min)/(t_grid[1] - t_grid[0])

    if float:
        return residual
    else:
        grid_idx = int(np.floor(residual))
        return grid_idx
    

def retrieve_univariate_integral_value(t0, min_max, t0_grid, univariate_integral_table):
    """ Compute a smoothly interpolated value for the univariate integral,
    based on the discrete values stored in the integral table. """
    t = t0
    t_min, t_max = (min_max[0][0], min_max[1][0])
    T_GRID = t0_grid
    INTEGRAL_TABLE = univariate_integral_table

    residual = map_to_grid_idx(t, t_min, t_max, T_GRID, float=True)
    floored_residual = int(np.floor(residual))
    w = residual - floored_residual
    if floored_residual == len(T_GRID) - 1:
        return INTEGRAL_TABLE[floored_residual]
    else:
        return (
            w * INTEGRAL_TABLE[floored_residual + 1]
            + (1-w) * INTEGRAL_TABLE[floored_residual]
        )
    

def retrieve_bivariate_integral_value(
        t0, t1, min_max, t0_grid, t1_grid, bivariate_integral_table
    ):
    """ 
    Compute a smoothly interpolated value for the bivariate integral,
    based on the discrete values stored in the integral table.
    """
    t0_bounds = [ min_max[0][0], min_max[1][0] ]
    t1_bounds = [ min_max[0][1], min_max[1][1] ]
    
    assert (t0 >= t0_bounds[0]) and (t0 <= t0_bounds[1]), f"t0 must be between {t0_bounds[0]} and {t0_bounds[1]}"
    assert (t1 >= t1_bounds[0]) and (t1 <= t1_bounds[1]), f"t1 must be between {t1_bounds[0]} and {t1_bounds[1]}"
    
    residual_0 = map_to_grid_idx(t0, t0_bounds[0], t0_bounds[1], t0_grid, float=True)
    residual_1 = map_to_grid_idx(t1, t1_bounds[0], t1_bounds[1], t1_grid, float=True)
    floored_residual_0 = int(np.floor(residual_0))
    floored_residual_1 = int(np.floor(residual_1))
    w0 = residual_0 - floored_residual_0
    w1 = residual_1 - floored_residual_1
    # Take care not to increment beyond the bounds
    incremented_residual_0 = floored_residual_0 + 1 if not np.allclose(w0, 0.0) else floored_residual_0
    incremented_residual_1 = floored_residual_1 + 1 if not np.allclose(w1, 0.0) else floored_residual_1

    return (
        bivariate_integral_table[floored_residual_0, floored_residual_1]
        + w0 * ( 
            bivariate_integral_table[incremented_residual_0, floored_residual_1] 
            - bivariate_integral_table[floored_residual_0, floored_residual_1] 
            )
        + w1 * ( 
            bivariate_integral_table[floored_residual_0, incremented_residual_1]
            - bivariate_integral_table[floored_residual_0, floored_residual_1]
        )
    )

def make_full_data(calibrated_conf):
    return np.array(calibrated_conf).transpose()

def train_probability_model(
        full_data, bivariate_n_grid=20, univariate_n_grid=100, 
        bivariate_integral_n_grid=25, univariate_integral_n_grid=250
    ):
    """
    TBD
    """
    n_models = full_data.shape[1]
    prob_models = []
    univariate_integral_tables = [-1] * n_models
    bivariate_integral_tables = [ [] for _ in range(n_models) ]
    unconditional_expected_correctness = {}
    bounds = {}

    for i in tqdm(range(n_models)):
        unconditional_expected_correctness[i] = np.mean(full_data[:,i])
        bounds[i] = [np.min(full_data[:,i]), np.max(full_data[:,i])]
        prob_models.append([])
        for j in range(n_models):
            if j <= i: 
                prob_models[i].append("N/A")
                bivariate_integral_tables[i].append("N/A")
            else:
                # gather all probabilistic data
                pair_data = full_data[:,[i,j]]
                min_max = np.min(pair_data,axis=0), np.max(pair_data,axis=0)
                cases_01 = get_01_cases(pair_data, min_max=min_max)
                joint_base_probs = get_joint_base_probs(pair_data, cases_01=cases_01)
                marginal_base_probs = get_marginal_base_probs(pair_data, cases_01=cases_01)
                offdiagonal_marginals = fit_offdiagonal_marginals(pair_data, cases_01=cases_01, min_max=min_max)
                smooth_marginals = fit_smooth_marginals(pair_data, cases_01=cases_01, min_max=min_max)
                smooth_copula_model = fit_smooth_copula_model(pair_data, smooth_marginals, cases_01=cases_01)

                # compute univariate integral table (only once for index i)
                if j == i+1:
                    T_GRID_0 = np.linspace(min_max[0][0], min_max[1][0], univariate_n_grid)

                    UNIVARIATE_INTEGRAL_TABLE = np.array(
                        [
                            compute_expected_correctness(
                                t0, marginal_base_probs, smooth_marginals,
                                n_grid=univariate_integral_n_grid, cases_01=cases_01, min_max=min_max
                            )
                            for t0 in T_GRID_0
                        ]
                    )
                    univariate_integral_tables[i] = {
                        "t0_grid": T_GRID_0,
                        "univariate_integral_table": UNIVARIATE_INTEGRAL_TABLE
                    }

                # compute bivariate integral table for expected correctness
                T_GRID_0 = np.linspace(min_max[0][0], min_max[1][0], bivariate_n_grid)
                T_GRID_1 = np.linspace(min_max[0][1], min_max[1][1], bivariate_n_grid)

                BIVARIATE_INTEGRAL_TABLE = np.array(
                    [
                        [ 
                            compute_expected_correctness_upon_deferral(
                                t0, t1, marginal_base_probs, joint_base_probs, 
                                offdiagonal_marginals, smooth_marginals, smooth_copula_model,
                                n_grid=bivariate_integral_n_grid, cases_01=cases_01, min_max=min_max
                            ) for t1 in T_GRID_1
                        ] for t0 in T_GRID_0
                    ]
                )
                bivariate_integral_tables[i].append({
                        "t0_grid": T_GRID_0,
                        "t1_grid": T_GRID_1,
                        "bivariate_integral_table": BIVARIATE_INTEGRAL_TABLE
                })

                # combine the data into a record
                prob_model_data = {
                    "min_max": min_max,
                    "cases_01": cases_01,
                    "joint_base_probs": joint_base_probs,
                    "marginal_base_probs": marginal_base_probs,
                    "offdiagonal_marginals": offdiagonal_marginals,
                    "smooth_marginals": smooth_marginals,
                    "smooth_copula_model": smooth_copula_model
                }
                prob_models[i].append(prob_model_data)

    return {
        'unconditional_expected_correctness': unconditional_expected_correctness,
        'bounds': bounds,
        'prob_models': prob_models,
        'univariate_integral_tables': univariate_integral_tables,
        'bivariate_integral_tables': bivariate_integral_tables,
    }


def compute_metrics(T, model_indices, expected_uncumulated_costs, results, record_all_values=False):
    """
    Compute the expected error and expected cost.
    """
    # assert model indices are increasing and thresholds are within bounds
    assert len(model_indices) > 0, "number of model indices cannot be zero"
    assert len(T) == len(model_indices)-1, "number of thresholds must be one less than number of models"

    if len(model_indices) == 1:
        # for a cascade consisting of a single model, directly return the metrics
        expected_correctness = results['unconditional_expected_correctness'][model_indices[0]]
        expected_cost = expected_uncumulated_costs[model_indices[0]]
        return {
            "expected_correctness": expected_correctness,
            "expected_cost": expected_cost,
            "conditional_deferral_probs": [0.0]
        }
    
    # cumulate the costs: these are the total costs when terminating at a certain model
    effective_costs = np.cumsum([ expected_uncumulated_costs[idx] for idx in model_indices ])

    assert np.all(np.diff(model_indices) > 0.0)

    # for t,i in zip(T, model_indices[:-1]):
    #     if t < results['bounds'][i][0]:
    #         t = results['bounds'][i][0]
    #     elif t > results['bounds'][i][1]:
    #         t = results['bounds'][i][1]

    assert np.all([ 
        (t >= results['bounds'][i][0]) and (t <= results['bounds'][i][1])
            for t,i in zip(T, model_indices[:-1]) 
    ]), str(T) + "\n" + str(results['bounds'])

    if record_all_values:
        deferral_terms = (-1)*np.ones(shape=(len(model_indices)-1,))
        terminal_terms = (-1)*np.ones(shape=(len(model_indices),))

    full_deferral_prob = 1.0
    full_expected_correctness = 0.0
    full_expected_cost = 0.0

    conditional_deferral_probs = []

    # THE FIRST MODEL: EVALUATE WITHOUT REFERENCE TO OTHER MODELS
    first_model_prob_model = results['prob_models'][model_indices[0]][model_indices[1]]
    first_model_deferral_prob = compute_marginal_thresholded_prob(T[0], **first_model_prob_model)
    first_model_terminal_correctness_term = retrieve_univariate_integral_value(
        T[0], first_model_prob_model['min_max'], 
        **results['univariate_integral_tables'][model_indices[0]]
    )
    first_model_terminal_cost_term = (1-first_model_deferral_prob) * effective_costs[0]
    # Accumulate
    full_expected_correctness += first_model_terminal_correctness_term
    full_expected_cost += first_model_terminal_cost_term
    full_deferral_prob *= first_model_deferral_prob
    # Build the record of deferral probabilities
    conditional_deferral_probs.append(first_model_deferral_prob)

    # INTERMEDIATE MODELS (IF ANY): TRANSITION BETWEEN MODELS
    for step_idx in range(1, len(model_indices)-1):
        # t1 is curr_model_idx, t0 is prev_model_idx
        curr_model_idx = model_indices[step_idx]
        prev_model_idx = model_indices[step_idx-1]
        prob_model_data = results['prob_models'][prev_model_idx][curr_model_idx]

        deferral_prob = compute_conditional_deferral_prob(T[step_idx-1], T[step_idx], **prob_model_data)    
        terminal_correctness_term = retrieve_bivariate_integral_value(
            T[step_idx-1], T[step_idx], prob_model_data['min_max'],
            **results['bivariate_integral_tables'][prev_model_idx][curr_model_idx]
        )
        terminal_cost_term = (1-deferral_prob) * effective_costs[step_idx]
        # Accumulate
        full_expected_correctness += full_deferral_prob * terminal_correctness_term
        full_expected_cost += full_deferral_prob * terminal_cost_term
        full_deferral_prob *= deferral_prob
        # Build the record of deferral probabilities
        conditional_deferral_probs.append(deferral_prob)

    # THE LAST MODEL: ONLY COMPUTE TERMINAL TERMS
    # NOTE: T[-1] is the threshold of the penultimate model
    last_min_max = results['prob_models'][model_indices[-2]][model_indices[-1]]['min_max']
    last_terminal_correctness_term = retrieve_bivariate_integral_value(
        t0=T[-1], t1=last_min_max[0][1], min_max=last_min_max,
        **results['bivariate_integral_tables'][model_indices[-2]][model_indices[-1]]
    )
    last_terminal_cost_term = effective_costs[-1]
    conditional_deferral_probs.append(0.0)

    full_expected_correctness += full_deferral_prob * last_terminal_correctness_term
    full_expected_cost += full_deferral_prob * last_terminal_cost_term

    return {
        "expected_correctness": full_expected_correctness,
        "expected_cost": full_expected_cost,
        "conditional_deferral_probs": conditional_deferral_probs
    }


def all_subsets(iterable):
    """Yield all subsets (as tuples) of the given iterable."""
    s = list(iterable)
    for r in range(len(s) + 1):
        for combo in itertools.combinations(s, r):
            yield combo


def loss_fn(cost_sensitivity, metrics):
    return (1-metrics['expected_correctness']) + cost_sensitivity*metrics['expected_cost']

def unpenalized_loss_fn(metrics):
    return (1-metrics['expected_correctness'])

def compute_loss(T, model_indices, cost_sensitivity, expected_uncumulated_costs, prob_models):
    metrics = compute_metrics(T, model_indices, expected_uncumulated_costs, prob_models)
    return (1-metrics['expected_correctness']) + cost_sensitivity*metrics['expected_cost']

def compute_cost(T, model_indices, expected_uncumulated_costs, prob_models):
    metrics = compute_metrics(T, model_indices, expected_uncumulated_costs, prob_models)
    return metrics['expected_cost']

def compute_unpenalized_loss(T, model_indices, expected_uncumulated_costs, prob_models):
    metrics = compute_metrics(T, model_indices, expected_uncumulated_costs, prob_models)
    return (1-metrics['expected_correctness'])

def make_loss(model_indices, cost_sensitity, expected_uncumulated_costs, prob_models):
    def output_fun(T):
        return compute_loss(
            T, model_indices, cost_sensitity, expected_uncumulated_costs, prob_models
        )
    return output_fun

def make_unpenalized_loss(model_indices, expected_uncumulated_costs, prob_models):
    def output_fun(T):
        return compute_unpenalized_loss(
            T, model_indices, expected_uncumulated_costs, prob_models
        )
    return output_fun

def make_cost_constraint(model_indices, max_cost, expected_uncumulated_costs, prob_models):
    def output_fun(T):
        cost = compute_cost(
            T, model_indices, expected_uncumulated_costs, prob_models
        )
        return cost - max_cost
    return output_fun


def optimize_cascade_thresholds(
        model_indices, cost_sensitivity, expected_uncumulated_costs, prob_models, eps=1e-3,
        max_cost=None, mode='lagrange'
    ):
    """ 
    Optimize the thresholds of a chain.

    The cost sensitivity specifies how many percentage points of error the user
    is willing to incur in order to save $1 per million queries.
    """
    bounds = [ 
        (prob_models['bounds'][idx][0]+eps, prob_models['bounds'][idx][1]-eps) 
            for idx in model_indices[:-1]
    ]
    T0 = [ np.mean(interval) for interval in bounds ]

    if mode=='lagrange':
        objective_function = make_loss(
            model_indices, cost_sensitivity, expected_uncumulated_costs, prob_models
        )
        optimization_result = minimize(
            objective_function, T0, method='L-BFGS-B', bounds=bounds
        )
        return optimization_result
    
    elif mode=='constrained':
        objective_function = make_unpenalized_loss(
            model_indices, expected_uncumulated_costs, prob_models
        )
        constraint = {
            'type': 'ineq',
            'fun': make_cost_constraint(
                model_indices, max_cost, expected_uncumulated_costs, prob_models
            )
        }
        optimization_result = minimize(
            objective_function, T0, method='trust-constr', bounds=bounds, constraints=[constraint]
        )
    return optimization_result


def get_expected_uncumulated_costs(raw_model_costs, results):
    """
    Compute expected uncumulated costs per million queries.

    Specifically, expected costs are computed as
        Mean( #tok_in * $/mtok_in + #tok_out * $/mtok_out )
    """
    model_names = list(raw_model_costs.keys())
    expected_costs_per_million_queries = { 
        model_name: np.mean([
            ( record['in']*raw_model_costs[model_name]['in']
            + record['out']*raw_model_costs[model_name]['out'] ) 
            for record in tok_list
        ])
        for model_name, tok_list in results['model_tokens'].items() 
    }
    expected_uncumulated_costs = np.array(
        [ expected_costs_per_million_queries[name] for name in model_names ]
    )
    return expected_uncumulated_costs


def estimate_max_cost_sensitivity(model_indices, expected_uncumulated_costs, results, compare_idx=0, data=None, mode='hard'):
    """ 
    Estimate the maximum cost sensitivity -- the point at which the smallest and
    least performant model in the cascade is preferable over the biggest and most
    performant.
    """
    if mode=='hard':
        min_perf = 0
        max_perf = 1
    elif mode=='soft':
        min_perf = np.mean(data[compare_idx])
        max_perf = np.mean(data[-1])

    cumulated_costs = np.cumsum(expected_uncumulated_costs)
    min_cost = cumulated_costs[compare_idx]
    max_cost = cumulated_costs[-1]

    max_cost_sensitivity = np.abs(max_perf-min_perf)/(0.01 + max_cost-min_cost)
    return max_cost_sensitivity

def profile_cascade_adaptively(
        model_indices, expected_uncumulated_costs, results, 
        start_sensitivities=[0, 1e-12, 1e-10, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4],
        cost_threshold_multiplier=1.25,
        sensitivity_increase_factor=1.5,
        stop_val=1e3,
        max_iterations=100,
        data=None,
        mode='lagrange',
    ):
    """ 
    Profile a cascade by adaptively choosing cost sensitivities.
    
    Args:
        start_sensitivities: Initial list of cost sensitivities to try
        cost_threshold_multiplier: If metrics['expected_cost'] > multiplier*expected_uncumulated_costs[0],
                                 continue with higher sensitivities
        sensitivity_increase_factor: Factor to increase sensitivity by in adaptive phase
        stop_val: Maximum cost sensitivity value to try before stopping
        max_iterations: Maximum number of iterations in the adaptive phase
    """
    loss_list = []
    metrics_list = []
    optimal_thresholds_list = []
    cost_sensitivity_values = []

    # First run through initial cost sensitivities
    for c in tqdm(start_sensitivities):
        if len(model_indices) == 1:
            metrics = compute_metrics(
                [], model_indices, expected_uncumulated_costs, results
            )
            optimal_thresholds = 0.0
        else:
            optim_result = optimize_cascade_thresholds(
                model_indices=model_indices, 
                cost_sensitivity=c,
                expected_uncumulated_costs=expected_uncumulated_costs,
                prob_models=results
            )
            metrics = compute_metrics(
                optim_result.x, model_indices, expected_uncumulated_costs, results
            )
            optimal_thresholds = optim_result.x

        loss = loss_fn(c, metrics)
        metrics_list.append(metrics)
        loss_list.append(loss)
        optimal_thresholds_list.append(optimal_thresholds)
        cost_sensitivity_values.append(c)

    # Check if we need to continue with higher sensitivities
    if metrics['expected_cost'] > cost_threshold_multiplier * expected_uncumulated_costs[0]:
        # Continue with adaptive phase
        current_sensitivity = cost_sensitivity_values[-1]
        iteration_count = 0
        while (metrics['expected_cost'] > cost_threshold_multiplier * expected_uncumulated_costs[0] 
               and current_sensitivity < stop_val
               and iteration_count < max_iterations):
            # Increase sensitivity
            current_sensitivity *= sensitivity_increase_factor
            iteration_count += 1
            
            # Optimize with new sensitivity
            if len(model_indices) == 1:
                metrics = compute_metrics(
                    [], model_indices, expected_uncumulated_costs, results
                )
                optimal_thresholds = 0.0
            else:
                optim_result = optimize_cascade_thresholds(
                    model_indices=model_indices, 
                    cost_sensitivity=current_sensitivity,
                    expected_uncumulated_costs=expected_uncumulated_costs,
                    prob_models=results
                )
                metrics = compute_metrics(
                    optim_result.x, model_indices, expected_uncumulated_costs, results
                )
                optimal_thresholds = optim_result.x

            loss = loss_fn(current_sensitivity, metrics)
            metrics_list.append(metrics)
            loss_list.append(loss)
            optimal_thresholds_list.append(optimal_thresholds)
            cost_sensitivity_values.append(current_sensitivity)

    return (
        { "cost_sensitivity": cost_sensitivity_values,
            "loss": loss_list,
            "optimal_thresholds": optimal_thresholds_list }
        | { key: [ record[key] for record in metrics_list ] for key in metrics_list[0].keys() }
    )

def profile_cascade(
        model_indices, expected_uncumulated_costs, results, 
        n_grid=40, multiply_max_cost_sensitivity=1.0, data=None,
        density_multiplier_for_last_bucket=3.0, mode='lagrange',
        multiply_sensitivity=1.0,
    ):
    """ 
    Profile a cascade by optimizing the thresholds for each cost sensitivity. 

    The parameter n_grid chooses at how many different cost sensitivities to evaluate
    each cascade.
    """
    loss_list = []
    metrics_list = []
    optimal_thresholds_list = []

    if mode=='lagrange':
        min_cost_sensitivity = 0.0
        max_cost_sensitivity = multiply_max_cost_sensitivity * estimate_max_cost_sensitivity(model_indices, expected_uncumulated_costs, results)
        middle_pegs = sorted([
            multiply_sensitivity * estimate_max_cost_sensitivity(model_indices, expected_uncumulated_costs, results, compare_idx=i, data=data, mode='soft')
                for i in model_indices
        ])
        pegs = np.unique([min_cost_sensitivity, *middle_pegs])

        mid_cost_sensitivity_values = np.concatenate(
            [ 10**np.linspace(
                    1e-14+np.log10(pegs[i-1]), 1e-14+np.log10(pegs[i]),  int(np.floor(n_grid / (len(pegs)-1)))
                )[:-1] for i in range(1, len(pegs)) ]
        )
        mid_cost_sensitivity_values[np.isnan(mid_cost_sensitivity_values)] = 0.0

        high_cost_sensitivity_values = 10**np.linspace(
            np.log10(middle_pegs[-1]), np.log10(max_cost_sensitivity),
            int(density_multiplier_for_last_bucket*(np.floor(n_grid / (len(pegs)-1))))
        )

        cost_sensitivity_values = np.concatenate(
            [mid_cost_sensitivity_values, high_cost_sensitivity_values]
        )

        print(cost_sensitivity_values)

        for c in tqdm(cost_sensitivity_values):
            if len(model_indices) == 1:
                metrics = compute_metrics(
                    [], model_indices, expected_uncumulated_costs, results
                )
                optimal_thresholds = 0.0
            else:
                optim_result = optimize_cascade_thresholds(
                    model_indices=model_indices, 
                    cost_sensitivity=c,
                    expected_uncumulated_costs=expected_uncumulated_costs,
                    prob_models=results
                )
                metrics = compute_metrics(
                    optim_result.x, model_indices, expected_uncumulated_costs, results
                )
                optimal_thresholds = optim_result.x

            loss = loss_fn(c, metrics)
            metrics_list.append(metrics)
            loss_list.append(loss)
            optimal_thresholds_list.append(optimal_thresholds)
        
        return (
            { "cost_sensitivity": cost_sensitivity_values,
                "loss": loss_list,
                "optimal_thresholds": optimal_thresholds_list }
            | { key: [ record[key] for record in metrics_list ] for key in metrics_list[0].keys() }
        )


def profile_all_cascades(expected_uncumulated_costs, results, n_grid=10):
    cascade_records = []
    n_models = len(expected_uncumulated_costs)

    for last_model_idx in tqdm(range(n_models)):
        possible_ancestors = { i for i in range(0, last_model_idx) }
        if len(possible_ancestors) == 0:
            model_indices = [last_model_idx]
            output = profile_cascade(model_indices, expected_uncumulated_costs, results, n_grid)
            cascade_records.append({"model_indices": model_indices, "output": output})
            continue
        # Consider all possible subsets of ancestors
        for preceding_models in all_subsets(possible_ancestors):
            model_indices = sorted(preceding_models) + [last_model_idx]
            output = profile_cascade(model_indices, expected_uncumulated_costs, results, n_grid)
            cascade_records.append({"model_indices": model_indices, "output": output})

    return cascade_records


def route_and_score_query(thresholds, calibrated_conf, score_test, start_model_index=0):
    """ 
    Recursively evaluate the cascade. Return the score (True/False), and the index
    of the model responsible for the output (the local index, not the global index).

    Thresholds should 
    """
    if len(thresholds) == 0:
        assert len(score_test) == 1 
        return (score_test[0], start_model_index)
    elif len(thresholds) > 0:
        assert len(calibrated_conf) == len(thresholds)
        if calibrated_conf[0] <= thresholds[0]:
            # defer the query
            return route_and_score_query(thresholds[1:], calibrated_conf[1:], score_test[1:], start_model_index+1)
        else:
            return (score_test[0], start_model_index)
        
    
def score_cascade(
        thresholds, model_indices, expected_uncumulated_costs_test, test_data):
    """
    Score the cascade's performance on the test set.

    The parameter test_data is a dictionary with keys "calib_conf" and "corr",
    whose corresponding values are arrays whose columns are the calibrated
    confidences and correctness values on the test set.

    Test data should contain the data for ALL models; we subset to the model indices
    inside this function.
    """
    full_covariates_test = test_data['calib_conf']
    full_corr_test = test_data['corr']

    covariates_test = full_covariates_test[:, model_indices[:-1]]
    scores_test = full_corr_test[:, model_indices]

    cumulated_costs = np.cumsum([ expected_uncumulated_costs_test[idx] for idx in model_indices ])
    corr_and_cost_records = []

    for test_query_idx, (calib_conf_, score_test_) in enumerate(zip(covariates_test, scores_test)):
        # Simulate the working of the chain
        score, local_model_idx = route_and_score_query(thresholds, calib_conf_, score_test_)
        cost = cumulated_costs[local_model_idx]
        corr_and_cost_records.append([score, cost, local_model_idx])

    corr_and_cost = np.array(corr_and_cost_records)
    expected_correctness = np.mean(corr_and_cost[:, 0])
    expected_cost = np.mean(corr_and_cost[:, 1])

    return {
        "expected_correctness_test": expected_correctness,
        "expected_cost_test": expected_cost,
        "responsible_models": {
            i: np.mean(corr_and_cost[:,-1] == i) for i in range(len(model_indices))
        }
    }
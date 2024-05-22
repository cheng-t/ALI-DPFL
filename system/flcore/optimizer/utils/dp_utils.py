
import numpy as np
import math
from scipy import special


def cartesian_to_polar(x):
    r = np.linalg.norm(x)
    theta = np.arccos(x[0] / r)  
    phi = [1. for i in range(len(x) - 1)]
    for i in range(len(phi)):
        phi[i] = np.arctan2(x[i + 1], x[0])  
    return np.concatenate(([r, theta], phi))


def polar_to_cartesian(p):
    r = p[0]
    theta = p[1]
    phi = p[2:]
    x = [1. for i in range(len(phi) + 1)]
    x[0] = r * np.cos(theta)  
    for i in range(len(phi)):
        x[i + 1] = x[0] * np.tan(phi[i])
    for j in range(len(x)):
        x[j] = round(x[j], 4)  
    return x


def vector_to_matrix(vector, shape):
    shape = tuple(shape)
    if len(shape) == 0 or np.prod(shape) != len(vector):
        raise ValueError("Invalid input dimensions")
    matrix = np.zeros(shape)
    strides = [np.prod(shape[i + 1:]) for i in range(len(shape) - 1)] + [1]
    for i in range(len(vector)):
        index = [0] * len(shape)
        for j in range(len(shape)):
            index[j] = (i // strides[j]) % shape[j]
        matrix[tuple(index)] = vector[i]
    return matrix



def cartesian_add_noise(p, sigma1, C1, sigma2):
    
    r = p[0]
    r += C1 * sigma1 * np.random.normal(0, 1)

    theta = p[1:]  
    theta += 2 * math.pi * sigma2 * np.random.normal(0, 1)

    return np.concatenate(([r], theta))


def devide_epslion(sigma, q, n):
    orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(11, 64)) + [128, 256, 512]
    eps, opt_order = apply_dp_sgd_analysis(q, sigma, 1, orders, 10 ** (-5))

    eps_sum = n * eps
   
    eps1 = eps_sum * 0.000001
    
    eps2 = eps_sum - eps1
   
    sigma1 = get_noise_multiplier(target_epsilon=eps1, target_delta=1e-5, sample_rate=512 / 60000, steps=1,
                                  alphas=orders)
    sigma2 = get_noise_multiplier(target_epsilon=eps2, target_delta=1e-5, sample_rate=512 / 60000, steps=1,
                                  alphas=orders)
    return sigma1, sigma2


def get_noise_multiplier(
        target_epsilon: float,
        target_delta: float,
        sample_rate: float,
        steps: int,
        alphas,
        epsilon_tolerance: float = 0.01,
) -> float:
    r"""
    Computes the noise level sigma to reach a total budget of (target_epsilon, target_delta)
    at the end of epochs, with a given sample_rate
    Args:
        target_epsilon: the privacy budget's epsilon
        target_delta: the privacy budget's delta
        sample_rate: the sampling rate (usually batch_size / n_data)
        steps: number of steps to run
        epsilon_tolerance: precision for the binary search
    Returns:
        The noise level sigma to ensure privacy budget of (target_epsilon, target_delta)
    """

    sigma_low, sigma_high = 0, 1000  

    eps_high, best_alpha = apply_dp_sgd_analysis(sample_rate, sigma_high, steps, alphas, target_delta)

    if eps_high > target_epsilon:
        raise ValueError("The target privacy budget is too low. 当前可供搜索的最大的sigma只到100")

    while target_epsilon - eps_high > epsilon_tolerance:  
        sigma = (sigma_low + sigma_high) / 2

        eps, best_alpha = apply_dp_sgd_analysis(sample_rate, sigma, steps, alphas, target_delta)

        if eps < target_epsilon:
            sigma_high = sigma
            eps_high = eps
        else:
            sigma_low = sigma

    return round(sigma_high, 2)


def compute_dp_sgd_privacy(n, batch_size, noise_multiplier, epochs, delta):
    """Compute epsilon based on the given hyperparameters.
    Args:
      n: Number of examples in the training data. 训练集样本总数
      batch_size: Batch size used in training. 一批采样的样本数
      noise_multiplier: Noise multiplier used in training. 噪声系数
      epochs: Number of epochs in training. 本地迭代轮次（还没有算上本地一次迭代中的多个batch迭代）
      delta: Value of delta for which to compute epsilon.
      S:sensitivity      这个原本的库是没有的
    Returns:
      Value of epsilon corresponding to input hyperparameters.  返回epsilon
    """
    q = batch_size / n  # q - the sampling ratio. 
    if q > 1:
        print('n must be larger than the batch size.')
    orders = (list(range(2, 64)) + [128, 256, 512])  

 
    steps = int(math.ceil(epochs * (n / batch_size)))

    return apply_dp_sgd_analysis(q, noise_multiplier, steps, orders, delta)


def apply_dp_sgd_analysis(q, sigma, steps, orders, delta):
    """Compute and print results of DP-SGD analysis."""

    # compute_rdp requires that sigma be the ratio of the standard deviation of
    # the Gaussian noise to the l2-sensitivity of the function to which it is
    # added. Hence, sigma here corresponds to the `noise_multiplier` parameter   sigma=noise_multilpier
    # in the DP-SGD implementation found in privacy.optimizers.dp_optimizer
    rdp = compute_rdp(q, sigma, steps, orders)  

    eps, opt_order = compute_eps(orders, rdp, delta)  

    return eps, opt_order


def compute_rdp(q, noise_multiplier, steps, orders):
    """Computes RDP of the Sampled Gaussian Mechanism.
    Args:
      q: The sampling rate.
      noise_multiplier: The ratio of the standard deviation of the Gaussian noise    STD标准差，敏感度应该包含在这里面了
        to the l2-sensitivity of the function to which it is added.
      steps: The number of steps.
      orders: An array (or a scalar) of RDP orders.
    Returns:
      The RDPs at all orders. Can be `np.inf`.
    """
    if np.isscalar(orders):  
        rdp = _compute_rdp(q, noise_multiplier, orders)  
    else:  
        rdp = np.array(
            [_compute_rdp(q, noise_multiplier, order) for order in orders])

    return rdp * steps  


def compute_eps(orders, rdp, delta):
    """Compute epsilon given a list of RDP values and target delta.
    Args:
      orders: An array (or a scalar) of orders.
      rdp: A list (or a scalar) of RDP guarantees.
      delta: The target delta.
    Returns:
      Pair of (eps, optimal_order).
    Raises:
      ValueError: If input is malformed.
    """
    orders_vec = np.atleast_1d(orders)  
    rdp_vec = np.atleast_1d(rdp)

    if delta <= 0:  
        raise ValueError("Privacy failure probability bound delta must be >0.")
    if len(orders_vec) != len(rdp_vec): 
        raise ValueError("Input lists must have the same length.")

    eps_vec = []
    for (a, r) in zip(orders_vec, rdp_vec):
        if a < 1:
            raise ValueError("Renyi divergence order must be >=1.")
        if r < 0:
            raise ValueError("Renyi divergence must be >=0.")

        if delta ** 2 + math.expm1(-r) >= 0:  # delta的约束条件
            # In this case, we can simply bound via KL divergence:
            # delta <= sqrt(1-exp(-KL)).
            eps = 0  # No need to try further computation if we have eps = 0.
        elif a > 1.01:
            # This bound is not numerically stable as alpha->1.Thus we have a min value of alpha.
            eps = (r - (np.log(delta) + np.log(a)) / (a - 1) + np.log((a - 1) / a))
        else:
            # In this case we can't do anything. E.g., asking for delta = 0.
            eps = np.inf  
        eps_vec.append(eps)

    idx_opt = np.argmin(eps_vec)  
    return max(0, eps_vec[idx_opt]), orders_vec[idx_opt]


def _compute_rdp(q, sigma, alpha):

    if q == 0:
        return 0

    # no privacy
    if sigma == 0:
        return np.inf

    
    if q == 1.: 
        return alpha / (
                2 * sigma ** 2)  

    if np.isinf(alpha):
        return np.inf

    if float(alpha).is_integer():  
        return _compute_log_a_for_int_alpha(q, sigma, int(alpha))
    else:  
        return _compute_log_a_for_frac_alpha(q, sigma, alpha)



def _compute_log_a_for_int_alpha(q, sigma, alpha):
    assert isinstance(alpha, int)
    rdp = -np.inf

    for i in range(alpha + 1):
        log_b = (
                math.log(special.binom(alpha, i))
                + i * math.log(q)
                + (alpha - i) * math.log(1 - q)
                + (i * i - i) / (2 * (sigma ** 2))
        )

        a, b = min(rdp, log_b), max(rdp, log_b)
        if a == -np.inf: 
            rdp = b
        else:
            rdp = math.log(math.exp(
                a - b) + 1) + b  

    rdp = float(rdp) / (alpha - 1)
    return rdp


def _compute_log_a_for_frac_alpha(q: float, sigma: float, alpha: float) -> float:

    # The two parts of A_alpha, integrals over (-inf,z0] and [z0, +inf), are
    # initialized to 0 in the log space:
    log_a0, log_a1 = -np.inf, -np.inf
    i = 0

    z0 = sigma ** 2 * math.log(1 / q - 1) + 0.5

    while True:  # do ... until loop
        coef = special.binom(alpha, i)
        log_coef = math.log(abs(coef))
        j = alpha - i

        log_t0 = log_coef + i * math.log(q) + j * math.log(1 - q)
        log_t1 = log_coef + j * math.log(q) + i * math.log(1 - q)

        log_e0 = math.log(0.5) + _log_erfc((i - z0) / (math.sqrt(2) * sigma))
        log_e1 = math.log(0.5) + _log_erfc((z0 - j) / (math.sqrt(2) * sigma))

        log_s0 = log_t0 + (i * i - i) / (2 * (sigma ** 2)) + log_e0
        log_s1 = log_t1 + (j * j - j) / (2 * (sigma ** 2)) + log_e1

        if coef > 0:
            log_a0 = _log_add(log_a0, log_s0)
            log_a1 = _log_add(log_a1, log_s1)
        else:
            log_a0 = _log_sub(log_a0, log_s0)
            log_a1 = _log_sub(log_a1, log_s1)

        i += 1
        if max(log_s0, log_s1) < -30:
            break

    return _log_add(log_a0, log_a1) / (alpha - 1)


def _log_add(logx: float, logy: float) -> float:
    r"""Adds two numbers in the log space.

    Args:
        logx: First term in log space.
        logy: Second term in log space.

    Returns:
        Sum of numbers in log space.
    """
    a, b = min(logx, logy), max(logx, logy)
    if a == -np.inf:  
        return b
    return math.log1p(math.exp(a - b)) + b  


def _log_sub(logx: float, logy: float) -> float:
    r"""Subtracts two numbers in the log space.

    Args:
        logx: First term in log space. Expected to be greater than the second term.
        logy: First term in log space. Expected to be less than the first term.

    Returns:
        Difference of numbers in log space.

    Raises:
        ValueError
            If the result is negative.
    """
    if logx < logy:
        raise ValueError("The result of subtraction must be non-negative.")
    if logy == -np.inf:  # subtracting 0
        return logx
    if logx == logy:
        return -np.inf  # 0 is represented as -np.inf in the log space.

    try:
        # Use exp(x) - exp(y) = (exp(x - y) - 1) * exp(y).
        return math.log(math.expm1(logx - logy)) + logy  # expm1(x) = exp(x) - 1
    except OverflowError:
        return logx


def _log_erfc(x: float) -> float:

    return math.log(2) + special.log_ndtr(-x * 2 ** 0.5)
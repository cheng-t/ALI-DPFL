
from flcore.optimizer.utils.RDP.compute_dp_sgd import apply_dp_sgd_analysis


def get_max_steps(
        target_epsilon: float,
        target_delta: float,
        sample_rate: float,
        sigma: float,
        alphas,
        epsilon_tolerance: float = 0.01,
) -> int:
    steps_low, steps_high = 0, 100000

    eps_high, best_alpha = apply_dp_sgd_analysis(sample_rate, sigma, steps_high, alphas, target_delta)

    if eps_high < target_epsilon:
        raise ValueError("The privacy budget is too high.")

    while eps_high - target_epsilon > epsilon_tolerance:
        steps = (steps_low + steps_high) / 2
        eps, best_alpha = apply_dp_sgd_analysis(sample_rate, sigma, steps, alphas, target_delta)

        if eps > target_epsilon:
            steps_high = steps
            eps_high = eps
        else:
            steps_low = steps

    return int(steps_high)

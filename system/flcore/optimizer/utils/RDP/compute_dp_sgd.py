
import math

from flcore.optimizer.utils.RDP.compute_rdp import compute_rdp
from flcore.optimizer.utils.RDP.rdp_convert_dp import compute_eps


def compute_dp_sgd_privacy(n, batch_size, noise_multiplier, epochs, delta):

    q = batch_size / n  
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


def apply_dp_sgd_analysis_old(q, sigma, steps, orders, delta):
    """Compute and print results of DP-SGD analysis."""

    # compute_rdp requires that sigma be the ratio of the standard deviation of
    # the Gaussian noise to the l2-sensitivity of the function to which it is
    # added. Hence, sigma here corresponds to the `noise_multiplier` parameter   sigma=noise_multilpier
    # in the DP-SGD implementation found in privacy.optimizers.dp_optimizer
    rdp = compute_rdp(q, sigma, steps, orders)  

    eps, opt_order = compute_eps(orders, rdp, delta)  

    return eps, opt_order


'''
orders = (list(range(2, 64)) + [128, 256, 512])  
eps, opt_order=apply_dp_sgd_analysis(256/60000, 1.1, 17470, orders, 10**(-5))
print("eps:",format(eps)+"| order:",format(opt_order))
'''

if __name__ == "__main__":
    orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(11, 64)) + [128, 256, 512]

    eps, opt_order = apply_dp_sgd_analysis(512 / 60000, 0.01, 1, orders, 10 ** (-5))
    print("eps:", format(eps) + "| order:", format(opt_order))

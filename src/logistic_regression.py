import numpy as np
from scipy import optimize, stats
import scipy.linalg
import matplotlib.pyplot as plt
import string
import tensorflow as tf

# function to simulate data
# simulate prior, xs from N(0, I), Y from bernoulli.
def gen_data(N, beta, sigma_x):
    D = beta.shape[0]
    X = sigma_x*np.random.normal(size=[N, D])
    ps = (1. + np.exp(-X.dot(beta)))**-1
    Y = np.random.binomial(1, p=ps)
    Y = 2*Y -1
    Y = np.array(Y, dtype=np.float)
    return X, Y


# function to find MLE
@tf.function
def log_lik(beta, X, Y):
    Xbeta = tf.einsum('nd,id->in', X, beta)
    #p_1 = (1+tf.exp(-Xbeta))**-1
    log_lik_by_datapoint = -tf.math.log(1+tf.exp(-Y[None]*Xbeta))
    log_lik_val = tf.reduce_sum(log_lik_by_datapoint, axis=1)
    return log_lik_val

@tf.function
def grad_log_lik(beta, X, Y):
    with tf.GradientTape() as t:
        t.watch(beta)
        log_lik_val = log_lik(beta, X, Y)
    return t.gradient(log_lik_val, beta)

@tf.function(experimental_relax_shapes=True)
def lik_hess(X, Y, beta_MLE):
    """hessian of negative log likelihood.
    """
    with tf.GradientTape() as t1:
        t1.watch(beta_MLE)
        with tf.GradientTape() as t2:
            t2.watch(beta_MLE)
            neg_log_lik = -log_lik(beta_MLE, X, Y)[0]
        jac = t2.jacobian(neg_log_lik, beta_MLE)[0]
    hess = t1.jacobian(jac, beta_MLE)[:, 0]
    return hess

# For posterior
@tf.function
def log_post(beta, X, Y):
    log_prior = -(1./2)*tf.reduce_sum(beta**2, axis=1)
    log_lik_val = log_lik(beta, X, Y)
    return log_prior + log_lik_val

@tf.function
def grad_log_post(beta, X, Y):
    with tf.GradientTape() as t:
        t.watch(beta)
        log_post_val = log_post(beta, X, Y)
    return t.gradient(log_post_val, beta)

@tf.function(experimental_relax_shapes=True)
def post_hess(X, Y, beta_MAP):
    """hessian of negative log posterior.
    """
    with tf.GradientTape() as t1:
        t1.watch(beta_MAP)
        with tf.GradientTape() as t2:
            t2.watch(beta_MAP)
            neg_log_post = -log_post(beta_MAP, X, Y)[0]
        jac = t2.jacobian(neg_log_post, beta_MAP)[0]
    hess = t1.jacobian(jac, beta_MAP)[:, 0]
    return hess


# for optimising
def find_max(f, grad_f, x_0):
    res = optimize.minimize(f, x_0, method='L-BFGS-B', jac=grad_f)
    if not res.success:
        print("Optimisation Failed!")
        print("res = ", res)
    return res.x

def MLE(X, Y):
    def f(beta):
        beta_tf = tf.convert_to_tensor(beta[None])
        return -log_lik(beta_tf, X, Y).numpy()

    def grad_f(beta):
        beta_tf = tf.convert_to_tensor(beta[None])
        return -grad_log_lik(beta_tf, X, Y).numpy()

    D = X.shape[1]
    beta_hat = find_max(f, grad_f, x_0=np.zeros([D]))
    return beta_hat

def MAP(X, Y):
    def f(beta):
        beta_tf = tf.convert_to_tensor(beta[None])
        return -log_post(beta_tf, X, Y).numpy()

    def grad_f(beta):
        beta_tf = tf.convert_to_tensor(beta[None])
        return -grad_log_post(beta_tf, X, Y).numpy()
    D = X.shape[1]
    beta_0 = np.zeros([D])
    beta_hat = find_max(f, grad_f, x_0=beta_0)
    return beta_hat


def grid_pts(n_grid_spaces, scale=3):
    delta = scale/n_grid_spaces

    # Create vector of locations at which to query predictive distribution
    x, y = np.arange(-scale, scale, delta), np.arange(-scale, scale, delta)
    X, Y = np.meshgrid(x, y)
    X_long, Y_long = X.reshape([-1]), Y.reshape([-1])
    query_pts = np.array(list(zip(X_long, Y_long)))

    return X, Y, query_pts

def posterior_density_grid(X_covs, Y_labels, n_grid_spaces=20, scale=3):
    """posterior_density_grid evaluate the posterior dictive density at a grid of points.

    Args:
        n_grid_spaces: granularity of locations at which to compute density
    """
    X, Y, query_pts = grid_pts(n_grid_spaces, scale=scale)

    # Compute predictive density and reshape into a grid for easy visualization
    post_dens = log_post(query_pts, X_covs, Y_labels)
    post_dens_grid = tf.reshape(post_dens, X.shape)
    X_Y_pd = np.array([X, Y, post_dens_grid])
    return X_Y_pd

def lik_grid(X_covs, Y_labels, n_grid_spaces=20, scale=3):
    """posterior_density_grid evaluate the posterior dictive density at a grid of points.

    Args:
        n_grid_spaces: granularity of locations at which to compute density
    """
    X, Y, query_pts = grid_pts(n_grid_spaces, scale=scale)

    # Compute predictive density and reshape into a grid for easy visualization
    log_lik_vals = log_lik(query_pts, X_covs, Y_labels)
    log_lik_grid = tf.reshape(log_lik_vals, X.shape)
    X_Y_lik = np.array([X, Y, log_lik_grid])
    return X_Y_lik

def plot_density(X, Y, Z, title=None):
    """plot_density shows a predictive density with a contour plot.

    Args:
        X, Y: x and y positions of density values, both np.array of shape [n_grid_spaces, n_grid_spaces]
        Z: density values

    """
    fig, ax = plt.subplots()
    CS = ax.contour(X, Y, Z)
    ax.clabel(CS, inline=1, fontsize=10)
    ax.set_title(title)
    plt.show()

### Code for Gaus-Hermite Quadrature approximation to the MLE
def GHQ_about_MAP(log_g, H, beta_MAP, n_grid_pts=5, param_idx_for_mean=None):
    """GHQ_about_MAP using Gauss-Hermite quadrature to integrate g

    Approximates only 2D indefinite integrals.


    Toss jacobian term, since it could worsen stability - and we are normalizing anyways.

    Args:
        log_g: log posterior up to an additive constant
        H: hessian of negative log posterior
        beta_MAP: max a posteriori value -- about which to compute integral
        n_grid_pts: size of grid for each dimension
        param_idx_for_mean: set None to get normalizing constant, or set to 0 or 1
            for computing term for posterior mean of that index.

    Returns:
        approximate integral
    """
    # Define change of variables
    H_inv_sqrt = scipy.linalg.sqrtm(np.linalg.inv(H))
    phi = lambda beta: np.sqrt(2)*H_inv_sqrt.dot(beta.T).T + beta_MAP[None]

    # this is the entire expression after the weights in the equation above
    g_scaled_density = lambda gamma: np.exp(np.sum(gamma**2, axis=1) + log_g(phi(gamma)))
    if param_idx_for_mean is not None:
        g_scaled = lambda gamma: g_scaled_density(gamma)*(phi(gamma)[:, param_idx_for_mean])
    else:
        g_scaled = g_scaled_density

    GHQ_pts, GHQ_wts = np.polynomial.hermite.hermgauss(n_grid_pts)

    D = H.shape[0]
    grids_by_dim = np.meshgrid(*[GHQ_pts for _ in range(D)])
    idcs = string.ascii_lowercase[:D]
    GHQ_wts_grid = np.einsum("%s->%s"%(",".join(idcs), idcs), *[GHQ_wts for _ in range(D)])

    # reshape into long arrays for vectorized evaluation
    query_pts = np.array(list(zip(*[list(grid.reshape([-1])) for grid in grids_by_dim])))
    GHQ_wts_long = GHQ_wts_grid.reshape([-1])

    # Evaluate the function the query points (gamma) and add together
    approx = g_scaled(query_pts).dot(GHQ_wts_long)

    # Multiply by jacobian determinant to account for reparameterisation
    # We can skip this, since we only need up to proportionality.
    #jac_det = np.linalg.det(np.sqrt(2)*H_inv_sqrt)
    #approx *= jac_det

    return approx

def post_mean_GHQ(X, Y, n_grid_pts=5):
    """post_mean_GHQ uses Gauss-Hermite Quadrature to approximate the posterior mean.

    Defined only for 2D integrals

    Args:
        X, Y : Covariates and responses
        n_grid_pts: size of grid for each dimension

    Returns:
        Estimate of posterior mean
    """

    # Compute MAP and hessian of the negative log posterior, to center GHQ approximation
    beta_MAP = MAP(X, Y)
    beta_MAP_tf = tf.convert_to_tensor(beta_MAP[None])
    hess_at_MAP = post_hess(X, Y, beta_MAP_tf).numpy()
    log_post_at_beta_MAP = log_post(beta_MAP[None], X, Y)[0]

    # Compute joint density at MAP and scale to avoid underflow

    # approximate normalizing constant
    log_g = lambda beta: log_post(beta, X, Y) - log_post_at_beta_MAP
    post_const = GHQ_about_MAP(log_g, hess_at_MAP, beta_MAP, n_grid_pts=n_grid_pts)

    # approximate numerator
    D = X.shape[1]
    post_mean_unscaled = np.array([
        GHQ_about_MAP(log_g, hess_at_MAP, beta_MAP, n_grid_pts=n_grid_pts, param_idx_for_mean=i)
        for i in range(D)])

    # Scale by normalizing constant
    post_mean_est = post_mean_unscaled / post_const
    return post_mean_est

def approx_post_mean(X, Y, beta_MLE, return_tranform_and_cov=False):
    """approx_post_mean forms an approximation to the posterior mean using a Gaussian
    approximation to the likelihood.

    The Gaussian approximation of the likelihood is formed at the MLE.

    Args:
        X, Y: covariates and responses
        beta_MLE: maximum likelihood estimate of beta

    Returns:
        Approximation to the posterior mean.
    """
    beta_MLE_tf = tf.convert_to_tensor(beta_MLE[None])
    H = lik_hess(X, Y, beta_MLE_tf).numpy()
    D = beta_MLE.shape[0]
    transform = np.linalg.inv(H + np.eye(D))@H
    post_mean_est = transform.dot(beta_MLE)
    if not return_tranform_and_cov:
        return post_mean_est

    return post_mean_est, transform, H

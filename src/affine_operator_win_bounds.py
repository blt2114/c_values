import numpy as np
import scipy
from scipy import optimize, stats

def W_lower_bound_offset_and_variance_term1(A, b, C, d, Sigma, y):
    # First compute offset
    theta_hat = A.dot(y)+b
    theta_star = C.dot(y)+d
    offset = (np.linalg.norm(theta_hat - y)**2 - np.linalg.norm(theta_star - y)**2) + 2*np.trace((A-C).dot(Sigma))

    # Next compute variance_term1
    Sigma_sqrt = scipy.linalg.sqrtm(Sigma)
    variance_term1 = (1/2)*np.sum((Sigma_sqrt.dot(A + A.T - C - C.T).dot(Sigma_sqrt))**2)
    return offset, variance_term1

def W_lower_bound_from_offset_and_variance_term(Gtheta_norm, alpha, offset, variance_term1):
    """W_lower_bound_from_offset_and_variance_term efficiently computes bound using precomputed terms.

    """
    variance = variance_term1 + Gtheta_norm
    quantile_term = 2*np.sqrt(variance)*stats.norm.ppf(alpha)
    return offset + quantile_term

def W_lower_bound(A, b, C, d, Sigma, Gtheta_norm, y, alpha):
    """W_lower_bound returns the alpha quantile of the normal approximation to distribution of the win.

    The squared, Sigma quadratic norm of G(theta) is assumed known (and provided).

    In particular the bound is
     (|theta_hat - y|^2 - |theta_star - y|^2) + 2Tr[(A-C)Sigma] + 2*sqrt{
        Gtheta_norm + (1/2)|Sigma^(1/2) (A + A^T - C - C^T)Sigma^(1/2)|_F^2 } z_alpha,
    where
        theta_hat = Ay + b
        theta_star = Cy + d
        Tr[ -- ] denotes trace
        Gtheta_norm = | G(\theta)|_Sigma^2
        | -- |_F denotes the Frobenius norm
        z_alpha is the alpha quantile of the standard normal


    The bound is computed in two stages --
        first computing the offset and one of the components of the variance,
        and second putting these together with Gtheta_norm and alpha to compute the bound.

    This decomposition permits fast computation of c-values by allowing the more intensive
    steps to be computed just once.
    """
    offset, variance_term1 = W_lower_bound_offset_and_variance_term1(A, b, C, d, Sigma, y)
    bound_value = W_lower_bound_from_offset_and_variance_term(Gtheta_norm, alpha, offset, variance_term1)
    return bound_value

def Gtheta_norm_upper_bound_gamma_kappa_delta(A, C, Sigma, Gy_norm):
    Sigma_sqrt = scipy.linalg.sqrtm(Sigma)

    # compute shortened variables for root finding
    gamma = Gy_norm - np.sum((Sigma_sqrt@(A-C)@(Sigma_sqrt))**2)
    kappa = 2*np.sum((Sigma_sqrt@(A-C)@Sigma@((A-C).T)@Sigma_sqrt)**2)
    delta = 4*np.linalg.norm(Sigma_sqrt@(A-C)@Sigma_sqrt, ord=2)**2
    return gamma, kappa, delta

def Gtheta_norm_upper_bound_from_gamma_kappa_delta(alpha, gamma, kappa, delta):
    eta = stats.norm.ppf(alpha)

    # compute coefficients of quadratic
    a = 1                           # quadratic coefficient
    b = -(2*gamma + (eta**2)*delta) # linear coefficient
    c = gamma**2 - (eta**2)*kappa   # constant

    # find larger root and confirm that it is (at least approximately) real
    roots = np.roots([a, b, c])
    Gtheta_norm_upper = max(roots)
    if not np.isclose(Gtheta_norm_upper, np.real(Gtheta_norm_upper)):
        print("roots:", roots)
        print("returning zero as the upper bound")
        return 0.
    return Gtheta_norm_upper

def Gtheta_norm_upper_bound(A, C, Sigma, Gy_norm, alpha):
    """Gtheta_norm_upper_bound computes an upper bound for a level 1-alpha confidence interval for the squared,
    Sigma quadratic norm of G(theta).

    Specifically, this is supremum of the set of all c's such that
     Gy_norm >= (c + | Sigma^(1/2) (A- C) Sigma^(1/2)|_F^2 + sqrt{
         2 |Sigma^(1/2) (A-C)Sigma (A-C)^T Sigma|_F^2 + 4|Sigma^(1/2) (A-C)Sigma^(1/2)|_OP^2 * c
     }*z_alpha,

    Where
        Gy_norm = | G(y)|_Sigma^2
        | -- |_OP is the operator norm
    and the other notation is the same as in W_lower_bound.

    We solve this by finding the roots of the quadratic (in x) of
        x^2 - (2*gamma + eta^2 * delta)*x + (gamma^2 - eta^2 * kappa) = 0,
    where
        gamma = Gy_norm - | Sigma^(1/2) (A- C) Sigma^(1/2)|_F^2
        eta = z_alpha
        kappa = 2 |Sigma^(1/2) (A-C)Sigma (A-C)^T Sigma|_F^2
        delta = 4|Sigma^(1/2) (A-C)Sigma^(1/2)|_OP^2

    As with W_lower_bound, we separate computation into two stages, where the first
    stage need be computed just once for computation of c-values.
    """
    gamma, kappa, delta = Gtheta_norm_upper_bound_gamma_kappa_delta(A, C, Sigma, Gy_norm)
    Gtheta_norm_upper = Gtheta_norm_upper_bound_from_gamma_kappa_delta(alpha, gamma, kappa, delta)
    return Gtheta_norm_upper

def b_bound(A, b, C, d, Sigma, y, alpha):
    """b_bayes computes b(y, alpha) and an upper bound on norm.

    If b_bound >0 for large alpha (close to 1), we shoud be confident in an improvement.
    """
    Gy = (A-C).dot(y) + (b-d)
    Gy_norm = Gy.dot(Sigma).dot(Gy)

    Gtheta_norm_upper = Gtheta_norm_upper_bound(A, C, Sigma, Gy_norm, (1.-alpha)/2)
    bya = W_lower_bound(A, b, C, d, Sigma, Gtheta_norm_upper, y, (1.-alpha)/2)
    return bya, Gtheta_norm_upper

def c_value(y, A, b, C, d, Sigma):
    ### Pre-compute quantities from the first stages of Gtheta_norm_upper_bound and W_lower_bound.
    # first for Gtheta_norm_upper_bound
    Gy = (A-C).dot(y) + (b-d)
    Gy_norm = Gy.dot(Sigma).dot(Gy)
    gamma, kappa, delta = Gtheta_norm_upper_bound_gamma_kappa_delta(A, C, Sigma, Gy_norm)

    # next for W_lower_bound
    offset, variance_term1 = W_lower_bound_offset_and_variance_term1(A, b, C, d, Sigma, y)

    ## Define local function for computing the bound, using the quantities above
    def b_bound_local(alpha):
        Gtheta_norm_upper = Gtheta_norm_upper_bound_from_gamma_kappa_delta(
            (1-alpha)/2, gamma, kappa, delta)
        b_bound_val = W_lower_bound_from_offset_and_variance_term(
            Gtheta_norm_upper, (1.-alpha)/2, offset, variance_term1)
        assert np.isreal(b_bound_val)
        return b_bound_val

    ## If bound is vacuous return immediately
    if b_bound_local(alpha=0.) < 0.: return 0.

    ## If bound is too close to 1 (very high power...)
    if b_bound_local(alpha=1.-1e-8) > 0.: return 1.

    ### Perform binary search for c-value
    c_val = optimize.bisect(
        f=lambda lmbda: b_bound_local(alpha=lmbda),
        a=0, b=1.-1e-8)
    return c_val

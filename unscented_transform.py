from typing import Callable, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


class Distribution:
    """
    Represents a generic probability distribution.
    """
    def dimension(self):
        raise NotImplementedError

    def mean(self):
        raise NotImplementedError
    
    def covariance(self):
        raise NotImplementedError
    
    def compute_sigma_points(self):
        raise NotImplementedError
    
    def sample(self):
        raise NotImplementedError
    
    def from_samples(self, samples: List[np.ndarray]):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError
    

class Gaussian(Distribution):
    """
    Represents a Gaussian distribution.
    """
    def __init__(self, mean, covariance, rng=None):
        self._mean = mean
        self._covariance = covariance
        self._rng = np.random.default_rng(rng)

    @property
    def rng(self):
        return self._rng

    def mean(self) -> np.ndarray:
        """
        Return the mean of the Gaussian distribution.
        """
        return self._mean
    
    def covariance(self) -> np.ndarray:
        """
        Return the covariance of the Gaussian distribution.
        """
        return self._covariance
    
    def dimension(self) -> int:
        """
        Return the dimension of the Gaussian distribution.
        """
        return len(self.mean())

    def sqrt_covariance(self) -> np.ndarray:
        """
        Return the square root of the covariance matrix.
        """
        return np.linalg.cholesky(self.covariance())
    
    def compute_sigma_points(self) -> List[np.ndarray]:
        """
        Compute the sigma points for the Gaussian distribution.
        """
        mean = self.mean()
        sqrt_covariance = self.sqrt_covariance()
        n = self.dimension()
        kappa = n - 3
        factor = np.sqrt(n + kappa)
        sigma_points = [mean]
        for i in range(n): # 2n + 1 sigma points
            sigma_points.append(mean + factor*sqrt_covariance[:, i])
            sigma_points.append(mean - factor*sqrt_covariance[:, i])
        assert len(sigma_points) == 2*n + 1
        return sigma_points
    
    def compute_weights(self) -> List[float]:
        """
        Compute the weights for the sigma points.
        """
        n = self.dimension()
        kappa = n - 3
        weights = [kappa / (n + kappa)]
        for _ in range(2*n):
            weights.append(1.0 / (2.0*(n + kappa)))
        assert len(weights) == 2*n + 1
        return weights

    def from_sigma_points(self, sigma_points: List[np.ndarray], weights: List[float]):
        """
        Create a Gaussian distribution from the given sigma points.
        Equations coming from https://arxiv.org/pdf/2104.01958
        """
        new_mean = np.zeros(self.dimension())
        for i in range(len(sigma_points)):
            new_mean += weights[i] * sigma_points[i]
        new_covariance = np.zeros((self.dimension(), self.dimension()))
        for i in range(len(sigma_points)):
            diff = sigma_points[i] - new_mean
            new_covariance += weights[i] * np.outer(diff, diff)
        return Gaussian(new_mean, new_covariance, self.rng)

    def sample(self) -> np.ndarray:
        """
        Sample from the Gaussian distribution.
        """
        return self.rng.multivariate_normal(self.mean(), self.covariance())
    
    def from_samples(self, samples: List[np.ndarray]):
        """
        Create a Gaussian distribution from the given samples.
        """
        sample_array = np.array(samples, dtype=np.float64)
        new_mean = np.mean(sample_array, axis=0)
        new_covariance = np.cov(sample_array, rowvar=False)
        return Gaussian(new_mean, new_covariance, self.rng)

    def __repr__(self):
        return f'Gaussian(mean={self.mean()}, covariance={self.covariance()})'
    

def unscented_transform(
    distribution: Distribution,
    non_linear_function: Callable,
) -> Distribution:
    """
    Propagate the distribution through the non-linear function using the unscented transform.
    """
    sigma_points = distribution.compute_sigma_points()
    transformed_sigma_points = [non_linear_function(sp) for sp in sigma_points]
    weights = distribution.compute_weights()
    return distribution.from_sigma_points(transformed_sigma_points, weights)


def propagate_samples(
    distribution: Distribution,
    non_linear_function: Callable,
    num_samples: int,
) -> Distribution:
    """
    Sample a distribution and propagate the samples through the non-linear function.
    Reestimate the distribution from the transformed samples.
    """
    samples = [distribution.sample() for _ in range(num_samples)]
    transformed_samples = [non_linear_function(sample) for sample in samples]
    return distribution.from_samples(transformed_samples)


def plot_ellipse(mean: np.ndarray, cov: np.ndarray, ax, n_std=1, **kwargs):
    """
    Plot an ellipse representing the Gaussian distribution.
    """
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1,1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0,0), width=ell_radius_x * 2, height=ell_radius_y * 2, **kwargs)

    # calculating the stdandarddeviation of x from  the squareroot of the variance
    # np.sqrt(cov[0, 0])
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = mean[0]
    
    # calculating the stdandarddeviation of y from  the squareroot of the variance
    # np.sqrt(cov[1, 1])
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = mean[1]
    
    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)
        
    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)
        
    return pearson


def analyse_unscented_transform():
    """
    Analyse the unscented transform by transforming a Gaussian distribution through a non-linear function.
    Compares the transformed distribution with the distribution obtained by sampling and transforming the samples.
    """
    mean = np.array([15.0, 15.0])
    covariance = np.array([[1.0, 0.0], [0.0, 1.0]])
    gaussian = Gaussian(mean, covariance)

    def non_linear_function(x):
        return np.array([0.25*x[0]**2 -20.0, 0.01*(x[0] * x[1]**2) - 10.0])

    transformed_gaussian = unscented_transform(gaussian, non_linear_function)
    print(transformed_gaussian)

    num_samples = 10_000
    sampled_gaussian = propagate_samples(gaussian, non_linear_function, num_samples)
    print(sampled_gaussian)
    
    plt.figure()
    sigma_points = gaussian.compute_sigma_points()
    transformed_sigma_points = [non_linear_function(sp) for sp in sigma_points]
    plot_ellipse(gaussian.mean(), gaussian.covariance(), plt.gca(), n_std=1, facecolor='none', edgecolor='blue')
    plt.scatter([x[0] for x in sigma_points], [x[1] for x in sigma_points], color='blue')

    # Draw an oval for the sampled Gaussian
    ax = plt.gca()
    transformed_samples = [non_linear_function(gaussian.sample()) for _ in range(num_samples)]
    plt.scatter([x[0] for x in transformed_samples], [x[1] for x in transformed_samples], color='orange', marker='+', alpha=0.1)
    plot_ellipse(sampled_gaussian.mean(), sampled_gaussian.covariance(), ax, n_std=1, facecolor='none', edgecolor='red')
    plot_ellipse(transformed_gaussian.mean(), transformed_gaussian.covariance(), ax, n_std=1, facecolor='none', edgecolor='green')
    plt.scatter([x[0] for x in transformed_sigma_points], [x[1] for x in transformed_sigma_points], color='green')
    plt.gca().axis('equal')
    plt.legend(['Original Gaussian', 'Original Sigma Points', 'Propogated Samples', 'Gaussian from Propogated Samples', 'Unscented Transformed Gaussian', 'Unscented Transformed Sigma Points'])
    plt.tight_layout()
    plt.show()


def test_unscented_transform_linear_func():
    """
    Test the unscented transform with a linear function.
    """
    mean = np.array([1.0, 2.0])
    covariance = np.array([[1.0, 0.0], [0.0, 1.0]])
    gaussian = Gaussian(mean, covariance)

    def linear_function(x):
        return np.array([2.0*x[0], 3.0*x[1]])
    
    jacobian = np.array([[2.0, 0.0], [0.0, 3.0]])

    transformed_gaussian = unscented_transform(gaussian, linear_function)
    expected_mean = linear_function(mean)
    expected_covariance = jacobian @ covariance @ jacobian.T
    assert np.allclose(transformed_gaussian.mean(), expected_mean)
    assert np.allclose(transformed_gaussian.covariance(), expected_covariance)


if __name__ == '__main__':
    test_unscented_transform_linear_func()
    analyse_unscented_transform()
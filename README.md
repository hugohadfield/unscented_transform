# Simple Unscented Transform Implementation

This repository provides a simple implementation of the Unscented Transform using Gaussian distributions. The main goal is to demonstrate how to propagate distributions through non-linear functions using the Unscented Transform.

## Table of Contents

- [Dependencies](#dependencies)
- [Usage](#usage)
- [Classes](#classes)
  - [Distribution](#distribution)
  - [Gaussian](#gaussian)
- [Functions](#functions)
  - [unscented_transform](#unscented_transform)
  - [propagate_samples](#propagate_samples)
  - [analyse_unscented_transform](#analyse_unscented_transform)
  - [test_unscented_transform_linear_func](#test_unscented_transform_linear_func)
- [Examples](#examples)
- [License](#license)

## Dependencies

To use this code, you need to have Python installed on your machine along with the necessary libraries. You can install the required packages using pip:

```bash
pip install numpy matplotlib
```

## Usage

You can use this library to create Gaussian distributions, compute sigma points, and propagate these points through non-linear functions. Below are the main classes and functions provided.

## Classes

### Distribution

`Distribution` is an abstract base class that defines the interface for different types of probability distributions.

#### Methods

- `dimension()`: Returns the dimension of the distribution.
- `mean()`: Returns the mean of the distribution.
- `covariance()`: Returns the covariance matrix of the distribution.
- `compute_sigma_points()`: Computes the sigma points for the distribution.
- `sample()`: Samples from the distribution.
- `from_samples(samples)`: Creates a distribution from the given samples.
- `__repr__()`: Returns a string representation of the distribution.

### Gaussian

`Gaussian` is a concrete implementation of the `Distribution` class that represents a Gaussian distribution.

#### Methods

- `__init__(mean, covariance, rng=None)`: Initializes the Gaussian distribution with the given mean and covariance.
- `mean()`: Returns the mean of the Gaussian distribution.
- `covariance()`: Returns the covariance of the Gaussian distribution.
- `dimension()`: Returns the dimension of the Gaussian distribution.
- `sqrt_covariance()`: Returns the square root of the covariance matrix.
- `compute_sigma_points()`: Computes the sigma points for the Gaussian distribution.
- `compute_weights()`: Computes the weights for the sigma points.
- `from_sigma_points(sigma_points, weights)`: Creates a Gaussian distribution from the given sigma points.
- `sample()`: Samples from the Gaussian distribution.
- `from_samples(samples)`: Creates a Gaussian distribution from the given samples.
- `__repr__()`: Returns a string representation of the Gaussian distribution.

## Functions

### unscented_transform

```python
def unscented_transform(distribution: Distribution, non_linear_function: Callable) -> Distribution:
```

Propagates the distribution through the non-linear function using the Unscented Transform.

### propagate_samples

```python
def propagate_samples(distribution: Distribution, non_linear_function: Callable, num_samples: int) -> Distribution:
```

Samples a distribution and propagates the samples through the non-linear function, re-estimating the distribution from the transformed samples.

### analyse_unscented_transform

```python
def analyse_unscented_transform():
```

Analyzes the Unscented Transform by transforming a Gaussian distribution through a non-linear function and compares the transformed distribution with the distribution obtained by sampling and transforming the samples.

### test_unscented_transform_linear_func

```python
def test_unscented_transform_linear_func():
```

Tests the Unscented Transform with a linear function to validate the implementation.

## Examples

### Analyzing the Unscented Transform

To analyze the Unscented Transform, you can run the following script:

```python
if __name__ == '__main__':
    analyse_unscented_transform()
```

This will create a Gaussian distribution, propagate it through a non-linear function, and compare the results with the distribution obtained by sampling and transforming the samples. A plot will be displayed showing the results.

### Testing the Unscented Transform

To test the Unscented Transform with a linear function, you can run the following script:

```python
if __name__ == '__main__':
    test_unscented_transform_linear_func()
```

This will validate the Unscented Transform implementation against a known linear function and its Jacobian.

## License

This project is licensed under the MIT License.
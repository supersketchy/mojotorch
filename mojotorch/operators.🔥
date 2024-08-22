import math
from builtin.math import abs


# Task 0.1 in mojo
fn mul(x: Float64, y: Float64) -> Float64:
    """Multiplies two floats together.

    Args:
        x: The first Float64 to multiply.
        y: The second Float64 to multiply.

    Returns:
        Float64 result of x * y.
    """
    return x * y


fn id(x: Float64) -> Float64:
    """Identifies the value of x.

    Args:
        x: The Float64 to identify.

    Returns:
        Float64 of x.
    """
    return x


fn eq(x: Float64, y: Float64) -> Float64:
    """Determines which value is greater between x and y. Prefer using is_close, as the floats may be inaccurate.

    Args:
        x: The first Float64 to compare.
        y: The second Float64 to compare.

    Returns:
        1.0 if x and y are equal, and 0.0 if they are not.
    """
    if x == y:
        return 1.0
    return 0.0


fn neg(x: Float64) -> Float64:
    """Returns the negation of x.

    Args:
        x: The Float64 to negate.

    Returns:
        The negation of x.
    """
    return -1 * x


fn add(x: Float64, y: Float64) -> Float64:
    """Adds two floats together.

    Args:
        x: The first Float64 to add.
        y: The second Float64 to add.

    Returns:
        Float64 result of x + y.
    """
    return x + y


fn max(x: Float64, y: Float64) -> Float64:
    """Returns the greater number between x and y.

    Args:
        x: The first Float64 to compare.
        y: The second Float64 to compare.

    Returns:
        Float64 of the greater number. If x = y, then y is returned.
    """
    if x > y:
        return x
    return y


fn lt(x: Float64, y: Float64) -> Float64:
    """Returns 1.0 if x is less than y, else return 0.0.

    Args:
        x: The first Float64 to compare.
        y: The second Float64 to compare.

    Returns:
        A Float64 representing if x is less than y.
    """
    if x < y:
        return 1.0
    return 0.0


var EPS: Float64 = 10**-6


fn log(x: Float64) -> Float64:
    """Returns the natural logarithm function for a given input x.

    Args:
        x: The Float64 to take the natural logarithm of.

    Returns:
        A Float64 representing the natural log of x.
    """
    return math.log(x + EPS)


fn exp(x: Float64) -> Float64:
    """Returns the exponential function for a given input x.

    Args:
        x: The Float64 to take the exponential function of.

    Returns:
        A Float64 representing the exponential function of x.
    """
    return math.exp(x)


fn sigmoid(x: Float64) -> Float64:
    """Returns the sigmoid function of x. Refer to the way that the sigmoid function is calculated here: https://minitorch.github.io/module0/module0/.

    Args:
        x: The Float64 to have the sigmoid function run on.

    Returns:
        A Float64 representing the result of calculating the sigmoid function.
    """
    if x >= 0:
        return 1.0 / (1 + exp(-x))
    else:
        return exp(x) / (1 + exp(x))


fn relu(x: Float64) -> Float64:
    """Returns the relu function of x.

    Args:
        x: The Float64 to have the relu function run on.

    Returns:
        A Float64 representing the result of calculating the relu function.
    """
    if x >= 0:
        return x
    else:
        return 0


fn inv(x: Float64) -> Float64:
    """Returns the inverse of x.

    Args:
        x: The Float64 to invert.

    Returns:
        A Float64 representing the inverse of x.
    """
    return 1 / x


fn inv_back(x: Float64, d: Float64) -> Float64:
    """If f(x) = 1/x, find d * f'(x).

    Args:
        x: The Float64 to find the value.
        d: The Float64 of the prior derivatives, used for backpropagation.

    Returns:
        A Float64 representing the value calculated as above.
    """
    return d * (-1 / x**2)


fn relu_back(x: Float64, d: Float64) -> Float64:
    """If f(x) is the relu function, find d * f'(x). For relu, if x > 0, return d, else return 0.

    Args:
        x: The Float64 to find the value.
        d: The Float64 of the prior derivatives, used for backpropagation.

    Returns:
        A Float64 representing the value calculated as above.
    """
    if x > 0:
        return d
    return 0


fn log_back(x: Float64, d: Float64) -> Float64:
    """If f(x) is the log function, find d * f'(x). For log, it is 1/x.

    Args:
        x: The Float64 to find the value.
        d: The Float64 of the prior derivatives, used for backpropagation.

    Returns:
        A Float64 representing the value calculated as above.
    """
    return (1 / x) * d


fn is_close(x: Float64, y: Float64) -> Float64:
    """Check if |x - y| < 1e-2.

    Args:
        x: The first Float64 to find the value.
        y: The second Float64 to compare.

    Returns:
        A Float64 representing the value calculated as above.
    """
    if abs(x - y) < 1e-2:
        return 1.0
    return 0.0

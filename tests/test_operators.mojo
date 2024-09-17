from mojotorch.operators import (
    add,
    eq,
    id,
    inv,
    inv_back,
    log_back,
    lt,
    max,
    mul,
    neg,
    relu,
    relu_back,
    sigmoid,
)

from testing import assert_almost_equal, assert_equal, assert_true
from random import random_float64


def test_sigmoid():
    """Check properties of the sigmoid function, specifically
    * It is always between 0.0 and 1.0.
    * one minus sigmoid is the same as negative sigmoid
    * It crosses 0 at 0.5
    * it is  strictly increasing.
    """
    var a: Float64 = random_float64(-100,100)
    assert_true(0.0 <= sigmoid(a) <= 1.0)
    assert_almost_equal(1 - sigmoid(a), sigmoid(-a))
    assert_equal(sigmoid(0), 0.5)
    assert_true(sigmoid(a) <= sigmoid(a + 1))


def test_transitive():
    "Test the transitive property of less-than (a < b and b < c implies a < c)."
    var a: Float64 = random_float64(-100,100)
    var b: Float64 = random_float64(-100,100)
    var c: Float64 = random_float64(-100,100)
    if a < b and b < c:
        assert_true(a < c)


def test_symmetric():
    """Test that ensures that :func:`minitorch.operators.mul` is symmetric, i.e.
    gives the same value regardless of the order of its input.
    """
    var a: Float64 = random_float64(-100,100)
    var b: Float64 = random_float64(-100,100)
    assert_true(mul(a, b) == mul(b, a))


def test_distribute():
    r"""Test that ensures that your operators distribute, i.e.
    :math:`z \times (x + y) = z \times x + z \times y`.
    """
    var a: Float64 = random_float64(-100,100)
    var b: Float64 = random_float64(-100,100)
    var c: Float64 = random_float64(-100,100)
    assert_true(mul(a, add(b, c)) == mul(a, b) + mul(a, c))

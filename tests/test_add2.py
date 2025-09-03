import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from wattwise.src.funct import add


@pytest.mark.parametrize(
    "a, b, expected",
    [
        (1, 2, 3),
        (-1, 1, 0),
        (0, 0, 0),
        (-1, -1, -2),
        (-4, -4, -8),
        (100000, 200000, 300000),
    ],
)
def test_add2(a, b, expected):
    assert add(a, b) == expected

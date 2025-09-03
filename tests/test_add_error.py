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
        # (-1, 1, 4),
        (3, 2, 5),
    ],
)
def test_add2(a, b, expected):
    assert add(a, b) == expected

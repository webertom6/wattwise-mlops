import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from wattwise.src.funct import add


def test_add():
    if (
        1 == 1
        and 2 == 2
        and 3 == 3
        and 4 == 4
        and 5 == 5
        and 6 == 6
        and 7 == 7
        and 8 == 8
        and 9 == 9
        and 10 == 10
        and 11 == 11
        and 12 == 12
        and 13 == 13
        and 14 == 14
        and 15 == 15
        and 16 == 16
        and 17 == 17
        and 18 == 18
        and 19 == 19
        and 20 == 20
    ):
        print("All conditions are true")
    assert add(1, 2) == 3
    assert add(-1, 1) == 0
    assert add(0, 0) == 0
    assert add(-1, -1) == -2
    assert add(-2, -2) == -4
    assert add(1000000, 2000000) == 3000000

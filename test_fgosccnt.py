import pytest  # type: ignore

import fgosccnt


params_check_page_mismatch = [
    # drop: 20
    (True, 21, 20, 1, 1, 3),
    # drop: 21
    (True, 22, 21, 1, 2, 4),
    (True, 15, 21, 2, 2, 4),
    # drop: 31
    (True, 22, 31, 1, 2, 5),
    (True, 18, 31, 2, 2, 5),
    # drop: 41
    (True, 21, 41, 1, 2, 6),
    (True, 21, 41, 2, 2, 6),
    # drop: 42
    (True, 21, 42, 1, 3, 7),
    (True, 21, 42, 2, 3, 7),
    (True, 15, 42, 3, 3, 7),
    # drop: 52
    (True, 21, 52, 1, 3, 8),
    (True, 21, 52, 2, 3, 8),
    (True, 18, 52, 3, 3, 8),
    # drop: 62
    (True, 21, 62, 1, 3, 9),
    (True, 21, 62, 2, 3, 9),
    (True, 21, 62, 3, 3, 9),
    # drop: 63
    (True, 21, 63, 1, 4, 10),
    (True, 21, 63, 2, 4, 10),
    (True, 21, 63, 3, 4, 10),
    (True, 15, 63, 4, 4, 10),
    # drop: 73
    (True, 21, 73, 1, 4, 11),
    (True, 21, 73, 2, 4, 11),
    (True, 21, 73, 3, 4, 11),
    (True, 18, 73, 4, 4, 11),
    # drop: 83
    (True, 21, 83, 1, 4, 12),
    (True, 21, 83, 2, 4, 12),
    (True, 21, 83, 3, 4, 12),
    (True, 21, 83, 3, 4, 12),
    # drop: 84
    (True, 21, 84, 1, 5, 13),
    (True, 21, 84, 2, 5, 13),
    (True, 21, 84, 3, 5, 13),
    (True, 21, 84, 4, 5, 13),
    (True, 15, 84, 5, 5, 13),
    # drop: 94
    (True, 21, 94, 1, 5, 14),
    (True, 21, 94, 2, 5, 14),
    (True, 21, 94, 3, 5, 14),
    (True, 21, 94, 4, 5, 14),
    (True, 18, 94, 5, 5, 14),
    # drop: 104
    (True, 21, 104, 1, 5, 15),
    (True, 21, 104, 2, 5, 15),
    (True, 21, 104, 3, 5, 15),
    (True, 21, 104, 4, 5, 15),
    (True, 21, 104, 5, 5, 15),
    # drop: 105
    (True, 21, 105, 1, 6, 16),
    (True, 21, 105, 2, 6, 16),
    (True, 21, 105, 3, 6, 16),
    (True, 21, 105, 4, 6, 16),
    (True, 21, 105, 5, 6, 16),
    (True, 15, 105, 6, 6, 16),
]


@pytest.mark.parametrize("expected, items_total, chestnum, pagenum, pages, lines", params_check_page_mismatch)
def test_check_page_mismatch(expected, items_total, chestnum, pagenum, pages, lines):
    assert expected == fgosccnt.check_page_mismatch(items_total, chestnum, pagenum, pages, lines)

import pytest

from app.stats import bootstrap_ci, compare, mcnemar_exact


def test_mcnemar_exact_values():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(10, 0) == pytest.approx(2 / 2**10)
    assert mcnemar_exact(3, 7) == mcnemar_exact(7, 3)
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(8, 2) == pytest.approx(2 * (1 + 10 + 45) / 2**10)  # 0.109


def test_bootstrap_interval_is_reproducible_and_brackets_the_gap():
    a = [True] * 30 + [False] * 10
    b = [True] * 20 + [False] * 20
    lo, hi = bootstrap_ci(a, b)
    assert (lo, hi) == bootstrap_ci(a, b)
    assert lo < 0.25 < hi and lo > 0
    assert bootstrap_ci([True] * 5, [True] * 5) == (0.0, 0.0)
    assert bootstrap_ci([], []) == (0.0, 0.0)


def test_compare_pairs_by_question_id():
    a = [{"question_id": 1, "correct": True}, {"question_id": 2, "correct": True}, {"question_id": 3, "correct": False}]
    b = [{"question_id": 3, "correct": True}, {"question_id": 1, "correct": False}, {"question_id": 9, "correct": True}]
    t = compare(a, b)
    assert t["n"] == 2  # questions 1 and 3; 2 and 9 ran in only one mode
    assert (t["only_a"], t["only_b"]) == (1, 1)
    assert t["ex_diff"] == 0.0 and t["p_mcnemar"] == 1.0

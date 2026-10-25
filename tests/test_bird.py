import json

import pytest

from app.bird import Question, db_path, load_questions, safe_join, split


def _ids(qs):
    return {q.question_id for q in qs}


def test_load_questions_filters_by_db(bird_dir):
    qs = load_questions("shop", bird_dir)
    assert len(qs) == 80
    assert all(q.db_id == "shop" for q in qs)
    assert qs[1] == Question(1, "shop", "question 1", "evidence 1", "SELECT 1", "simple")
    assert qs[0].evidence == ""


def test_db_path(bird_dir):
    assert db_path("shop", bird_dir) == bird_dir / "dev_databases" / "shop" / "shop.sqlite"


def test_split_deterministic_disjoint_and_complete(bird_dir):
    qs = load_questions("shop", bird_dir)
    seed_a, eval_a = split(qs)
    seed_b, eval_b = split(list(reversed(qs)))
    assert _ids(seed_a) == _ids(seed_b)
    assert _ids(eval_a) == _ids(eval_b)
    assert not _ids(seed_a) & _ids(eval_a)
    assert _ids(seed_a) | _ids(eval_a) == _ids(qs)
    assert 10 <= len(seed_a) <= 40  # roughly 30% of 80


def test_split_seed_changes_assignment(bird_dir):
    qs = load_questions("shop", bird_dir)
    assert _ids(split(qs, seed=1)[0]) != _ids(split(qs, seed=2)[0])


def test_split_is_recorded(bird_dir, tmp_path):
    qs = load_questions("shop", bird_dir)
    out = tmp_path / "out" / "split-shop.json"
    seed_qs, eval_qs = split(qs, record_to=out)
    rec = json.loads(out.read_text())
    assert rec["seed_ids"] == sorted(_ids(seed_qs))
    assert rec["eval_ids"] == sorted(_ids(eval_qs))


@pytest.mark.parametrize("name", ["../x", "a/../../x", "/etc/x", "C:/x", r"C:\x", r"..\x", "a/b:c"])
def test_safe_join_rejects_unsafe(tmp_path, name):
    with pytest.raises(ValueError):
        safe_join(tmp_path, name)


def test_safe_join_accepts_normal(tmp_path):
    assert safe_join(tmp_path, "financial/financial.sqlite") == tmp_path.resolve() / "financial" / "financial.sqlite"

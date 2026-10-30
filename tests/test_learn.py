import json

import pytest

from app import bird, evaluate, learn
from app.llm import Draft, ProposedRule
from tests.test_evaluate import GOLD, WRONG, world  # noqa: F401  (world is a fixture)

RULE_BODY = "total revenue is the sum of sale.amount"


class Model:
    """Writes the revenue query right only when some rule shown says how revenue is computed."""

    def __init__(self):
        self.seen = []

    def generate(self, question, schema_text, items):
        self.seen.append(question)
        if "total revenue" in question:
            sql = GOLD["total revenue"] if any("sum of sale.amount" in i.body for i in items) else WRONG
        else:
            sql = GOLD["number of sales"]
        return Draft(sql=sql, items_used=[i.id for i in items], assumptions="")

    def repair(self, question, schema_text, items, sql, error):
        return self.generate(question, schema_text, items)


class User:
    def __init__(self, correction="Revenue means adding up the amount of every sale."):
        self.correction, self.corrected, self.distilled = correction, [], 0

    def correct(self, question, schema_text, sql, rows, gold_sql, gold_rows):
        self.corrected.append((question, gold_sql))
        return self.correction

    def distill(self, question, schema_text, sql, correction, items):
        self.distilled += 1
        return [ProposedRule(kind="metric", title="Revenue", terms=["revenue"], body=RULE_BODY,
                             sql="SUM(sale.amount)")]


def _ids(qs):
    return [q.question_id for q in qs]


def test_stream_and_test_halves_are_disjoint_and_stable(world):
    qs = evaluate.eval_questions("shop")
    stream, test = learn.split_stream(qs)
    assert stream and test and not set(_ids(stream)) & set(_ids(test))
    assert sorted(_ids(stream) + _ids(test)) == sorted(_ids(qs))
    assert learn.split_stream(list(reversed(qs)))[0] == list(reversed(stream))


@pytest.mark.parametrize("text, leaks", [
    ("Revenue is the amount of each sale, added up.", False),
    ("Use SELECT SUM(amount) FROM sale", True),
    ("join loan as l on the account", True),
    ("you should group by district", True),
    ("Select the region column instead.", False),
])
def test_reveals_sql(text, leaks):
    assert learn.reveals_sql(text, "SELECT SUM(amount) FROM sale") is leaks


def test_corrections_on_the_stream_lift_the_held_out_curve(world, tmp_path, capsys):
    out = tmp_path / "learn.json"
    user = User()
    assert learn.main(["--db", "shop", "--start", "empty", "--checkpoint-every", "3", "--yes", "--json", str(out)],
                      llm=Model(), learner=user) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    cps = data["checkpoints"]
    assert cps[0]["after"] == 0 and cps[-1]["after"] == len(data["run"]["stream_ids"])
    assert cps[-1]["ex"] == 1.0 > cps[0]["ex"]
    assert data["summary"]["rules_added"] == 1  # the second identical rule is a duplicate title
    assert data["summary"]["held_out"]["only_b"] == 0 and data["summary"]["held_out"]["only_a"] > 0
    assert data["baseline_schema"] == cps[0]["ex"]
    # only stream questions are ever corrected; the test half never feeds the store
    corrected = {q for q, _ in user.corrected}
    test_questions = {q.question for q in bird.load_questions("shop") if q.question_id in data["run"]["test_ids"]}
    assert corrected and not corrected & test_questions
    assert [e["question_id"] for e in data["events"]] == data["run"]["stream_ids"]
    shown = capsys.readouterr().out
    assert "Held-out change" in shown and "Schema only" in shown


def test_drafts_confirmed_by_good_answers_are_promoted(world, tmp_path):
    out = tmp_path / "learn.json"
    assert learn.main(["--db", "shop", "--start", "empty", "--yes", "--json", str(out)], llm=Model(), learner=User()) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    confirmations = sum(e["correct"] and any(i.startswith("fb") for i in e["items_used"]) for e in data["events"])
    assert confirmations >= 2
    assert data["summary"]["promoted"] == 1
    assert data["checkpoints"][-1]["items"] == {"draft": 0, "reviewed": 1, "certified": 0}


def test_corrections_that_reveal_sql_are_not_learned(world, tmp_path):
    out = tmp_path / "learn.json"
    user = User("Just run SELECT SUM(amount) FROM sale")
    assert learn.main(["--db", "shop", "--start", "empty", "--yes", "--json", str(out)], llm=Model(), learner=user) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert user.distilled == 0 and data["summary"]["rules_added"] == 0
    assert data["summary"]["revealed_sql"] == data["summary"]["corrections"] > 0
    assert data["checkpoints"][-1]["ex"] == data["checkpoints"][0]["ex"]


def test_seeded_start_already_knows_revenue(world, tmp_path):
    out = tmp_path / "learn.json"
    user = User()
    assert learn.main(["--db", "shop", "--yes", "--json", str(out)], llm=Model(), learner=user) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["checkpoints"][0]["ex"] == 1.0 and user.corrected == []
    assert data["run"]["store"].startswith("seeded")


def test_options_are_checked(world, capsys):
    assert learn.main(["--db", "shop", "--stream-fraction", "0.999", "--yes"], llm=Model(), learner=User()) == 1
    assert "empty" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        learn.main(["--db", "shop", "--stream-fraction", "1.5", "--yes"])

import json
import pickle
from copy import deepcopy

import pytest
from pydantic import TypeAdapter

import dspy
from dspy.adapters.utils import parse_value
from typesafe_dspy import Score, plan_signature


def test_score_is_a_validated_number_with_a_visible_rubric():
    relevance = Score["Unrelated", "Partial", "Direct"]
    value = relevance(1.6)
    assert isinstance(value, float)
    assert value > 1.5
    assert value * 10 == 16
    assert type(value + 1) is float
    assert json.dumps(value) == "1.6"
    assert type(pickle.loads(pickle.dumps(value))) is relevance
    assert type(deepcopy(value)) is relevance
    adapter = TypeAdapter(relevance)
    assert isinstance(adapter.validate_json("1.6"), relevance)
    assert adapter.dump_json(value) == b"1.6"
    assert isinstance(parse_value("1.6", relevance), relevance)
    schema = adapter.json_schema()
    assert schema["type"] == "number"
    assert (schema["minimum"], schema["maximum"]) == (0, 2)
    assert "2: Direct" in schema["description"]
    for endpoint in (0, 2):
        assert adapter.validate_python(endpoint) == endpoint
    for invalid in (-0.01, 2.01, float("nan"), float("inf"), True, "1.6"):
        with pytest.raises(ValueError):
            relevance(invalid)
        with pytest.raises(ValueError):
            adapter.validate_python(invalid)


def test_signature_transformations_preserve_each_score_rubric():
    class Review(dspy.Signature):
        text: str = dspy.InputField()
        relevance: Score["Unrelated", "Partial", "Direct"] = dspy.OutputField()  # noqa: F821 - runtime rubric strings
        severity: Score["Harmless", "Minor", "Serious", "Critical"] = dspy.OutputField()  # noqa: F821

    updated = Review.with_instructions("Evaluate the evidence.")
    restored = Review.load_state(json.loads(json.dumps(updated.dump_state())))
    for signature in (Review, updated, restored):
        plans = plan_signature(signature).prompt_plans
        assert plans[0].score_levels == {0: "Unrelated", 1: "Partial", 2: "Direct"}
        assert plans[1].score_levels == {0: "Harmless", 1: "Minor", 2: "Serious", 3: "Critical"}
    with dspy.context(lm=dspy.utils.DummyLM([{"relevance": "1.6", "severity": "2.4"}])):
        result = dspy.Predict(Review)(text="Evidence")
    assert isinstance(result.relevance, Review.output_fields["relevance"].annotation)
    assert result.severity == 2.4


def test_bare_score_is_not_an_unspecified_rubric():
    with pytest.raises(TypeError, match="at least two"):
        TypeAdapter(Score)


def test_explicit_anchors_preserve_range_and_serialization():
    severity = Score[(-1.5, "Low"), (3, "Medium"), (10, "Critical")]
    value = severity(6.3)
    adapter = TypeAdapter(severity)
    assert adapter.dump_json(value) == b"6.3"
    assert isinstance(adapter.validate_json("6.3"), severity)
    for restored in (deepcopy(value), pickle.loads(pickle.dumps(value))):
        assert type(restored) is severity
        assert restored == 6.3
    schema = adapter.json_schema()
    assert (schema["minimum"], schema["maximum"]) == (-1.5, 10)
    assert "-1.5: Low; 3: Medium; 10: Critical" in schema["description"]
    for endpoint in (-1.5, 10):
        assert adapter.validate_python(endpoint) == endpoint
    for invalid in (-1.51, 10.01):
        with pytest.raises(ValueError):
            adapter.validate_python(invalid)


@pytest.mark.parametrize(
    "levels",
    [
        ((1, "Low"), (1, "High")),
        ((5, "Low"), (1, "High")),
        ((True, "Low"), (5, "High")),
        ((float("nan"), "Low"), (5, "High")),
        ((1, "Low"), (float("inf"), "High")),
        ((1, "Low"), "High"),
        ((1, "Low"), (5, "Low")),
        ((1, "Low"), (5, "")),
    ],
)
def test_invalid_explicit_anchors(levels):
    with pytest.raises(ValueError):
        Score[levels]


def test_multiple_explicit_rubrics_survive_signature_transformations():
    class Review(dspy.Signature):
        text: str = dspy.InputField()
        severity: Score[(1, "Low"), (3, "Medium"), (10, "Critical")] = dspy.OutputField()  # noqa: F821
        sentiment: Score[(-1, "Negative"), (1, "Positive")] = dspy.OutputField()  # noqa: F821

    updated = Review.with_instructions("Use evidence.")
    restored = Review.load_state(json.loads(json.dumps(updated.dump_state())))
    for signature in (Review, updated, restored):
        plans = plan_signature(signature).prompt_plans
        assert plans[0].score_levels == {1: "Low", 3: "Medium", 10: "Critical"}
        assert plans[1].score_levels == {-1: "Negative", 1: "Positive"}

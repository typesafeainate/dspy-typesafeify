"""Rubric-bearing numeric output types for DSPy signatures."""

import itertools
import math
from functools import lru_cache

from pydantic_core import core_schema


class Score(float):
    """A finite rubric score: Score["Low", "High"] or Score[(1, "Low"), (5, "High")].

    Arithmetic returns ordinary floats. The rubric belongs to the type, not to
    individual values; native probabilities remain in typesafe_results().
    """

    levels: tuple[tuple[int | float, str], ...] = ()

    def __new__(cls, value):
        if not cls.levels:
            raise TypeError("Use Score with at least two ordered descriptions")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Score requires a number, not a boolean or string")
        value = float(value)
        low, high = cls.levels[0][0], cls.levels[-1][0]
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"Score must be finite and between {low} and {high}")
        return super().__new__(cls, value)

    def __class_getitem__(cls, levels):
        if cls is not Score:
            raise TypeError("Specialize Score, not an existing rubric")
        if not isinstance(levels, tuple) or len(levels) < 2:
            raise ValueError("Score requires at least two distinct nonempty descriptions")
        if all(isinstance(level, str) for level in levels):
            levels = tuple(enumerate(levels))
        if any(
            not isinstance(level, tuple)
            or len(level) != 2
            or isinstance(level[0], bool)
            or not isinstance(level[0], (int, float))
            or not math.isfinite(level[0])
            or not isinstance(level[1], str)
            or not level[1].strip()
            for level in levels
        ):
            raise ValueError("Use all descriptions or all (finite numeric anchor, description) pairs")
        if any(a[0] >= b[0] for a, b in itertools.pairwise(levels)):
            raise ValueError("Score anchors must be strictly increasing")
        if len({description for _, description in levels}) != len(levels):
            raise ValueError("Score descriptions must be distinct")
        return _score_type(levels)

    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        if not cls.levels:
            raise TypeError("Use Score with at least two ordered descriptions")
        return core_schema.no_info_before_validator_function(
            cls,
            core_schema.is_instance_schema(cls),
            json_schema_input_schema=core_schema.float_schema(
                ge=cls.levels[0][0], le=cls.levels[-1][0], allow_inf_nan=False
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(float),
        )

    @classmethod
    def __get_pydantic_json_schema__(cls, schema, handler):
        return {
            "type": "number",
            "minimum": cls.levels[0][0],
            "maximum": cls.levels[-1][0],
            "description": "Ordered rubric: " + "; ".join(f"{anchor}: {label}" for anchor, label in cls.levels),
        }

    def __reduce__(self):
        return _restore_score, (self.levels, float(self))


@lru_cache(maxsize=128)
def _score_type(levels):
    return type("Score[" + ", ".join(repr(level) for level in levels) + "]", (Score,), {"levels": levels})


def _restore_score(levels, value):
    return Score[levels](value)

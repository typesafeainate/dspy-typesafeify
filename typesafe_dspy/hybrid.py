from __future__ import annotations

import json
import logging
import textwrap
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Protocol, Sequence, get_args, get_origin

import anyio.to_thread
from pydantic_core import PydanticUndefined

import dspy
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.clients.base_lm import BaseLM
from dspy.dsp.utils.settings import settings
from dspy.predict.predict import Predict, _get_type_name, _is_value_compatible_with_type, serialize_object
from dspy.primitives.prediction import Prediction
from dspy.signatures.signature import Signature, ensure_signature, make_signature
from dspy.utils.constants import IS_TYPE_UNDEFINED
from typesafe_dspy.score import Score

logger = logging.getLogger(__name__)

JsonValue = str | list[Any] | dict[str, Any]
TypesafeKind = Literal["noul", "choice", "score", "disable"]
DocumentBuilder = Callable[[type[Signature], dict[str, Any]], JsonValue]
ScoreFieldSpec = Mapping[str, Sequence[float] | Mapping[float, JsonValue]]

_TYPESAFE_ENABLED_ATTR = "__typesafe_dspy_enabled__"
_TYPESAFE_CONFIG_KEY = "typesafe_dspy_config"
_TYPESAFE_AUTO_KEY = "typesafe_dspy_auto"
_ORIGINAL_PREDICT_FORWARD = Predict.forward
_ORIGINAL_PREDICT_AFORWARD = Predict.aforward
_PREDICT_PATCHED = False


class PromptFactory(Protocol):
    """Build question objects for the active Typesafe runtime."""

    def create_noul(self, *, instructions: JsonValue) -> Any:
        """Create a Typesafe Noul question."""

    def create_choice(
        self,
        *,
        instructions: JsonValue,
        options: Mapping[str, JsonValue],
    ) -> Any:
        """Create a Typesafe Choice question."""

    def create_score(
        self,
        *,
        instructions: JsonValue,
        levels: Mapping[float, JsonValue],
    ) -> Any:
        """Create a Typesafe Score question."""


class ImportedTypesafePromptFactory:
    """Lazily import prompt classes so the integration stays optional."""

    def __init__(self) -> None:
        self._prompt_types: tuple[type[Any], type[Any], type[Any]] | None = None

    def _load(self) -> tuple[type[Any], type[Any], type[Any]]:
        if self._prompt_types is None:
            try:
                from typesafe_sdk import Choice, Noul, Score
            except ImportError as exc:
                raise ImportError(
                    "typesafe-sdk 0.6.0 or newer is required to build real Typesafe questions. "
                    "Install the `typesafe` extra."
                ) from exc
            self._prompt_types = (Noul, Choice, Score)
        return self._prompt_types

    def create_noul(self, *, instructions: JsonValue) -> Any:
        noul_prompt, _, _ = self._load()
        return noul_prompt(instructions=instructions)

    def create_choice(
        self,
        *,
        instructions: JsonValue,
        options: Mapping[str, JsonValue],
    ) -> Any:
        _, choice_prompt, _ = self._load()
        return choice_prompt(instructions=instructions, criteria=options)

    def create_score(
        self,
        *,
        instructions: JsonValue,
        levels: Mapping[float, JsonValue],
    ) -> Any:
        _, _, score_prompt = self._load()
        criteria = [description for _, description in sorted(levels.items())]
        return score_prompt(instructions=instructions, criteria=criteria)


@dataclass(frozen=True)
class TypesafeFieldConfig:
    """Optional per-field overrides for the hybrid plan."""

    kind: TypesafeKind | None = None
    instructions: JsonValue | None = None
    choice_options: Mapping[Any, JsonValue] | None = None
    score_levels: Mapping[float, JsonValue] | None = None


@dataclass(frozen=True)
class TypesafeConfig:
    """Runtime configuration for a Typesafe-backed predictor."""

    client: Any
    model: str
    field_configs: Mapping[str, TypesafeFieldConfig] = field(default_factory=dict)
    prompt_factory: PromptFactory | None = None
    document_builder: DocumentBuilder | None = None
    noul_true_threshold: float = 0.5


@dataclass(frozen=True)
class ChoiceOptionPlan:
    """Map an internal prompt option key back to the literal field value."""

    prompt_key: str
    value: Any
    description: JsonValue


@dataclass(frozen=True)
class TypesafePromptPlan:
    """Describe how one DSPy output field should be evaluated by Typesafe."""

    field_name: str
    kind: Literal["noul", "choice", "score"]
    instructions: JsonValue
    choice_options: tuple[ChoiceOptionPlan, ...] = ()
    score_levels: Mapping[float, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True)
class TypesafeFieldResult:
    """Carry the raw Typesafe decision data alongside the resolved field value."""

    kind: Literal["noul", "choice", "score"]
    value: Any
    probability: float | None = None
    probabilities: Mapping[Any, float] | None = None
    confidence: float | None = None
    expectation: float | None = None


@dataclass(frozen=True)
class TypesafeTiming:
    """Capture the split runtime for a hybrid Typesafe call."""

    typesafe_seconds: float
    dspy_seconds: float
    total_seconds: float


@dataclass(frozen=True)
class PredictionFieldDelta:
    """Describe how one output field differs between two predictions."""

    field_name: str
    baseline_value: Any
    candidate_value: Any
    baseline_display: str
    candidate_display: str
    delta: str
    changed: bool


@dataclass(frozen=True)
class PredictionComparison:
    """Carry a field-by-field comparison for two predictions."""

    signature: type[Signature]
    baseline_label: str
    candidate_label: str
    field_deltas: tuple[PredictionFieldDelta, ...]
    baseline_timings: TypesafeTiming | None = None
    candidate_timings: TypesafeTiming | None = None

    @property
    def changed_field_names(self) -> tuple[str, ...]:
        return tuple(delta.field_name for delta in self.field_deltas if delta.changed)


@dataclass(frozen=True)
class SignaturePlan:
    """A compiled hybrid execution plan for one DSPy signature."""

    original_signature: type[Signature]
    prompt_plans: tuple[TypesafePromptPlan, ...]
    remaining_output_names: tuple[str, ...]
    reduced_signature: type[Signature] | None

    @property
    def typesafe_field_names(self) -> tuple[str, ...]:
        return tuple(plan.field_name for plan in self.prompt_plans)

    @property
    def requires_dspy(self) -> bool:
        return bool(self.remaining_output_names)


def default_document_builder(signature: type[Signature], inputs: dict[str, Any]) -> JsonValue:
    """Build a structured Typesafe document from a DSPy signature and inputs."""

    return {
        "signature": {
            "name": signature.__name__,
            "instructions": signature.instructions,
            "inputs": {name: _field_context(name, field) for name, field in signature.input_fields.items()},
            "outputs": {name: _field_context(name, field) for name, field in signature.output_fields.items()},
        },
        "inputs": serialize_object(inputs),
    }


def annotate_signature(
    signature: type[Signature],
    *,
    fields: Mapping[str, TypesafeFieldConfig],
) -> type[Signature]:
    """Attach Typesafe field metadata to an existing DSPy signature class."""

    metadata = dict(getattr(signature, "__typesafe_dspy__", {}))
    metadata.update(fields)
    signature.__typesafe_dspy__ = metadata
    setattr(signature, _TYPESAFE_ENABLED_ATTR, True)
    return signature


def typesafe_signature(
    *,
    fields: Mapping[str, TypesafeFieldConfig] | None = None,
    score_fields: ScoreFieldSpec | None = None,
    copy: bool = False,
    signature_name: str | None = None,
):
    """Decorator form of `annotate_signature`."""

    return typesafeify(
        fields=fields,
        score_fields=score_fields,
        copy=copy,
        signature_name=signature_name,
    )


def typesafeify(
    signature: type[Signature] | None = None,
    *,
    fields: Mapping[str, TypesafeFieldConfig] | None = None,
    score_fields: ScoreFieldSpec | None = None,
    copy: bool = False,
    signature_name: str | None = None,
):
    """Opt a DSPy signature into Typesafe-backed execution with minimal changes."""

    def decorator(signature_cls: type[Signature]) -> type[Signature]:
        target_signature = signature_cls
        if copy:
            target_signature = _clone_signature(
                signature_cls,
                signature_name=signature_name,
            )
        elif signature_name is not None:
            raise ValueError("`signature_name` only applies when `copy=True`.")

        return annotate_signature(
            target_signature,
            fields=_expanded_field_configs(fields=fields, score_fields=score_fields),
        )

    if signature is None:
        return decorator
    return decorator(signature)


def is_typesafeified(signature: str | type[Signature]) -> bool:
    """Return whether a signature has been marked for Typesafe execution."""

    signature = ensure_signature(signature)
    return bool(getattr(signature, _TYPESAFE_ENABLED_ATTR, False))


def install() -> None:
    """Patch `dspy.Predict` so decorated signatures transparently use Typesafe."""

    global _PREDICT_PATCHED

    if _PREDICT_PATCHED:
        return

    Predict.forward = _patched_predict_forward
    Predict.aforward = _patched_predict_aforward
    _PREDICT_PATCHED = True


def uninstall() -> None:
    """Restore the original `dspy.Predict` methods."""

    global _PREDICT_PATCHED

    if not _PREDICT_PATCHED:
        return

    Predict.forward = _ORIGINAL_PREDICT_FORWARD
    Predict.aforward = _ORIGINAL_PREDICT_AFORWARD
    _PREDICT_PATCHED = False


def configure_typesafe(
    *,
    client: Any,
    model: str,
    field_configs: Mapping[str, TypesafeFieldConfig] | None = None,
    prompt_factory: PromptFactory | None = None,
    document_builder: DocumentBuilder | None = None,
    noul_true_threshold: float = 0.5,
    auto: bool = False,
) -> TypesafeConfig:
    """Configure the global Typesafe runtime for normal `dspy.Predict` calls."""

    install()
    config = TypesafeConfig(
        client=client,
        model=model,
        field_configs=field_configs or {},
        prompt_factory=prompt_factory,
        document_builder=document_builder,
        noul_true_threshold=noul_true_threshold,
    )
    dspy.settings.configure(**{_TYPESAFE_CONFIG_KEY: config, _TYPESAFE_AUTO_KEY: auto})
    return config


def disable_typesafe() -> None:
    """Disable globally configured Typesafe execution without removing the patch."""

    dspy.settings.configure(**{_TYPESAFE_CONFIG_KEY: None, _TYPESAFE_AUTO_KEY: False})


def plan_signature(
    signature: str | type[Signature],
    *,
    field_configs: Mapping[str, TypesafeFieldConfig] | None = None,
) -> SignaturePlan:
    """Split a DSPy signature into Typesafe-resolved and LM-resolved outputs."""

    signature = ensure_signature(signature)
    field_configs = _merged_field_configs(signature, field_configs)
    _validate_field_configs(signature, field_configs)

    prompt_plans: list[TypesafePromptPlan] = []
    remaining_output_names: list[str] = []

    reduced_fields: dict[str, tuple[type[Any], Any]] = {
        name: (field.annotation, deepcopy(field)) for name, field in signature.input_fields.items()
    }

    for name, output_field in signature.output_fields.items():
        prompt_plan = _build_prompt_plan(
            signature,
            name,
            output_field,
            field_configs.get(name),
        )
        if prompt_plan is None:
            remaining_output_names.append(name)
            reduced_fields[name] = (output_field.annotation, deepcopy(output_field))
            continue

        prompt_plans.append(prompt_plan)
        reduced_fields[name] = (
            output_field.annotation,
            _clone_output_as_input(name, output_field),
        )

    reduced_signature = None
    if prompt_plans and remaining_output_names:
        reduced_signature = make_signature(
            reduced_fields,
            instructions=_fallback_instructions(signature, prompt_plans),
            signature_name=f"{signature.__name__}TypesafeFallback",
        )

    return SignaturePlan(
        original_signature=signature,
        prompt_plans=tuple(prompt_plans),
        remaining_output_names=tuple(remaining_output_names),
        reduced_signature=reduced_signature,
    )


def typesafe_results(prediction: Prediction) -> Mapping[str, TypesafeFieldResult]:
    """Return the raw Typesafe decision metadata attached to a prediction."""

    return getattr(prediction, "_typesafe_results", {})


def typesafe_timings(prediction: Prediction) -> TypesafeTiming | None:
    """Return execution timing metadata for a Typesafe-backed prediction."""

    return getattr(prediction, "_typesafe_timings", None)


def compare_predictions(
    signature: str | type[Signature],
    baseline: Prediction,
    candidate: Prediction,
    *,
    baseline_label: str = "dspy",
    candidate_label: str = "typesafe",
) -> PredictionComparison:
    """
    Compare two predictions for the same signature field by field.
    """

    signature = ensure_signature(signature)
    field_deltas: list[PredictionFieldDelta] = []

    for field_name in signature.output_fields:
        baseline_value = getattr(baseline, field_name)
        candidate_value = getattr(candidate, field_name)
        changed = _comparison_token(baseline_value) != _comparison_token(candidate_value)
        field_deltas.append(
            PredictionFieldDelta(
                field_name=field_name,
                baseline_value=baseline_value,
                candidate_value=candidate_value,
                baseline_display=_display_value(baseline_value),
                candidate_display=_display_value(candidate_value),
                delta=_delta_value(baseline_value, candidate_value, changed=changed),
                changed=changed,
            )
        )

    return PredictionComparison(
        signature=signature,
        baseline_label=baseline_label,
        candidate_label=candidate_label,
        field_deltas=tuple(field_deltas),
        baseline_timings=typesafe_timings(baseline),
        candidate_timings=typesafe_timings(candidate),
    )


def render_prediction_comparison(
    comparison: PredictionComparison,
    *,
    max_value_width: int = 30,
) -> str:
    """
    Render a compact side-by-side comparison for two predictions.
    """

    if not comparison.field_deltas:
        return "No output fields to compare."

    field_width = max(len("field"), *(len(delta.field_name) for delta in comparison.field_deltas))
    baseline_width = max(len(comparison.baseline_label), max_value_width)
    candidate_width = max(len(comparison.candidate_label), max_value_width)
    delta_width = max(len("delta"), *(len(delta.delta) for delta in comparison.field_deltas))

    header = (
        f"{'field':<{field_width}}  "
        f"{comparison.baseline_label:<{baseline_width}}  "
        f"{comparison.candidate_label:<{candidate_width}}  "
        f"{'delta':>{delta_width}}"
    )
    divider = f"{'-' * field_width}  {'-' * baseline_width}  {'-' * candidate_width}  {'-' * delta_width}"

    lines = [header, divider]
    for field_delta in comparison.field_deltas:
        baseline_display = _short_display(field_delta.baseline_display, width=baseline_width)
        candidate_display = _short_display(field_delta.candidate_display, width=candidate_width)
        lines.append(
            f"{field_delta.field_name:<{field_width}}  "
            f"{baseline_display:<{baseline_width}}  "
            f"{candidate_display:<{candidate_width}}  "
            f"{field_delta.delta:>{delta_width}}"
        )
    return "\n".join(lines)


class TypesafePredict(Predict):
    """Resolve typed outputs with Typesafe, optionally rejecting all LM fallback.

    Use strict=True for a TypeSafe-only predictor.
    """

    def __init__(
        self,
        signature: str | type[Signature],
        typesafe_config: TypesafeConfig,
        callbacks=None,
        *,
        strict: bool = False,
        **config,
    ) -> None:
        super().__init__(signature, callbacks=callbacks, **config)
        self.strict = strict
        self.typesafe_config = typesafe_config
        self.prompt_factory = typesafe_config.prompt_factory or ImportedTypesafePromptFactory()
        self.document_builder = typesafe_config.document_builder or default_document_builder
        if strict:
            _validate_strict_plan(self.plan_signature(), self.config)

    @classmethod
    def from_predictor(
        cls,
        predictor: Predict,
        *,
        typesafe_config: TypesafeConfig,
    ) -> TypesafePredict:
        """Wrap an existing predictor without changing its external DSPy shape."""

        wrapped = cls(
            predictor.signature,
            typesafe_config=typesafe_config,
            callbacks=getattr(predictor, "callbacks", None),
            strict=getattr(predictor, "strict", False),
            **predictor.config,
        )
        wrapped.lm = predictor.lm
        wrapped.traces = list(predictor.traces)
        wrapped.train = list(predictor.train)
        wrapped.demos = list(predictor.demos)
        wrapped.stage = predictor.stage
        return wrapped

    def plan_signature(self, signature: str | type[Signature] | None = None) -> SignaturePlan:
        """Return the current hybrid plan for the configured signature."""

        return plan_signature(
            signature or self.signature,
            field_configs=self.typesafe_config.field_configs,
        )

    def forward(self, **kwargs):
        return _hybrid_forward(
            self,
            typesafe_config=self.typesafe_config,
            prompt_factory=self.prompt_factory,
            document_builder=self.document_builder,
            **kwargs,
        )

    async def aforward(self, **kwargs):
        return await anyio.to_thread.run_sync(
            lambda: _hybrid_forward(
                self,
                typesafe_config=self.typesafe_config,
                prompt_factory=self.prompt_factory,
                document_builder=self.document_builder,
                **kwargs,
            )
        )


def enable_typesafe(module: dspy.Module, *, typesafe_config: TypesafeConfig) -> dspy.Module:
    """Wrap every predictor in a DSPy program with `TypesafePredict`."""

    return module.map_named_predictors(
        lambda predictor: TypesafePredict.from_predictor(
            predictor,
            typesafe_config=typesafe_config,
        )
        if not isinstance(predictor, TypesafePredict)
        else predictor
    )


def _patched_predict_forward(self: Predict, **kwargs):
    runtime = _active_runtime_config(kwargs.get("signature", self.signature))
    if runtime is None:
        return _ORIGINAL_PREDICT_FORWARD(self, **kwargs)

    prompt_factory = runtime.prompt_factory or ImportedTypesafePromptFactory()
    document_builder = runtime.document_builder or default_document_builder
    return _hybrid_forward(
        self,
        typesafe_config=runtime,
        prompt_factory=prompt_factory,
        document_builder=document_builder,
        **kwargs,
    )


async def _patched_predict_aforward(self: Predict, **kwargs):
    runtime = _active_runtime_config(kwargs.get("signature", self.signature))
    if runtime is None:
        return await _ORIGINAL_PREDICT_AFORWARD(self, **kwargs)

    prompt_factory = runtime.prompt_factory or ImportedTypesafePromptFactory()
    document_builder = runtime.document_builder or default_document_builder
    return await anyio.to_thread.run_sync(
        lambda: _hybrid_forward(
            self,
            typesafe_config=runtime,
            prompt_factory=prompt_factory,
            document_builder=document_builder,
            **kwargs,
        )
    )


def _active_runtime_config(signature: str | type[Signature]) -> TypesafeConfig | None:
    signature = ensure_signature(signature)
    runtime = settings.get(_TYPESAFE_CONFIG_KEY)
    if runtime is None:
        return None

    if settings.get(_TYPESAFE_AUTO_KEY, False):
        return runtime

    if is_typesafeified(signature):
        return runtime

    return None


def _hybrid_forward(
    predictor: Predict,
    *,
    typesafe_config: TypesafeConfig,
    prompt_factory: PromptFactory,
    document_builder: DocumentBuilder,
    **kwargs,
) -> Prediction:
    total_start = time.perf_counter()
    lm, config, signature, demos, inputs, plan = _prepare_hybrid_call(
        predictor,
        typesafe_config=typesafe_config,
        **kwargs,
    )

    if not plan.prompt_plans:
        dspy_start = time.perf_counter()
        prediction = _run_dspy(predictor, lm, config, signature, demos, inputs)
        prediction._typesafe_timings = TypesafeTiming(
            typesafe_seconds=0.0,
            dspy_seconds=time.perf_counter() - dspy_start,
            total_seconds=time.perf_counter() - total_start,
        )
        return prediction

    typesafe_start = time.perf_counter()
    resolved_outputs, field_results = _evaluate_typesafe(
        typesafe_config=typesafe_config,
        prompt_factory=prompt_factory,
        document_builder=document_builder,
        signature=signature,
        inputs=inputs,
        plan=plan,
    )
    typesafe_seconds = time.perf_counter() - typesafe_start

    dspy_seconds = 0.0
    if plan.requires_dspy:
        fallback_inputs = {**inputs, **resolved_outputs}
        dspy_start = time.perf_counter()
        completions = _run_dspy_for_signature(
            predictor,
            lm=lm,
            config=config,
            signature=plan.reduced_signature,
            demos=demos,
            inputs=fallback_inputs,
        )
        dspy_seconds = time.perf_counter() - dspy_start
        completions = [{**completion, **resolved_outputs} for completion in completions]
    else:
        completions = [resolved_outputs]

    prediction = predictor._forward_postprocess(completions, signature, **inputs)
    prediction._typesafe_results = field_results
    prediction._typesafe_timings = TypesafeTiming(
        typesafe_seconds=typesafe_seconds,
        dspy_seconds=dspy_seconds,
        total_seconds=time.perf_counter() - total_start,
    )
    return prediction


def _prepare_hybrid_call(
    predictor: Predict,
    *,
    typesafe_config: TypesafeConfig,
    **kwargs,
):
    assert "new_signature" not in kwargs, "new_signature is no longer a valid keyword argument."

    signature = ensure_signature(kwargs.pop("signature", predictor.signature))
    demos = kwargs.pop("demos", predictor.demos)
    config = {**predictor.config, **kwargs.pop("config", {})}
    lm = kwargs.pop("lm", predictor.lm) or settings.lm
    merged_field_configs = {**typesafe_config.field_configs, **_merged_field_configs(signature, None)}
    plan = plan_signature(signature, field_configs=merged_field_configs)

    if getattr(predictor, "strict", False):
        _validate_strict_plan(plan, config)

    if plan.requires_dspy:
        _configure_residual_dspy(lm, config)

    _move_prediction_to_config(kwargs, config)
    _apply_input_defaults(signature, kwargs)
    _warn_for_extra_inputs(signature, kwargs)
    _warn_for_type_mismatches(signature, kwargs)
    _warn_for_missing_inputs(signature, kwargs)

    return lm, config, signature, demos, kwargs, plan


def _validate_strict_plan(plan: SignaturePlan, config: dict[str, Any]) -> None:
    if config:
        raise ValueError("Strict TypesafePredict does not support LM generation options.")
    if not plan.prompt_plans or plan.remaining_output_names:
        raise ValueError(
            f"Strict TypesafePredict requires every output to be handled by Typesafe. "
            f"Unsupported outputs: {list(plan.remaining_output_names)}"
        )
    for prompt in plan.prompt_plans:
        output = plan.original_signature.output_fields[prompt.field_name]
        if output.metadata:
            raise ValueError(f"Field `{prompt.field_name}` has unsupported output constraints.")
        if prompt.kind == "choice" and len(prompt.choice_options) < 2:
            raise ValueError(f"Field `{prompt.field_name}` requires at least two Choice options.")
        if prompt.kind == "score":
            if not isinstance(output.annotation, type) or not issubclass(output.annotation, float):
                raise ValueError(f"Field `{prompt.field_name}` must be a float output for Score.")
            if len(prompt.score_levels) < 2:
                raise ValueError(f"Field `{prompt.field_name}` requires at least two Score levels.")


def _configure_residual_dspy(lm: BaseLM | str | None, config: dict[str, Any]) -> None:
    if lm is None:
        raise ValueError(
            "No LM is loaded for the residual DSPy outputs. Configure an LM with "
            "`dspy.configure(lm=dspy.LM(...))` or use a signature that is fully handled by Typesafe."
        )
    if isinstance(lm, str):
        raise ValueError(
            f"LM must be an instance of `dspy.BaseLM`, not a string. Instead of using a string like "
            f"'dspy.configure(lm=\"{lm}\")', please configure the LM like 'dspy.configure(lm=dspy.LM(\"{lm}\"))'"
        )
    if not isinstance(lm, BaseLM):
        raise ValueError(f"LM must be an instance of `dspy.BaseLM`, not {type(lm)}. Received `lm={lm}`.")

    temperature = config.get("temperature") or lm.kwargs.get("temperature")
    num_generations = config.get("n") or lm.kwargs.get("n") or lm.kwargs.get("num_generations") or 1
    if (temperature is None or temperature <= 0.15) and num_generations > 1:
        config["temperature"] = 0.7


def _move_prediction_to_config(kwargs: dict[str, Any], config: dict[str, Any]) -> None:
    prediction = kwargs.get("prediction")
    if isinstance(prediction, dict) and prediction.get("type") == "content" and "content" in prediction:
        config["prediction"] = kwargs.pop("prediction")


def _apply_input_defaults(signature: type[Signature], inputs: dict[str, Any]) -> None:
    for name, input_field in signature.input_fields.items():
        if name not in inputs and input_field.default is not PydanticUndefined:
            inputs[name] = input_field.default


def _warn_for_extra_inputs(signature: type[Signature], inputs: dict[str, Any]) -> None:
    extra_fields = [name for name in inputs if name not in signature.input_fields]
    if extra_fields:
        logger.warning(
            "Input contains fields not in signature. These fields will be ignored: %s. Expected fields: %s.",
            extra_fields,
            list(signature.input_fields.keys()),
        )


def _warn_for_type_mismatches(signature: type[Signature], inputs: dict[str, Any]) -> None:
    if not settings.warn_on_type_mismatch:
        return

    for field_name, field_info in signature.input_fields.items():
        if field_name not in inputs:
            continue
        value = inputs[field_name]
        expected_type = field_info.annotation

        if value is None or field_info.json_schema_extra.get(IS_TYPE_UNDEFINED, False):
            continue

        if not _is_value_compatible_with_type(value, expected_type):
            logger.warning(
                "Type mismatch for field '%s': expected %s based on given Signature, "
                "but the provided value is incompatible: %s.",
                field_name,
                _get_type_name(expected_type),
                value,
            )


def _warn_for_missing_inputs(signature: type[Signature], inputs: dict[str, Any]) -> None:
    if all(name in inputs for name in signature.input_fields):
        return

    present = [name for name in signature.input_fields if name in inputs]
    missing = [name for name in signature.input_fields if name not in inputs]
    logger.warning(
        "Not all input fields were provided to module. Present: %s. Missing: %s.",
        present,
        missing,
    )


def _evaluate_typesafe(
    *,
    typesafe_config: TypesafeConfig,
    prompt_factory: PromptFactory,
    document_builder: DocumentBuilder,
    signature: type[Signature],
    inputs: dict[str, Any],
    plan: SignaturePlan,
) -> tuple[dict[str, Any], dict[str, TypesafeFieldResult]]:
    document = document_builder(signature, inputs)
    questions = {
        prompt_plan.field_name: _build_prompt(prompt_factory, prompt_plan) for prompt_plan in plan.prompt_plans
    }
    evaluation = typesafe_config.client.system_one(document, questions, model=typesafe_config.model)

    resolved_outputs: dict[str, Any] = {}
    field_results: dict[str, TypesafeFieldResult] = {}

    for prompt_plan in plan.prompt_plans:
        if prompt_plan.kind == "noul":
            response = evaluation.answers[prompt_plan.field_name]
            value = response.noul >= typesafe_config.noul_true_threshold
            resolved_outputs[prompt_plan.field_name] = value
            field_results[prompt_plan.field_name] = TypesafeFieldResult(
                kind="noul",
                value=value,
                probability=response.noul,
            )
            continue

        if prompt_plan.kind == "choice":
            response = evaluation.answers[prompt_plan.field_name]
            option_lookup = {option.prompt_key: option.value for option in prompt_plan.choice_options}
            value = option_lookup[response.choice]
            resolved_outputs[prompt_plan.field_name] = value
            probabilities = {
                option_lookup[key]: probability
                for key, probability in response.probabilities.items()
                if key in option_lookup
            }
            field_results[prompt_plan.field_name] = TypesafeFieldResult(
                kind="choice",
                value=value,
                probabilities=probabilities,
                confidence=response.confidence,
            )
            continue

        response = evaluation.answers[prompt_plan.field_name]
        # JSON object keys are strings; SDK versions may expose integer indices.
        probabilities_by_index = {int(index): value for index, value in response.probabilities.items()}
        score = _score_expectation_on_configured_scale(
            response.score,
            prompt_plan.score_levels,
            probabilities_by_index,
        )
        probabilities = _score_probabilities_by_anchor(probabilities_by_index, prompt_plan.score_levels)
        annotation = signature.output_fields[prompt_plan.field_name].annotation
        if isinstance(annotation, type) and issubclass(annotation, Score):
            score = annotation(score)
        resolved_outputs[prompt_plan.field_name] = score
        field_results[prompt_plan.field_name] = TypesafeFieldResult(
            kind="score",
            value=score,
            probabilities=probabilities,
            confidence=response.confidence,
            expectation=score,
        )

    return resolved_outputs, field_results


def _build_prompt(prompt_factory: PromptFactory, prompt_plan: TypesafePromptPlan) -> Any:
    if prompt_plan.kind == "noul":
        return prompt_factory.create_noul(
            instructions=prompt_plan.instructions,
        )

    if prompt_plan.kind == "choice":
        options = {option.prompt_key: option.description for option in prompt_plan.choice_options}
        return prompt_factory.create_choice(
            instructions=prompt_plan.instructions,
            options=options,
        )

    return prompt_factory.create_score(
        instructions=prompt_plan.instructions,
        levels=prompt_plan.score_levels,
    )


def _score_probabilities_by_anchor(
    probabilities: Mapping[int, float],
    levels: Mapping[float, JsonValue],
) -> Mapping[float, float]:
    anchors = sorted(levels)
    if all(index in probabilities for index in range(len(anchors))):
        return {
            anchor: probabilities[index]
            for index, anchor in enumerate(anchors)
        }
    return dict(probabilities)


def _score_expectation_on_configured_scale(
    score: float,
    levels: Mapping[float, JsonValue],
    probabilities: Mapping[int, float],
) -> float:
    """Preserve the configured numeric expectation across v1's positional scale."""
    anchors = sorted(levels)
    if not anchors:
        raise ValueError("Score levels must contain at least one numeric anchor.")
    if len(anchors) == 1:
        return float(anchors[0])

    if all(index in probabilities for index in range(len(anchors))):
        total_probability = sum(probabilities[index] for index in range(len(anchors)))
        if total_probability > 0:
            return sum(
                anchor * probabilities[index]
                for index, anchor in enumerate(anchors)
            ) / total_probability

    bounded_score = min(max(float(score), 0.0), len(anchors) - 1)
    lower_index = min(int(bounded_score), len(anchors) - 2)
    fraction = bounded_score - lower_index
    lower = anchors[lower_index]
    upper = anchors[lower_index + 1]
    return lower + fraction * (upper - lower)


def _run_dspy(
    predictor: Predict,
    lm: BaseLM | None,
    config: dict[str, Any],
    signature: type[Signature],
    demos: list[dict[str, Any]],
    inputs: dict[str, Any],
) -> Prediction:
    completions = _run_dspy_for_signature(
        predictor,
        lm=lm,
        config=config,
        signature=signature,
        demos=demos,
        inputs=inputs,
    )
    return predictor._forward_postprocess(completions, signature, **inputs)


def _run_dspy_for_signature(
    predictor: Predict,
    *,
    lm: BaseLM | None,
    config: dict[str, Any],
    signature: type[Signature] | None,
    demos: list[dict[str, Any]],
    inputs: dict[str, Any],
) -> list[dict[str, Any]]:
    adapter = settings.adapter or ChatAdapter()

    if predictor._should_stream():
        with settings.context(caller_predict=predictor):
            return adapter(lm, lm_kwargs=config, signature=signature, demos=demos, inputs=inputs)

    with settings.context(send_stream=None):
        return adapter(lm, lm_kwargs=config, signature=signature, demos=demos, inputs=inputs)


def _merged_field_configs(
    signature: type[Signature],
    overrides: Mapping[str, TypesafeFieldConfig] | None,
) -> dict[str, TypesafeFieldConfig]:
    merged = dict(getattr(signature, "__typesafe_dspy__", {}))
    if overrides:
        merged.update(overrides)
    return merged


def _clone_signature(
    signature: type[Signature],
    *,
    signature_name: str | None,
) -> type[Signature]:
    cloned_fields = {name: (field.annotation, deepcopy(field)) for name, field in signature.fields.items()}
    clone = make_signature(
        cloned_fields,
        instructions=signature.instructions,
        signature_name=signature_name or f"{signature.__name__}Typesafe",
    )
    clone.__module__ = signature.__module__
    clone.__qualname__ = signature_name or f"{signature.__qualname__}Typesafe"
    if is_typesafeified(signature):
        clone.__typesafe_dspy__ = deepcopy(getattr(signature, "__typesafe_dspy__", {}))
        setattr(clone, _TYPESAFE_ENABLED_ATTR, True)
    return clone


def _expanded_field_configs(
    *,
    fields: Mapping[str, TypesafeFieldConfig] | None,
    score_fields: ScoreFieldSpec | None,
) -> dict[str, TypesafeFieldConfig]:
    expanded = dict(fields or {})
    for field_name, spec in (score_fields or {}).items():
        expanded[field_name] = TypesafeFieldConfig(
            kind="score",
            score_levels=_score_levels_from_spec(field_name, spec),
        )
    return expanded


def _validate_field_configs(
    signature: type[Signature],
    field_configs: Mapping[str, TypesafeFieldConfig],
) -> None:
    unknown_fields = sorted(set(field_configs) - set(signature.output_fields))
    if unknown_fields:
        raise ValueError(
            f"Typesafe field overrides must target output fields on {signature.__name__}. Unknown: {unknown_fields}"
        )


def _score_levels_from_spec(
    field_name: str,
    spec: Sequence[float] | Mapping[float, JsonValue],
) -> Mapping[float, JsonValue]:
    if isinstance(spec, Mapping):
        return dict(spec)

    anchors = list(spec)
    if len(anchors) not in {2, 3}:
        raise ValueError(
            f"Score field `{field_name}` must use either [min, max], [min, mid, max], "
            "or a mapping of explicit score levels."
        )

    if len(anchors) == 2:
        low, high = anchors
        if low >= high:
            raise ValueError(f"Score field `{field_name}` requires an increasing range, received {anchors}.")
        midpoint = (low + high) / 2
        anchors = sorted({low, midpoint, high})
    else:
        anchors = list(anchors)
        if anchors != sorted(anchors):
            raise ValueError(f"Score field `{field_name}` must be in increasing order, received {anchors}.")

    levels: dict[float, JsonValue] = {}
    for index, anchor in enumerate(anchors):
        if index == 0:
            meaning = f"`{field_name}` is near the low end of the range ({anchor})."
        elif index == len(anchors) - 1:
            meaning = f"`{field_name}` is near the high end of the range ({anchor})."
        else:
            meaning = f"`{field_name}` is around the middle of the range ({anchor})."
        levels[anchor] = meaning
    return levels


def _build_prompt_plan(
    signature: type[Signature],
    field_name: str,
    field: Any,
    config: TypesafeFieldConfig | None,
) -> TypesafePromptPlan | None:
    config = config or TypesafeFieldConfig()
    if isinstance(field.annotation, type) and issubclass(field.annotation, Score):
        if config.kind not in (None, "score") or config.score_levels is not None or config.choice_options is not None:
            raise ValueError(f"Field `{field_name}` has conflicting overrides for its Score annotation.")
        return TypesafePromptPlan(
            field_name=field_name,
            kind="score",
            instructions=config.instructions or _default_prompt_instructions(signature, field_name, field, "score"),
            score_levels=dict(field.annotation.levels),
        )
    kind = _resolve_kind(field.annotation, config)
    if kind is None or kind == "disable":
        return None

    instructions = config.instructions or _default_prompt_instructions(signature, field_name, field, kind)

    if kind == "noul":
        if field.annotation is not bool:
            raise ValueError(f"Field `{field_name}` must be annotated as `bool` to use a Noul prompt.")
        return TypesafePromptPlan(field_name=field_name, kind="noul", instructions=instructions)

    if kind == "choice":
        option_values = config.choice_options or _literal_choice_options(field.annotation)
        if not option_values:
            raise ValueError(
                f"Field `{field_name}` must define `choice_options` or use a supported `Literal[...]` annotation."
            )

        choice_options = []
        for index, (value, description) in enumerate(option_values.items()):
            choice_options.append(
                ChoiceOptionPlan(
                    prompt_key=f"option_{index}",
                    value=value,
                    description=description,
                )
            )
        return TypesafePromptPlan(
            field_name=field_name,
            kind="choice",
            instructions=instructions,
            choice_options=tuple(choice_options),
        )

    if not config.score_levels:
        raise ValueError(f"Field `{field_name}` must define `score_levels` to use a Score prompt.")
    return TypesafePromptPlan(
        field_name=field_name,
        kind="score",
        instructions=instructions,
        score_levels=config.score_levels,
    )


def _resolve_kind(annotation: Any, config: TypesafeFieldConfig) -> TypesafeKind | None:
    if config.kind is not None:
        return config.kind

    if annotation is bool:
        return "noul"
    if _literal_choice_options(annotation):
        return "choice"
    return None


def _literal_choice_options(annotation: Any) -> Mapping[Any, JsonValue] | None:
    if get_origin(annotation) is not Literal:
        return None

    values = get_args(annotation)
    if not values or not all(isinstance(value, str | int | float | bool) for value in values):
        return None

    return {
        value: {
            "value": serialize_object(value),
            "meaning": f"The field should be {value!r}.",
        }
        for value in values
    }


def _clone_output_as_input(name: str, field: Any) -> Any:
    cloned = deepcopy(field)
    extras = dict(cloned.json_schema_extra or {})
    extras["__dspy_field_type"] = "input"

    description = _field_description(field)
    extras["desc"] = (
        f"Resolved earlier for `{name}`. {description}" if description else f"Resolved earlier for `{name}`."
    )
    cloned.json_schema_extra = extras
    return cloned


def _fallback_instructions(
    signature: type[Signature],
    prompt_plans: list[TypesafePromptPlan],
) -> str:
    resolved_names = ", ".join(f"`{plan.field_name}`" for plan in prompt_plans)
    return (
        f"{signature.instructions}\n\n"
        f"The fields {resolved_names} have already been decided and are now trusted inputs. "
        "Only produce the remaining output fields."
    )


def _default_prompt_instructions(
    signature: type[Signature],
    field_name: str,
    field: Any,
    kind: Literal["noul", "choice", "score"],
) -> JsonValue:
    return {
        "task": signature.instructions,
        "output_field": _field_context(field_name, field),
        "decision": {
            "kind": kind,
            "goal": _decision_goal(field_name, kind),
        },
    }


def _field_context(name: str, field: Any) -> dict[str, Any]:
    context = {
        "name": name,
        "annotation": _get_type_name(field.annotation),
    }
    description = _field_description(field)
    if description:
        context["description"] = description
    constraints = (field.json_schema_extra or {}).get("constraints")
    if constraints:
        context["constraints"] = constraints
    return context


def _field_description(field: Any) -> str | None:
    description = (field.json_schema_extra or {}).get("desc")
    if not description or description.startswith("${"):
        return None
    return description


def _decision_goal(
    field_name: str,
    kind: Literal["noul", "choice", "score"],
) -> str:
    if kind == "noul":
        return f"Estimate the probability that `{field_name}` should be true."
    if kind == "choice":
        return f"Choose the best value for `{field_name}`."
    return f"Estimate the numeric expectation for `{field_name}`."


def _comparison_token(value: Any) -> Any:
    return serialize_object(value)


def _display_value(value: Any) -> str:
    serialized = serialize_object(value)
    if isinstance(serialized, str):
        return serialized
    return json.dumps(serialized, sort_keys=True)


def _delta_value(baseline: Any, candidate: Any, *, changed: bool) -> str:
    if not changed:
        return "same"
    if _is_numeric_delta(baseline, candidate):
        return f"{float(candidate) - float(baseline):+.2f}"
    return "changed"


def _is_numeric_delta(baseline: Any, candidate: Any) -> bool:
    numeric_types = (int, float)
    return (
        isinstance(baseline, numeric_types)
        and not isinstance(baseline, bool)
        and isinstance(candidate, numeric_types)
        and not isinstance(candidate, bool)
    )


def _short_display(value: str, *, width: int) -> str:
    return textwrap.shorten(value.replace("\n", " "), width=width, placeholder="...")

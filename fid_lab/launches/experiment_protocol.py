"""Pre-registered scale and evidence contract for every new Launch Review."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from hashlib import sha256
import json
from math import ceil
from pathlib import Path
from statistics import NormalDist


PROTOCOL_VERSION = "recommendation-experiment-protocol-v1"
SMOKE_USERS = 100_000
SCREEN_USERS_PER_SALT = 100_000
SCREEN_SALTS = 3
MIN_POWERED_SALTS = 3


class ExperimentPhase(str, Enum):
    SMOKE = "smoke"
    SCREEN = "screen"
    POWERED_AB = "powered_ab"
    SCALE_BENCHMARK = "scale_benchmark"


def payload_fingerprint(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return f"sha256:{sha256(encoded).hexdigest()}"


@dataclass(frozen=True)
class ExperimentPlan:
    launch_id: str
    phase: ExperimentPhase
    hypothesis: str
    isolated_change: str
    primary_metric: str
    mde_absolute: float
    alpha: float
    power: float
    pilot_total_users: int
    pilot_primary_standard_error: float
    users_per_salt: int
    salts: tuple[int, ...]
    control_fingerprint: str
    treatment_fingerprint: str
    scenario_fingerprint: str
    predecessor_report: str | None
    predecessor_report_sha256: str | None
    registered_before_evidence: bool
    protocol_version: str = PROTOCOL_VERSION

    def __post_init__(self):
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError("unsupported experiment protocol")
        if not self.launch_id or not self.hypothesis or not self.isolated_change:
            raise ValueError("experiment identity and isolated change are required")
        if not 0 < self.alpha < 0.5 or not 0.5 < self.power < 1.0:
            raise ValueError("experiment alpha or power is invalid")
        if self.mde_absolute <= 0:
            raise ValueError("pre-registered absolute MDE must be positive")
        if self.users_per_salt <= 0 or not self.salts:
            raise ValueError("planned users and salts are required")
        if len(set(self.salts)) != len(self.salts):
            raise ValueError("experiment salts must be unique")
        if not self.registered_before_evidence:
            raise ValueError("post-hoc experiment plans cannot authorize a launch")
        if not self.control_fingerprint.startswith("sha256:"):
            raise ValueError("control fingerprint is invalid")
        if not self.treatment_fingerprint.startswith("sha256:"):
            raise ValueError("treatment fingerprint is invalid")
        if not self.scenario_fingerprint.startswith("sha256:"):
            raise ValueError("scenario fingerprint is invalid")
        self._validate_predecessor_contract()
        self._validate_phase_scale()

    def _validate_predecessor_contract(self):
        requires_predecessor = self.phase in {
            ExperimentPhase.SCREEN, ExperimentPhase.POWERED_AB
        }
        present = bool(self.predecessor_report and self.predecessor_report_sha256)
        if requires_predecessor != present:
            raise ValueError("experiment predecessor contract is incomplete")

    def _validate_phase_scale(self):
        validators = {
            ExperimentPhase.SMOKE: self._validate_smoke,
            ExperimentPhase.SCREEN: self._validate_screen,
            ExperimentPhase.POWERED_AB: self._validate_powered,
            ExperimentPhase.SCALE_BENCHMARK: self._validate_benchmark,
        }
        validators[self.phase]()

    def _validate_smoke(self):
        if self.users_per_salt != SMOKE_USERS or len(self.salts) != 1:
            raise ValueError("smoke is exactly 100k users and one salt")

    def _validate_screen(self):
        correct_users = self.users_per_salt == SCREEN_USERS_PER_SALT
        if not correct_users or len(self.salts) != SCREEN_SALTS:
            raise ValueError("screen is exactly three 100k-user salts")

    def _validate_powered(self):
        if len(self.salts) < MIN_POWERED_SALTS:
            raise ValueError("powered A/B requires at least three salts")
        if self.planned_total_users < self.required_total_users:
            raise ValueError("powered A/B plan is below its pre-registered MDE")

    def _validate_benchmark(self):
        if len(self.salts) != 1 or self.users_per_salt < 1_000_000:
            raise ValueError("scale benchmark requires one >=1M-user run")

    @property
    def planned_total_users(self) -> int:
        return self.users_per_salt * len(self.salts)

    @property
    def required_total_users(self) -> int:
        if self.pilot_total_users <= 0 or self.pilot_primary_standard_error <= 0:
            raise ValueError("positive control-variance pilot evidence is required")
        normal = NormalDist()
        critical = normal.inv_cdf(1.0 - self.alpha / 2.0)
        target = normal.inv_cdf(self.power)
        multiplier = (
            (critical + target)
            * self.pilot_primary_standard_error
            / self.mde_absolute
        ) ** 2
        return ceil(self.pilot_total_users * multiplier)

    @property
    def plan_fingerprint(self) -> str:
        return payload_fingerprint(self.manifest())

    def manifest(self) -> dict[str, object]:
        value = asdict(self)
        value["phase"] = self.phase.value
        value["salts"] = list(self.salts)
        value["planned_total_users"] = self.planned_total_users
        value["required_total_users"] = self.required_total_users
        return value

    def validate_run(
        self, control: dict, treatment: dict, scenario: dict,
        users: int, salt: int,
    ) -> None:
        if payload_fingerprint(control) != self.control_fingerprint:
            raise ValueError("runtime control differs from the registered plan")
        if payload_fingerprint(treatment) != self.treatment_fingerprint:
            raise ValueError("runtime treatment differs from the registered plan")
        if payload_fingerprint(scenario) != self.scenario_fingerprint:
            raise ValueError("runtime scenario differs from the registered plan")
        if users != self.users_per_salt:
            raise ValueError("runtime users differ from the registered plan")
        if salt not in self.salts:
            raise ValueError("runtime salt is absent from the registered plan")

    def validate_predecessor(self, root: Path) -> None:
        if self.predecessor_report is None:
            return
        path = root / self.predecessor_report
        if not path.exists():
            raise ValueError("registered predecessor report does not exist")
        actual = sha256(path.read_bytes()).hexdigest()
        if actual != self.predecessor_report_sha256:
            raise ValueError("registered predecessor report hash mismatch")
        report = json.loads(path.read_text())
        previous = experiment_plan_from_manifest(report["experiment_plan"])
        expected_phase = (
            ExperimentPhase.SMOKE
            if self.phase == ExperimentPhase.SCREEN
            else ExperimentPhase.SCREEN
        )
        if previous.launch_id != self.launch_id or previous.phase != expected_phase:
            raise ValueError("experiment predecessor phase or launch id differs")
        for name in (
            "control_fingerprint", "treatment_fingerprint",
            "scenario_fingerprint",
        ):
            if getattr(previous, name) != getattr(self, name):
                raise ValueError("experiment artifacts changed between phases")
        expected_decision = (
            "smoke_pass"
            if self.phase == ExperimentPhase.SCREEN
            else "advance_to_powered"
        )
        if report.get("decision") != expected_decision:
            raise ValueError("predecessor decision does not authorize this phase")


def load_experiment_plan(path: Path, root: Path | None = None) -> ExperimentPlan:
    payload = json.loads(path.read_text())
    if payload.get("schema") != "recommendation-experiment-plan-v1":
        raise ValueError("experiment plan schema mismatch")
    fields = {key: value for key, value in payload.items() if key != "schema"}
    plan = experiment_plan_from_manifest(fields)
    if root is not None:
        plan.validate_predecessor(root)
    return plan


def experiment_plan_from_manifest(payload: dict) -> ExperimentPlan:
    fields = {
        key: value for key, value in payload.items()
        if key not in {"planned_total_users", "required_total_users"}
    }
    fields["phase"] = ExperimentPhase(fields["phase"])
    fields["salts"] = tuple(fields["salts"])
    return ExperimentPlan(**fields)


def phase_decision(
    plan: ExperimentPlan,
    statistical_decision: str,
    completed_salts: tuple[int, ...],
) -> str:
    if set(completed_salts) != set(plan.salts):
        return "partial_evidence"
    if plan.phase == ExperimentPhase.SMOKE:
        return "smoke_pass" if statistical_decision != "hold_or_reject" else "smoke_fail"
    if plan.phase == ExperimentPhase.SCREEN:
        return (
            "advance_to_powered"
            if statistical_decision in {"pass", "continue_powered_online_experiment"}
            else "reject"
        )
    if plan.phase == ExperimentPhase.SCALE_BENCHMARK:
        return (
            "benchmark_fail"
            if statistical_decision == "hold_or_reject"
            else "benchmark_pass"
        )
    if statistical_decision == "pass":
        return "pass"
    return "hold" if statistical_decision == "continue_powered_online_experiment" else "reject"

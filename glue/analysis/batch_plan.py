"""``BatchPlan`` — the concrete output of a selection: ``m`` parallel synthesis lanes.

A :class:`Lane` is one chosen hub plus the ``k`` products selected from its
diversification set — physically, "make this shared intermediate once, then run these
``k`` final reactions in parallel". A :class:`BatchPlan` is the ``m`` lanes together:
the full parallel-batch proposal for one strategy configuration.

A plan serialises to the project's **standard candidate dataset**
(``glue.datasets.candidates``) so the selected molecules are immediately readable by the
same tooling/harness as every other entrant — plus a ``lanes.csv`` sidecar recording
which hub/lane each candidate came from, the hub's flow, and the shared route, which is
what makes the concurrency/cost analysis possible after the fact.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from glue.active_learning.route import route_to_str
from glue.analysis.hub_graph import Child, Hub
from glue.datasets.candidates import CandidateDataset


@dataclass
class Lane:
    """One hub and the products selected from it (one parallel synthesis lane)."""

    lane_id: int
    hub: Hub
    children: List[Child]

    @property
    def n(self) -> int:
        return len(self.children)


@dataclass
class BatchPlan:
    """``m`` lanes = the full parallel-diversification proposal for one configuration."""

    lanes: List[Lane]
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def concurrency(self) -> int:
        """Number of parallel lanes (hubs) == the concurrency axis."""
        return len(self.lanes)

    def all_children(self) -> List[Child]:
        return [c for lane in self.lanes for c in lane.children]

    def all_smiles(self) -> List[str]:
        return [c.smiles for c in self.all_children()]

    def all_scores(self) -> List[Optional[float]]:
        return [c.score for c in self.all_children()]

    # -------------------------------------------------------------- serialise
    def to_candidate_dataset(
        self,
        out_dir,
        generator: str = "rgfn_hub_batch",
        oracle: Optional[str] = None,
        system: Optional[str] = None,
        seed: Optional[int] = None,
        higher_is_better: Optional[bool] = None,
        score_units: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> Path:
        """Write the selected molecules as a standard candidate dataset + lanes sidecar.

        Each candidate row carries extra columns tying it to its lane: ``lane_id``,
        ``hub_smiles``, ``hub_depth``, ``hub_flow``, ``child_source``. A ``lanes.csv``
        summarises the plan (one row per lane: hub, flow, route-to-hub, #products).
        """
        out_dir = Path(out_dir)
        ds = CandidateDataset(
            out_dir,
            generator=generator,
            oracle=oracle,
            system=system,
            seed=seed,
            score_higher_is_better=higher_is_better,
            score_units=score_units,
            notes=notes or "Hub-batched late-stage-diversification plan (glue.analysis).",
        )
        for lane in self.lanes:
            for child in lane.children:
                extra = {
                    "lane_id": lane.lane_id,
                    "hub_smiles": lane.hub.smiles,
                    "hub_depth": lane.hub.num_reactions,
                    "hub_flow": lane.hub.visit_count,
                    "child_source": child.source,
                }
                if child.route is not None:
                    extra["route_str"] = route_to_str(child.route)
                ds.add(
                    smiles=child.smiles, score=child.score, step=1, route=child.route, extra=extra
                )
        ds.write()
        self._write_lanes(out_dir / "lanes.csv")
        return out_dir

    def _write_lanes(self, path: Path) -> None:
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(
                ["lane_id", "hub_smiles", "hub_depth", "hub_flow", "n_products", "route_to_hub"]
            )
            for lane in self.lanes:
                w.writerow(
                    [
                        lane.lane_id,
                        lane.hub.smiles,
                        lane.hub.num_reactions,
                        lane.hub.visit_count,
                        lane.n,
                        route_to_str(lane.hub.route_to_hub) if lane.hub.route_to_hub else "",
                    ]
                )

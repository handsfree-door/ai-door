"""Geometry-only object/person proximity association; no claim of physical possession."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 (pixels)
    model: str


def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _ioa(a, b):
    """Area of intersection divided by area of a (object)."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy / max(_area(a), 1e-9)


def match_objects(persons: list[dict], objects: list[Detection],
                  min_overlap: float = 0.15, max_distance: float = 0.65):
    """Assign each object to at most one person; output proximity, not carrying.

    Score uses intersection over object area and center distance normalized by
    person box diagonal. Larger objects (e.g., stroller) may only partly overlap.
    """
    assignments: dict[int, list[dict]] = {int(p['track_id']): [] for p in persons}
    unmatched: list[dict] = []
    for obj in objects:
        best = None
        best_score = float('-inf')
        ox = (obj.box[0] + obj.box[2]) / 2
        oy = (obj.box[1] + obj.box[3]) / 2
        for person in persons:
            pb = person['bbox_xyxy']
            cx = (pb[0] + pb[2]) / 2
            cy = (pb[1] + pb[3]) / 2
            diag = max(((pb[2]-pb[0])**2 + (pb[3]-pb[1])**2)**0.5, 1.0)
            d = ((ox-cx)**2+(oy-cy)**2)**0.5 / diag
            overlap = _ioa(obj.box, pb)
            if overlap < min_overlap and d > max_distance:
                continue
            score = 0.7 * overlap + 0.3 * max(0.0, 1.0-d)
            if score > best_score:
                best_score = score
                best = int(person['track_id'])
        entry = {
            'label': obj.label, 'confidence': round(obj.confidence, 4),
            'bbox_xyxy': [round(float(v), 2) for v in obj.box],
            'model': obj.model,
            'association_score': round(max(best_score, 0.0), 4) if best is not None else None,
        }
        if best is None:
            unmatched.append(entry)
        else:
            assignments[best].append(entry)
    return assignments, unmatched

import numpy as np

from moving_det.models import Annotation, Component, OBB, Proposal, Tubelet


def ann(
    track: int,
    cx: float,
    cy: float = 20.0,
    width: float = 12.0,
    height: float = 6.0,
    theta: float = 0.0,
    class_name: str = "car",
) -> Annotation:
    return Annotation(
        obb=OBB(cx, cy, width, height, theta),
        class_name=class_name,
        track_id=track,
    )


def proposal(
    cx: float,
    cy: float = 20.0,
    width: float = 12.0,
    height: float = 6.0,
    theta: float = 0.0,
    frame: int = 1,
    tubelet_id: int = 1,
) -> Proposal:
    return Proposal(
        frame_index=frame,
        obb=OBB(cx, cy, width, height, theta),
        motion_score=1.0,
        tubelet_id=tubelet_id,
    )


def component_at(
    frame: int,
    x: int,
    y: int,
    width: int = 12,
    height: int = 6,
    component_id: int = 1,
) -> Component:
    xx, yy = np.meshgrid(
        np.arange(x, x + width),
        np.arange(y, y + height),
    )
    points = np.column_stack((xx.ravel(), yy.ravel())).astype(np.float32)
    return Component(
        component_id=component_id,
        frame_index=frame,
        points_xy=points,
        bbox_xyxy=(x, y, x + width, y + height),
        area=len(points),
        mean_score=1.0,
    )


def tubelet_at(frame: int, cx: float, cy: float) -> Tubelet:
    component = component_at(
        frame=frame,
        x=round(cx - 6),
        y=round(cy - 3),
    )
    return Tubelet(tubelet_id=1, components=(component,))

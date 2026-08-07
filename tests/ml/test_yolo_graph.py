import pytest
import torch

from moving_det.ml.models.baseline import create_p2_obb_detector
from moving_det.ml.yolo_graph import (
    execute_yolo_graph,
    extract_backbone_features,
)


def test_p2_detector_builds_exact_four_scale_graph():
    detector = create_p2_obb_detector(weights=None, nc=4)

    head = detector.model[-1]
    assert head.nc == 4
    assert head.nl == 4
    assert tuple(int(value) for value in detector.stride) == (4, 8, 16, 32)
    assert head.f == [19, 22, 25, 28]
    assert [layer.f for layer in detector.model[11:29]] == [
        -1,
        [-1, 6],
        -1,
        -1,
        [-1, 4],
        -1,
        -1,
        [-1, 2],
        -1,
        -1,
        [-1, 16],
        -1,
        -1,
        [-1, 13],
        -1,
        -1,
        [-1, 10],
        -1,
    ]


def test_graph_override_changes_p2_without_changing_training_schema():
    torch.manual_seed(7)
    detector = create_p2_obb_detector(weights=None, nc=4).train()
    image = torch.rand(1, 3, 128, 128)

    with torch.no_grad():
        p2 = extract_backbone_features(detector, image, (2,))[2]
        normal = execute_yolo_graph(detector, image)
        changed = execute_yolo_graph(
            detector,
            image,
            {2: torch.zeros_like(p2)},
        )

    assert normal.keys() == changed.keys() == {
        "boxes",
        "scores",
        "feats",
        "angle",
    }
    assert not torch.equal(normal["scores"], changed["scores"])


def test_graph_override_rejects_wrong_layer_shape():
    detector = create_p2_obb_detector(weights=None, nc=4).train()
    image = torch.rand(1, 3, 128, 128)

    with pytest.raises(ValueError, match=r"layer 2.*shape"):
        execute_yolo_graph(detector, image, {2: torch.zeros(1, 1, 1, 1)})


def test_backbone_extraction_returns_requested_spatial_scales():
    detector = create_p2_obb_detector(weights=None, nc=4).eval()
    image = torch.rand(1, 3, 128, 128)

    with torch.no_grad():
        features = extract_backbone_features(detector, image, (2, 4))

    assert tuple(features) == (2, 4)
    assert features[2].shape[-2:] == (32, 32)
    assert features[4].shape[-2:] == (16, 16)

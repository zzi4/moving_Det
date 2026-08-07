import pytest
import torch
from torch.nn import BatchNorm2d

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


def test_p2_detector_preserves_complete_yaml_module_and_argument_topology():
    detector = create_p2_obb_detector(weights=None, nc=4)

    assert detector.yaml["scale"] == "m"
    assert detector.yaml["backbone"] == [
        [-1, 1, "Conv", [64, 3, 2]],
        [-1, 1, "Conv", [128, 3, 2]],
        [-1, 2, "C3k2", [256, False, 0.25]],
        [-1, 1, "Conv", [256, 3, 2]],
        [-1, 2, "C3k2", [512, False, 0.25]],
        [-1, 1, "Conv", [512, 3, 2]],
        [-1, 2, "C3k2", [512, True]],
        [-1, 1, "Conv", [1024, 3, 2]],
        [-1, 2, "C3k2", [1024, True]],
        [-1, 1, "SPPF", [1024, 5]],
        [-1, 2, "C2PSA", [1024]],
    ]
    assert detector.yaml["head"] == [
        [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
        [[-1, 6], 1, "Concat", [1]],
        [-1, 2, "C3k2", [512, False]],
        [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
        [[-1, 4], 1, "Concat", [1]],
        [-1, 2, "C3k2", [256, False]],
        [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
        [[-1, 2], 1, "Concat", [1]],
        [-1, 2, "C3k2", [128, False]],
        [-1, 1, "Conv", [128, 3, 2]],
        [[-1, 16], 1, "Concat", [1]],
        [-1, 2, "C3k2", [256, False]],
        [-1, 1, "Conv", [256, 3, 2]],
        [[-1, 13], 1, "Concat", [1]],
        [-1, 2, "C3k2", [512, False]],
        [-1, 1, "Conv", [512, 3, 2]],
        [[-1, 10], 1, "Concat", [1]],
        [-1, 2, "C3k2", [1024, True]],
        [[19, 22, 25, 28], 1, "OBB", ["nc", 1]],
    ]
    assert [type(layer).__name__ for layer in detector.model] == [
        "Conv",
        "Conv",
        "C3k2",
        "Conv",
        "C3k2",
        "Conv",
        "C3k2",
        "Conv",
        "C3k2",
        "SPPF",
        "C2PSA",
        "Upsample",
        "Concat",
        "C3k2",
        "Upsample",
        "Concat",
        "C3k2",
        "Upsample",
        "Concat",
        "C3k2",
        "Conv",
        "Concat",
        "C3k2",
        "Conv",
        "Concat",
        "C3k2",
        "Conv",
        "Concat",
        "C3k2",
        "OBB",
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


def test_graph_override_rejects_wrong_layer_dtype():
    detector = create_p2_obb_detector(weights=None, nc=4).eval()
    image = torch.rand(1, 3, 128, 128)
    with torch.no_grad():
        p2 = extract_backbone_features(detector, image, (2,))[2]

    with pytest.raises(ValueError, match=r"layer 2.*dtype"):
        execute_yolo_graph(detector, image, {2: p2.to(torch.float64)})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_graph_override_rejects_wrong_layer_device():
    detector = create_p2_obb_detector(weights=None, nc=4).eval()
    image = torch.rand(1, 3, 128, 128)
    with torch.no_grad():
        p2 = extract_backbone_features(detector, image, (2,))[2]

    with pytest.raises(ValueError, match=r"layer 2.*device"):
        execute_yolo_graph(detector, image, {2: p2.cuda()})


def test_graph_override_rejects_unknown_and_non_tensor_layers():
    detector = create_p2_obb_detector(weights=None, nc=4).eval()
    image = torch.rand(1, 3, 128, 128)

    with pytest.raises(ValueError, match=r"unknown.*99"):
        execute_yolo_graph(detector, image, {99: image})
    with pytest.raises(ValueError, match=r"layer 2.*tensor"):
        execute_yolo_graph(detector, image, {2: object()})


def test_backbone_extraction_returns_requested_spatial_scales():
    detector = create_p2_obb_detector(weights=None, nc=4).eval()
    image = torch.rand(1, 3, 128, 128)

    with torch.no_grad():
        features = extract_backbone_features(detector, image, (2, 4))

    assert tuple(features) == (2, 4)
    assert features[2].shape[-2:] == (32, 32)
    assert features[4].shape[-2:] == (16, 16)


def test_backbone_extraction_stops_exactly_at_largest_requested_layer():
    detector = create_p2_obb_detector(weights=None, nc=4).train()
    image = torch.rand(1, 3, 128, 128)
    executed = []
    handles = [
        layer.register_forward_hook(
            lambda module, _inputs, _output: executed.append(module.i)
        )
        for layer in detector.model
    ]

    try:
        with torch.no_grad():
            extract_backbone_features(detector, image, (2,))
    finally:
        for handle in handles:
            handle.remove()

    assert executed == [0, 1, 2]


def test_backbone_extraction_does_not_update_downstream_batch_norm():
    detector = create_p2_obb_detector(weights=None, nc=4).train()
    image = torch.rand(1, 3, 128, 128)
    downstream_bn = next(
        module
        for module in detector.model[28].modules()
        if isinstance(module, BatchNorm2d)
    )
    tracked_before = downstream_bn.num_batches_tracked.clone()
    mean_before = downstream_bn.running_mean.clone()

    with torch.no_grad():
        extract_backbone_features(detector, image, (2,))

    torch.testing.assert_close(
        downstream_bn.num_batches_tracked,
        tracked_before,
    )
    torch.testing.assert_close(downstream_bn.running_mean, mean_before)


def _assert_eval_outputs_equal(actual, expected):
    assert isinstance(actual, tuple)
    assert len(actual) == 2
    torch.testing.assert_close(actual[0], expected[0])
    assert actual[1].keys() == expected[1].keys() == {
        "boxes",
        "scores",
        "feats",
        "angle",
    }
    for key in ("boxes", "scores", "angle"):
        torch.testing.assert_close(actual[1][key], expected[1][key])
    assert len(actual[1]["feats"]) == len(expected[1]["feats"]) == 4
    for actual_feature, expected_feature in zip(
        actual[1]["feats"],
        expected[1]["feats"],
        strict=True,
    ):
        torch.testing.assert_close(actual_feature, expected_feature)


def test_eval_graph_matches_direct_model_for_normal_and_identity_override():
    torch.manual_seed(17)
    detector = create_p2_obb_detector(weights=None, nc=4).eval()
    image = torch.rand(1, 3, 128, 128)

    with torch.no_grad():
        direct = detector(image)
        normal = execute_yolo_graph(detector, image)
        p2 = extract_backbone_features(detector, image, (2,))[2]
        overridden = execute_yolo_graph(detector, image, {2: p2})

    _assert_eval_outputs_equal(normal, direct)
    _assert_eval_outputs_equal(overridden, direct)

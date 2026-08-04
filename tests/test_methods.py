from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from moving_det.models import FrameSample, SequenceData
from moving_det.motion.alignment import AlignmentResult
from moving_det.motion import methods as methods_module
from moving_det.motion.methods import create_method


def _write_frame(path: Path, square_left: int) -> None:
    yy, xx = np.indices((96, 128))
    image = ((3 * xx + 5 * yy) % 96).astype(np.uint8)
    image[36:44, square_left : square_left + 10] = 255
    Image.fromarray(image).save(path)


def _sequence(
    tmp_path: Path,
    frame_indices: tuple[int, ...] = (1, 2),
    ignored_on_first: bool = False,
) -> SequenceData:
    frames = []
    for position, frame_index in enumerate(frame_indices):
        path = tmp_path / f"{frame_index:06d}.png"
        _write_frame(path, square_left=20 + 8 * position)
        ignored = (
            (((8.0, 8.0), (18.0, 8.0), (18.0, 18.0), (8.0, 18.0)),)
            if ignored_on_first and position == 0
            else ()
        )
        frames.append(
            FrameSample(
                sequence_id="method_sequence",
                frame_index=frame_index,
                timestamp=position / 30,
                image_path=path,
                annotations=(),
                ignore_polygons=ignored,
            )
        )
    return SequenceData(
        sequence_id="method_sequence",
        width=128,
        height=96,
        fps=30,
        frames=tuple(frames),
    )


@pytest.mark.parametrize(
    "method_name",
    ["frame_diff", "mog2", "temporal_median", "multiscale"],
)
def test_each_method_returns_one_evidence_map_per_frame(
    method_name,
    synthetic_sequence,
    config,
):
    method = create_method(method_name, config)

    results = method.run(synthetic_sequence, scale=1.0)

    assert tuple(results) == tuple(range(1, 81))
    assert all(item.fused_z.shape == (96, 128) for item in results.values())


def test_mog2_detects_moving_square_after_warmup(
    synthetic_sequence,
    config,
):
    result = create_method("mog2", config, var_threshold=16).run(
        synthetic_sequence,
        scale=1.0,
    )

    assert result[70].fused_z.max() == config.mad_clip
    assert result[1].fused_z.max() == 0


@pytest.mark.parametrize(
    "method_name",
    [
        "frame_diff",
        "mog2",
        "temporal_median",
        "multiscale",
        "multiscale_tubelet",
    ],
)
def test_each_method_scales_frames_isotropically(
    method_name,
    synthetic_sequence,
    config,
):
    method = create_method(method_name, config)

    results = method.run(synthetic_sequence, scale=0.7)

    assert tuple(results) == tuple(range(1, 81))
    assert all(item.fused_z.shape == (67, 90) for item in results.values())


def test_method_specific_channels_and_boundary_support(
    synthetic_sequence,
    config,
):
    frame_diff = create_method("frame_diff", config).run(
        synthetic_sequence,
        scale=1.0,
    )
    temporal = create_method("temporal_median", config).run(
        synthetic_sequence,
        scale=1.0,
    )
    multiscale = create_method("multiscale", config).run(
        synthetic_sequence,
        scale=1.0,
    )

    assert tuple(frame_diff[1].channel_z) == ("d1",)
    assert frame_diff[1].support_indices == (1,)
    assert frame_diff[70].support_indices == (69, 70)
    assert tuple(temporal[1].channel_z) == ("dbg",)
    assert temporal[1].support_indices == tuple(range(1, 17))
    assert temporal[80].support_indices == tuple(range(65, 81))
    assert tuple(multiscale[1].channel_z) == (
        "d1",
        "d3",
        "d7",
        "d15",
        "dbg",
    )
    assert multiscale[1].support_indices == tuple(range(1, 17))
    assert multiscale[80].support_indices == tuple(range(65, 81))


def test_multiscale_tubelet_evidence_is_identical_to_multiscale(
    synthetic_sequence,
    config,
):
    multiscale = create_method("multiscale", config).run(
        synthetic_sequence,
        scale=1.0,
    )
    tubelet = create_method("multiscale_tubelet", config).run(
        synthetic_sequence,
        scale=1.0,
    )

    for frame_index in multiscale:
        assert multiscale[frame_index].support_indices == (
            tubelet[frame_index].support_indices
        )
        for channel_name in multiscale[frame_index].channel_z:
            np.testing.assert_array_equal(
                multiscale[frame_index].channel_z[channel_name],
                tubelet[frame_index].channel_z[channel_name],
            )
        np.testing.assert_array_equal(
            multiscale[frame_index].fused_z,
            tubelet[frame_index].fused_z,
        )
        np.testing.assert_array_equal(
            multiscale[frame_index].fused_score,
            tubelet[frame_index].fused_score,
        )


def test_two_pass_alignment_scales_ignore_mask_and_records_fallbacks(
    tmp_path,
    config,
    monkeypatch,
):
    sequence = _sequence(tmp_path, ignored_on_first=True)
    exclude_masks = []
    returned = (
        AlignmentResult(
            matrix=np.eye(2, 3, dtype=np.float32),
            correlation=0.0,
            used_fallback=True,
            reason="ecc_failed",
        ),
        AlignmentResult(
            matrix=np.eye(2, 3, dtype=np.float32),
            correlation=0.0,
            used_fallback=True,
            reason="low_correlation",
        ),
    )

    def controlled_ecc(reference, moving, cfg, exclude_mask=None):
        del reference, moving, cfg
        exclude_masks.append(exclude_mask.copy())
        return returned[len(exclude_masks) - 1]

    monkeypatch.setattr(
        methods_module,
        "estimate_euclidean_ecc",
        controlled_ecc,
    )
    method = create_method("frame_diff", config)

    results = method.run(sequence, scale=0.7)

    assert len(exclude_masks) == 2
    assert exclude_masks[0].shape == (67, 90)
    assert exclude_masks[0][7, 7]
    assert not exclude_masks[0][30, 20]
    assert np.all(exclude_masks[1][exclude_masks[0]])
    assert np.count_nonzero(exclude_masks[1]) > np.count_nonzero(
        exclude_masks[0],
    )
    assert results[2].fused_z.max() == config.mad_clip
    diagnostic = method.diagnostics[2][0]
    assert diagnostic.reference_index == 2
    assert diagnostic.support_index == 1
    assert diagnostic.mode == "ecc_two_pass"
    assert diagnostic.first_pass.reason == "ecc_failed"
    assert diagnostic.second_pass.reason == "low_correlation"
    assert diagnostic.first_pass.correlation == 0.0
    assert diagnostic.second_pass.correlation == 0.0


def test_mog2_outputs_binary_z_and_identity_diagnostics(
    synthetic_sequence,
    config,
):
    method = create_method("mog2", config, var_threshold=16)

    results = method.run(synthetic_sequence, scale=1.0)

    assert tuple(results[70].channel_z) == ("foreground",)
    assert set(np.unique(results[70].fused_z)) <= {0.0, config.mad_clip}
    np.testing.assert_array_equal(
        results[70].fused_score,
        results[70].fused_z / config.mad_clip,
    )
    assert results[70].support_indices == (70,)
    diagnostic = method.diagnostics[70][0]
    assert diagnostic.mode == "identity"
    assert diagnostic.first_pass is None
    assert diagnostic.second_pass is None


def test_results_are_sorted_by_frame_index_and_evidence_is_read_only(
    tmp_path,
    config,
):
    sequence = _sequence(tmp_path, frame_indices=(2, 1))

    results = create_method("frame_diff", config).run(sequence, scale=1.0)

    assert tuple(results) == (1, 2)
    assert tuple(item.frame_index for item in results.values()) == (1, 2)
    with pytest.raises(TypeError):
        results[3] = results[2]
    with pytest.raises(TypeError):
        results[2].channel_z["extra"] = results[2].fused_z
    for item in results.values():
        arrays = (*item.channel_z.values(), item.fused_z, item.fused_score)
        assert all(not array.flags.writeable for array in arrays)
        for array in arrays:
            with pytest.raises(ValueError):
                array.setflags(write=True)


@pytest.mark.parametrize("scale", [0.0, 0.5, 1.5, np.nan, True])
def test_run_rejects_unplanned_scales(scale, synthetic_sequence, config):
    method = create_method("frame_diff", config)

    with pytest.raises(ValueError, match="scale"):
        method.run(synthetic_sequence, scale=scale)


@pytest.mark.parametrize("name", ["", "optical_flow", "MOG2"])
def test_factory_rejects_unknown_method_names(name, config):
    with pytest.raises(ValueError, match="method"):
        create_method(name, config)


@pytest.mark.parametrize("var_threshold", [8, 12, np.inf, True])
def test_factory_rejects_unplanned_mog2_var_thresholds(
    var_threshold,
    config,
):
    with pytest.raises(ValueError, match="var_threshold"):
        create_method("mog2", config, var_threshold=var_threshold)


def test_mog2_rejects_non_poc_history(config):
    with pytest.raises(ValueError, match="history"):
        create_method("mog2", replace(config, mog2_history=30))

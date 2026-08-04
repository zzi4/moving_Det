from dataclasses import replace

import numpy as np
import pytest

from moving_det.motion.masks import extract_components, threshold_and_clean
from moving_det.motion.tubelets import (
    link_tubelets,
    proposals_for_frame,
    proposals_from_components,
)
from moving_det.models import Tubelet
from tests.helpers import component_at, tubelet_at


def test_threshold_is_inclusive_and_returns_binary_uint8_without_mutation(config):
    fused_z = np.zeros((8, 8), dtype=np.float32)
    fused_z[2, 2:6] = 3.0
    before = fused_z.copy()
    fused_z.flags.writeable = False

    mask = threshold_and_clean(fused_z, 3.0, config)

    expected = np.zeros((8, 8), dtype=np.uint8)
    expected[2, 2:6] = 1
    np.testing.assert_array_equal(mask, expected)
    np.testing.assert_array_equal(fused_z, before)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)) <= {0, 1}


def test_threshold_cleanup_fills_holes_and_removes_small_components(config):
    fused_z = np.zeros((10, 10), dtype=np.float32)
    fused_z[2:5, 2:5] = 4.0
    fused_z[3, 3] = 0.0
    fused_z[8, 8] = 4.0

    mask = threshold_and_clean(fused_z, 4.0, config)

    expected = np.zeros((10, 10), dtype=np.uint8)
    expected[2:5, 2:5] = 1
    np.testing.assert_array_equal(mask, expected)


def test_threshold_cleanup_uses_three_by_three_elliptical_close_once(config):
    fused_z = np.zeros((7, 7), dtype=np.float32)
    fused_z[2, 3:5] = 3.0
    fused_z[4, 2] = 3.0

    mask = threshold_and_clean(fused_z, 3.0, config)

    expected = np.zeros((7, 7), dtype=np.uint8)
    expected[2, 3:5] = 1
    expected[3, 3] = 1
    expected[4, 2] = 1
    np.testing.assert_array_equal(mask, expected)


def test_extract_components_uses_eight_connectivity_and_mean_score(config):
    mask = np.zeros((5, 6), dtype=np.uint8)
    mask[1, 2] = 1
    mask[2, 3] = 1
    score = np.zeros((5, 6), dtype=np.float64)
    score[1, 2] = 2.0
    score[2, 3] = 6.0
    mask_before = mask.copy()
    score_before = score.copy()
    mask.flags.writeable = False
    score.flags.writeable = False

    components = extract_components(
        frame_index=7,
        mask=mask,
        score=score,
        cfg=replace(config, min_component_area=1),
    )

    assert len(components) == 1
    component = components[0]
    assert component.component_id == 1
    assert component.frame_index == 7
    np.testing.assert_array_equal(
        component.points_xy,
        np.array([[2.0, 1.0], [3.0, 2.0]], dtype=np.float32),
    )
    assert component.bbox_xyxy == (2, 1, 4, 3)
    assert component.area == 2
    assert component.mean_score == pytest.approx(4.0)
    np.testing.assert_array_equal(mask, mask_before)
    np.testing.assert_array_equal(score, score_before)


def test_empty_masks_components_links_and_proposals_are_empty(config):
    fused_z = np.zeros((5, 7), dtype=np.float32)

    mask = threshold_and_clean(fused_z, 3.0, config)

    np.testing.assert_array_equal(mask, np.zeros((5, 7), dtype=np.uint8))
    assert extract_components(1, mask, fused_z, config) == ()
    assert link_tubelets({}, config) == ()
    assert proposals_from_components(1, (), (), config) == ()
    assert proposals_for_frame(1, (), (), config) == ()


def test_one_frame_noise_is_not_a_tubelet(config):
    components = {
        1: (),
        2: (component_at(frame=2, x=20, y=20),),
        3: (),
    }

    assert link_tubelets(components, config) == ()


def test_components_do_not_link_across_a_frame_gap(config):
    components = {
        1: (component_at(frame=1, x=20, y=20),),
        3: (component_at(frame=3, x=21, y=20),),
    }

    assert link_tubelets(components, config) == ()


def test_expanded_box_link_threshold_is_inclusive(config):
    components = {
        1: (component_at(frame=1, x=0, y=0, width=2, height=2),),
        2: (component_at(frame=2, x=42, y=0, width=2, height=2),),
    }

    tubelets = link_tubelets(components, config)

    assert len(tubelets) == 1
    assert tuple(c.frame_index for c in tubelets[0].components) == (1, 2)


def test_components_beyond_both_link_thresholds_remain_noise(config):
    components = {
        1: (component_at(frame=1, x=0, y=0, width=2, height=2),),
        2: (component_at(frame=2, x=43, y=0, width=2, height=2),),
    }

    assert link_tubelets(components, config) == ()


def test_large_component_half_diagonal_can_link_components(config):
    components = {
        1: (component_at(frame=1, x=0, y=0, width=100, height=2),),
        2: (component_at(frame=2, x=49, y=50, width=2, height=2),),
    }

    assert len(link_tubelets(components, config)) == 1


def test_all_qualifying_temporal_graph_edges_form_one_component(config):
    components = {
        1: (
            component_at(frame=1, x=20, y=20, component_id=2),
            component_at(frame=1, x=18, y=20, component_id=1),
        ),
        2: (
            component_at(frame=2, x=21, y=20, component_id=2),
            component_at(frame=2, x=19, y=20, component_id=1),
        ),
    }

    tubelets = link_tubelets(components, config)

    assert len(tubelets) == 1
    assert tuple(
        (component.frame_index, component.component_id)
        for component in tubelets[0].components
    ) == ((1, 1), (1, 2), (2, 1), (2, 2))


def test_persistence_counts_distinct_frames(config):
    components = {
        1: (
            component_at(frame=1, x=20, y=20, component_id=1),
            component_at(frame=1, x=21, y=20, component_id=2),
        ),
        2: (component_at(frame=2, x=20, y=20),),
    }

    assert (
        link_tubelets(
            components,
            replace(config, tubelet_min_frames=3),
        )
        == ()
    )


def test_tubelet_ids_and_component_order_are_deterministic(config):
    components = {
        2: (
            component_at(frame=2, x=101, y=20, component_id=2),
            component_at(frame=2, x=21, y=20, component_id=1),
        ),
        1: (
            component_at(frame=1, x=100, y=20, component_id=2),
            component_at(frame=1, x=20, y=20, component_id=1),
        ),
    }

    tubelets = link_tubelets(components, config)

    assert tuple(tubelet.tubelet_id for tubelet in tubelets) == (1, 2)
    assert tuple(
        tuple(
            (component.frame_index, component.component_id)
            for component in tubelet.components
        )
        for tubelet in tubelets
    ) == (((1, 1), (2, 1)), ((1, 2), (2, 2)))


def test_two_neighboring_components_form_padded_obb(config):
    components = {
        1: (component_at(frame=1, x=20, y=20, width=12, height=6),),
        2: (component_at(frame=2, x=25, y=20, width=12, height=6),),
    }

    tubelets = link_tubelets(components, config)
    proposals = proposals_for_frame(2, tubelets, (), config)

    assert len(proposals) == 1
    assert proposals[0].obb.width == pytest.approx(15.0, abs=1.0)
    assert proposals[0].obb.height == pytest.approx(7.5, abs=1.0)


def test_per_frame_proposals_have_deterministic_negative_ids_and_scores(config):
    first = replace(
        component_at(frame=3, x=40, y=20, component_id=2),
        mean_score=5.5,
    )
    second = replace(
        component_at(frame=3, x=20, y=20, component_id=1),
        mean_score=4.5,
    )

    proposals = proposals_from_components(
        frame_index=3,
        components=(first, second),
        ignore_polygons=(),
        cfg=config,
    )

    assert tuple(proposal.tubelet_id for proposal in proposals) == (
        -300001,
        -300002,
    )
    assert tuple(proposal.motion_score for proposal in proposals) == (4.5, 5.5)
    assert all(proposal.frame_index == 3 for proposal in proposals)


def test_tubelet_proposals_only_include_requested_frame(config):
    tubelet = Tubelet(
        tubelet_id=9,
        components=(
            component_at(frame=1, x=20, y=20),
            component_at(frame=2, x=21, y=20),
        ),
    )

    proposals = proposals_for_frame(2, (tubelet,), (), config)

    assert len(proposals) == 1
    assert proposals[0].frame_index == 2
    assert proposals[0].tubelet_id == 9


def test_proposal_inside_ignore_polygon_is_removed(config):
    tubelet = tubelet_at(frame=2, cx=20, cy=20)
    ignore = (((0, 0), (40, 0), (40, 40), (0, 40)),)

    assert proposals_for_frame(2, (tubelet,), ignore, config) == ()


def test_proposal_with_majority_obb_overlap_is_removed_when_center_is_outside(
    config,
):
    tubelet = tubelet_at(frame=2, cx=20, cy=20)
    c_shape = (
        (10, 15),
        (30, 15),
        (30, 18.5),
        (14, 18.5),
        (14, 21.5),
        (30, 21.5),
        (30, 25),
        (10, 25),
    )

    assert proposals_for_frame(2, (tubelet,), (c_shape,), config) == ()


def test_proposal_with_less_than_half_obb_overlap_is_retained(config):
    tubelet = tubelet_at(frame=2, cx=20, cy=20)
    c_shape = (
        (10, 15),
        (30, 15),
        (30, 18.05),
        (12.4, 18.05),
        (12.4, 21.95),
        (30, 21.95),
        (30, 25),
        (10, 25),
    )

    proposals = proposals_for_frame(2, (tubelet,), (c_shape,), config)

    assert len(proposals) == 1

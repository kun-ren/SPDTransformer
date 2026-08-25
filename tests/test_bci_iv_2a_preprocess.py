from __future__ import annotations

from unittest.mock import patch

import numpy as np

from src.datasets.BCICompetitionIV2a_preprocess import (
    BCI_IV_2A_CHANNELS,
    _encode_labels_in_event_order,
    _encode_session_run_labels,
    preprocess_bci_iv_2a_spd,
)
from src.datasets.PhysioNetMI_preprocess import (
    PHYSIONET_BRAIN_REGION_PRESETS,
    resolve_brain_region_indices,
)


def test_physionet_motor_7_preset_allows_intentional_region_overlap():
    channels = list(
        dict.fromkeys(
            channel
            for region in PHYSIONET_BRAIN_REGION_PRESETS["motor_7"].values()
            for channel in region
        )
    )

    names, indices = resolve_brain_region_indices(channels, "motor_7")

    assert names == ["left_motor", "central_motor", "right_motor"]
    assert indices.shape == (3, 7)


def test_bci_preprocessing_matches_region_shape_and_scale_normalization():
    rng = np.random.default_rng(42)
    signals = rng.normal(size=(8, len(BCI_IV_2A_CHANNELS), 40))
    labels = np.asarray(
        ["left_hand", "right_hand", "feet", "tongue"] * 2,
        dtype=np.str_,
    )
    subjects = np.asarray(["A01"] * 4 + ["A02"] * 4, dtype=np.str_)
    runs = np.asarray([1] * 4 + [2] * 4, dtype=np.int16)

    def fake_load_bci_iv_2a_epochs(**_kwargs):
        return {
            "X": signals.copy(),
            "y": labels.copy(),
            "subject": subjects.copy(),
            "run": runs.copy(),
            "metadata": None,
            "events": ["left_hand", "right_hand", "feet", "tongue"],
            "ch_names": list(BCI_IV_2A_CHANNELS),
        }

    with patch(
        "src.datasets.BCICompetitionIV2a_preprocess.load_bci_iv_2a_epochs",
        side_effect=fake_load_bci_iv_2a_epochs,
    ):
        x_spd, y, class_names, subject_labels, run_labels = (
            preprocess_bci_iv_2a_spd(
                filter_bank=[[8, 13], [13, 20]],
                subjects=[1, 2],
                estimator="lwf",
                eps=1.0e-6,
                sfreq=20,
                segment_duration=1.0,
                stride_duration=1.0,
                events="left_hand,right_hand,feet,tongue",
                brain_region_mode="motor_7",
                return_subjects=True,
                return_runs=True,
                moabb_accept_terms=False,
            )
        )

    assert x_spd.shape == (8, 2, 2, 3, 7, 7)
    assert y.tolist() == [0, 1, 2, 3, 0, 1, 2, 3]
    assert class_names == ["left_hand", "right_hand", "feet", "tongue"]
    np.testing.assert_array_equal(subject_labels, subjects)
    np.testing.assert_array_equal(run_labels, runs)
    assert np.isfinite(x_spd).all()

    # Normalization is applied independently inside each frequency-band pass.
    diagonal_sum = np.diagonal(x_spd, axis1=-2, axis2=-1).sum(axis=(1, 3, 4))
    expected_sum = 2 * 3 * 7 * (1.0 + 1.0e-6)
    np.testing.assert_allclose(diagonal_sum, expected_sum, rtol=1.0e-5)


def test_original_event_semantics_are_encoded_in_configured_order():
    labels = ["tongue", "left_hand", "feet", "right_hand"]
    y, names = _encode_labels_in_event_order(
        labels,
        ["left_hand", "right_hand", "feet", "tongue"],
    )

    assert names == ["left_hand", "right_hand", "feet", "tongue"]
    assert y.tolist() == [3, 0, 2, 1]


def test_session_and_run_pairs_get_unique_stable_ids():
    metadata = {
        "session": ["0train", "0train", "1test", "1test"],
        "run": ["0", "1", "0", "1"],
    }

    assert _encode_session_run_labels(metadata).tolist() == [1, 2, 3, 4]

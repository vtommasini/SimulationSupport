# Distributed under the MIT License.
# See LICENSE.txt for details.

import json

import numpy.testing as npt
import pytest
from click.testing import CliRunner

from SimulationSupport.EccentricityControl.InitialOrbitalParameters import (
    initial_orbital_parameters,
    initial_orbital_parameters_command,
)


def test_initial_orbital_parameters():
    target_params = {
        "MassRatio": 1.0,
        "MassA": 0.5,
        "MassB": 0.5,
        "DimensionlessSpinA": [0.0, 0.0, 0.0],
        "DimensionlessSpinB": [0.0, 0.0, 0.0],
        "Eccentricity": 0.0,
    }
    # Expected results are computed from SpEC's ZeroEccParamsFromPN.py
    npt.assert_allclose(
        initial_orbital_parameters(
            target_params,
            separation=20.0,
            orbital_angular_velocity=0.01,
            radial_expansion_velocity=-1.0e-5,
        ),
        [20.0, 0.01, -1.0e-5],
    )
    npt.assert_allclose(
        initial_orbital_parameters(
            target_params,
            separation=16.0,
        ),
        [16.0, 0.014474280975952748, -4.117670632867514e-05],
    )
    npt.assert_allclose(
        initial_orbital_parameters(
            target_params,
            orbital_angular_velocity=0.015,
        ),
        [15.6060791015625, 0.015, -4.541705362753467e-05],
    )
    npt.assert_allclose(
        initial_orbital_parameters(
            {**target_params, "NumOrbits": 20},
        ),
        [16.0421142578125, 0.014419921875000002, -4.0753460821644916e-05],
    )
    npt.assert_allclose(
        initial_orbital_parameters(
            {**target_params, "TimeToMerger": 6000},
        ),
        [16.1357421875, 0.01430025219917298, -3.9831982447244026e-05],
    )


def test_initial_orbital_parameters_pn_requires_zero_eccentricity():
    # PN method only supports zero eccentricity
    target_params = {
        "MassRatio": 1.0,
        "MassA": 0.5,
        "MassB": 0.5,
        "DimensionlessSpinA": [0.0, 0.0, 0.0],
        "DimensionlessSpinB": [0.0, 0.0, 0.0],
        "Eccentricity": 0.1,
        "MeanAnomalyFraction": 0.5,
    }
    with pytest.raises(AssertionError, match="zero eccentricity"):
        initial_orbital_parameters(
            target_params,
            separation=16.0,
            method="PN",
        )


def test_initial_orbital_parameters_gpr_requires_zero_eccentricity(
    gpr_checkpoint_dir,
):
    # GPR method currently only supports zero eccentricity
    target_params = {
        "MassRatio": 1.0,
        "MassA": 0.5,
        "MassB": 0.5,
        "DimensionlessSpinA": [0.0, 0.0, 0.0],
        "DimensionlessSpinB": [0.0, 0.0, 0.0],
        "Eccentricity": 0.1,
        "MeanAnomalyFraction": 0.5,
    }
    with pytest.raises(AssertionError, match="zero eccentricity"):
        initial_orbital_parameters(
            target_params,
            separation=16.0,
            method="GPR",
            gpr_checkpoints={
                "Omega0": str(gpr_checkpoint_dir / "gpr_model_omega.pth")
            },
        )


def test_initial_orbital_parameters_gpr_requires_checkpoints():
    # method = "GPR" requires a non-empty gpr_checkpoints dict
    target_params = {
        "MassRatio": 1.0,
        "MassA": 0.5,
        "MassB": 0.5,
        "DimensionlessSpinA": [0.0, 0.0, 0.0],
        "DimensionlessSpinB": [0.0, 0.0, 0.0],
        "Eccentricity": 0.0,
    }
    with pytest.raises(AssertionError, match="gpr_checkpoints"):
        initial_orbital_parameters(
            target_params,
            separation=16.0,
            method="GPR",
        )


def test_initial_orbital_parameters_gpr_rejects_unknown_quantities(
    gpr_checkpoint_dir,
):
    """
    Test that the keys of the checkpoint files match the parameter names used.
    """
    target_params = {
        "MassRatio": 1.0,
        "MassA": 0.5,
        "MassB": 0.5,
        "DimensionlessSpinA": [0.0, 0.0, 0.0],
        "DimensionlessSpinB": [0.0, 0.0, 0.0],
        "Eccentricity": 0.0,
    }
    with pytest.raises(ValueError, match="Unknown quantity") as excinfo:
        initial_orbital_parameters(
            target_params,
            separation=16.0,
            method="GPR",
            gpr_checkpoints={
                "omega": str(gpr_checkpoint_dir / "gpr_model_omega.pth")
            },
        )
    assert "Omega0" in str(excinfo.value)


def test_initial_orbital_parameters_gpr_rejects_mismatched_checkpoint(
    gpr_checkpoint_dir,
):
    """
    Test that supplying a checkpoint trained for a different quantity is
    prevented, rather than adding to the wrong correction.
    """
    target_params = {
        "MassRatio": 1.0,
        "MassA": 0.5,
        "MassB": 0.5,
        "DimensionlessSpinA": [0.0, 0.0, 0.0],
        "DimensionlessSpinB": [0.0, 0.0, 0.0],
        "Eccentricity": 0.0,
    }
    with pytest.raises(ValueError, match="trained to predict"):
        initial_orbital_parameters(
            target_params,
            separation=16.0,
            method="GPR",
            gpr_checkpoints={
                "Omega0": str(gpr_checkpoint_dir / "gpr_model_adot.pth")
            },
        )


def test_initial_orbital_parameters_gpr_omega_and_adot_correction(
    gpr_checkpoint_dir,
):
    """
    Test the GPR method with the real, trained checkpoints. The
    expected deltas are computed directly from gpr_model_omega.pth
    and gpr_model_adot.pth.
    """
    target_params = {
        "MassRatio": 1.0,
        "MassA": 0.5,
        "MassB": 0.5,
        "DimensionlessSpinA": [0.0, 0.0, 0.0],
        "DimensionlessSpinB": [0.0, 0.0, 0.0],
        "Eccentricity": 0.0,
    }
    pn_separation, pn_omega, pn_adot = (
        16.0,
        0.014474280975952748,
        -4.117670632867514e-05,
    )
    omega_delta = -3.0704395612701774e-05
    adot_delta = 8.766858081799e-05

    separation, omega, adot = initial_orbital_parameters(
        target_params,
        separation=16.0,
        method="GPR",
        gpr_checkpoints={
            "Omega0": str(gpr_checkpoint_dir / "gpr_model_omega.pth"),
            "Adot0": str(gpr_checkpoint_dir / "gpr_model_adot.pth"),
        },
    )

    npt.assert_allclose(separation, pn_separation)
    npt.assert_allclose(omega, pn_omega + omega_delta, rtol=1e-4)
    npt.assert_allclose(adot, pn_adot + adot_delta, rtol=1e-4)


def test_cli_gpr(gpr_checkpoint_dir):
    runner = CliRunner()
    result = runner.invoke(
        initial_orbital_parameters_command,
        [
            "--mass-ratio",
            "1.0",
            "--dimensionless-spin-a",
            "0.0",
            "0.0",
            "0.0",
            "--dimensionless-spin-b",
            "0.0",
            "0.0",
            "0.0",
            "--eccentricity",
            "0.0",
            "--separation",
            "16.0",
            "--method",
            "GPR",
            "--gpr-omega-checkpoint",
            str(gpr_checkpoint_dir / "gpr_model_omega.pth"),
            "--gpr-adot-checkpoint",
            str(gpr_checkpoint_dir / "gpr_model_adot.pth"),
            "--output-json",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    output = json.loads(result.output)
    npt.assert_allclose(
        output["Omega0"],
        0.014474280975952748 - 3.0704395612701774e-05,
        rtol=1e-4,
    )
    npt.assert_allclose(
        output["Adot0"],
        -4.117670632867514e-05 + 8.766858081799e-05,
        rtol=1e-4,
    )

#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.
"""Estimate initial orbital parameters."""

import json
import logging
from typing import Optional, Tuple

import click
import numpy as np
import rich
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


def initial_orbital_parameters(
    target_params: dict,
    separation: Optional[float] = None,
    orbital_angular_velocity: Optional[float] = None,
    radial_expansion_velocity: Optional[float] = None,
    method: str = "PN",
    gpr_checkpoints: Optional[dict] = None,
) -> Tuple[float, float, float]:
    r"""Estimate initial orbital parameters from PN or GPR.

    Estimates initial orbital parameters from a Post-Newtonian (PN) approximation,
    or from a Gaussian Process Regression (GPR) correction to the PN approximation.

    Given the target eccentricity and one other orbital parameter, this
    routine estimates the initial separation ``D_0``, orbital angular velocity
    ``Omega_0``, and radial expansion velocity ``adot_0`` for a binary system.
    The resulting parameters can be fed into an eccentricity control loop to
    refine the starting parameters.

    Parameters
    ----------
    target_params : dict
        Simulation parameters that describe the binary. The dictionary must
        include the following keys:

        * ``"MassRatio"``: Mass ratio :math:`q = M_A / M_B \ge 1`.
        * ``"DimensionlessSpinA"``: Dimensionless spin vector of the larger
          black hole (length 3).
        * ``"DimensionlessSpinB"``: Dimensionless spin vector of the smaller
          black hole (length 3).
        * ``"Eccentricity"``: Target orbital eccentricity. Provide this value
          together with exactly one of the orbital parameters below.

        Optional keys:

        * ``"MeanAnomalyFraction"``: Mean anomaly divided by :math:`2\pi`
          (between 0 and 1). Required for nonzero eccentricity.
        * ``"NumOrbits"``: Desired number of inspiral orbits until merger.
        * ``"TimeToMerger"``: Desired time to merger.
    separation : float, optional
        Coordinate separation ``D_0`` of the black holes.
    orbital_angular_velocity : float, optional
        Orbital angular velocity ``Omega_0``.
    radial_expansion_velocity : float, optional
        Radial expansion velocity ``adot_0``.
    method: str, optional
        Either ``PN`` to compute parameters from the PN approximation,
        or ``GPR`` to apply a learned correction from a trained GPR model
        to the PN approximation
    gpr_checkpoints: dict, optional
        Required when ``method = GPR``. Maps the quantities to correct to the
        path of their trained GPR checkpoint file produced by ``save_gpr_checkpoint``.
        Recognized keys are ``Omega0`` and ``Adot0``. Any quantity without
        an entry is left at its PN value and no correction is applied. ``D_0`` is
        currently never corrected and is always returned at its PN value. Each checkpoint
        is checked against the quantity it is supplied for, so passing a checkpoint trained
        for a different quantity raises a ``ValueError`` instead of silently producing
        a wrong correction.


    Returns
    -------
    tuple[float, float, float]
        A tuple ``(D_0, Omega_0, adot_0)`` with the initial separation,
        orbital angular velocity, and radial expansion velocity.
    """
    # If all orbital parameters are already specified, return early
    if (
        separation is not None
        and orbital_angular_velocity is not None
        and radial_expansion_velocity is not None
    ):
        return separation, orbital_angular_velocity, radial_expansion_velocity

    # Unpack the pieces of target_params we need. Everything is derived from this dict
    # instead of passed as individual arguments, so callers only have to build one
    # dict per system
    mass_ratio = target_params["MassRatio"]
    dimensionless_spin_a = np.asarray(target_params["DimensionlessSpinA"])
    dimensionless_spin_b = np.asarray(target_params["DimensionlessSpinB"])
    eccentricity = target_params["Eccentricity"]
    mean_anomaly_fraction = target_params.get("MeanAnomalyFraction")
    num_orbits = target_params.get("NumOrbits")
    time_to_merger = target_params.get("TimeToMerger")

    # Check input parameters for consistency
    assert eccentricity is not None, (
        "Specify all orbital parameters 'separation',"
        " 'orbital_angular_velocity', and 'radial_expansion_velocity', or"
        " specify an 'eccentricity' plus one orbital parameter."
    )
    if eccentricity != 0.0:
        assert mean_anomaly_fraction is not None, (
            "If you specify a nonzero 'eccentricity' you must also specify a"
            " 'mean_anomaly_fraction'."
        )
    assert radial_expansion_velocity is None, (
        "Can't use the 'radial_expansion_velocity' to compute orbital"
        " parameters. Remove it and choose another orbital parameter."
    )
    assert (
        (separation is not None)
        ^ (orbital_angular_velocity is not None)
        ^ (num_orbits is not None)
        ^ (time_to_merger is not None)
    ), (
        "Specify an 'eccentricity' plus _one_ of the following orbital"
        " parameters: 'separation', 'orbital_angular_velocity', 'num_orbits',"
        " 'time_to_merger'."
    )
    assert method in (
        "PN",
        "GPR",
    ), f"Unknown method '{method}'. Choose either 'PN' or 'GPR'."

    # GPR method. This will be modified later to accept
    # both non-eccentric and eccentric GPR models.
    if method == "GPR":
        assert gpr_checkpoints, (
            "The GPR method requires a 'gpr_checkpoints' dict mapping the"
            " quantities to correct to their trained checkpoint file paths."
        )
        return _initial_orbital_parameters_gpr(
            mass_ratio=mass_ratio,
            dimensionless_spin_a=dimensionless_spin_a,
            dimensionless_spin_b=dimensionless_spin_b,
            eccentricity=eccentricity,
            separation=separation,
            orbital_angular_velocity=orbital_angular_velocity,
            num_orbits=num_orbits,
            time_to_merger=time_to_merger,
            gpr_checkpoints=gpr_checkpoints,
        )

    # Compute the initial orbital parameters from the Post-Newtonian approximation.
    return _initial_orbital_parameters_pn(
        mass_ratio=mass_ratio,
        dimensionless_spin_a=dimensionless_spin_a,
        dimensionless_spin_b=dimensionless_spin_b,
        eccentricity=eccentricity,
        separation=separation,
        orbital_angular_velocity=orbital_angular_velocity,
        num_orbits=num_orbits,
        time_to_merger=time_to_merger,
    )


def _initial_orbital_parameters_pn(
    mass_ratio,
    dimensionless_spin_a,
    dimensionless_spin_b,
    eccentricity,
    separation,
    orbital_angular_velocity,
    num_orbits,
    time_to_merger,
) -> Tuple[float, float, float]:
    """Zero-eccentricity initial orbital parameters from the PN approximation.

    This is the PN-only implementation, which also serves as the baseline guess
    the GPR models learn to correct."""

    assert eccentricity == 0.0, (
        "Initial orbital parameters from PN can currently only be computed for"
        " zero eccentricity."
    )

    # Import functions from SpEC. These functions currently work only for zero
    # eccentricity. We will need to generalize this for eccentric orbits.
    # These functions call old Fortran code (LSODA) through
    # scipy.integrate.odeint, which leads to lots of noise in stdout. We should
    # modernize them to use scipy.integrate.solve_ivp.
    from SimulationSupport.EccentricityControl.ZeroEccParamsFromPN import (
        nOrbitsAndTotalTime,
        omegaAndAdot,
    )

    # If the caller specifies a desired number of orbits or time to merger
    # instead of an orbital angular velocity, root-find for the omega0 that
    # produces it.
    if num_orbits is not None or time_to_merger is not None:
        opt_result = minimize(
            lambda x: (
                abs(
                    nOrbitsAndTotalTime(
                        q=mass_ratio,
                        chiA0=dimensionless_spin_a,
                        chiB0=dimensionless_spin_b,
                        omega0=x[0],
                    )[0 if num_orbits is not None else 1]
                    - (num_orbits if num_orbits is not None else time_to_merger)
                )
            ),
            x0=[0.01],
            method="Nelder-Mead",
        )
        if not opt_result.success:
            raise ValueError(
                "Failed to find an orbital angular velocity that gives the"
                " desired number of orbits or time to merger. Error:"
                f" {opt_result.message}"
            )
        orbital_angular_velocity = opt_result.x[0]
        logger.debug(
            f"Found orbital angular velocity: {orbital_angular_velocity}"
        )

    # Given an orbital angular velocity, either passed in directly, or solved for above,
    # root-find for the coordinate separation that produces it under the PN approximation
    if orbital_angular_velocity is not None:
        opt_result = minimize(
            lambda x: abs(
                omegaAndAdot(
                    r=x[0],
                    q=mass_ratio,
                    chiA=dimensionless_spin_a,
                    chiB=dimensionless_spin_b,
                    rPrime0=1.0,  # Choice also made in SpEC
                )[0]
                - orbital_angular_velocity
            ),
            x0=[10.0],
            method="Nelder-Mead",
        )
        if not opt_result.success:
            raise ValueError(
                "Failed to find a separation that gives the desired orbital"
                f" angular velocity. Error: {opt_result.message}"
            )
        separation = opt_result.x[0]
        logger.debug(f"Found initial separation: {separation}")

    # Now that we have a separation, either passed in directly, or solved for above,
    # find the radial expansion velocity at that separation
    new_orbital_angular_velocity, radial_expansion_velocity = omegaAndAdot(
        r=separation,
        q=mass_ratio,
        chiA=dimensionless_spin_a,
        chiB=dimensionless_spin_b,
        rPrime0=1.0,
    )
    if orbital_angular_velocity is None:
        orbital_angular_velocity = new_orbital_angular_velocity
    else:
        assert np.isclose(
            new_orbital_angular_velocity, orbital_angular_velocity, rtol=1e-4
        ), (
            "Orbital angular velocity is inconsistent with separation."
            " Maybe the rootfind failed to reach sufficient accuracy."
        )

    # Estimate number of orbits and time to merger
    num_orbits, time_to_merger = nOrbitsAndTotalTime(
        q=mass_ratio,
        chiA0=dimensionless_spin_a,
        chiB0=dimensionless_spin_b,
        omega0=orbital_angular_velocity,
    )
    logger.info(
        "Selected approximately circular orbit. Number of orbits:"
        f" {num_orbits:g}. Time to merger: {time_to_merger:g} M."
    )
    return separation, orbital_angular_velocity, radial_expansion_velocity


# Map the user-facing quantity names to the 'output_name' stored inside the
# trained GPR checkpoints. The checkpoint names are decided within the training
# pipelines, so we map them here instead of renaming them. This can be changed in the future.
_CHECKPOINT_OUTPUT_NAMES = {"Omega0": "omega", "Adot0": "adot"}


def _apply_gpr_correction(
    quantity_name, baseline_value, available_values, checkpoint_path
):
    """Load a GPR checkpoint and add its predicted correction to a PN baseline.

    Note: GPR checkpoints predict a correction to the PN baseline, not the direct
    quantity itself.
    """
    from SimulationSupport.gpr import (
        load_gpr_checkpoint,
        predict_with_gpr_model,
    )

    model, likelihood, meta = load_gpr_checkpoint(checkpoint_path)
    # Guard against a checkpoint being passed for the wrong quantity, which
    # would add the wrong delta and silently produce an incorrect number.
    expected_output = _CHECKPOINT_OUTPUT_NAMES[quantity_name]
    if meta["output_name"] != expected_output:
        raise ValueError(
            f"Checkpoint '{checkpoint_path}' was trained to predict"
            f" '{meta['output_name']}', but it is being applied to"
            f" '{quantity_name}', which expects a checkpoint predicting"
            f" '{expected_output}'. Check that the checkpoint matches the"
            " quantity."
        )

    # Assemble a raw feature array, in the order the GPR checkpoint expects
    try:
        raw_x = [available_values[name] for name in meta["input_features"]]
    except KeyError as missing_feature:
        raise KeyError(
            f"GPR checkpoint expects input feature {missing_feature},"
            " which is not available. Available features:"
            f" {sorted(available_values.keys())}. Update the"
            " 'available_values' mapping in this module to match the"
            " checkpoint's input_features metadata."
        ) from missing_feature

    raw_x = np.asarray([raw_x], dtype=float)
    delta_mean, delta_std = predict_with_gpr_model(raw_x, model, likelihood)
    # The noneccentric GPR predicts a correction (delta), to the PN approximation.
    # This value is then added to the PN baseline to get the final corrected quantity.
    corrected_value = baseline_value + float(delta_mean[0])
    logger.debug(
        f"GPR correction for {quantity_name}: baseline={baseline_value:g},"
        f" delta={float(delta_mean[0]):g} +/- {float(delta_std[0]):g},"
        f" corrected={corrected_value:g}"
    )
    return corrected_value


def _initial_orbital_parameters_gpr(
    mass_ratio,
    dimensionless_spin_a,
    dimensionless_spin_b,
    eccentricity,
    separation,
    orbital_angular_velocity,
    num_orbits,
    time_to_merger,
    gpr_checkpoints,
) -> Tuple[float, float, float]:
    """Zero-eccentricity initial orbital parameters, PN baseline, and
    GPR correction.
    """
    assert eccentricity == 0.0, (
        "Initial orbital parameters from GPR can currently only be computed for"
        " zero eccentricity."
    )
    # Start from the baseline PN guess, which the GPR models are trained to correct
    pn_separation, pn_omega, pn_adot = _initial_orbital_parameters_pn(
        mass_ratio=mass_ratio,
        dimensionless_spin_a=dimensionless_spin_a,
        dimensionless_spin_b=dimensionless_spin_b,
        eccentricity=eccentricity,
        separation=separation,
        orbital_angular_velocity=orbital_angular_velocity,
        num_orbits=num_orbits,
        time_to_merger=time_to_merger,
    )
    # Feature names match SimulationSupport.gpr (see the GPR tutorial notebook
    # for a detailed explanation of how to train, save, and load the GPR model
    # with real data). Each checkpoint selects the subset of these values it
    # was trained on via its 'input_features' metadata. The aligned-spin example
    # checkpoints in Examples use 'initial_separation', 'initial_mass_ratio',
    # 'initial_dimensionless_spin1_z', and 'initial_dimensionless_spin2_z'.
    available_values = {
        "initial_separation": pn_separation,
        "initial_mass_ratio": mass_ratio,
        "initial_dimensionless_spin1_x": dimensionless_spin_a[0],
        "initial_dimensionless_spin1_y": dimensionless_spin_a[1],
        "initial_dimensionless_spin1_z": dimensionless_spin_a[2],
        "initial_dimensionless_spin2_x": dimensionless_spin_b[0],
        "initial_dimensionless_spin2_y": dimensionless_spin_b[1],
        "initial_dimensionless_spin2_z": dimensionless_spin_b[2],
        "pn_guess_omega": pn_omega,
        "pn_guess_adot": pn_adot,
    }

    corrected = {"Omega0": pn_omega, "Adot0": pn_adot}
    for quantity_name, checkpoint_path in gpr_checkpoints.items():
        if quantity_name not in corrected:
            raise ValueError(
                f"Unknown quantity `{quantity_name}` in `gpr_checkpoints`."
                f" Expected one of {','.join(sorted(corrected))}."
            )
        corrected[quantity_name] = _apply_gpr_correction(
            quantity_name,
            corrected[quantity_name],
            available_values,
            checkpoint_path,
        )
    orbital_angular_velocity = corrected["Omega0"]
    radial_expansion_velocity = corrected["Adot0"]
    logger.info(
        "Selected approximately circular orbit using GPR corrected PN guess."
        f" D0={pn_separation:g}, Omega0={orbital_angular_velocity:g},"
        f" Adot0={radial_expansion_velocity:g}."
    )
    return pn_separation, orbital_angular_velocity, radial_expansion_velocity


# CLI
# The function can be imported and called from Python directly, or it can be called with the CLI.
@click.command(
    name="initial-orbital-parameters",
)
@click.option(
    "--mass-ratio",
    "-q",
    type=float,
    required=True,
    help=r"Mass ratio, q = M_A / M_B \ge 1, of the two black holes.",
)
@click.option(
    "--dimensionless-spin-a",
    nargs=3,
    type=float,
    required=True,
    help=(
        "Dimensionless spin vector of the larger black hole, for example,"
        "written as '--dimensionless-spin-a 0.0 0.1 0.1'."
    ),
)
@click.option(
    "--dimensionless-spin-b",
    nargs=3,
    type=float,
    required=True,
    help=(
        "Dimensionless spin vector of the smaller black hole, for example,"
        " written as '--dimensionless-spin-b 0.0 0.0 0.1'."
    ),
)
@click.option(
    "--eccentricity",
    "-e",
    type=float,
    required=True,
    help="Desired orbital eccentricity.",
)
@click.option(
    "--mean-anomaly-fraction",
    type=float,
    help=(
        "Mean anomaly divided by 2pi (between 0 and 1). Required if"
        " eccentricity is nonzero."
    ),
)
@click.option(
    "--separation",
    "-D",
    type=float,
    help="Coordinate separation, D_0, between the black holes.",
)
@click.option(
    "--orbital-angular-velocity",
    "-w",
    type=float,
    help="Orbital angular velocity, Omega_0.",
)
@click.option(
    "--num-orbits",
    type=float,
    help="Desired number of orbits until merger.",
)
@click.option("--time-to-merger", type=float, help="Desired time until merger.")
@click.option(
    "--method",
    type=click.Choice(["PN", "GPR"]),
    default="PN",
    show_default=True,
    help=(
        "Compute from PN or from GPR, which applies a learned"
        " correction to the PN baseline."
    ),
)
@click.option(
    "--gpr-omega-checkpoint",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help="Path to a trained GPR checkpoint providing Omega0 corrections.",
)
@click.option(
    "--gpr-adot-checkpoint",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help="Path to a trained GPR checkpoint providing Adot0 corrections.",
)
@click.option(
    "--output-json",
    is_flag=True,
    help="Print the result as a JSON file instead of text.",
)
def initial_orbital_parameters_command(
    mass_ratio,
    dimensionless_spin_a,
    dimensionless_spin_b,
    eccentricity,
    mean_anomaly_fraction,
    separation,
    orbital_angular_velocity,
    num_orbits,
    time_to_merger,
    method,
    gpr_omega_checkpoint,
    gpr_adot_checkpoint,
    output_json,
):
    """Estimate the initial orbital parameters for a BBH evolution.

    Estimates the initial coordinate separation, D_0, orbital angular velocity, Omega_0, and
    radial expansion velocity, adot_0, from a Post-Newtonian approximation, optionally
    corrected by a trained Gaussian Process Regression (GPR) model.

    Specify the target eccentricity and either '--separation', '--orbital-angular-velocity',
    '--num-orbits', or '--time-to-merger'.
    """
    _rich_traceback_guard = True

    target_params = {
        "MassRatio": mass_ratio,
        "DimensionlessSpinA": list(dimensionless_spin_a),
        "DimensionlessSpinB": list(dimensionless_spin_b),
        "Eccentricity": eccentricity,
    }
    if mean_anomaly_fraction is not None:
        target_params["MeanAnomalyFraction"] = mean_anomaly_fraction
    if num_orbits is not None:
        target_params["NumOrbits"] = num_orbits
    if time_to_merger is not None:
        target_params["TimeToMerger"] = time_to_merger

    gpr_checkpoints = None
    if method == "GPR":
        gpr_checkpoints = {}
        if gpr_omega_checkpoint:
            gpr_checkpoints["Omega0"] = gpr_omega_checkpoint
        if gpr_adot_checkpoint:
            gpr_checkpoints["Adot0"] = gpr_adot_checkpoint
        if not gpr_checkpoints:
            raise click.UsageError(
                "'--method GPR' requires either '--gpr-omega-checkpoint',"
                " '--gpr-adot-checkpoint', or both."
            )

    D0, Omega0, Adot0 = initial_orbital_parameters(
        target_params,
        separation=separation,
        orbital_angular_velocity=orbital_angular_velocity,
        method=method,
        gpr_checkpoints=gpr_checkpoints,
    )

    if output_json:
        print(
            json.dumps({"D0": D0, "Omega0": Omega0, "Adot0": Adot0}, indent=2)
        )
    else:
        rich.print(f"D0     = {D0}")
        rich.print(f"Omega0 = {Omega0}")
        rich.print(f"Adot0  = {Adot0}")

    return D0, Omega0, Adot0


if __name__ == "__main__":
    initial_orbital_parameters_command(help_option_names=["-h", "--help"])

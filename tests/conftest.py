# Distributed under the MIT License.
# See LICENSE.txt for details.

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def gpr_checkpoint_dir():
    """Directory containing example trained GPR checkpoints,
    used for in Test_InitialOrbitalParameters.py."""
    return (
        REPO_ROOT
        / "src"
        / "SimulationSupport"
        / "EccentricityControl"
        / "Examples"
    )

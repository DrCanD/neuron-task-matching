"""
neuron-task-matching: experiment harness for the cross-domain double dissociation study.

Importing this package ensures the physics-motivated spiking-neuron library
(dikmen-spiking-neurons, which exposes NeuronRegistry) is importable, installing it
from GitHub on first use if necessary, exactly as the original Colab runs did.
"""
import sys
import subprocess

__version__ = "1.1.0"


def _ensure_dikmen():
    for k in [k for k in sys.modules if k.startswith("dikmen")]:
        del sys.modules[k]
    try:
        import dikmen_neurons  # noqa: F401
    except ImportError:
        url = "git+https://github.com/DrCanD/dikmen-spiking-neurons.git"
        print("[setup] installing dikmen-spiking-neurons from GitHub ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", url], check=True)
        for k in [k for k in sys.modules if k.startswith("dikmen")]:
            del sys.modules[k]
        import dikmen_neurons  # noqa: F401


_ensure_dikmen()

from . import core  # noqa: E402,F401

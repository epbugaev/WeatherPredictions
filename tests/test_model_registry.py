"""Regression tests for the active model registry declarations.

The training entry point discovers architectures by importing ``Models`` and
looking up constructors in ``utils.registry.MODELS``.  Importing the package
requires the full ML stack, so this lightweight test parses the registry
declarations directly and documents the post-cleanup contract: only IAM4VP,
SimVP, and PredRNN-family architectures are registered for YAML training.
"""

import ast
from pathlib import Path
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_INIT = REPO_ROOT / "Models" / "__init__.py"


def registered_model_names() -> list[str]:
    """Return model keys declared through ``register_model("name")`` calls."""
    tree = ast.parse(MODELS_INIT.read_text())
    names: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Call):
            continue
        register_call = node.func
        if not isinstance(register_call.func, ast.Name):
            continue
        if register_call.func.id != "register_model" or not register_call.args:
            continue
        name_arg = register_call.args[0]
        if isinstance(name_arg, ast.Constant) and isinstance(name_arg.value, str):
            names.append(name_arg.value)
    return sorted(names)


class ModelRegistryTest(unittest.TestCase):
    """Validate that only active thesis architectures are public training keys."""

    def test_only_active_models_are_registered(self) -> None:
        """The registry should expose IAM4VP, SimVP, and PredRNN-family models."""
        self.assertEqual(
            registered_model_names(),
            ["PI-IAM4VP", "PredRNN", "PredRNNv2", "SimVP"],
        )


if __name__ == "__main__":
    unittest.main()

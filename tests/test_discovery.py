import importlib
import sys
import unittest
from unittest.mock import patch

from sde.core.definitions import DatasetDefinition
from sde.discovery import discover_and_register_components
from sde.registry import registry

# Capture the real import_module function before it's patched in the tests
_real_import = importlib.import_module


class TestDiscovery(unittest.TestCase):
    def setUp(self):
        # Reset the registry before each test
        registry.challenges.clear()
        registry.models.clear()
        registry.schedulers.clear()
        registry.strict_mode = False

        # Unload all component modules to ensure a clean slate for discovery
        modules_to_unload = [
            k
            for k in sys.modules
            if k.startswith("sde.challenges")
            or k.startswith("sde.models")
            or k.startswith("sde.exploration")
        ]
        for k in modules_to_unload:
            del sys.modules[k]

    def test_discovery_and_registration(self):
        """Test that components are discovered and registered correctly."""
        discover_and_register_components()

        self.assertIn("MNIST", registry.list_challenges())
        self.assertIn("CIFAR-10", registry.list_challenges())
        self.assertIn("Fashion-MNIST", registry.list_challenges())

        mnist_challenge = registry.get_challenge("MNIST")
        self.assertIsInstance(mnist_challenge, DatasetDefinition)

    @patch("importlib.import_module")
    def test_discovery_with_partial_import_error(self, mock_import_module):
        """Test that discovery handles a single module import error gracefully."""

        def import_module_side_effect(name):
            if "cifar10" in name:
                raise ImportError("Simulated import error for cifar10")
            return _real_import(name)

        mock_import_module.side_effect = import_module_side_effect

        discover_and_register_components()

        self.assertIn("MNIST", registry.list_challenges())
        self.assertIn("Fashion-MNIST", registry.list_challenges())
        self.assertNotIn("CIFAR-10", registry.list_challenges())

    def test_get_challenge(self):
        """Test retrieving a registered challenge."""
        discover_and_register_components()
        challenge = registry.get_challenge("MNIST")
        self.assertEqual(challenge.name, "MNIST")

    def test_metadata_is_registered(self):
        """Test that version and dependency metadata is registered."""
        discover_and_register_components()
        challenge_component = registry.challenges["MNIST"]
        self.assertEqual(challenge_component.version, "1.0.0")
        self.assertEqual(
            challenge_component.dependencies, {"torchvision": ">=0.15.0"}
        )

    def test_strict_mode_raises_error_on_duplicate(self):
        """Test that strict mode raises an error on duplicate registration."""
        from sde.core.definitions import DatasetType

        registry.strict_mode = True

        # Create a dummy challenge to register
        dummy_challenge = DatasetDefinition(
            name="dummy_challenge",
            type=DatasetType.IMAGE_CLASSIFICATION,
            description="A test challenge.",
            loader_factory=lambda: (None, None),
            input_shape=(1, 1, 1),
            output_shape=1,
            loss_function_factory=lambda: None,
            performance_metric_name="accuracy",
        )

        # Register it once, which should succeed
        registry.register_challenge(dummy_challenge, origin_file="test_file.py")

        # Try to register it again, which should fail in strict mode
        with self.assertRaises(ValueError):
            registry.register_challenge(dummy_challenge, origin_file="test_file.py")


if __name__ == "__main__":
    unittest.main()

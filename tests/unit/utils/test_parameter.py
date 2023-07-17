import os
import unittest

from council.utils import MissingEnvVariableException, EnvVariableValueException
from council.utils.parameter import ParameterValueException, Parameter


def tv(x: float):
    """
    Temperature is an Optional float valid between 0.0 and 2.0, default value 0.0
    """
    if x < 0.0 or x > 2.0:
        raise Exception("must be in the range [0.0..2.0]")


class TestParameter(unittest.TestCase):
    def test_default_optional_parameter_not_set(self) -> None:
        temperature: Parameter[float] = Parameter.float(name="temperature", required=False, default=1.0, validator=tv)
        self.assertEqual(temperature.unwrap(), 1.0)

    def test_default_required_parameter_not_set(self) -> None:
        temperature: Parameter[float] = Parameter.float(name="temperature", required=True, default=0.0, validator=tv)
        with self.assertRaises(MissingEnvVariableException):
            temperature.from_env("MISSING_LLM_TEMPERATURE")

    def test_parameter_set_to_valid_value(self) -> None:
        os.environ["TEST_LLM_TEMPERATURE"] = "1.23456"

        temperature: Parameter[float] = Parameter.float(name="temperature", required=False, default=0.0, validator=tv)
        temperature.from_env("TEST_LLM_TEMPERATURE")
        self.assertEqual(temperature.unwrap(), 1.23456)

    def test_parameter_set_to_invalid_value(self) -> None:
        os.environ["TEST_LLM_TEMPERATURE"] = "4"

        temperature: Parameter[float] = Parameter.float(name="temperature", required=False, default=0.0, validator=tv)
        with self.assertRaises(ParameterValueException) as cm:
            temperature.from_env("TEST_LLM_TEMPERATURE")
        self.assertEqual(
            str(cm.exception),
            "'temperature' parameter value '4.0' is invalid. Value must be must be in the range [0.0..2.0]",
        )

    def test_parameter_set_to_invalid_type_value(self) -> None:
        os.environ["TEST_LLM_TEMPERATURE"] = "4Gb"

        temperature: Parameter[float] = Parameter.float(name="temperature", required=False, default=0.0, validator=tv)
        with self.assertRaises(EnvVariableValueException) as cm:
            temperature.from_env("TEST_LLM_TEMPERATURE")
        self.assertIsInstance(cm.exception.__cause__, ValueError)

    def test_parameter_default_none_value(self) -> None:
        os.environ["TEST_LLM_TEMPERATURE"] = "4Gb"

        temperature: Parameter[float] = Parameter.float(name="temperature", required=False, validator=tv)
        self.assertTrue(temperature.is_none())

    def test_parameter_no_validation(self) -> None:
        os.environ["TEST_LLM_TEMPERATURE"] = "9876.54321"

        temperature: Parameter[float] = Parameter.float(name="temperature", required=False)
        temperature.from_env("TEST_LLM_TEMPERATURE")
        self.assertEqual(temperature.unwrap(), 9876.54321)

    def test_parameter_set_to_invalid_default_value(self) -> None:
        with self.assertRaises(ParameterValueException):
            _ = Parameter.float(name="temperature", required=False, default=-1.0, validator=tv)

import unittest

from council.utils import MissingEnvVariableException, EnvVariableValueException, OsEnviron
from council.utils.parameter import ParameterValueException, Parameter, Undefined


def tv(x: float):
    """
    Temperature is an Optional float valid between 0.0 and 2.0, default value 0.0
    """
    if x < 0.0 or x > 2.0:
        raise ValueError("must be in the range [0.0..2.0]")


class TestParameter(unittest.TestCase):
    def test_default_optional_parameter_set(self) -> None:
        temperature: Parameter[float] = Parameter.float(name="temperature", required=False, default=1.0, validator=tv)
        self.assertEqual(temperature.unwrap(), 1.0)
        self.assertEqual("Parameter(optional) `temperature` with value `1.0`. Default value `1.0`.", f"{temperature}")

    def test_set_valid_value(self) -> None:
        temperature: Parameter[float] = Parameter.float(name="temperature", required=False, value=0.123, validator=tv)
        self.assertEqual(temperature.unwrap(), 0.123)
        self.assertEqual("Parameter(optional) `temperature` with value `0.123`.", f"{temperature}")

    def test_set_to_invalid_value(self) -> None:
        with self.assertRaises(ParameterValueException):
            _ = Parameter.float(name="temperature", required=False, value=-1.0, validator=tv)

    def test_set_to_invalid_default_value(self) -> None:
        with self.assertRaises(ParameterValueException):
            _ = Parameter.float(name="temperature", required=False, default=-1.0, validator=tv)

    def test_from_env_required_parameter_not_set(self) -> None:
        temperature: Parameter[float] = Parameter.float(name="temperature", required=True, default=0.0, validator=tv)
        with self.assertRaises(MissingEnvVariableException):
            temperature.from_env("MISSING_LLM_TEMPERATURE")

    def test_default_none_value(self) -> None:
        temperature: Parameter[float] = Parameter.float(name="temperature", required=False, validator=tv)
        self.assertTrue(temperature.is_none())
        self.assertEqual("Parameter(optional) `temperature` with undefined value.", f"{temperature}")

    def test_from_env_set_to_valid_value(self) -> None:
        temperature: Parameter[float] = Parameter.float(name="temperature", required=False, default=0.0, validator=tv)
        with OsEnviron("TEST_LLM_TEMPERATURE", "1.234"):
            temperature.from_env("TEST_LLM_TEMPERATURE")
        self.assertEqual(temperature.unwrap(), 1.234)
        self.assertEqual("Parameter(optional) `temperature` with value `1.234`. Default value `0.0`.", f"{temperature}")

    def test_from_env_set_to_invalid_value(self) -> None:
        temperature: Parameter[float] = Parameter.float(name="temperature", required=False, default=0.0, validator=tv)
        with OsEnviron("TEST_LLM_TEMPERATURE", "4"), self.assertRaises(ParameterValueException) as cm:
            temperature.from_env("TEST_LLM_TEMPERATURE")
        self.assertEqual(
            str(cm.exception),
            "'temperature' parameter value '4.0' is invalid. Value must be must be in the range [0.0..2.0]",
        )

    def test_from_env_set_to_invalid_type_value(self) -> None:
        temperature: Parameter[float] = Parameter.float(name="temperature", required=False, default=0.0, validator=tv)
        with OsEnviron("TEST_LLM_TEMPERATURE", "0.5Gb"), self.assertRaises(EnvVariableValueException) as cm:
            temperature.from_env("TEST_LLM_TEMPERATURE")
        self.assertIsInstance(cm.exception.__cause__, ValueError)

    def test_from_env_no_validation(self) -> None:
        temperature: Parameter[float] = Parameter.float(name="temperature", required=False)
        with OsEnviron("TEST_LLM_TEMPERATURE", "9876.54321"):
            temperature.from_env("TEST_LLM_TEMPERATURE")
        self.assertEqual(temperature.unwrap(), 9876.54321)

    def test_equal(self) -> None:
        temperature1: Parameter[float] = Parameter.float(name="temperature1", required=False, value=12.34)
        temperature2: Parameter[float] = Parameter.float(name="temperature2", required=True, value=12.34)
        temperature3: Parameter[float] = Parameter.float(name="temperature3", required=True, value=43.21)

        self.assertTrue(temperature1 == temperature2)
        self.assertTrue(temperature1 == 12.34)

        self.assertFalse(temperature1 == temperature3)
        self.assertFalse(temperature1 == 43.21)
        self.assertFalse(temperature1 == Undefined())

import os
import unittest

from council.utils import (
    must_read_env_str,
    read_env_str,
    read_env_int,
    read_env_float,
    read_env_bool,
    MissingEnvVariableException,
    EnvVariableValueException,
    OsEnviron,
)


class TestPrompt(unittest.TestCase):
    def test_env(self):
        os.environ["STR"] = "VALUE"
        ev = must_read_env_str("STR")
        self.assertEqual(ev, "VALUE")

        os.environ["INT"] = "2"
        ev = read_env_int("INT").unwrap()
        self.assertEqual(ev, 2)

        os.environ["FLOAT"] = "1.23456"
        ev = read_env_float("FLOAT").unwrap()
        self.assertEqual(ev, 1.23456)

        os.environ["T_BOOL"] = "True"
        ev = read_env_bool("T_BOOL").unwrap()
        self.assertEqual(ev, True)

        os.environ["F_BOOL"] = "False"
        ev = read_env_bool("F_BOOL").unwrap()
        self.assertEqual(ev, False)

    def test_missing_env_with_default(self):
        ev = read_env_int("MISSING_INT", required=False, default=2).unwrap()
        self.assertEqual(ev, 2)

    def test_missing_env(self):
        r = read_env_str("MISSING", required=False)
        self.assertTrue(r.is_none())

        with self.assertRaises(MissingEnvVariableException):
            _ = read_env_str("MISSING")

    def test_invalid_env(self):
        os.environ["INVALID_FLOAT"] = "2Gb"
        with self.assertRaises(EnvVariableValueException):
            _ = read_env_int("INVALID_FLOAT").unwrap()

        os.environ["INVALID_BOOL"] = "Tue"
        with self.assertRaises(EnvVariableValueException):
            _ = read_env_int("INVALID_BOOL").unwrap()

    def test_os_env_new_var(self):
        self.assertIsNone(os.environ.get("NEW_VAR", None))
        with OsEnviron("NEW_VAR", "I'm here"):
            self.assertEqual(os.environ["NEW_VAR"], "I'm here")
        self.assertIsNone(os.environ.get("NEW_VAR", None))

    def test_os_env_overwrite_existing(self):
        os.environ["EXISTING_VAR"] = "hi"
        self.assertEqual(os.environ["EXISTING_VAR"], "hi")
        with OsEnviron("NEW_VAR", "hey"):
            self.assertEqual(os.environ["NEW_VAR"], "hey")
        self.assertEqual(os.environ["EXISTING_VAR"], "hi")

        with OsEnviron("NEW_VAR"):
            self.assertIsNone(os.environ.get("NEW_VAR", None))
        self.assertEqual(os.environ["EXISTING_VAR"], "hi")

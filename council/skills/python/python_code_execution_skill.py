"""

A module for executing Python code within a chat context. It provides an execution skill that allows running Python code provided in chat messages, capturing their output, and handling errors accordingly.

This module defines:
- `PYTHON_EXECUTABLE`: The path to the Python executable to use, configurable via environment variable.
- `PythonCodeExecutionSkill`: A skill class that executes Python code from chat messages.

Classes:
    PythonCodeExecutionSkill(SkillBase): A skill for executing Python code in a chat interaction

Functions:
    <None>

Exceptions:
    <None>


"""
import os
import subprocess
from typing import Mapping, Optional

from council.contexts import ChatMessage, SkillContext
from council.skills import SkillBase

from .llm_helper import extract_code_block

PYTHON_EXECUTABLE = os.environ.get("COUNCIL_PYTHON_EXECUTABLE", "python")


class PythonCodeExecutionSkill(SkillBase):
    """
    A class to execute Python code strings within a skill-based system and manage its output.
    This class provides a method for executing Python code taken from a chat context and handles the
    process of running the code in a separate subprocess, capturing its output and errors, and
    optionally decoding the standard output stream. It extends a base class for skills and can be
    used in a system where multiple skills are managed and executed.
    
    Attributes:
        _env_var (Mapping[str, str], optional):
             A dictionary with environment variables
            for the subprocess running the code. Defaults to a copy of the current environment
            variables extended with any additional variables provided.
        _decode_stdout (bool):
             A flag to determine if the standard output bytes should be decoded.
            Defaults to True.
    
    Methods:
        __init__(env_var:
             Optional[Mapping[str, str]], decode_stdout: bool): Initializes the
            PythonCodeExecutionSkill with optional environment variables and a flag to decode stdout.
        execute(context:
             SkillContext): Executes the Python code from the last message in the context,
            logs various debugging information, and returns a ChatMessage indicating the result of
            the execution including success or error information, output, and return code.

    """

    def __init__(self, env_var: Optional[Mapping[str, str]] = None, decode_stdout: bool = True):
        """
        Initializes the Python code runner with optional environment variables and stdout decoding behavior.
        This constructor takes optional environment variables to modify the subprocess environment when running Python code.
        It also allows for the standard output to be decoded based on a boolean value.
        
        Args:
            env_var (Optional[Mapping[str, str]]):
                 A mapping of environment variable names to their values
                that should be set for the subprocess. If not specified, the subprocess will inherit
                the current process's environment variables.
            decode_stdout (bool):
                 A flag indicating whether to decode the standard output from the
                subprocess. If True, the output will be decoded; otherwise, it will be treated as binary data.
                The method copies the existing environment variables in the system via the `os.environ.copy()` function
                and then updates them with any additional environment variables specified in the `env_var` parameter.
                The `decode_stdout` argument determines how the standard output is handled by the code runner. If it's set
                to True (which is the default behavior), the standard output will be decoded from bytes to a string using
                the default system encoding. If False, the output will remain as a byte string.
            

        """

        super().__init__("python code runner")
        self._env_var = os.environ.copy()
        self._env_var.update(env_var or {})
        self._decode_stdout = decode_stdout

    def execute(self, context: SkillContext) -> ChatMessage:
        """
        Executes a block of Python code within a given context.
        This method parses the last chat message from a context, extracts a block of Python code flagged as 'python',
        and runs the extracted code using a Python subprocess. The execution details, such as return code, standard output,
        and standard error, are logged and encapsulated in a data object which is used to build either a success or error
        ChatMessage based on the execution result. Optionally, the standard output can be decoded based on an instance variable.
        
        Args:
            context (SkillContext):
                 The context containing information about the last message and the execution environment.
        
        Returns:
            (ChatMessage):
                 A ChatMessage object containing the result of the code execution, along with additional execution details.
        
        Raises:
            LLMParsingException:
                 If no code block can be found in the last message or the block cannot be extracted.

        """
        last_message = context.try_last_message.unwrap("last message")
        python_code = extract_code_block(last_message.message, "python")
        context.logger.debug(f"running python code: \n {python_code}")
        execution = subprocess.run([PYTHON_EXECUTABLE, "-c", python_code], capture_output=True, env=self._env_var)

        return_code = execution.returncode
        stdout_bytes = execution.stdout
        stderr = execution.stderr.decode()
        data = {
            "return_code": return_code,
            "stderr": stderr,
        }

        context.logger.debug(f"process completed with code: {return_code}")
        context.logger.debug(f"std err: \n{stderr}")
        if self._decode_stdout:
            stdout = stdout_bytes.decode()
            data["stdout"] = stdout
            context.logger.debug(f"std out: \n{stdout}")
        else:
            data["stdout_bytes"] = stdout_bytes
            context.logger.debug(f"std out is {len(stdout_bytes)} bytes")

        if return_code == 0:
            return self.build_success_message("Python code execution succeeded", data=data)

        context.logger.debug(f"Python code execution failed:\n{stderr}")
        return self.build_error_message(f"Python code execution failed:\n{stderr}", data=data)

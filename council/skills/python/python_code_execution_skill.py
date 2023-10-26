import os
import subprocess
from typing import Mapping, Optional

from council.contexts import ChatMessage, SkillContext
from council.skills import SkillBase

from .llm_helper import extract_code_block


class PythonCodeExecutionSkill(SkillBase):
    """
    Skills that execute python code and provides the results.

    The python code is retrieved from the message content from `context.try_last_message`,
    looking for a markdown `python` code block.

    The return message data contains a dictionary with the status code, stdout and stderr.
    """

    def __init__(self, env_var: Optional[Mapping[str, str]] = None, decode_stdout: bool = True):
        """
        Initialize a new instance

        Args:
            env_var: Optional list of environment variable to be set for the code execution
            decode_stdout: either or not the stdout should be returns as a string (`True`), or as a bytes (`False`)
        """

        super().__init__("python code runner")
        self._env_var = os.environ.copy() | (env_var or {})
        self._decode_stdout = decode_stdout

    def execute(self, context: SkillContext) -> ChatMessage:
        last_message = context.try_last_message.unwrap("last message")
        python_code = extract_code_block(last_message.message, "python")
        context.logger.debug(f"running python code: \n {python_code}")
        execution = subprocess.run(["python", "-c", python_code], capture_output=True, env=self._env_var)

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

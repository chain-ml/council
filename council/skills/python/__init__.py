import itertools
from typing import List, Optional

from .python_code_generation_skill import PythonCodeGenerationSkill
from .python_code_verification_skill import PythonCodeVerificationSkill
from .python_code_execution_skill import PythonCodeExecutionSkill
from council.contexts import ChainContext
from council.runners import RunnerBase, DoWhile, If, Sequential


def build_code_generation_loop(
    code_generation: PythonCodeGenerationSkill,
    verification: Optional[PythonCodeVerificationSkill] = None,
    execution: Optional[PythonCodeExecutionSkill] = None,
    max_iteration: int = 10,
) -> RunnerBase:
    """
    Helper function to build a python code generation loop,
    running the code generation skill until it successfully pass the verification and execution.

    Args:
        code_generation (PythonCodeGenerationSkill): skill to generate the python code
        verification (Optional[PythonCodeVerificationSkill]): optional static code verification skill
        execution (Optional[PythonCodeExecutionSkill]): optional code execution skill
        max_iteration (int): the maximum number of iteration for the generation/correction loop

    Returns:
        RunnerBase:
    """

    def while_predicate(context: ChainContext) -> bool:
        error_messages = filter(lambda m: m.is_error, context.messages)
        count = sum(1 for _ in itertools.islice(error_messages, max_iteration))
        last_message = context.last_message
        return last_message is not None and last_message.is_error and count < max_iteration

    def is_ok(context: ChainContext) -> bool:
        return context.last_message is not None and context.last_message.is_ok

    sequence: List[RunnerBase] = [code_generation]
    if verification is not None:
        sequence.append(If(is_ok, verification))
    if execution is not None:
        sequence.append(If(is_ok, execution))

    return DoWhile(while_predicate, Sequential(*sequence))

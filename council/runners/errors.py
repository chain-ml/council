class RunnerError(Exception):
    """
    An error occurred in a :class:`.RunnerBase`
    """

    pass


class RunnerTimeoutError(RunnerError):
    """
    Timeout during the execution of a :class:`.RunnerBase`
    """

    pass


class RunnerSkillError(RunnerError):
    """
    An error occurred during the execution of a :class:`.SkillBase`
    """

    pass


class RunnerPredicateError(RunnerError):
    """
    An error occurred during the execution of a :class:`.RunnerPredicate`
    """

    pass


class RunnerGeneratorError(RunnerError):
    """
    An error occurred during the execution of a :class:`.RunnerGenerator`
    """

    pass

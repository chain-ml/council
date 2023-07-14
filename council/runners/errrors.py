class RunnerError(Exception):
    pass


class RunnerTimeoutError(RunnerError):
    pass


class RunnerSkillError(RunnerError):
    pass


class RunnerPredicateError(RunnerError):
    pass


class RunnerGeneratorError(RunnerError):
    pass

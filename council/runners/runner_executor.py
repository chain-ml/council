from concurrent import futures


RunnerExecutor = futures.ThreadPoolExecutor


def new_runner_executor(name: str = "skill_runner") -> RunnerExecutor:
    return RunnerExecutor(thread_name_prefix=name, max_workers=10)

"""


Module runner_executor

Provides a convenient way to create a new thread pool executor tailored for running skills.

The module defines RunnerExecutor as an alias for futures.ThreadPoolExecutor, which leverages
a pool of threads to execute calls asynchronously.

Functions:
    new_runner_executor(name: str='skill_runner') -> RunnerExecutor
        Creates a new instance of a ThreadPoolExecutor with a specified name prefix for the threads and a fixed
        number of worker threads. The prefix helps in identifying the threads associated with the executor
        during debugging or logging.

        Args:
            name (str, optional): The name prefix for the threads created by the ThreadPoolExecutor. Defaults to 'skill_runner'.

        Returns:
            RunnerExecutor: An instance of ThreadPoolExecutor initialized with the specified thread name prefix
                            and a default maximum of 10 worker threads.


"""
from concurrent import futures


RunnerExecutor = futures.ThreadPoolExecutor


def new_runner_executor(name: str = "skill_runner") -> RunnerExecutor:
    """
    Creates a new RunnerExecutor object with the specified thread name prefix and maximum number of worker threads.
    
    Args:
        name (str, optional):
             The prefix to use for thread names created by the RunnerExecutor. Defaults to 'skill_runner'.
    
    Returns:
        (RunnerExecutor):
             A new instance of RunnerExecutor with the specified configuration.

    """
    return RunnerExecutor(thread_name_prefix=name, max_workers=10)

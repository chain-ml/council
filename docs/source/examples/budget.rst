Budget
------

In the context of agent execution, a budget refers to a predetermined limit on the amount of resources or actions an agent is allowed to consume during its execution. The purpose of a budget is to control and manage the agent's behavior, preventing it from using excessive resources, making too many calls to external services, or executing for an extended period.

Budgets are particularly relevant when working with AI agents or language models that interact with external systems, use computational resources, or have certain usage restrictions. It ensures that the agent operates within specified boundaries, making it more predictable and manageable.

Possible Types of Budget
========================

1. **Time Budget:** A time budget imposes a limit on the execution time for the agent. It ensures that the agent's execution does not exceed a specified time limit, preventing it from running indefinitely.

2. **Token Budget:** In natural language processing tasks, tokens are fundamental units representing words or characters. A token budget restricts the number of tokens an agent can use during execution. This is useful for limiting the length of generated responses or controlling the number of tokens processed.

3. **Call Budget:** If an agent interacts with external services or APIs, a call budget limits the number of API calls or external requests it can make. This helps prevent abuse or overloading external systems.


Importance of Budgets
=====================

1. **Fair Usage:** Budgets promote fair usage of resources, especially in multi-user environments. They prevent a single agent from monopolizing system resources and allow fair access to other agents.

2. **Cost Control:** For services where costs are associated with resource consumption (e.g., cloud-based AI services), budgets help control expenses by limiting resource usage.

3. **Safety and Security:** Budgets act as safety guards, preventing agents from executing indefinitely and causing resource exhaustion or crashes.

4. **Predictability:** With budgets, the behavior of an agent becomes more predictable, making it easier to plan and manage system resources.

5. **Preventing Abuse:** Budgets protect against malicious or poorly designed agents that might otherwise overload systems with excessive usage.

Example
=======

The example below demonstrates how to use the budget to limit the amount of resources an agent is allowed to consume.

.. testcode::

    from council.contexts import Budget, Consumption

    # Limit the agent's execution time
    maximum_time_execution = 100

    # Limit the number of token the "gpt-35-turbo" model can produce.
    limit_token = Consumption(120, "token", "gpt-35-turbo")

    # Limit the number of call to the "LLMSkill"
    limit_call = Consumption(5, "call", "LLMSkill")

    budget = Budget(maximum_time_execution, limits=[limit_token, limit_call])


Classes
=======

.. autoclass:: council.contexts.Budget
.. autoclass:: council.contexts.Consumption
.. autoclass:: council.contexts.ConsumptionEvent

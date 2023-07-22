ParallelFor
===========

.. autoclass::
    council.runners.ParallelFor

Example 1
---------

The example below demonstrate how to use the parallel for in a chain.

.. testcode::

    from council.chains import Chain
    from council.contexts import ChainContext
    from council.runners import Budget, ParallelFor
    from council.mocks import MockSkill

    def generator(context: ChainContext, budget: Budget):
        for i in range(0, 5):
            yield "hi"

    chain = Chain(name="name", description="parallel for", runners=[ParallelFor(generator, MockSkill())])

Example 2
---------

This example builds on the previous one and shows how to consume the iteration into a skill.

.. testcode::

    from council.chains import Chain
    from council.contexts import ChatMessage, ChainContext, SkillContext
    from council.runners import Budget, ParallelFor
    from council.skills import SkillBase

    def generator(context: ChainContext, budget: Budget):
        for i in range(0, 5):
            yield f"hi {i}"

    class MySkill(SkillBase):
        def __init__(self):
            super().__init__("mySkill")

        def execute(self, context: SkillContext, budget: Budget) -> ChatMessage:
            it = context.iteration.unwrap()
            message = f"index {it.index}, {it.value}"
            print(message)
            return self.build_success_message(message=message)

    chain = Chain(name="name", description="parallel for", runners=[ParallelFor(generator, MySkill(), parallelism=5)])
    context = ChainContext.empty()
    context.new_iteration()
    chain.execute(context, Budget(1))

The output would looks like.

.. testoutput::

    index 0, hi 0
    index 1, hi 1
    index 2, hi 2
    index 3, hi 3
    index 4, hi 4

Parallel
--------

When building a :ref:`chain-ref-label` containing independent :ref:`skill-ref-label`, we can leverage the :ref:`parallel-ref-label` implementation of the
:ref:`runners-ref-label` abstraction to actually run the skills in parallel and enhance throughput.

Example
=======

Here is a simple example showing how to instantiate a chain with skills running in parallel.

.. testcode::

   from council.chains import Chain
   from council.runners import Parallel
   from council.mocks import MockSkill

   chain = Chain(name="name", description="parallel", runners=[Parallel(MockSkill(), MockSkill())])

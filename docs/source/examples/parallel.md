# Parallel

When building a [chain](chain-ref-label) containing independent [skill](skill-ref-label), we can leverage the [parallel](parallel-ref-label) implementation of the [runners](runners-ref-label) abstraction to actually run the skills in parallel and enhance throughput.

## Example

Here is a simple example showing how to instantiate a chain with skills running in parallel.

```{eval-rst}
.. testcode::

   from council.chains import Chain
   from council.runners import Parallel
   from council.mocks import MockSkill

   chain = Chain(name="name", description="parallel", runners=[Parallel(MockSkill(), MockSkill())])
```

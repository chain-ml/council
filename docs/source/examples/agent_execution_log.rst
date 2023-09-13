Agent Execution Log
---------------------

..  testcode::

    from council.agents import Agent
    from council.contexts import AgentContext
    from council.chains import Chain
    from council.controllers import BasicController
    from council.evaluators import BasicEvaluator
    from council.filters import BasicFilter
    from council.mocks import MockSkill

    chains = [Chain("a chain", "do something", [MockSkill("a skill")])]
    agent = Agent(BasicController(chains), BasicEvaluator(), BasicFilter(), name="an agent")

    context = AgentContext.from_user_message("run")
    agent.execute(context)
    print(context.execution_log_to_json())


.. Skip test output as output contains duration which may vary and actual execution date
.. testoutput::
    :options: +SKIP

    {
      "entries": [
        {
          "source": "agent",
          "start": "2023-09-13T22:09:52.358559+00:00",
          "duration": 0.000377,
          "consumptions": [],
          "messages": []
        },
        {
          "source": "agent/iterations[0]",
          "start": "2023-09-13T22:09:52.358678+00:00",
          "duration": 0.000253,
          "consumptions": [],
          "messages": []
        },
        {
          "source": "agent/iterations[0]/controller",
          "start": "2023-09-13T22:09:52.358686+00:00",
          "duration": 7e-06,
          "consumptions": [],
          "messages": []
        },
        {
          "source": "agent/iterations[0]/execution(a chain)",
          "start": "2023-09-13T22:09:52.358701+00:00",
          "duration": 0.000199,
          "consumptions": [],
          "messages": []
        },
        {
          "source": "agent/iterations[0]/execution(a chain)/chain(a chain)",
          "start": "2023-09-13T22:09:52.358710+00:00",
          "duration": 0.000187,
          "consumptions": [],
          "messages": []
        },
        {
          "source": "agent/iterations[0]/execution(a chain)/chain(a chain)/runner",
          "start": "2023-09-13T22:09:52.358731+00:00",
          "duration": 0.000157,
          "consumptions": [],
          "messages": [
            {
              "is_error": false,
              "kind": "SKILL",
              "message": "",
              "source": "a skill"
            }
          ]
        },
        {
          "source": "agent/iterations[0]/evaluator",
          "start": "2023-09-13T22:09:52.358903+00:00",
          "duration": 1.5e-05,
          "consumptions": [],
          "messages": []
        },
        {
          "source": "agent/iterations[0]/filter",
          "start": "2023-09-13T22:09:52.358921+00:00",
          "duration": 5e-06,
          "consumptions": [],
          "messages": []
        }
      ]
    }

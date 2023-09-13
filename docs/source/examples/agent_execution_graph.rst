Agent Execution Graph
---------------------

..  testcode::

    from council.agents import Agent
    from council.chains import Chain
    from council.controllers import BasicController
    from council.evaluators import BasicEvaluator
    from council.filters import BasicFilter
    from council.mocks import MockSkill

    chains = [Chain("a chain", "do something", [MockSkill("a skill")])]
    agent = Agent(BasicController(chains), BasicEvaluator(), BasicFilter(), name="an agent")
    print(agent.render_as_json())


.. testoutput::

    {
      "properties": {
        "name": "an agent"
      },
      "type": "Agent",
      "baseType": "agent",
      "children": [
        {
          "name": "controller",
          "value": {
            "properties": {},
            "type": "BasicController",
            "baseType": "controller",
            "children": []
          }
        },
        {
          "name": "chains[0]",
          "value": {
            "properties": {
              "name": "a chain"
            },
            "type": "Chain",
            "baseType": "chain",
            "children": [
              {
                "name": "runner",
                "value": {
                  "properties": {
                    "name": "a skill"
                  },
                  "type": "MockSkill",
                  "baseType": "skill",
                  "children": []
                }
              }
            ]
          }
        },
        {
          "name": "evaluator",
          "value": {
            "properties": {},
            "type": "BasicEvaluator",
            "baseType": "evaluator",
            "children": []
          }
        },
        {
          "name": "filter",
          "value": {
            "properties": {},
            "type": "BasicFilter",
            "baseType": "filter",
            "children": []
          }
        }
      ]
    }

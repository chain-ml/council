# Proposed Design Update

- Skills: 
  - it consumes from the context and provides a new message(s). It has a clear scope and responsibility, similar to a function
  - it does not modify the context (Input is readonly context, Output is a List of messages)
- Runners:
  - Used to define a graph of operation (if, sequence or chain, parallel, while, ...)
  - Manage `context` update and ensure thread safety
  - new `Controller`? (as in flow controller, or control flow)
- Chain: 
  - would become `runner` + description
  - new `Agent`???
- Controller
  - would become a `runner`
- Evaluator
  - would become a _many to many_ skill
- Filter
  - would become a _many to many_ skill
- Agent 
  - is too complex
  - constraint workflow
  - would be removed in favor of the new `runner`/`controller`

## Design Goals

- Thread Safe
- 1 constraint => 1 or more benefits
- composability

## Categories

- one to one
- one to many
- many to one
- many to many

## Sequential

```mermaid
flowchart LR
    SkillA[Skill A]
    subgraph Sequence
        direction LR
        SkillB[Skill B]
        SkillC[Skill C]
    end
    SkillD[Skill D]
    SkillA --a--> SkillB --b--> SkillC --c--> SkillD
%%    SkillD --> |d, data = c,s|EvalA
%%    SkillD --> |c', data = c.data + s| EvalB
    
```
In the scenario, each skill takes one message as input, and produces one message as output. 

| skill   | visible messages | last message |
|---------|------------------|--------------|
| Skill A | -                | -            |
| Skill B | a                | a            |
| Skill C | a, b             | b            |
| Skill D | a, c             | c            |

## Parallel

It executes multiple branches against the same input message.
This is helpful to explore different execution strategies.

```mermaid
flowchart LR
    SkillA[Skill A]
    SkillB[Skill B]
    subgraph Parallel
        direction LR
        subgraph branchC[Branch C]
            SkillCA[Skill CA]
            SkillCB[Skill CB]
        end
        subgraph branchD[Branch D]
            SkillDA[Skill DA]
            SkillDB[Skill DB]
        end
    end
    SkillE[Skill E]
    SkillF[Skill F]
    SkillA --a--> SkillB 
    SkillB --b--> SkillCA
    SkillB --b--> SkillDA
    SkillCA --ca--> SkillCB --cb--> SkillE
    SkillDA --da--> SkillDB --db--> SkillE
    SkillE --e--> SkillF
```

This scenario leverages different types of skills:
- `Skill E` is a _many to one_ skill: multiple inputs, single outputs. It fans in
- All other skills are _one to one_ skills: one input, one output

| skill    | visible messages | last message(s) |
|----------|------------------|-----------------|
| Skill A  | -                | -               |
| Skill B  | a                | a               |
| Skill CA | a, b             | b               |
| Skill CB | a, b, ca         | ca              |
| Skill DA | a, b             | b               |
| Skill DB | a, b, da         | da              |
| Skill E  | a, b, cb, db     | cb, db          |
| Skill F  | a, b, e          | e               |

## Parallel For

It executes the same runner for each input message.

```mermaid
flowchart LR
    SkillA[Skill A]
    SkillB[Skill B]
    subgraph ParallelFor[Parallel For]
        direction LR
        subgraph branchC1[Iteration 1]
            SkillC1A[Skill CA]
            SkillC1B[Skill CB]
        end
        subgraph branchC2[Iteration 2]
            SkillC2A[Skill CA]
            SkillC2B[Skill CB]
        end
    end
    SkillD[Skill D]
    SkillE[Skill E]
    SkillA --a--> SkillB 
    SkillB --b1--> SkillC1A
    SkillB --b2--> SkillC2A
    SkillC1A --c1a--> SkillC1B --c1b--> SkillD
    SkillC2A --c2a--> SkillC2B --c2b--> SkillD
    SkillD --d--> SkillE
```

This scenario leverages different types of skills:
- `Skill B` is a _one to many_ skill: one input, multiple outputs. It fans out
- `Skill D` is a _many to one_ skill: multiple inputs, single outputs. It fans in
- All other skills are _one to one_ skills: one input, one output


| skill     | visible messages | last message(s) |
|-----------|------------------|-----------------|
| Skill A   | -                | -               |
| Skill B   | a                | a               |
| Skill C1A | a, b1            | b1              |
| Skill C1B | a, b1, c1a       | c1a             |
| Skill C2A | a, b2            | b2              |
| Skill C2B | a, b2, c2a       | c2a             |
| Skill D   | a, c1b, c2b      | c1b, c2b        |
| Skill E   | a, d             | d               |


## If

Runs one skill if a given predicate returns `True`; another skill otherwise.

```mermaid
flowchart LR
    SkillA[Skill A]
    SkillB[Skill B]
    subgraph If
        Predicate{Predicate}
        SkillCA[Skill CA]
        SkillCB[Skill CB]
        Either{Either}
    end
    SkillD[Skill D]
    SkillE[Skill E]
    SkillA --a--> SkillB
    SkillB --b--> Predicate
    Predicate --true--> SkillCA
    Predicate --false--> SkillCB
    SkillCA --ca--> Either
    SkillCB --cb--> Either
    Either --c?--> SkillD
    SkillD --d--> SkillE
    
```

Here, `Skill D` is a _one to one_ skill as it receives either `ca` or `cb` as input, never both.

| skill    | visible messages    | last message(s) |
|----------|---------------------|-----------------|
| Skill A  | -                   | -               |
| Skill B  | a                   | a               |
| Skill CA | a, b                | b               |
| Skill CB | a, b                | b               |
| Skill D  | a, b, (ca or cb)    | ca or cb        |
| Skill E  | a, b, (ca or cb), d | d               |

## While

Runs a skill as long as a given predicate is `true`. The predicate executes at the beginning of the iteration

```mermaid
flowchart LR
    SkillA[Skill A]
    SkillB[Skill B]
    subgraph While
        direction LR
        Predicate{Predicate}
        SkillCA[Skill CA]
        SkillCB[Skill CB]
    end
    SkillD[Skill D]
    SkillE[Skill E]
    
    SkillA --a--> SkillB
    SkillB --b--> Predicate
    Predicate -->|true| SkillCA
    SkillCA --cNa--> SkillCB
    SkillCB --cNb--> Predicate
    Predicate ---->|false| SkillD
    SkillD --d--> SkillE
```


| skill    | visible messages                      | last message(s) |
|----------|---------------------------------------|-----------------|
| Skill A  | -                                     | -               |
| Skill B  | a                                     | a               |
| Skill CA | a, b                                  | b               |
| Skill CB | a, b, c1a                             | c1a             |
| Skill CA | a, b, c1a, c1b                        | c1b             |
| Skill CB | a, b, c1a, c1b, c2a                   | c2a             |
| ...      | ...                                   | ...             |
| Skill CA | a ,b, c1a, c1b, c2a, c2b, ... c(N-1)b | c(N-1)b         |
| Skill CB | a, b, c1a, c1b, c2a, c2b, ... cNa     | cNa             |
| Skill D  | a, b, CNb                             | cNb (or b)      |
| Skill E  | a, b, cNb, d                          | d               |

## DoWhile

Runs a skill as long as a given predicate is `true`. The predicate executes at the end of the iteration

```mermaid
flowchart LR
    SkillA[Skill A]
    SkillB[Skill B]
    subgraph While 
        direction LR
        SkillCA[Skill CA]
        SkillCB[Skill CB]
        Predicate{Predicate}
    end
    SkillD[Skill D]
    SkillE[Skill E]
    
    SkillA --a--> SkillB
    SkillB --b--> SkillCA
    SkillCA --cNa--> SkillCB
    SkillCB --cNb--> Predicate
    Predicate --true--> SkillCA
    Predicate --false--> SkillD
    SkillD --d--> SkillE
```


| skill    | visible messages                      | last message(s) |
|----------|---------------------------------------|-----------------|
| Skill A  | -                                     | -               |
| Skill B  | a                                     | a               |
| Skill CA | a, b                                  | b               |
| Skill CB | a, b, c1a                             | c1a             |
| Skill CA | a, b, c1a, c1b                        | c1b             |
| Skill CB | a, b, c1a, c1b, c2a                   | c2a             |
| ...      | ...                                   | ...             |
| Skill CA | a ,b, c1a, c1b, c2a, c2b, ... c(N-1)b | c(N-1)b         |
| Skill CB | a, b, c1a, c1b, c2a, c2b, ... cNa     | cNa             |
| Skill D  | a, b, cNb                             | cNb             |
| Skill E  | a, b, cNb, d                          | d               |

# Controller

A Controller is a dynamic runner: the graph of execution depends on the context. It has two main execution steps:
- building a graph of execution (or an execution plan). This produce a `runner`
- executing it

A `controller` is a `runner` by contract (executing a graph for a given context), it shares the same benefits as any other `runner`, including:
- traceability via the `context`
- logging via the `context`
- composability into complex graph of execution

```Python
class ControllerRunnerBase(RunnerBase):
    def _execute(self, context) -> List[Message]:
      return self.build_runner(context).execute(context).last_messages()
    
    def build_runner(self, context) -> Runner :
      pass
    
class PlanControllerRunner(ControllerRunnerBase):
    def build_runner(self, context) -> Runner:
        plan = self.get_plan()
        return self.build_runner_from_plan(plan)

    def get_plan(self) -> List[ExecutionUnit]:
        pass

class AgentRunner(PlanControllerRunner):
    def build_runner(self, context) -> Runner:
      runner = super().build_runner()
      return DoWhile(sequence(runner, filter, evaluator))
        
```

Below are a few examples of controller execution flow.

## Switch Controller

From a set of runners, pick one and execute it
```mermaid
flowchart LR
    User((User))
    Done((end))
    subgraph Controller
        direction LR
        llm
        switch{Switch}
        nyc[AirBNB NYC Specialist]
        olist[OList Specialist]
        
    end
    User --> |userMessage| Controller --> |response| Done 
    llm --> switch -->|most relevant| nyc 
```

## Tool Controller

From a set of runners, consume them as chainable tools to achieve a greater goal

```mermaid
flowchart LR
    
    User((User))
    done((end))
    subgraph Controller
        direction LR
        llm 
        CsvAnalyst 
        SqlAnalyst
        PlotSpecialist 
        TextCommentary
        TimeseriesForecast
    end
    
    User --> Controller --> done
    llm --> CsvAnalyst --> PlotSpecialist --> TextCommentary
```
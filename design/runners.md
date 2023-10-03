# Runners

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
    
```
In the scenario, each skill takes one message as input, and produces one message as output. 

| skill   | visible messages | last message |
|---------|------------------|--------------|
| Skill A | -                | -            |
| Skill B | a                | a            |
| Skill C | a, b             | b            |
| Skill D | a, b, c          | c            |

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


| skill    | visible messages                        | last message(s) |
|----------|-----------------------------------------|-----------------|
| Skill A  | -                                       | -               |
| Skill B  | a                                       | a               |
| Skill CA | a, b                                    | b               |
| Skill CB | a, b, c1a                               | c1a             |
| Skill CA | a, b, c1a, c1b                          | c1b             |
| Skill CB | a, b, c1a, c1b, c2a                     | c2a             |
| ...      | ...                                     | ...             |
| Skill CA | a ,b, c1a, c1b, c2a, c2b, ... c(N-1)b   | c(N-1)b         |
| Skill CB | a, b, c1a, c1b, c2a, c2b, ... cNa       | cNa             |
| Skill D  | a, b, c1a, c1b, c2a, c2b, ..., cNa, cNb | cNb (or b)      |
| Skill E  | a, b, c1a, c1b, c2a, c2b, ..., cNa, cNb | d               |

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


| skill    | visible messages                        | last message(s) |
|----------|-----------------------------------------|-----------------|
| Skill A  | -                                       | -               |
| Skill B  | a                                       | a               |
| Skill CA | a, b                                    | b               |
| Skill CB | a, b, c1a                               | c1a             |
| Skill CA | a, b, c1a, c1b                          | c1b             |
| Skill CB | a, b, c1a, c1b, c2a                     | c2a             |
| ...      | ...                                     | ...             |
| Skill CA | a ,b, c1a, c1b, c2a, c2b, ... c(N-1)b   | c(N-1)b         |
| Skill CB | a, b, c1a, c1b, c2a, c2b, ... cNa       | cNa             |
| Skill D  | a, b, c1a, c1b, c2a, c2b, ..., cNa, cNb | cNb             |
| Skill E  | a, b, c1a, c1b, c2a, c2b, ..., cNa, cNb | d               |

# Controller

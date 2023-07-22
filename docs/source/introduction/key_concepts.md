# Key Concepts


Key components of the framework are shown in below image and further introduced in this section.

![engine flow](engine_flow.png "engine")

## Agent
Agents encapsulate the end-to-end application logic from prompt input to final response across Controller, Evaluation and registered Chains of Skills. Agents itself can be recursively nested within other Agents in the form of AgentChains.

## Controller
Controllers determine user intent given a prompt and prompt history and route the prompt to one or multiple of registered Chains before leveraging one or multiple Evaluators to score returned results and determine further course of action (including the ability to determine whether a revision is needed in which case Chains might be prompted again). Controllers will control whether one or multiple Chains are called, whether calls happen in series or in parallel. They will also be responsible for the distribution of compute budget to Chains and handling Chains that are not able to return results within the allocated compute budget. A State Management component will allow Controllers to save and retrieve state across user sessions. Controllers can be implemented in Python or (soon) Rust (to meet the needs of performance-critical applications).

## Skill
Skills are services that receive a prompt / input and will return an output. Skills can represent a broad range of different tasks relevant in a conversational context. They could wrap general purpose calls to publicly available language model APIs such as OpenAI’s GPT-4, Anthropic’s Claude, or Google’s Bison. They could also encapsulate locally available smaller language models such as Stable-LM and Cerebras-GPT, or serve more specific purposes such as providing interfaces to application-specific knowledge bases or generate application-aware code snippets. 

## Chain
Chains are directed graphs of Skills that are co-located and expose a single entry point for execution to Controllers. Users will be able to define their own Chains or reuse existing implementations. Chains themselves can be Agents that are recursively nested into the execution of another Agent as AgentChain.

## Evaluator 
Evaluators are responsible for assessing the quality of one or multiple Skills / Chains at runtime for example by ranking and selecting the best response or responses that meet a given quality threshold. This can happen in multiple ways and is dependent on the implementation chosen. Users can extend standard Evaluators to achieve custom behavior that best matches their requirements. 

## State Management
Council provides native objects to facilitate management of Agent, Chain and Skill context. These objects make it easy to keep track of message history and intermediate results.

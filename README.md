# The Objective
Building a prototype using a large language model (LLM) agent that answers conversational questions based on ConvFinQA.


## Installations
### Prerequisites
- Python 3.12+
- [UV environment manager](https://docs.astral.sh/uv/getting-started/installation/)

### Setup
1. Clone this repository
2. Use the UV environment manager to install dependencies:

```bash
# install uv
brew install uv

# set up env
uv sync

# add python package to env
uv add <package_name>
```

### [optional] Use CLI to chat

We have created a boilerplate cli app using [typer](https://typer.tiangolo.com/) (sister of fastapi, built on click) so there is a simple chat interface, which you can extend to meet your needs if you so choose.  By default the chat responds with a standard message as shown below.


You can run it with:
```bash 
uv run main chat <record_id> [--model gpt-4o-mini] [--verbose]
uv run main chat <record_id> [--provider groq] [--verbose]

# example
uv run main chat Single_JKHY/2008/page_30.pdf-1 --provider groq
```

## Docker
Build the docker image using the following command:

```bash
docker build -t convfinqa .
```

and then run the chat using following example:

```bash
docker run -it --env-file .env convfinqa chat [--provider groq] [--verbose]

# example
docker run -it --env-file .env convfinqa chat Single_JKHY/2008/page_30.pdf-1 --provider groq
```



# Detailed Report

**The data**: ConvFinQA is a dataset for financial question and answering. The data conists of multi-hop numerical questions and answers. 

**The Task**: This task in the first glance appeared to be an Agentic-RAG task. However, since the data (per-document) is small enough to be loaded directly into the LLM context window, it does not need the typical Vanila RAG structure of chuncking, embeding, vector DB, and retrieval. The assignment is more focused on the Agentic design for financial documents using multi-hop approaches. 

## Method

### Literature
Several references were used to study the potential candidates for multi-hop agentic design that are mentioned below:

1. https://arxiv.org/pdf/2401.15391
2. https://blog.leena.ai/glossary/multi-hop-reasoning/
3. https://www.langchain.com/blog/plan-and-execute-agents?utm_source=chatgpt.com
4. https://c3.ai/blog/enhancing-enterprise-ai-with-multi-hop-orchestration-agents-advanced-reasoning-for-accurate-reliable-decision-making-part-2

### The agentic solution
Multiple candidates were selected for implementations and benchmarking. These are:

1. **Chain-of-Thoughts (CoT)**. Pros: breaks tasks into smaller steps for step-by-step solutions, it is also fast. Cons: if it makes a mistake in one of the steps, this will propagate into next steps.
2. **Reasoning and Acting (ReAct)**. Pros: combines step-by-step reasoning with real-time actions, allowing the model to interact with an environment and adjust based on feedback. Cons: it is slower than CoT, and it may get stuck in iterations and never return a result.
3. **Plan-and-Execute (Planner)**: Pros: the model first generates a high-level plan by breaking the task into a sequence of sub-tasks, then executes each step in order. Unlike ReAct, the planning phase is separated from execution, making the workflow more structured and often more efficient for complex, multi-step tasks. It is also provide insight when error happens. Cons: it is slow and token heavy.
4. Baseline agent (Direct): direct prompting is not a multi-hop solution and here is used as a baseline to compare the performance of the above three candidates.

There is also Graph-RAG approach which is ideal for complex multi-hoping but current dataset is not complex enough for Graph approach. Moreover, literatre has shown direct approch outperforms Grpah search for simple tasks.

Here are the diagram of these methods:

```
Direct                  ReAct                   CoT                     Planner

Question                Question                Question                Question
↓                       ↓                       ↓                       ↓
Tool (as needed)        Thought                 Reasoning               Plan (ordered steps)
↓                       ↓                       ↓                       ↓
Answer                  Action (Tool)           Tool                    Executor: Tool
                        ↓                       ↓                       ↓
                        Observation             More reasoning          Executor: Tool
                        ↓                       ↓                       ↓
                        Thought                 Tool                    ...
                        ↓                       ↓                       ↓
                        Action (Tool)           Answer                  Answer
                        ↓
                        Observation
                        ↓
                        ...
                        ↓
                        Answer
```

### LLMs

I evaluated the solution using two different LLMs to compare the behaviour of a lightweight commercial model and an open-weight model.

1. **GPT-4.1 Mini** was selected because it is a good balance between capability, latency, and cost. Although it is a smaller model compared with many 2026 models, it performs well on structured reasoning tasks.

2. **OpenAI/gpt-oss-120b**: this is a open weight model by OpenAI and was selected to evaluate whether an openly available model could achieve comparable performance. 

### Tools
1. **read_document**: it returns the complete financial document (pre-text, table, post-text) as a single block of text so the agent can read it directly instead of searching or chunking it. The table is in markdown format.

2. **calculator**: it evaluates an arithmetic expression (e.g. (206588 - 181001) / 181001) via an evaluator, so the agent never has to do math by hand.

3. **record_final_answer**: lets the agent submit its final numeric (or yes/no) answer plus a one-sentence rationale in a structured form, which the harness captures for scoring instead of parsing free text.

### Metrics
1. **Execution accuracy**: the result was compared against the numerical result (and not the string format), as it is being used in the original paper. It makes it comparable to them and it will prevent the penalizing due to formating (eg. "23.5%" vs 0.235). This is provided as "Turn accuracy": `|prediction - gold| <= max(1% relative, 0.001 absolute)` for numerical answers. For `yes/no` should be an exact match (non-case sensitive). Although the accuracy for the string format is also provided as "Conversation accuracy". The turn accuracy was also applied for the question types: Type I and Type II (hybrid questions and more difficult).

2. **Evaluation time**: the experiment time was also measured and reported. This is important for the user as a faster solution is more appealing. The elapsed time also correlates with the token usage. 

### Few shot prompting
Few shot prompting is known for helping the LLMs to process a given task. 
For all the 4 agents, we checked the effects of the few shot prompting on their performances, ie. with and without few shot.



## Result

All four were run on the **same 25-conversation dev sample (seed=42, 86 turns)** with the **same model (`gpt-4.1-mini` via OpenAI)**, so the comparison isolates the effect of agent architecture from model or sample differences:

| Agent | Turn execution accuracy | Conversation accuracy | Type I / Type II accuracy | Elapsed |
|---|---|---|---|---|
| **Direct** | **79.1%** | **72.0%** | **82.6%** / 64.7% | 321.3s |
| Direct (+ few-shot) | 73.3% | 68.0% | 75.4% / 64.7% | 305.8s |
| ReAct | 74.4% | 60.0% | 76.8% / 64.7% | 467.1s |
| ReAct (+ few-shot) | 74.4% | 56.0% | 78.3% / 58.8% | 503.5s |
| CoT | 73.3% | 64.0% | 75.4% / 64.7% | 339.5s |
| CoT (+ few-shot) | 76.7% | 64.0% | 79.7% / 64.7% | 334.0s |
| Planner | 67.4% | 56.0% | 71.0% / 52.9% | 564.3s |
| Planner (+ few-shot) | 74.4% | 64.0% | 76.8% / 64.7% | 696.2s |

### Result summary

1. **Direct agent**: the Direct agent was the fastest and achieved the highest accuracy in Type I questions. But in Type II questions it performed similar to others.
2. **CoT agent**: The CoT agent had a slightly lower performance to Direct agent both in terms of accuracy and processing time.
3. **ReAct agent**: was significantly slower than CoT and Direct agents. It also scored a lower accuracy than Direct agent. than 
4. **Planner agent**: The Planner agent achieved similar accuracy to the CoT agent albeit being more than twice as long. 

#### Impact of few shop prompting in performance
It appeared that adding a few examples to the system prompt for the agents led to mixed results. For example, CoT and Planner agents had improvements, while Direct agent accuracy dropped. This also could indicate that GPT 4.1 mini is more than capable of handling questions shown here, and adding examples in the prompt may disrupt its ability to reason on the questions.

### Checking on a free model (Groq)

The Direct agent (with few-shot) was also run against `openai/gpt-oss-120b` on Groq's free tier. Due to the token limit, only the first 15 conversation were used in the evaluation. The result is provided here and shows a competitivev performance in Type I questions but significantly lower accuracy in Type II.

| Provider / model | Turn accuracy | Conversation accuracy | Type I / Type II accuracy |
|---|---|---|---|
| OpenAI `gpt-4.1-mini` (+ few-shot), n=25 | 73.3% | 68.0% | 75.4% / 64.7% |
| Groq `openai/gpt-oss-120b` (+ few-shot), n=15 (partial) | 71.4% | 66.7% | 77.5% / 44.4% |

## Error Analysis

1. **Question ambiguity**: all the 4 agents failed on this question in file `Single_UNP/2014/page_75.pdf-3 turn 1`: {`question`: `what was the total expense for repairs and maintenance incurred in 2013?`}. All answered in dollar unit while the grountruth is `2.3`. Meanwhile, in the file it is mentioned `... $ 2.3 billion for 2013 ...`. So it appears that the question may not be correct as it is not mentioning the unit.
2. **Error propagation**: all the 4 agents failed in this file `Single_AAP/2012/page_12.pdf-3` about `variance of the foreign exchange products`. It also showed that if a turn fails, all the subsequent turns will fail as they will use the failed turn.
3. **Grountruth mismatch**: in `Single_AAP/2012/page_12.pdf-3` there is a mismatch between some documents. For example `"conv_answers": ["202.0", "101"] "executed_answers": [225.0, 112.5]`. This mis
4. **Groundtruth mistake**: in `Double_PNC/2013/page_207.pdf` question 1 is asking `which one is higher: this... or that...` but the groundtruth is `no`. 

## Potentials Improvements 
1. **Scaling the evaluation**: the evaluation was only run on 25 files, due to time and resource constraints. This is not sufficient to produce robust results. Nevertheless, it captured the agents performance on type I vs type II questions.
2. **Prompt fine tuning and few shot**: there was not enough work done to explore the effect of few shot prompting on the performance of the agents. The mixed result of few shot prompting indicates the prompting might need some fine tuning to help the LLMs with the task.
3. **Critic agent**: adding a critic agent could help the agents to cross-check the arithmetic operations.
4. **Multi-agents**: given that questions have difficulty levels, there could be two agents to deal with easy and difficult questions. For example, the open weight from OpenAI achieved the same score in type I questions. This means that GPT 4.1 mini is an overkill for type I questions.
5. Agentic RAG: The current solution relies on the fact that the input documents are significantly smaller (maximum of approximately 15,000 words) than the context window of the selected LLMs. If the solution were to be extended to support multiple documents or substantially larger files, relying solely on the LLM's context window would become inefficient. In such cases, a conventional RAG pipeline incorporating document chunking, embeddings, and retrieval would be required to retrieve only the relevant context before reasoning.

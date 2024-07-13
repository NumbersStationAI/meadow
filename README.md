# Meadow
A framework for building multi-agent data workflows with LLMs with interactive user feedback.

> I like to think (and the sooner the better!) of a cybernetic meadow where mammals and computers live together in mutually programming harmony like pure water touching clear sky.
>
> --Richard Brautigan

<p align="center">
<img src="assets/meadow_image.png" alt="Cybernetic Meadow"/>
</p>

Check out our [blog post](https://numbersstation.ai/introducing-meadow-llm-agents-for-data-tasks/) announcing Meadow.

# Setup
## Install
### Poetry
If you don't have `poetry` installed, follow instructions from [here](https://python-poetry.org/docs/#installing-with-the-official-installer). We recommend

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### Meadow
Then install Meadow with
```
cd meadow
make install
```

## Loading Data
If you want to load our sample data (required for the benchmark) in `examples/data`, you will need `sqlite3`. To load the clean data, run

```bash
cd meadow/examples/data
sqlite3 database_sqlite/sales_example/sales_example.sqlite < sales.sql
```

And to load the ambiguous data (also needed for benchmark), run
```bash
cd meadow/examples/data
sqlite3 database_sqlite/sales_ambiguous_joins_example/sales_ambiguous_joins_example.sqlite < sales_ambiguous_joins.sql
```

# Running

## Commands (Interactive)
Meadow is a plan-then-execute framework where a planner generates a sequence of sub-tasks for a collection of agents to solve based on the user's high level goal. Each each point in the plan, Meadow will return the current sub-task output to the user for feedback. The user can enter one of three things:

1. A text response of feedback for what to change.
2. '\<next\>' string which indicates the user is happy with the output and ready to move on.
3. '\<end\>' string which indicates the user wants to terminate the conversation.

Meadow supports auto-advance which means the feedback is always '\<next\>'.

## Text-to-SQL Use Case
To get started with a simple, interactive use case using text-to-SQL on a sqlite3 database, we have `examples/demo.py` to run. This use case has three main task agents: a attribute detector, a schema cleaner, and a text-to-SQL agent. Note, you will need an OpenAI API key or have it set as an environment variable.
```bash
poetry run python examples/demo.py \
  --api-key API_KEY \
  --db-path examples/data/database_sqlite/sales_example/sales_example.sqlite
```

If you want to auto-advance instead of giving feedback, add the flag `--auto-advance`.

## Benchmark
To repo results from our custom 50 question text-to-SQL benchmark, run
```bash
poetry run experiments/text2sql/predict.py predict \
  examples/data/custom_text2sql_benchmark.json \
  examples/data/sales_example_schema.json \
  examples/data/database_sqlite \
  --output-dir experiments/text2sql/predictions \
  --client-cache-path test_custom_cache.duckdb \
  --use-table-schema \
  --api-provider openai \
  --planner-api-provider openai \
  --api-key <OPEN_AI_API_KEY> \
  --planner-api-key <OPEN_AI_API_KEY> \
  --model gpt-4o \
  --planner-model gpt-4o \
  --num-run 50 \
  --num-print 0 \
  --add-reask \
  --add-empty-table \
  --add-attribute-selector \
  --add-schema-cleaner \
  --async-batch-size 20
```
make sure to setup the sqlite database in the instructions above (you will need both databases setup).

You can evaluate the results by first running
```bash
cd experiments/text2sql/
mkdir metrics
cd metrics
git clone git@github.com:ElementAI/test-suite-sql-eval.git test_suite_sql_eval
cd ../../..
```
to download spider evaluation code.

Then running
```bash
poetry run python3 experiments/text2sql/evaluate.py evaluate \         
  --gold examples/data/custom_text2sql_benchmark.json \
  --db examples/data/database_sqlite \
  --tables examples/data/sales_example_schema.json \
  --output-dir experiments/text2sql/predictions \
  --client-cache-path test_custom_cache.duckdb \
  --api-key <OPEN_AI_API_KEY> \
  --model gpt-4o \
  --pred <PREDICTION_FILE_FROM_ABOVE>
```

The same script can be used to run [Spider](https://yale-lily.github.io/spider) V1 eval numbers. V2 is coming soon!

## Customzing Agents
To see an example of how you can build your own data agents, see [new_agent.ipynb](examples/notebookes/new_agent.ipynb). This notebook will walk you through loading up the existing text-to-SQL pipeline, adding an agent that explains the response, and asking SQL questions.

# Meadow Framework
Meadow is a framework for building multi-agent workflows to solve data tasks with LLMs. Customizable agents all share access to a data layer and converse to solve a task with optional user feedback along the way. Meadow provides:

* A shared data layer to allow agents to solve data tasks and communicate updates to downstream tasks.
* Specialized executor agents that allow for iterative debugging against the data.
* Data-aware planners that break down complex data tasks into sub-tasks based on the available agents.
* A user agent that can give feedback and control the flow of the planned exection with a simple set of keyword commands.

## Chat Flow
Meadow is a plan-then-execute framework where a planner generates a sequence of sub-tasks for a collection of agents to solve based on the user's high level goal. At any point in time, there are two agents communicating to complete a sub-task. During this two agent communication, one agent is designated the `Supervisor` and is the one that initiates the task and provides feedback. The other agent is the `Task Handler` that performs the task. Very often, the user is the supervisor and an LLM-backed agent is the task handler.

## Agents
The core abstraction of Meadow is the [`Agent`](meadow/agent/agent.py) that must implement three methods: to send a message to another agent, to receive a message from another agent, and to generate a reply. An agent does not need to be based on a LLM, e.g. a user agent, but it often is.

An agent can be extended in three possible ways:

1. `PlannerAgent`: The agent must have a collection of agents it can coordinate between and must be able to move through a plan linearly. Planner agents can be hierarchical in that one planner agent can have another planner agent as one of its available agents to coordinate between.
2. `ExecutorAgent`: The agent must have an execution function that parses a message from another agent. The role of the executor is to catch bugs and throw back error messages upon mistakes.
3. `AgentWithExecutors`: This is an agent that has an associated executor agent that must execute on this agent's output. When an agent has an executor, a sub-chat is initiated between the agent and its executor so the executor and agent can debug if desired.

Agents can be extended in multiple ways, e.g., an agent can be both an executor agent and have its own executors.

### Controller
There is one special non-LLM agent in Meadow called the [`Controller`](meadow/agent/controller.py). The controller is determinisitc and has three main functions:

1. It is the mediator between any conversation between two agents. It sends and receives messages and stores the message coversation history for ease of access and observability.
2. It implements and understands the [`Commands`](meadow/agent/schema.py) sent by the supervisor to move through a plan.
3. It can start new chats between two agents. E.g. it initiates the chat between an agent and its executor as a "branch" off of the main chat. Once the sub-chat terminates, the main chat can continue.

## Data Layer
All agents have access to a [`Database`](meadow/database/database.py) that is a view over the user's data. The database allows agents to retrieve information via SQL queries as well as create views over the data. Each view that is created is shared to other agents solving the task.

We support backend databases connected to [SQLite](https://sqlite.org/) or [DuckDB](https://duckdb.org/).

## LLM Client
We support a simple LLM wrapper client that supports sending a chat requiest to various LLMs. The client integrates with a caching layer for faster and cheaper reruns of the same inputs. We currently support [OpenAI](https://platform.openai.com/docs/overview) and [Anthropic](https://www.anthropic.com/api).

# Acknowledge
We'd like to thank the follow packages for inspiring and helping Meadow.

* [AutoGen](https://github.com/microsoft/autogen)
* [MetaGPT](https://github.com/geekan/MetaGPT)
* [AgentVerse](https://github.com/OpenBMB/AgentVerse/tree/main)

# Contact and Cite
For any questions or comments, please leave an Issue or email `laurel.orr@numbersstation.ai`.

To cite
```
@misc{orr2024meadow,
  author = {Orr, Laurel and Chami, Ines},
  title = {Meadow},
  year = {2024},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/NumbersStationAI/meadow}},
}
```

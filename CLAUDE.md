# CLAUDE.md

## About Python code

Always follow these guidelines when writing or editing Python code in
this project:

- Always use `uv` as the package manager whenever a package needs to be
  installed (including in Colab).
- Write all docstrings in English.
- Docstrings must be complete: describe the input parameters
  (`Parameters`) and the return value (`Returns`).
- When a function or method is complex, include a usage example in the
  docstring (`Example` section).
- Respect the per-line character limit in docstrings according to
  PEP 8 / PEP 257 (79 characters).
- Variables, functions, classes and methods must have descriptive,
  meaningful names written in English.
- Use type hints on function and method parameters and return values.
- Do not use type hints on simple variable assignments inside the body
  of the code.
- Do not add comments in the code — keep only the docstrings.
- Whenever a function or method has 2 or more parameters, break the
  signature across multiple lines, one parameter per line, for example:

```python
def func1(
    param_1: type_1,
    param_2: type_2,
    ...
    param_n: type_n,
) -> type_of_return:
```

- Whenever production code is created or changed, add (or update) the
  corresponding unit tests under `tests/`, mirroring the structure of
  `src/generative_models/`.
- Always check whether the existing documentation (README, docstrings,
  files under `docs/`, if any) is still consistent with the change; if
  it needs an update, update it right away.

## About experiment tracking

- Whenever a PyTorch model is trained (a training loop with an
  optimizer and a loss function), always integrate
  `WandbExperimentTracker` from
  `generative_models.tracking.wandb_tracker` to track the experiment,
  following the pattern already used in
  `notebooks/regression_log_return_training.ipynb` and
  `notebooks/outlier_detection_comparison.ipynb`:
  - Instantiate the tracker with a `project_name` and a `config`
    dictionary covering the experiment's hyperparameters (dataset
    settings, random seed, epochs, learning rate, architecture sizes,
    optimizer, loss function, etc.), then call `start_run()` before
    the training loop begins.
  - Inside the training loop, call `log_metrics` at least once per
    epoch with the training loss (and any other relevant metric).
  - After training finishes, call `log_model` to register the trained
    model as a wandb artifact, passing a `description` and any extra
    `metadata` not already covered by `config` (for example, the
    final training loss).
  - Call `finish_run()` once the experiment (including evaluation) is
    complete.
- Do not hardcode a wandb API key in code or notebooks. The key must
  be read from a `WANDB_API_KEY` variable in a local `.env` file, as
  already handled by `WandbExperimentTracker`.

## About notebooks

- Before each code cell, add a text (Markdown) cell briefly explaining
  what the following code cell does or should do.
- Every notebook must start with a header (Markdown cell) containing:
  - Name: Renan Santos Mendes
  - Email: renansantosmendes@gmail.com
  - Course subject: `<COURSE_SUBJECT_PLACEHOLDER>`
  - Program: `<PROGRAM_PLACEHOLDER>`
- Do not use emojis or emoticons anywhere in the notebook (code or text
  cells), to keep the course material looking professional.

## About agents, MCP servers, subagents and skills

These rules apply whenever Claude creates or changes an agent, an MCP
server, a subagent or a skill for this project.

### General principles

- Prefer the smallest extension that solves the problem: a skill for
  reusable know-how, a subagent for an isolated task that would flood the
  main context, an MCP server only when an external system or data
  source must be reached through tools.
- Never hardcode secrets (API keys, tokens, passwords) in any of these
  artifacts. Read them from environment variables or from the local
  `.env` file, and document the required variable names in the README.
- Each artifact must have a single, clear purpose and a description that
  explains when it should be used. Vague descriptions cause wrong
  triggering.
- Keep instructions concrete: state inputs, expected outputs, constraints
  and what to do on failure.
- Whenever an artifact is created or changed, update the documentation
  that references it (README, `docs/`, or the file listing the available
  agents/skills/servers) in the same change.
- If the artifact contains Python code, follow every rule in "About
  Python code" above, including docstrings, type hints, `uv` and unit
  tests under `tests/`.

### Subagents

- Store project subagents as Markdown files in `.claude/agents/`, one
  file per subagent, named in kebab-case (for example
  `.claude/agents/code-reviewer.md`).
- Each file starts with YAML frontmatter containing at least `name`
  (matching the file name) and `description` (one or two sentences
  stating when to delegate to this subagent). Optional keys: `tools`
  (restrict to the minimum required), `model` (only when a specific model
  is justified).
- The body is the system prompt: role, scope, step-by-step procedure,
  output format expected by the caller, and explicit limits (what the
  subagent must not do).
- Grant only the tools the subagent needs. A read-only reviewer must not
  have write or shell-execution tools.
- Subagents do not see the main conversation. The delegating prompt must
  carry all the context the subagent needs (paths, constraints, the exact
  question to answer).
- A subagent must return a concise result to the caller. Ask it to
  summarize findings instead of dumping full file contents.

### Skills

- Store project skills in `.claude/skills/<skill-name>/SKILL.md`, with
  the folder name in kebab-case and matching the `name` field.
- `SKILL.md` starts with YAML frontmatter containing `name` and
  `description`. The description must state what the skill does and the
  triggers that should activate it (file types, keywords, situations).
- Keep `SKILL.md` focused and short. Move long reference material into
  additional files inside the skill folder (for example `reference/`)
  and link to them from `SKILL.md`, so they are loaded only when needed.
- Put deterministic or repetitive logic in scripts inside the skill
  folder (for example `scripts/`) instead of asking the model to
  reproduce it each time. Python scripts in a skill follow the Python
  rules of this file.
- Write instructions in the imperative, with a clear step order and the
  expected output. Include at least one example of invocation when the
  behavior is not obvious.
- Test a new skill on at least one realistic request before considering
  it finished, and refine the description if it does not trigger when it
  should.

### MCP servers

- Declare project-scoped MCP servers in `.mcp.json` at the project root,
  using the `mcpServers` object. Each entry defines `command`, `args` and,
  when needed, `env` with variable references such as `${API_KEY}`
  instead of literal values.
- Prefer existing, well-maintained MCP servers over writing new ones.
  Before adding a server, confirm its source, its maintainer and the
  permissions it requests.
- A custom MCP server written in Python must:
  - be managed with `uv` and declare its dependencies in `pyproject.toml`;
  - expose each tool with a precise name, a docstring that explains the
    tool's purpose, its parameters and its return value, and type hints;
  - validate its inputs and return structured, readable errors instead
    of raising raw exceptions to the client;
  - never log or return secrets;
  - have unit tests under `tests/` that cover each tool, with external
    calls mocked.
- Restrict filesystem and network access of a server to what its tools
  require. Tools that modify external state (send messages, write files,
  call paid APIs) must be clearly named and documented as side-effecting.
- Document each server in the README: purpose, tools exposed, required
  environment variables, and how to start it.

### Agents (autonomous workflows)

- An agent is a named workflow that combines a system prompt, a set of
  tools and, optionally, skills or subagents. Define it in a single
  Markdown or Python file with a clear goal, its tools, its stopping
  conditions and its failure behavior.
- An agent must not take irreversible actions (deleting data, sending
  messages, spending money, pushing to remote repositories) without an
  explicit confirmation step, unless the user has explicitly asked for
  that action.
- Log the key decisions and results of an agent run in a way that a
  human can audit afterwards. Never log secrets or full personal data.
- If the agent calls a model, record the model identifier and the main
  hyperparameters in its configuration, and track experiments following
  the "About experiment tracking" section when training is involved.

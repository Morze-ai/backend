# Backend

backend repository for the project

## Setup

Uses `uv` as a tool to manage the environment.

Run the following:

```bash
uv sync --group dev
```

or

```bash
make setup
```

After installation you can run the commands using `uv run` all use the Makefile to run already defined commands as follows: `make ruff-fix`

## How to use

whenever you make changes, run `make checks` to make sure that:

- format is proper
- it gets linted
- static type checks are good
- tests pass

## Code information regarding commits

Commitzen will stop a commit (it will get stashed, so you will have to unstash it) if it fails to adhere to the following format:

- **feat**: Introduces a new feature.
- **fix**: Fixes an issue, patches a bug.
- **docs**: Documentation changes, comment changes. NO code changes.
- **style**: Code style changes and formatting.
- **refactor**: Code refactor without functionality changes.
- **perf**: Performance improvemenet.
- **test**: Adding or updating tests.
- **build**: Changes to build or dependencies.
- **ci**: Changes to configuration files.
- **chore**: Other changes that don't modify code or test (like initial commits).

### Good

```yaml
fix(commands): handle missing user input gracefully
feat(api): add pagination support
```

### Avoid

```yaml
fix: stuff
feat: commit command introduced
```

## Strucutre

### CLI

Command entry line config and logic

### Data

Data loading and preprocessing

### Models

Model definitions and configuration

### Training

Model training logic

### Utils

Shared utilities across `CLI` and other modules

### Visualization

Plot generation

### Tests

Testing config, experiments, data preprocessing and trainer to ensure it all work correctly

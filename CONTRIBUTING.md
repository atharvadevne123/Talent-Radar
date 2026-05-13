# Contributing to Talent-Radar

Thank you for considering a contribution!

## Development Setup

```bash
git clone https://github.com/atharvadevne123/Talent-Radar
cd Talent-Radar
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pre-commit install
```

## Making Changes

1. Create a feature branch: `git checkout -b feat/my-feature`
2. Write your code with type annotations and docstrings
3. Add tests for any new functionality
4. Run `make lint` and `make test` — both must pass
5. Open a pull request against `main`

## Code Style

- All Python code is linted with **ruff** (`make lint`)
- Type annotations are required on all public functions
- Google-style docstrings for all classes and public methods

## Testing

```bash
make test
```

Coverage must not drop below 80% for new code.

## Commit Message Format

```
type(scope): short description

# Types: feat, fix, docs, test, refactor, ci, chore
```

# Contributing to PreML

Thank you for contributing to PreML.

## Development Setup

1. Clone the repository.
2. Create a virtual environment.
3. Install dependencies.

```bash
git clone https://github.com/pqun7/preml.git
cd preml
python -m venv .venv
. .venv/Scripts/activate
python -m pip install --upgrade pip
pip install -e .
pip install -r requirements.txt
pip install pytest pytest-cov
```

## Running Tests

```bash
pytest
```

## Coding Guidelines

- Follow PEP 8 and clear naming.
- Keep module responsibilities focused.
- Preserve backward compatibility where practical.
- Add tests for every behavior change.
- Keep user-facing errors actionable.

## Pull Request Process

1. Create a feature branch.
2. Add or update tests.
3. Ensure tests pass locally.
4. Update docs (including Usage Guide) for API changes.
5. Open a pull request with a clear summary and rationale.

## Reporting Issues

Please include:

- Python version
- Operating system
- Minimal reproducible example
- Full traceback/error message

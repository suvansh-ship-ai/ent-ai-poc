# Entain AI POC

Sample codebase for the AI Virtual Engineer proof-of-concept.

## Structure

```
src/
├── models.py        # Domain models (User, Payment, BettingSlip)
├── config.py        # App configuration (env vars)
├── exceptions.py    # Custom exception classes
└── __init__.py
tests/
└── __init__.py
requirements.txt
README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

```bash
uvicorn src.main:app --reload
```

## Testing

```bash
pytest tests/ -v
```

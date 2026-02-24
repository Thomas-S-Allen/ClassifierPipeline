# Django Classifier DB UI

This is a Django implementation of the classifier DB web UI, kept alongside:
- `classifier_db_ui.py` (desktop Qt)
- `classifier_db_web.py` (WSGI web app)

## Install

From the repo root, install Django in your environment if needed:

```bash
./venv/bin/pip install "Django>=4.2,<5"
```

## Run

```bash
cd /Users/thomasallen/Code/ClassifierPipeline_codex_ui/django_classifier_ui
../venv/bin/python manage.py runserver 127.0.0.1:8010
```

Open:

- http://127.0.0.1:8010/

## Behavior

- Same connection/query/update flow as the other UI versions
- Score column is sortable by clicking the table header
- ADS title and abstract lookup support is included

import csv
import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
ARTIFACTS = ROOT / 'artifacts'
LABELS = ['Chinese', 'English', 'French', 'German', 'Italian', 'Japanese',
          'Korean', 'Russian', 'Spanish', 'Vietnamese']
DISPLAY = {label: label for label in LABELS}
MAX_LENGTH = 64


def model_text(value):
    """Keep Latin diacritics as useful features; normalize case and punctuation."""
    value = unicodedata.normalize('NFKC', value).strip().lower()
    value = value.replace('’', "'").replace('ʼ', "'").replace('‐', '-')
    value = re.sub(r'\s+', ' ', value)
    if not value or not any(c.isalpha() for c in value):
        raise ValueError('Enter a name using Latin letters, with optional accents, spaces or hyphens.')
    for c in value:
        if c not in " '-" and not (c.isalpha() and 'LATIN' in unicodedata.name(c, '')):
            raise ValueError('Enter a name using Latin letters, with optional accents, spaces or hyphens.')
    if len(value) > MAX_LENGTH:
        raise ValueError(f'The normalized name must contain no more than {MAX_LENGTH} characters.')
    return value


def normalize_name(value):
    """Accent-insensitive key for deduplication, group splitting and ambiguity lookup."""
    value = model_text(value)
    value = value.replace('đ', 'd').replace('ł', 'l').replace('ø', 'o')
    value = value.replace('æ', 'ae').replace('œ', 'oe').replace('ß', 'ss')
    value = ''.join(c for c in unicodedata.normalize('NFKD', value)
                    if not unicodedata.combining(c))
    value = re.sub(r'\s+', ' ', value)
    if not re.fullmatch(r"[a-z '\-]+", value) or not re.search('[a-z]', value):
        raise ValueError('Enter a name using Latin letters, with optional accents, spaces or hyphens.')
    if len(value) > MAX_LENGTH:
        raise ValueError(f'The normalized name must contain no more than {MAX_LENGTH} characters.')
    return value


def read_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_rows(path, rows, fields):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

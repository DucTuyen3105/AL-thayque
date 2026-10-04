"""Download five public sources, remove ambiguity, split independent name groups."""
import argparse
import hashlib
import re
import urllib.request
import zipfile
from collections import Counter, defaultdict

from sklearn.model_selection import train_test_split

from .common import DATA, LABELS, normalize_name, read_rows, write_json, write_rows

SOURCES = {
    'pytorch_names.zip': 'https://download.pytorch.org/tutorial/data.zip',
    'behind_name.csv': 'https://raw.githubusercontent.com/Anwarvic/Behind-The-Name/master/dataset.csv',
    'popular_common-forenames-by-country.csv': 'https://raw.githubusercontent.com/sigpwned/popular-names-by-country-dataset/main/common-forenames-by-country.csv',
    'popular_common-surnames-by-country.csv': 'https://raw.githubusercontent.com/sigpwned/popular-names-by-country-dataset/main/common-surnames-by-country.csv',
    'vietnamese_boy.txt': 'https://raw.githubusercontent.com/duyet/vietnamese-namedb/master/boy.txt',
    'vietnamese_girl.txt': 'https://raw.githubusercontent.com/duyet/vietnamese-namedb/master/girl.txt',
    'wikidata_given.csv': 'https://raw.githubusercontent.com/reglab/disaggregation/main/data/name_race_data/p_race_given_firstname.csv',
    'wikidata_surname.csv': 'https://raw.githubusercontent.com/reglab/disaggregation/main/data/name_race_data/p_race_given_surname.csv',
}
COUNTRIES = {'CN': 'Chinese', 'GB': 'English', 'FR': 'French', 'DE': 'German',
             'IT': 'Italian', 'JP': 'Japanese', 'KR': 'Korean', 'RU': 'Russian',
             'ES': 'Spanish', 'VN': 'Vietnamese'}


class Groups:
    def __init__(self):
        self.parent = {}

    def find(self, key):
        self.parent.setdefault(key, key)
        if self.parent[key] != key:
            self.parent[key] = self.find(self.parent[key])
        return self.parent[key]

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        self.parent[max(a, b)] = min(a, b)


def prepare(download=True, seed=42):
    raw = DATA / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    previous_path = raw / 'manifest.json'
    import json
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
    manifest = {}
    for filename, url in SOURCES.items():
        path = raw / filename
        if not path.exists():
            if not download:
                raise FileNotFoundError(f'{path}: run prepare without --offline first')
            print(f'Downloading {url}', flush=True)
            request = urllib.request.Request(url, headers={'User-Agent': 'NameOrigin-Educational/1.0'})
            blob = urllib.request.urlopen(request, timeout=90).read()
            path.write_bytes(blob)
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        if filename in previous and previous[filename]['sha256'] != checksum:
            raise ValueError(f'Checksum changed: {filename}. Review source before regenerating manifest.')
        manifest[filename] = {'url': url, 'sha256': checksum, 'bytes': path.stat().st_size}
    write_json(previous_path, manifest)

    entries = {}
    groups = Groups()
    aliases = {}
    source_counts = Counter()
    skipped = Counter()

    def add(name, label, source, kind, family=None):
        # Behind the Name uses (1), (2), ... as disambiguation IDs, not name text.
        name = re.sub(r'\s*\(\d+\)\s*$', '', name).strip()
        try:
            key = normalize_name(name)
        except ValueError:
            skipped[source] += 1
            return
        groups.find(key)
        row = entries.setdefault(key, {'names': set(), 'labels': set(), 'sources': set(), 'kinds': set()})
        row['names'].add(name)
        row['labels'].add(label)
        row['sources'].add(source)
        row['kinds'].add(kind)
        source_counts[source] += 1
        if family:
            if family in aliases:
                groups.union(key, aliases[family])
            aliases[family] = key

    with zipfile.ZipFile(raw / 'pytorch_names.zip') as archive:
        for label in LABELS:
            for name in archive.read(f'data/names/{label}.txt').decode('utf-8-sig').splitlines():
                add(name, label, 'pytorch', 'surname')
    all_given = read_rows(raw / 'behind_name.csv')
    for row in all_given:
        # Only explicit modern usage labels; do not turn Ancient Germanic into German.
        origins = [re.sub(r'\s*\([^)]*\)', '', x).strip()
                   for x in row['origin'].strip('[]').split(',')]
        for label in sorted(set(origins) & set(LABELS)):
            add(row['name'], label, 'behind_name', 'given_name')
    # Use independent components, not thousands of recombinations of a few syllables.
    viet_components = set()
    for filename in ['vietnamese_boy.txt', 'vietnamese_girl.txt']:
        for name in (raw / filename).read_text(encoding='utf-8-sig').splitlines():
            viet_components.update(name.split())
    for component in sorted(viet_components):
        add(component, 'Vietnamese', 'vietnamese_namedb', 'given_component')
    wiki_counts = {}
    wiki_rejected = 0
    for kind, filename in [('given_name', 'wikidata_given.csv'), ('surname', 'wikidata_surname.csv')]:
        rows = read_rows(raw / filename)
        wiki_counts[kind] = len(rows)
        for row in rows:
            distribution = {k: float(v) for k, v in row.items() if k != 'name'}
            winner = max(distribution, key=distribution.get)
            # Silver labels: association within this six-group cohort, not etymology.
            if winner.capitalize() not in LABELS or distribution[winner] < .95:
                wiki_rejected += 1
                continue
            add(row['name'], winner.capitalize(), 'wikidata_silver', kind)
    raw_country_rows = {}
    for kind, filename in [('given_name', 'popular_common-forenames-by-country.csv'),
                           ('surname', 'popular_common-surnames-by-country.csv')]:
        rows = read_rows(raw / filename)
        raw_country_rows[kind] = len(rows)
        for row in rows:
            label = COUNTRIES.get(row['Country'])
            if not label:
                continue
            family = f"popular:{kind}:{row['Name Group']}"
            # Country popularity is a proxy for naming tradition, not proof of etymology.
            for name in set([row['Romanized Name'], row['Localized Name']]):
                if name:
                    add(name, label, 'popular', kind, family)

    bundles = defaultdict(list)
    for key in sorted(entries):
        bundles[groups.find(key)].append(key)
    clean, ambiguous = [], []
    for group, keys in sorted(bundles.items()):
        labels = sorted(set().union(*(entries[k]['labels'] for k in keys)))
        for key in keys:
            row = entries[key]
            representative = min(row['names'], key=lambda n: (-sum(ord(c) > 127 for c in n), n))
            record = {'name': representative, 'normalized': key,
                      'language': '|'.join(labels), 'group': group,
                      'sources': '|'.join(sorted(row['sources'])),
                      'kind': '|'.join(sorted(row['kinds']))}
            (clean if len(labels) == 1 else ambiguous).append(record)
    group_ids = sorted({r['group'] for r in clean})
    group_label = {r['group']: r['language'] for r in clean}
    train_ids, other_ids = train_test_split(group_ids, test_size=.30, random_state=seed,
                                          stratify=[group_label[g] for g in group_ids])
    val_ids, test_ids = train_test_split(other_ids, test_size=.50, random_state=seed,
                                       stratify=[group_label[g] for g in other_ids])
    assignment = {g: split for split, ids in [('train', train_ids), ('val', val_ids), ('test', test_ids)] for g in ids}
    for row in clean:
        row['split'] = assignment[row['group']]
    fields = ['name', 'normalized', 'language', 'group', 'sources', 'kind']
    write_rows(DATA / 'ambiguous.csv', ambiguous, fields)
    write_rows(DATA / 'names.csv', clean, fields + ['split'])
    for split in ['train', 'val', 'test']:
        write_rows(DATA / f'{split}.csv', [r for r in clean if r['split'] == split], fields + ['split'])
    stats = {'seed': seed, 'labels': LABELS, 'raw_behind_name_rows': len(all_given),
             'raw_wikidata_rows': wiki_counts, 'wikidata_association_threshold': .95,
             'wikidata_rows_not_selected': wiki_rejected, 'vietnamese_unique_components': len(viet_components),
             'raw_popular_rows': raw_country_rows, 'accepted_name_label_records': dict(source_counts),
             'rejected_non_latin_or_invalid': dict(skipped),
             'unique_normalized_names_before_conflict_removal': len(entries),
             'clean_names': len(clean), 'independent_groups': len(group_ids),
             'ambiguous_names_excluded': len(ambiguous),
             'split_counts': dict(Counter(r['split'] for r in clean)),
             'per_class': {label: {**dict(Counter(r['split'] for r in clean if r['language'] == label)),
                                  'total': sum(r['language'] == label for r in clean),
                                  'groups': sum(group_label[g] == label for g in group_ids)} for label in LABELS}}
    write_json(DATA / 'stats.json', stats)
    print(stats, flush=True)
    return stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    prepare(not args.offline, args.seed)

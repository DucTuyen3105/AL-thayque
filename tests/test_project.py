import hashlib
import json
import unittest
from collections import Counter, defaultdict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from app import create_app
from name_origin.common import ARTIFACTS, DATA, LABELS, model_text, normalize_name, read_rows
from name_origin.model import encode
from name_origin.predict import Predictor


class DataIntegrityTests(unittest.TestCase):
    def test_source_checksums(self):
        manifest = json.loads((DATA / 'raw/manifest.json').read_text())
        self.assertEqual(len(manifest), 8)
        for filename, metadata in manifest.items():
            self.assertEqual(hashlib.sha256((DATA / 'raw' / filename).read_bytes()).hexdigest(), metadata['sha256'])

    def test_no_name_or_alias_leakage(self):
        rows = read_rows(DATA / 'names.csv')
        assignments = defaultdict(set)
        keys = set()
        for r in rows:
            self.assertEqual(r['normalized'], normalize_name(r['name']))
            self.assertIn(r['language'], LABELS)
            self.assertNotIn(r['normalized'], keys)
            keys.add(r['normalized'])
            assignments[r['group']].add(r['split'])
        self.assertTrue(all(len(v) == 1 for v in assignments.values()))
        ambiguous = {r['normalized'] for r in read_rows(DATA / 'ambiguous.csv')}
        self.assertFalse(keys & ambiguous)
        for split in ['train', 'val', 'test']:
            self.assertEqual({r['normalized'] for r in rows if r['split'] == split},
                             {r['normalized'] for r in read_rows(DATA / f'{split}.csv')})
            self.assertEqual(set(Counter(r['language'] for r in rows if r['split'] == split)), set(LABELS))

    def test_known_romanizations_in_same_split(self):
        from name_origin.prepare import COUNTRIES
        clean = {r['normalized']: r for r in read_rows(DATA / 'names.csv')}
        families = defaultdict(set)
        for kind in ['forenames', 'surnames']:
            for row in read_rows(DATA / f'raw/popular_common-{kind}-by-country.csv'):
                if row['Country'] not in COUNTRIES:
                    continue
                try:
                    key = normalize_name(row['Romanized Name'])
                except ValueError:
                    continue
                if key in clean:
                    families[kind + row['Name Group']].add(clean[key]['split'])
        self.assertTrue(all(len(v) <= 1 for v in families.values()))

    def test_exported_scores_match_predictions(self):
        report = json.loads((ARTIFACTS / 'metrics.json').read_text())
        rows = read_rows(ARTIFACTS / 'test_predictions.csv')
        y, pred = [r['truth'] for r in rows], [r['prediction'] for r in rows]
        self.assertAlmostEqual(accuracy_score(y, pred), report['test']['accuracy'])
        self.assertAlmostEqual(f1_score(y, pred, labels=LABELS, average='macro'), report['test']['macro_f1'])
        self.assertEqual(len(rows), len(read_rows(DATA / 'test.csv')))


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = Predictor()
        cls.client = create_app().test_client()

    def test_accents_and_padding_preserve_prediction(self):
        self.assertEqual(normalize_name('NGUYỄN'), normalize_name('Nguyen'))
        self.assertNotEqual(model_text('Nguyễn'), model_text('Nguyen'))
        self.assertEqual(self.predictor.predict('Nguyễn')['candidates'],
                         self.predictor.predict('NGUYỄN')['candidates'])
        x, lengths = encode(['nguyen'], self.predictor.vocab)
        with torch.inference_mode():
            full = self.predictor.model(x, lengths)
            shortened = self.predictor.model(x[:, :lengths.max()], lengths)
        torch.testing.assert_close(full, shortened, atol=1e-6, rtol=1e-5)

    def test_full_name_aggregation_is_explicit(self):
        result = self.predictor.predict('Nguyễn Đức Tuyên', top_k=10)
        self.assertEqual(result['method'], 'mean_of_components')
        self.assertEqual(len(result['components']), 3)
        self.assertTrue(result['warnings'])
        self.assertAlmostEqual(sum(p['score'] for p in result['candidates']), 1, places=5)
        x, lengths = encode(['nguyễn', 'đức', 'tuyên'], self.predictor.vocab)
        with torch.inference_mode():
            expected = (self.predictor.model(x, lengths) / self.predictor.temperature).softmax(1).mean(0)
        for candidate in result['candidates']:
            self.assertAlmostEqual(candidate['score'], float(expected[LABELS.index(candidate['language'])]), places=5)

    def test_shared_name_retains_ambiguity(self):
        result = self.predictor.predict('Lee')
        self.assertTrue(result['uncertain'])
        self.assertIn('lee', result['ambiguous_components'])
        self.assertGreater(len(result['ambiguous_components']['lee']), 1)

    def test_invalid_requests_and_web_rendering(self):
        self.assertEqual(self.client.get('/').status_code, 200)
        self.assertEqual(self.client.get('/health').json['status'], 'ok')
        with self.client.get('/report/confusion_matrix.png') as response:
            self.assertEqual(response.status_code, 200)
        self.assertEqual(self.client.get('/report/model.pt').status_code, 404)
        for payload in [{}, {'name': ''}, {'name': '12345'}, {'name': '山本'},
                        {'name': 'a' * 65}, {'name': ['Nguyen']}, {'name': 'Nguyen', 'mode': 'invalid'},
                        {'name': '<script>alert(1)</script>'}]:
            with self.subTest(payload=payload):
                self.assertEqual(self.client.post('/api/predict', json=payload).status_code, 400)
        self.assertEqual(self.client.post('/api/predict', data='not-json').status_code, 400)
        valid = self.client.post('/api/predict', json={'name': 'Yamamoto'})
        self.assertEqual(valid.status_code, 200)
        self.assertEqual(len(valid.json['candidates']), 3)


if __name__ == '__main__':
    unittest.main()

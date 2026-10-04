import argparse
import json
import re

import torch

from .common import ARTIFACTS, DATA, DISPLAY, model_text, normalize_name, read_rows
from .model import NameLSTM, encode


class Predictor:
    def __init__(self, model_path=ARTIFACTS / 'model.pt'):
        torch.set_num_threads(2)
        if not model_path.exists():
            raise FileNotFoundError('No trained model found. Run: python -m name_origin.train')
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        self.vocab = checkpoint['vocab']
        self.labels = checkpoint['labels']
        self.temperature = checkpoint['temperature']
        self.model = NameLSTM(len(self.vocab) + 2, len(self.labels), hidden_dim=checkpoint['hidden_dim'])
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        self.ambiguities = {r['normalized']: r['language'].split('|')
                            for r in read_rows(DATA / 'ambiguous.csv')}

    def predict(self, name, mode='auto', top_k=3):
        if not isinstance(name, str):
            raise ValueError('The name must be a text string.')
        if mode not in ['auto', 'single', 'components']:
            raise ValueError('Mode must be auto, single or components.')
        normalized = model_text(name)
        parts = normalized.split() if mode == 'components' or (mode == 'auto' and ' ' in normalized) else [normalized]
        # Keep hyphenated names intact: min-jun is a single sequence.
        if any(not re.search('[a-z]', p) for p in parts):
            raise ValueError('Each name component must contain a letter.')
        x, lengths = encode(parts, self.vocab)
        with torch.inference_mode():
            probabilities = (self.model(x, lengths) / self.temperature).softmax(1)
        # Full-name support is an explicit, unvalidated heuristic; no dictionary label overrides.
        averaged = probabilities.mean(0)
        order = averaged.argsort(descending=True)
        best, second = float(averaged[order[0]]), float(averaged[order[1]])
        ambiguous = {p: self.ambiguities[normalize_name(p)] for p in parts if normalize_name(p) in self.ambiguities}
        uncertain = best < .55 or best - second < .15 or bool(ambiguous)
        candidates = [{'language': self.labels[int(i)], 'display': DISPLAY[self.labels[int(i)]],
                       'score': round(float(averaged[i]), 6)} for i in order[:top_k]]
        warnings = []
        if len(parts) > 1:
            warnings.append('Full-name scores average individual name components. This approach has not been evaluated on full names.')
        if ambiguous:
            warnings.append('Some name components appear in multiple origin groups, so a single origin cannot be confirmed.')
        if any(len(p) < 3 for p in parts):
            warnings.append('Very short names provide fewer clues about their origin.')
        if any(self.vocab.get(c, 1) == 1 for c in normalized):
            warnings.append('Some characters were not seen during training.')
        return {'name': name.strip(), 'normalized': normalized, 'candidates': candidates,
                'uncertain': uncertain, 'status': 'Uncertain result' if uncertain else 'Most likely name origin',
                'method': 'mean_of_components' if len(parts) > 1 else 'single_name_lstm',
                'components': [{'name': part, 'language': self.labels[int(p.argmax())],
                                'score': round(float(p.max()), 6)} for part, p in zip(parts, probabilities)],
                'ambiguous_components': ambiguous, 'warnings': warnings,
                'note': 'Model scores across 10 name-origin groups do not determine a person’s nationality, ethnicity or spoken language.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict linguistic naming tradition using the trained LSTM.')
    parser.add_argument('names', nargs='+')
    parser.add_argument('--mode', choices=['auto', 'single', 'components'], default='auto')
    args = parser.parse_args()
    predictor = Predictor()
    for name in args.names:
        try:
            print(json.dumps(predictor.predict(name, args.mode), ensure_ascii=False, indent=2))
        except ValueError as error:
            parser.error(str(error))

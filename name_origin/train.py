import argparse
import copy
import json
import os
import random
import time

import matplotlib
os.environ.setdefault('MPLCONFIGDIR', '/tmp/name-origin-matplotlib')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, log_loss, top_k_accuracy_score)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .common import ARTIFACTS, DATA, LABELS, model_text, read_rows, write_json, write_rows
from .model import NameLSTM, build_vocab, encode


def metrics(y, probabilities):
    pred = probabilities.argmax(axis=1)
    return {'accuracy': float(accuracy_score(y, pred)),
            'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
            'macro_f1': float(f1_score(y, pred, labels=list(range(len(LABELS))), average='macro', zero_division=0)),
            'top3_accuracy': float(top_k_accuracy_score(y, probabilities, k=3, labels=list(range(len(LABELS))))),
            'log_loss': float(log_loss(y, probabilities, labels=list(range(len(LABELS))))),
            'per_class': classification_report(y, pred, labels=list(range(len(LABELS))), target_names=LABELS,
                                                output_dict=True, zero_division=0)}


def train(args):
    ARTIFACTS.mkdir(exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    rows = {split: read_rows(DATA / f'{split}.csv') for split in ['train', 'val', 'test']}
    # Assert both exact normalization and known romanization groups stay in one split.
    for column in ['normalized', 'group']:
        sets = {split: {r[column] for r in records} for split, records in rows.items()}
        assert not (sets['train'] & sets['val'] or sets['train'] & sets['test'] or sets['val'] & sets['test'])
    names = {split: [model_text(r['name']) for r in records] for split, records in rows.items()}
    labels = {split: np.array([LABELS.index(r['language']) for r in records]) for split, records in rows.items()}
    # Train-only accent removal augmentation; both forms share the original group.
    extra_names, extra_labels = [], []
    for row in rows['train']:
        if model_text(row['name']) != row['normalized']:
            extra_names.append(row['normalized'])
            extra_labels.append(LABELS.index(row['language']))
    names['train'] += extra_names
    labels['train'] = np.concatenate([labels['train'], np.array(extra_labels, dtype=np.int64)])
    vocab = build_vocab(names['train'])
    loaders = {}
    for split in rows:
        x, lengths = encode(names[split], vocab)
        dataset = TensorDataset(x, lengths, torch.tensor(labels[split], dtype=torch.long))
        loaders[split] = DataLoader(dataset, batch_size=args.batch_size, shuffle=split == 'train')
    model = NameLSTM(len(vocab) + 2, len(LABELS), hidden_dim=args.hidden_dim)
    counts = np.bincount(labels['train'], minlength=len(LABELS))
    # Moderate weighting: fully inverse-frequency weights can overfit tiny classes.
    weights = torch.tensor((counts.max() / counts) ** .5, dtype=torch.float32)
    weights /= weights.mean()
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=.03)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history, best_f1, best_state, stale, best_epoch = [], -1, None, 0, 0

    def logits_for(split):
        model.eval()
        with torch.inference_mode():
            return torch.cat([model(x, lengths) for x, lengths, _ in loaders[split]])

    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        for x, lengths, y in loaders['train']:
            optimizer.zero_grad()
            loss = loss_fn(model(x, lengths), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            total_loss += loss.item() * len(y)
        val_probs = logits_for('val').softmax(1).numpy()
        val = metrics(labels['val'], val_probs)
        item = {'epoch': epoch, 'train_loss': total_loss / len(names['train']),
                'val_accuracy': val['accuracy'], 'val_macro_f1': val['macro_f1']}
        history.append(item)
        print(f"Epoch {epoch:02d}: loss={item['train_loss']:.4f}, val_acc={val['accuracy']:.4f}, val_macro_f1={val['macro_f1']:.4f}", flush=True)
        if val['macro_f1'] > best_f1 + .0005:
            best_f1, best_state, stale, best_epoch = val['macro_f1'], copy.deepcopy(model.state_dict()), 0, epoch
        else:
            stale += 1
        if stale >= args.patience:
            print('Early stopping; restore best validation macro-F1 checkpoint.', flush=True)
            break
    model.load_state_dict(best_state)
    # Calibration uses validation only. Test is evaluated once after all selections.
    val_logits = logits_for('val')
    val_y = torch.tensor(labels['val'], dtype=torch.long)
    temperatures = np.linspace(.5, 4, 71)
    losses = [nn.functional.cross_entropy(val_logits / float(t), val_y).item() for t in temperatures]
    temperature = float(temperatures[int(np.argmin(losses))])
    validation = metrics(labels['val'], (val_logits / temperature).softmax(1).numpy())
    # Baseline selection on the same validation split, never on the test results.
    best_baseline, best_baseline_f1, baseline_c = None, -1, None
    for c in [.5, 2, 8]:
        baseline = make_pipeline(TfidfVectorizer(analyzer='char', ngram_range=(1, 4), sublinear_tf=True),
                                 LogisticRegression(C=c, class_weight='balanced', max_iter=500))
        baseline.fit(names['train'], labels['train'])
        score = f1_score(labels['val'], baseline.predict(names['val']), average='macro')
        if score > best_baseline_f1:
            best_baseline, best_baseline_f1, baseline_c = baseline, score, c
    test_logits = logits_for('test')
    test_probs = (test_logits / temperature).softmax(1).numpy()
    test = metrics(labels['test'], test_probs)
    baseline_test = metrics(labels['test'], best_baseline.predict_proba(names['test']))
    ascii_x, ascii_lengths = encode([r['normalized'] for r in rows['test']], vocab)
    model.eval()
    with torch.inference_mode():
        ascii_logits = torch.cat([model(ascii_x[i:i+args.batch_size], ascii_lengths[i:i+args.batch_size])
                                  for i in range(0, len(ascii_x), args.batch_size)])
    ascii_test = metrics(labels['test'], (ascii_logits / temperature).softmax(1).numpy())
    source_subsets = {}
    for subset, mask in {
        'dictionary_or_tutorial': np.array([bool(set(r['sources'].split('|')) & {'pytorch', 'behind_name', 'vietnamese_namedb'}) for r in rows['test']]),
        'wikidata_silver_only': np.array([r['sources'] == 'wikidata_silver' for r in rows['test']]),
        'single_token': np.array([' ' not in r['normalized'] for r in rows['test']]),
    }.items():
        if mask.any():
            source_subsets[subset] = {'rows': int(mask.sum()), **metrics(labels['test'][mask], test_probs[mask])}
    group_counts = {split: len({r['group'] for r in records}) for split, records in rows.items()}
    report = {'architecture': 'Embedding(32) -> LSTM -> Dropout(0.3) -> Linear -> calibrated Softmax',
              'parameters': sum(p.numel() for p in model.parameters()), 'config': vars(args),
              'best_epoch': best_epoch, 'temperature': temperature, 'validation': validation,
              'test': test, 'baseline_tfidf_logistic': {'C': baseline_c, 'validation_macro_f1': best_baseline_f1, 'test': baseline_test},
              'test_subsets': source_subsets,
              'test_all_accents_removed': ascii_test, 'train_augmented_rows': len(extra_names),
              'majority_class_test_accuracy': float(np.mean(labels['test'] == int(counts.argmax()))),
              'split_rows': {s: len(v) for s, v in rows.items()}, 'split_independent_groups': group_counts,
              'elapsed_seconds': round(time.monotonic() - started, 2),
              'test_scope': 'Held-out Latin-script surnames and given names with a single label among ten selected classes; ambiguous names excluded. Includes Wikidata silver labels derived from subgroup association >=0.95. Not a full-name benchmark or independent etymology validation.',
              'source_manifest': json.loads((DATA / 'raw/manifest.json').read_text())}
    torch.save({'state_dict': model.state_dict(), 'vocab': vocab, 'labels': LABELS,
                'preprocessing': 'latin_keep_diacritics_v1',
                'hidden_dim': args.hidden_dim, 'temperature': temperature}, ARTIFACTS / 'model.pt')
    write_json(ARTIFACTS / 'metrics.json', report)
    write_json(ARTIFACTS / 'history.json', history)
    write_rows(ARTIFACTS / 'test_predictions.csv',
               [{'name': r['name'], 'truth': r['language'], 'prediction': LABELS[int(p.argmax())],
                 'score': float(p.max()), 'correct': int(LABELS[int(p.argmax())] == r['language']),
                 'sources': r['sources'], 'group': r['group']} for r, p in zip(rows['test'], test_probs)],
               ['name', 'truth', 'prediction', 'score', 'correct', 'sources', 'group'])
    cm = confusion_matrix(labels['test'], test_probs.argmax(1), labels=list(range(len(LABELS))))
    write_rows(ARTIFACTS / 'confusion_matrix.csv',
               [{'truth': label, **dict(zip(LABELS, map(int, cm[i])))} for i, label in enumerate(LABELS)], ['truth'] + LABELS)
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(cm / cm.sum(1, keepdims=True), cmap='Blues', vmin=0, vmax=1)
    ax.set(xticks=range(len(LABELS)), yticks=range(len(LABELS)), xticklabels=LABELS, yticklabels=LABELS,
           xlabel='Predicted', ylabel='True', title='Held-out test: row-normalized confusion matrix')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=8,
                    color='white' if cm[i, j] / cm[i].sum() > .5 else 'black')
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(ARTIFACTS / 'confusion_matrix.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot([h['epoch'] for h in history], [h['train_loss'] for h in history])
    axes[0].set(title='Training loss', xlabel='Epoch')
    for key in ['val_accuracy', 'val_macro_f1']:
        axes[1].plot([h['epoch'] for h in history], [h[key] for h in history], label=key)
    axes[1].set(title='Validation metrics', xlabel='Epoch', ylim=(0, 1))
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(ARTIFACTS / 'training_curves.png', dpi=180)
    plt.close(fig)
    print(json.dumps({k: v for k, v in test.items() if k != 'per_class'}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train LSTM, calibrate, evaluate and export reports.')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--lr', type=float, default=.002)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size, args.hidden_dim, args.threads) < 1 or args.lr <= 0:
        parser.error('Training parameters must be positive.')
    train(args)

"""Regression test: transformer predictions must come back in input order.

A previous version used `Trainer.predict`, which (transformers 5.x, length-grouped sampling) returned rows
shuffled: validation looked excellent while test accuracy was at chance level.
"""

import numpy as np
import pytest
import torch
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

from tvt.transformer import predict_proba

WORDS = ["film", "super", "nul", "pas", "bon", "acteurs", "je", "le", "recommande", "ennui"]


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    vocab = tmp_path_factory.mktemp("tok") / "vocab.txt"
    vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *WORDS]), encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(vocab))
    torch.manual_seed(0)
    config = BertConfig(vocab_size=len(WORDS) + 5, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                        intermediate_size=32, max_position_embeddings=64, num_labels=3)
    return BertForSequenceClassification(config).eval(), tokenizer


def test_predictions_follow_input_order(tiny_model):
    model, tokenizer = tiny_model
    rng = np.random.default_rng(0)
    # Varied lengths so that length-sorting actually reorders the batch.
    texts = [" ".join(rng.choice(WORDS, size=rng.integers(1, 30))) for _ in range(37)]

    batched = predict_proba(model, tokenizer, texts, max_length=32, batch_size=8)
    one_by_one = np.vstack([predict_proba(model, tokenizer, [t], max_length=32, batch_size=1) for t in texts])

    assert batched.shape == (37, 3)
    np.testing.assert_allclose(batched, one_by_one, atol=1e-5)
    np.testing.assert_allclose(batched.sum(axis=1), 1.0, atol=1e-6)

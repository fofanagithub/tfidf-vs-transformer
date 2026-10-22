import numpy as np
import pandas as pd
import pytest

from tvt.analysis import has_negation, slice_metrics
from tvt.baseline import build_vectorizer, load_predictions, run_baseline
from tvt.config import BaselineConfig, DataConfig, apply_overrides
from tvt.data import fingerprint, from_frames, normalize_text, stratified_sample

POS = ["film magnifique", "acteurs excellents", "un vrai chef d'oeuvre", "je recommande vivement", "superbe réalisation"]
NEG = ["film nul", "acteurs mauvais", "un vrai navet", "je ne recommande pas", "réalisation ratée"]


def toy_frame(n_per_class: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = [(f"{rng.choice(POS)} {rng.choice(POS)} numéro {seed}-{i}", 1) for i in range(n_per_class)]
    rows += [(f"{rng.choice(NEG)} {rng.choice(NEG)} numéro {seed}-{i}", 0) for i in range(n_per_class)]
    return pd.DataFrame(rows, columns=["text", "label"])


@pytest.fixture
def splits():
    train = toy_frame(150, seed=0)
    test = toy_frame(50, seed=1)
    test.loc[0, "text"] = train.loc[0, "text"]  # planted leak
    test.loc[0, "label"] = train.loc[0, "label"]
    return from_frames(train, test, ["neg", "pos"], DataConfig(name="toy", language="fr"), seed=0)


def test_normalize_text_collapses_whitespace_only():
    assert normalize_text("  Très  BON\n film ! ") == "Très BON film !"


def test_splits_are_disjoint_and_leak_free(splits):
    test_texts = set(splits.test["text"])
    assert test_texts.isdisjoint(splits.train["text"])
    assert test_texts.isdisjoint(splits.val["text"])
    assert set(splits.train["text"]).isdisjoint(splits.val["text"])


def test_splits_are_deterministic(splits):
    again = from_frames(toy_frame(150, 0), toy_frame(50, 1), ["neg", "pos"], DataConfig(name="toy"), seed=0)
    assert fingerprint(again.val) == fingerprint(splits.val)


def test_stratified_sample_preserves_balance():
    df = toy_frame(500, seed=3)
    sample = stratified_sample(df, 100, seed=0)
    assert len(sample) == 100
    assert sample["label"].value_counts().tolist() == [50, 50]
    assert len(stratified_sample(df, None, seed=0)) == len(df)


def test_vectorizer_spec_parsing():
    union = build_vectorizer("word_1_2+char_2_5", BaselineConfig())
    assert [name for name, _ in union.transformer_list] == ["word_1_2", "char_2_5"]
    with pytest.raises(ValueError):
        build_vectorizer("bigram", BaselineConfig())


def test_baseline_learns_toy_task(splits, tmp_path):
    cfg = BaselineConfig(feature_sets=["word_1_1", "word_1_2"], C_grid=[1.0, 4.0], min_df=1)
    meta = run_baseline(splits, cfg, tmp_path, save_model=False)
    y, pred, probs = load_predictions(tmp_path / "predictions.csv")
    assert meta["val_macro_f1"] > 0.95
    assert (pred == y).mean() > 0.95
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert not (tmp_path / "model.joblib").exists()
    assert (tmp_path / "val_grid.csv").exists()


def test_negation_lexicon():
    texts = pd.Series(["Je ne l'ai pas aimé", "Génial", "It wasn't good", "Great"])
    assert has_negation(texts[:2], "fr").tolist() == [True, False]
    assert has_negation(texts[2:], "en").tolist() == [True, False]


def test_slice_metrics_shapes(splits):
    y = splits.test["label"].to_numpy()
    out = slice_metrics(splits.test, {"baseline": y, "transformer": 1 - y}, "fr")
    assert (out["baseline_accuracy"] == 1.0).all()
    assert (out["transformer_accuracy"] == 0.0).all()


def test_config_overrides_parse_yaml_values():
    raw = apply_overrides({"transformer": {"num_epochs": 3}}, ["transformer.num_epochs=1", "transformer.seeds=[1, 2]", "data.max_train_samples=null"])
    assert raw["transformer"] == {"num_epochs": 1, "seeds": [1, 2]}
    assert raw["data"]["max_train_samples"] is None

# Transformer fine-tuné vs baseline TF-IDF

**Un fine-tuning de Transformer vaut-il vraiment son coût face à une baseline TF-IDF bien réglée ?**
Ce projet répond à la question par une comparaison contrôlée sur la classification de sentiment en français
(critiques Allociné, CamemBERT) et une expérience de contrôle en anglais (AG News, DistilBERT) :
performance avec intervalles de confiance, tests statistiques appariés, coût (temps, taille, débit),
efficacité en données et analyse d'erreurs.

> 📊 Résultats : voir [Résultats](#résultats) (le rapport complet est généré dans `results/<expérience>/report/report.md`).

---

## Question et hypothèses

| # | Hypothèse | Comment elle est testée |
|---|---|---|
| H1 | Sur le sentiment, le transformer dépasse significativement une baseline TF-IDF **bien réglée**. | Δ macro-F1 avec IC 95 % (bootstrap apparié) + test de McNemar, pour chaque seed |
| H2 | L'avantage du transformer est concentré sur les textes contenant une **négation** et sur les textes longs. | Exactitude par tranche (longueur en quartiles, présence de négation) |
| H3 | Avec peu de données annotées, l'écart se réduit ou s'inverse. | Courbe d'apprentissage de 100 à 10 000 exemples, plusieurs tirages |
| H4 | Sur une tâche thématique (AG News), les mots-clés suffisent : l'écart est faible. | Même protocole sur `configs/ag_news.yaml` |
| H5 | Le gain de performance a un coût de plusieurs ordres de grandeur en calcul. | Temps d'entraînement, débit d'inférence, taille du modèle |

## Protocole expérimental

Les choix ci-dessous visent à ce que l'écart mesuré soit dû **aux modèles et à rien d'autre**.

- **Mêmes données pour les deux modèles.** Un unique module (`tvt/data.py`) produit train / validation / test ;
  nettoyage minimal et identique (espaces), doublons supprimés, et **suppression des fuites** (textes du test
  présents dans train/val). Une empreinte SHA-256 du test est vérifiée au moment du rapport.
  _Constat sur Allociné :_ 41 avis du train et 15 de la validation sont aussi présents dans le test officiel ;
  ils sont retirés, sinon ils gonfleraient les scores, surtout ceux du modèle qui mémorise le plus.
- **Longueur de séquence mesurée, pas supposée.** Avec le tokenizer CamemBERT, les avis font 93 tokens en
  médiane, 350 au 95e centile et 448 au 99e : `max_length=384` (≈ 3 % de textes tronqués, taux enregistré
  à chaque run) ; batchs regroupés par longueur pour limiter le padding.
- **Le test n'est utilisé qu'une fois.** Toute sélection (hyperparamètres de la baseline, meilleure époque du
  transformer, seed « représentatif » des figures) se fait sur la **validation**.
- **Baseline forte, pas un homme de paille.** TF-IDF mots (1-2 grammes) + caractères (2-5 grammes, `char_wb`),
  TF sous-linéaire, régression logistique ; grille sur les familles de features × régularisation `C`.
  Les accents et les mots d'une lettre sont conservés (importants en français).
- **Variance du fine-tuning prise en compte.** 3 seeds ; on rapporte moyenne ± écart-type et la
  significativité **pour chaque seed** (conclusion valide seulement si elle tient pour tous).
- **Incertitude quantifiée.** IC 95 % par bootstrap percentile sur le test ; différence appariée par bootstrap ;
  test exact/χ² de McNemar sur les prédictions appariées.
- **Au-delà de l'exactitude.** Macro-F1 (robuste au déséquilibre), calibration (ECE + diagramme de fiabilité),
  matrices de confusion, coût d'entraînement et d'inférence (tokenisation incluse), taille du modèle.
- **Reproductibilité.** Configurations YAML versionnées, seeds fixés, versions des bibliothèques et matériel
  enregistrés dans `results/<exp>/run_*.json`, exécutions reprenables après interruption.

## Résultats

_À compléter avec les chiffres de `results/allocine/report/report.md` après l'exécution complète (GPU)._

| Modèle | Macro-F1 test (IC 95 %) | ECE | Entraînement | Inférence |
|---|---|---|---|---|
| TF-IDF + LogReg | – | – | – | – |
| CamemBERT-base (moy. ± σ, 3 seeds) | – | – | – | – |

Points clés à rédiger à partir du rapport : réponse à H1–H5, quelques exemples de `disagreements.csv`
illustrant ce que le transformer comprend (négation, ironie, contraste « mais ») et ce qu'il rate.

## Structure

```
configs/            allocine.yaml (principale), ag_news.yaml (contrôle), smoke.yaml (test CPU rapide)
src/tvt/
  data.py           chargement HF, splits stratifiés, déduplication, anti-fuite, empreinte
  baseline.py       TF-IDF + régression logistique, sélection sur validation
  transformer.py    fine-tuning (HF Trainer, early stopping, multi-seeds)
  evaluation.py     métriques, ECE, bootstrap, McNemar, différence appariée
  learning_curve.py efficacité en données (reprenable)
  analysis.py       tranches (longueur, négation), exemples de désaccord
  report.py         rapport Markdown + figures + summary.json
  cli.py            point d'entrée `tvt`
tests/              tests unitaires (sans réseau) des statistiques, des splits et de la baseline
notebooks/          run_on_colab.ipynb pour l'exécution GPU
```

## Utilisation

```bash
python -m venv .venv && .venv/Scripts/activate   # Linux/macOS : source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

```bash
tvt all --config configs/smoke.yaml          # vérification de bout en bout sur CPU (~5 min)
tvt baseline --config configs/allocine.yaml  # baseline seule (CPU suffit)
tvt transformer --config configs/allocine.yaml
tvt learning-curve --config configs/allocine.yaml
tvt report --config configs/allocine.yaml    # régénère le rapport sans réentraîner
```

Toute valeur se surcharge en ligne de commande, par ex. `--set transformer.num_epochs=3 --set data.max_train_samples=20000`.
Le fine-tuning complet demande un GPU : utiliser [notebooks/run_on_colab.ipynb](notebooks/run_on_colab.ipynb) (T4 gratuit).

## Limites

- Une seule famille de baseline (modèle linéaire) ; un SVM ou NB-SVM donnerait des résultats proches, une
  baseline à embeddings (fastText, sentence-transformers + LogReg) serait un point intermédiaire intéressant.
- Hyperparamètres du transformer fixés aux valeurs usuelles (lr 2e-5, 2 époques) plutôt que recherchés,
  par budget de calcul ; ce choix **favorise la baseline**, ce qui rend une victoire du transformer plus solide.
- Les débits d'inférence dépendent du matériel (baseline sur CPU, transformer sur GPU) : le matériel est
  indiqué dans chaque tableau.
- Le lexique de négation est une heuristique de surface.

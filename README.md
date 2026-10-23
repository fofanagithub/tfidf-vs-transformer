# Transformer fine-tuné vs baseline TF-IDF

**Un fine-tuning de Transformer vaut-il vraiment son coût face à une baseline TF-IDF bien réglée ?**
Ce projet répond à la question par une comparaison contrôlée sur la classification de sentiment en français
(critiques Allociné, CamemBERT) et une expérience de contrôle en anglais (AG News, DistilBERT) :
performance avec intervalles de confiance, tests statistiques appariés, coût (temps, taille, débit),
efficacité en données et analyse d'erreurs.

> 📄 **[Rapport scientifique complet](RAPPORT.md)** : méthode étape par étape, figures, résultats, discussion et limites.
> 📊 Résumé des résultats : [Résultats](#résultats) (tables générées dans `results/allocine/report/report.md`).

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

Allociné, test officiel dédupliqué (19 928 avis), train complet (159 420 avis). Rapport complet :
[results/allocine/report/report.md](results/allocine/report/report.md).

| Modèle | Macro-F1 test (IC 95 %) | ECE | Paramètres | Entraînement | Inférence |
|---|---|---|---|---|---|
| TF-IDF (mots 1-2 + car. 2-5) + LogReg, C=4 | 94.44 [94.11, 94.76] | 1.34 | 0.9 M | 4.7 min (CPU) | 662 ex./s (CPU) |
| CamemBERT-base, 3 seeds | **97.42 ± 0.02** (pire seed : 97.41 [97.20, 97.62]) | 1.63 | 110.6 M | 37 min / seed (T4) | 393 ex./s (T4) |

**H1 — confirmée.** +2.96 à +3.01 points de macro-F1 selon le seed (IC 95 % apparié ≈ [+2.7, +3.3]),
McNemar p < 10⁻⁷⁵ pour chacun des 3 seeds. Le taux d'erreur passe de 5.55 % à 2.58 % : **−54 % d'erreurs**.
Sur les 19 928 avis, environ 800 ne sont bien classés que par CamemBERT, contre environ 210 pour la baseline seule.
La variance entre seeds (σ = 0.02 pt) est négligeable devant l'écart.

![Courbe d'apprentissage](results/allocine/report/figures/learning_curve.png)

**H3 — confirmée, avec un point de bascule net entre 500 et 1 000 exemples.** Sous 500 exemples annotés, TF-IDF
l'emporte, et le fine-tuning est instable : à 250 exemples l'écart-type entre tirages atteint 22 points, certains
runs s'effondrant sur la classe majoritaire. Dès 1 000 exemples, CamemBERT (93.0) dépasse la baseline entraînée
sur **10 fois plus de données** (91.8 à 10 000), et avec 2 500 exemples (94.8) il dépasse la baseline entraînée
sur le train complet (94.4, soit 64 fois plus de données).

**H2 — partiellement confirmée, effet modeste.** Le gain croît avec la longueur des avis : +2.1 points sur le
quart le plus court (≤ 33 mots) contre +3.3 à +3.7 sur les deux quarts les plus longs. Le gain est un peu plus
fort sur les avis contenant une négation (+3.1 contre +2.4), mais le lexique de surface marque 76 % des avis :
il discrimine mal. Ces écarts entre tranches ne sont pas testés statistiquement.

**Analyse qualitative** (`disagreements.csv`). CamemBERT corrige surtout les **contrastes et concessions** que
le sac de n-grammes ne peut pas composer : « Le moins bon des 4 même si c'est un film sympa à voir » (négatif),
« ce n'est pas non plus un navet » (positif), « divertissant sans être très original. Mais… ». Les erreurs
communes aux deux modèles, commises avec une confiance ≈ 1.0, sont en majorité du **bruit d'étiquette** :
les classes viennent de la note, pas du texte (« Mon dieu, quel beau film ! […] (ceci est un fake) » étiqueté
négatif). Ces cas bornent le score atteignable.

**H5 — nuancée.** Le transformer a 120 fois plus de paramètres et un modèle 11 fois plus lourd. Il demande un
GPU pour s'entraîner en un temps raisonnable : 37 min sur T4 contre 4.7 min sur CPU pour la baseline. Même sur
GPU, il traite 1.7 fois moins d'exemples par seconde que la baseline sur CPU. L'écart de coût est donc d'un à
deux ordres de grandeur, pas « plusieurs ».

**Calibration.** Les deux modèles sont bien calibrés globalement (ECE 1.3 contre 1.6 point). CamemBERT est
cependant **surconfiant** dans la zone 0.5–0.9 et se trompe parfois avec une confiance de 0.999 ; un *temperature
scaling* sur la validation serait la prochaine étape si les probabilités devaient servir à décider.

**H4 — non testée** à ce stade (`configs/ag_news.yaml` est prêt).

**Conclusion.** Avec plus de ~1 000 exemples annotés, le fine-tuning de CamemBERT réduit les erreurs de moitié,
un gain robuste et significatif. Sous ce seuil, ou avec des contraintes de latence et de coût sans GPU, une
baseline TF-IDF bien réglée reste le meilleur choix.

### Incident détecté et corrigé

Le premier rapport montrait CamemBERT à 97.4 % en validation mais **50 % sur le test**, avec une ECE de 49 points.
Une telle divergence est incompatible avec un vrai modèle : elle signalait un défaut de pipeline. Cause : avec
transformers 5.x, l'échantillonnage « groupé par longueur » activé pour l'entraînement s'applique aussi à
`Trainer.predict`, qui renvoyait les prédictions dans un ordre mélangé. Les prédictions étaient donc associées aux
mauvaises étiquettes. Ce mélange étant déterministe (graine connue), [scripts/recover_predictions.py](scripts/recover_predictions.py)
reconstruit exactement la permutation et remet les probabilités dans l'ordre, sans réentraînement. La
récupération se valide d'elle-même : 97.41–97.45 % sur le test pour chaque seed, cohérent avec la validation,
alors qu'une permutation fausse donne ~50 %. L'inférence passe désormais par une boucle explicite qui préserve
l'ordre (`predict_proba`), couverte par un test de non-régression (`tests/test_transformer.py`).

## Structure

```
configs/            allocine.yaml (principale), ag_news.yaml (contrôle), smoke.yaml (test CPU rapide)
src/tvt/
  data.py           chargement HF, splits stratifiés, déduplication, anti-fuite, empreinte
  baseline.py       TF-IDF + régression logistique, sélection sur validation
  transformer.py    fine-tuning (HF Trainer, early stopping, multi-seeds), inférence ordonnée
  evaluation.py     métriques, ECE, bootstrap, McNemar, différence appariée
  learning_curve.py efficacité en données (reprenable)
  analysis.py       tranches (longueur, négation), exemples de désaccord
  report.py         rapport Markdown + figures + summary.json
  cli.py            point d'entrée `tvt`
tests/              tests unitaires (sans réseau) : statistiques, splits, baseline, ordre des prédictions
scripts/            recover_predictions.py (correction ponctuelle, voir « Incident détecté et corrigé »)
notebooks/          run_on_colab.ipynb pour l'exécution GPU
results/allocine/   rapport, figures, métadonnées et prédictions de l'expérience principale
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

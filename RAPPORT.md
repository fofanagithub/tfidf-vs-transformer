# Transformer fine-tuné vs baseline TF-IDF : une comparaison contrôlée sur la classification de sentiment en français

**Auteur :** Mamadou Moussa Fofana · **Code :** [github.com/fofanagithub/tfidf-vs-transformer](https://github.com/fofanagithub/tfidf-vs-transformer)

---

## Résumé

Ce travail mesure ce que le fine-tuning d'un Transformer pré-entraîné apporte, et ce qu'il coûte, par rapport à
une baseline classique TF-IDF + régression logistique **soigneusement réglée**, sur la classification de
sentiment de critiques de films en français (Allociné, ≈ 200 000 avis). Les deux modèles sont entraînés et évalués
sur des données strictement identiques, dédupliquées et débarrassées des fuites train → test. Toute sélection se
fait sur la validation, et le test n'est utilisé qu'une fois.

CamemBERT-base atteint **97,42 ± 0,02 %** de macro-F1 (3 seeds) contre **94,44 %** [IC 95 % : 94,11 – 94,76] pour
la baseline, soit **54 % d'erreurs en moins**. L'écart (+3,0 points) est significatif pour chaque seed
(McNemar, p < 10⁻⁷⁵). Une courbe d'apprentissage montre un **point de bascule entre 500 et 1 000 exemples
annotés** : en dessous, la baseline l'emporte ; au-dessus, CamemBERT entraîné sur 1 000 avis dépasse la baseline
entraînée sur 10 000. Le gain est concentré sur les phrases à **contraste ou concession**, que le sac de
n-grammes ne peut pas composer. Il a un coût : 120 fois plus de paramètres, un GPU pour l'entraînement et un
débit d'inférence plus faible.

---

## 1. Introduction

### 1.1 Contexte

Depuis BERT (Devlin et al., 2019), le fine-tuning de modèles pré-entraînés est devenu l'approche par défaut en
classification de texte. Pourtant, en production, une baseline linéaire sur features TF-IDF reste très utilisée :
elle est rapide, peu coûteuse, interprétable, et souvent étonnamment compétitive (Wang & Manning, 2012). Beaucoup
de comparaisons publiées opposent toutefois un Transformer à une baseline mal réglée, sans mesure d'incertitude
et sur un seul seed. L'écart qu'elles rapportent n'est donc pas fiable.

### 1.2 Question de recherche

> **Un Transformer fine-tuné surpasse-t-il significativement une baseline TF-IDF bien réglée, dans quelles
> conditions (quantité de données, type de texte) et à quel coût ?**

### 1.3 Hypothèses

Les hypothèses ont été formulées **avant** les expériences :

| # | Hypothèse | Test prévu |
|---|---|---|
| H1 | Sur le sentiment, le transformer dépasse significativement une baseline TF-IDF bien réglée. | Δ macro-F1 avec IC 95 % (bootstrap apparié) + McNemar, pour chaque seed |
| H2 | Son avantage se concentre sur les textes avec négation et les textes longs. | Exactitude par tranche de longueur et selon la présence de négation |
| H3 | Avec peu de données annotées, l'écart se réduit ou s'inverse. | Courbe d'apprentissage de 100 à 10 000 exemples, 3 tirages |
| H4 | Sur une tâche thématique, les mots-clés suffisent et l'écart est faible. | Même protocole sur AG News |
| H5 | Le gain se paie de plusieurs ordres de grandeur en coût de calcul. | Temps d'entraînement, débit d'inférence, taille, paramètres |

---

## 2. Données

### 2.1 Corpus

**Allociné** (Blard, 2019 ; `tblard/allocine` sur le Hugging Face Hub) : critiques de films en français
rédigées par des internautes, étiquetées *positif* / *négatif* d'après la note attribuée par l'auteur. Le corpus
est fourni avec un découpage officiel train / validation / test, conservé ici.

### 2.2 Nettoyage et contrôle des fuites

Le traitement est volontairement **minimal et identique pour les deux modèles** :

1. **Normalisation des espaces uniquement.** La casse, la ponctuation et les accents sont conservés : le
   tokenizer de CamemBERT les exploite, et le vectoriseur TF-IDF gère lui-même la mise en minuscules.
2. **Suppression des textes vides et des doublons** à l'intérieur de chaque split.
3. **Suppression des fuites entre splits.** Un texte présent à la fois dans le test et dans le train ou la
   validation biaise l'évaluation, surtout en faveur du modèle qui mémorise le plus (ici le transformer).
   Le contrôle a trouvé **41 avis du train et 15 avis de la validation présents mot pour mot dans le test
   officiel** ; ils ont été retirés du train et de la validation.
4. **Empreinte du test.** Un hachage SHA-256 du test (`c8157e050b1d8bd7`) est enregistré. Le rapport vérifie que
   toutes les prédictions portent bien sur ce même test.

### 2.3 Statistiques après nettoyage

| Split | Avis | Négatifs | Positifs | Mots (moyenne) | Mots (médiane) | Mots (95e centile) |
|---|---|---|---|---|---|---|
| Train | 159 420 | 79 103 (49,6 %) | 80 317 (50,4 %) | 91,4 | 66 | 254 |
| Validation | 19 933 | 10 177 (51,1 %) | 9 756 (48,9 %) | 92,6 | 66 | 260 |
| Test | 19 928 | 10 360 (52,0 %) | 9 568 (48,0 %) | 92,6 | 67 | 259 |

Les classes sont quasi équilibrées. La macro-F1 reste néanmoins la métrique principale, car elle pèse les deux
classes à égalité indépendamment de leur fréquence.

### 2.4 Longueur des séquences

La longueur maximale des séquences du transformer est un compromis entre coût et information perdue. Elle a été
**mesurée plutôt que supposée**, sur 5 000 avis tokenisés avec le tokenizer de CamemBERT :

| Médiane | 90e centile | 95e centile | 99e centile | > 256 tokens | > 512 tokens |
|---|---|---|---|---|---|
| 93 | 292 | 350 | 448 | 14,0 % | 0,1 % |

La valeur courante de 256 aurait tronqué 14 % des avis. Le choix s'est porté sur **384 tokens** : le taux de
troncature mesuré pendant l'entraînement est de **3,7 %**.

---

## 3. Méthodes

### 3.1 Baseline : TF-IDF + régression logistique

Le but est d'opposer au transformer **la meilleure baseline linéaire raisonnable**, pas un homme de paille.

- **Représentation.** Union de deux vectoriseurs TF-IDF :
  - **mots**, unigrammes et bigrammes, pour capter les expressions courtes comme « pas terrible » ou
    « vraiment nul » ;
  - **caractères**, 2 à 5 grammes (`char_wb`), robustes aux fautes d'orthographe, aux flexions et aux néologismes
    fréquents dans les avis d'internautes.

  Le TF est sous-linéaire ($1 + \log \mathrm{tf}$) pour atténuer l'effet des mots répétés. Les termes présents
  dans moins de 2 documents sont ignorés, et chaque vectoriseur est limité à 500 000 features. Les accents sont
  conservés (« où » ≠ « ou ») ainsi que les mots d'une lettre (« a », « y »).
- **Classifieur.** Régression logistique régularisée en L2. Elle produit des probabilités, nécessaires à la
  mesure de calibration.
- **Sélection de modèle.** Recherche sur grille de 3 familles de features × 4 valeurs de $C$. Chaque
  configuration est entraînée sur le train et évaluée sur la validation (macro-F1). La meilleure est réentraînée
  sur le train puis appliquée une seule fois au test.

**Résultats de la grille (macro-F1 en validation) :**

| Features | C = 0,25 | C = 1 | C = 4 | C = 16 | Nb de features |
|---|---|---|---|---|---|
| mots 1-1 | 91,80 | 92,54 | 92,81 | 92,54 | 86 068 |
| mots 1-2 | 92,79 | 93,91 | 94,25 | 94,31 | 500 000 |
| mots 1-2 + caractères 2-5 | 93,36 | 94,17 | **94,35** | 94,14 | 910 454 |

Les bigrammes apportent +1,5 point, les n-grammes de caractères un gain marginal (+0,04 point). Le plateau
atteint autour de 94,3 confirme que la baseline est proche de son maximum dans cette famille de modèles.

### 3.2 Transformer : CamemBERT-base fine-tuné

- **Modèle.** CamemBERT-base (Martin et al., 2020) : architecture RoBERTa, 110 M de paramètres, pré-entraînée sur
  138 Go de français (OSCAR). C'est le choix naturel pour du texte français. Une tête de classification linéaire
  est ajoutée sur le token `<s>`.
- **Hyperparamètres** (valeurs standard de la littérature, non recherchées faute de budget) :

  | Paramètre | Valeur |
  |---|---|
  | Taux d'apprentissage | 2 × 10⁻⁵, warmup linéaire sur 6 % des pas puis décroissance linéaire |
  | Weight decay | 0,01 |
  | Batch | 32 |
  | Époques | 2 (early stopping, patience 1, sur la macro-F1 de validation) |
  | Longueur max. | 384 tokens, padding dynamique, batchs regroupés par longueur |
  | Précision | fp16 mixte (GPU Tesla T4) |
  | Seeds | 13, 42, 2024 |

- **Sélection.** Le modèle est évalué sur la validation à chaque époque et la meilleure époque est conservée.
- **Plusieurs seeds.** Le fine-tuning est connu pour être sensible à l'initialisation de la tête et à l'ordre des
  données (Dodge et al., 2020). Une conclusion n'est retenue que si elle tient **pour chacun des 3 seeds**.

### 3.3 Protocole d'évaluation

| Principe | Mise en œuvre |
|---|---|
| Données identiques | Un seul module produit les splits ; l'empreinte du test est vérifiée |
| Test utilisé une seule fois | Hyperparamètres de la baseline, meilleure époque du transformer et seed représentatif des figures sont tous choisis sur la validation |
| Équité de l'inférence | Le temps mesuré inclut la tokenisation et la vectorisation à partir du texte brut |
| Reproductibilité | Configurations YAML versionnées, seeds fixés, versions des bibliothèques et matériel enregistrés à chaque exécution |

### 3.4 Métriques et tests statistiques

**Macro-F1.** C'est la moyenne non pondérée des F1 de chaque classe :
$\mathrm{F1}_{\text{macro}} = \frac{1}{K}\sum_{k=1}^{K} \frac{2 P_k R_k}{P_k + R_k}$.
L'exactitude (accuracy) est aussi rapportée.

**Intervalles de confiance.** Ils sont obtenus par **bootstrap percentile** (Efron & Tibshirani, 1993) : le
test est rééchantillonné avec remise 1 000 fois, la métrique est recalculée à chaque fois, et on retient les
quantiles 2,5 % et 97,5 %.

**Comparaison appariée.** Les deux modèles prédisent les mêmes exemples, donc leurs erreurs sont corrélées et
comparer deux intervalles séparés serait trop conservateur. Deux tests appariés sont utilisés :

- **Test de McNemar** (McNemar, 1947 ; recommandé par Dietterich, 1998). Soit $b$ le nombre d'exemples que seule
  la baseline classe correctement, et $c$ le nombre que seul le transformer classe correctement. La statistique
  $\chi^2 = \frac{(|b - c| - 1)^2}{b + c}$ suit une loi du χ² à 1 degré de liberté sous l'hypothèse d'égalité
  des performances. On utilise la version binomiale exacte si $b + c < 50$.
- **Bootstrap apparié** sur la différence de macro-F1. Les mêmes indices sont tirés pour les deux modèles ; on en
  déduit un IC 95 % de la différence et une p-value bilatérale. Celle-ci ne peut pas descendre sous 1/1 000 et
  est alors rapportée comme « < 0,001 ».

**Calibration.** On mesure l'erreur de calibration attendue (ECE, Guo et al., 2017). Les prédictions sont
réparties en $M = 15$ intervalles de confiance égaux $B_m$, et
$\mathrm{ECE} = \sum_{m} \frac{|B_m|}{n} \left| \mathrm{acc}(B_m) - \mathrm{conf}(B_m) \right|$.
Un modèle bien calibré qui annonce 90 % de confiance a raison 90 % du temps. Cette propriété compte dès que les
probabilités servent à décider (seuil, revue humaine).

### 3.5 Expériences complémentaires

- **Courbe d'apprentissage (H3).** Sous-échantillons stratifiés du train de 100, 250, 500, 1 000, 2 500, 5 000 et
  10 000 avis, 3 tirages chacun (seeds 13, 42, 2024). Les deux modèles sont réentraînés à chaque fois avec le même
  protocole : la grille complète pour la baseline, 2 époques pour le transformer. L'évaluation se fait sur le
  test complet.
- **Analyse par tranches (H2).** Le test est découpé en quartiles de longueur (en mots) et selon la présence d'au
  moins un marqueur de négation (*ne, n', pas, jamais, plus, rien, aucun(e), ni, sans*).
- **Analyse qualitative.** Pour chacun des cas suivants, on extrait les erreurs commises avec la plus forte
  confiance : seul le transformer a raison, seule la baseline a raison, les deux se trompent.

---

## 4. Résultats

### 4.1 Performance globale (H1)

| Modèle | Accuracy (IC 95 %) | Macro-F1 (IC 95 %) | ECE (pts) |
|---|---|---|---|
| TF-IDF + LogReg (mots 1-2 + car. 2-5, C = 4) | 94,45 [94,12 ; 94,77] | 94,44 [94,11 ; 94,76] | 1,34 |
| CamemBERT, seed 13 | 97,41 [97,20 ; 97,63] | 97,41 [97,20 ; 97,62] | 1,66 |
| CamemBERT, seed 42 | 97,41 [97,20 ; 97,64] | 97,41 [97,20 ; 97,64] | 1,64 |
| CamemBERT, seed 2024 | 97,45 [97,24 ; 97,66] | 97,45 [97,24 ; 97,66] | 1,58 |
| **CamemBERT, moyenne ± écart-type** | **97,42 ± 0,02** | **97,42 ± 0,02** | **1,63 ± 0,04** |

**Significativité de l'écart (CamemBERT − baseline) :**

| Seed | Δ Macro-F1 (pts) [IC 95 %] | p (bootstrap apparié) | Seule la baseline a raison ($b$) | Seul CamemBERT a raison ($c$) | χ² McNemar | p (McNemar) |
|---|---|---|---|---|---|---|
| 13 | +2,96 [+2,66 ; +3,28] | < 0,001 | 205 | 795 | 346,9 | 2,0 × 10⁻⁷⁷ |
| 42 | +2,96 [+2,66 ; +3,29] | < 0,001 | 215 | 805 | 340,1 | 6,0 × 10⁻⁷⁶ |
| 2024 | +3,01 [+2,70 ; +3,32] | < 0,001 | 209 | 807 | 350,8 | 2,8 × 10⁻⁷⁸ |

**Lecture.** Les intervalles de confiance des deux modèles sont disjoints. L'écart (+3,0 points) est 150 fois plus
grand que l'écart-type entre seeds (σ = 0,02 point). Le taux d'erreur passe de **5,55 % à 2,58 %, soit 54 %
d'erreurs en moins**. Pour chaque seed, CamemBERT corrige environ 800 erreurs de la baseline et n'en introduit
qu'environ 210. **H1 est confirmée.**

![Matrices de confusion](results/allocine/report/figures/confusion_matrices.png)

*Figure 1 – Matrices de confusion normalisées par ligne (pourcentage et effectif). À gauche la baseline, à droite
CamemBERT (seed 13, seed de validation médiane).*

| | Rappel nég. | Rappel pos. | Faux positifs (nég. → pos.) | Faux négatifs (pos. → nég.) |
|---|---|---|---|---|
| Baseline | 94,26 % | 94,66 % | 595 | 511 |
| CamemBERT (seed 13) | 96,93 % | 97,93 % | 318 | 198 |

Les deux modèles se trompent plus souvent sur les avis négatifs, classés à tort positifs. Ce point est rediscuté
en 5.2.

### 4.2 Dynamique d'entraînement

| Seed | Macro-F1 val. après époque 1 | après époque 2 | Perte val. après époque 1 | après époque 2 | Perte train (fin) |
|---|---|---|---|---|---|
| 13 | 97,27 | **97,41** | 0,079 | 0,090 | 0,055 |
| 42 | 97,28 | **97,46** | 0,085 | 0,090 | 0,054 |
| 2024 | 97,28 | **97,33** | 0,087 | 0,088 | 0,066 |

Pour les trois seeds, la meilleure époque est la 2ᵉ, et l'early stopping ne s'est pas déclenché. La **perte de
validation augmente légèrement** entre les époques 1 et 2 alors que la macro-F1 progresse. C'est la signature
d'un modèle qui classe mieux mais devient plus **sûr de lui sur ses erreurs**, ce que confirme l'analyse de
calibration (4.6). Une 3ᵉ époque aurait probablement accentué cette tendance sans gain notable.

### 4.3 Efficacité en données (H3)

![Courbe d'apprentissage](results/allocine/report/figures/learning_curve.png)

*Figure 2 – Macro-F1 sur le test en fonction du nombre d'exemples d'entraînement (échelle logarithmique).
Moyenne de 3 tirages ; la bande montre ± 1 écart-type.*

| Exemples | Baseline | CamemBERT | Écart |
|---|---|---|---|
| 100 | **74,85 ± 3,10** | 48,02 ± 8,33 | −26,8 |
| 250 | **81,26 ± 0,84** | 58,43 ± 22,36 | −22,8 |
| 500 | **84,77 ± 0,54** | 83,01 ± 10,37 | −1,8 |
| 1 000 | 87,36 ± 0,27 | **93,02 ± 0,47** | +5,7 |
| 2 500 | 89,62 ± 0,10 | **94,78 ± 0,14** | +5,2 |
| 5 000 | 90,75 ± 0,08 | **95,68 ± 0,04** | +4,9 |
| 10 000 | 91,79 ± 0,13 | **96,28 ± 0,13** | +4,5 |
| 159 420 (complet) | 94,44 | **97,42 ± 0,02** | +3,0 |

**Lecture.**
- **Le point de bascule se situe entre 500 et 1 000 exemples.** En dessous, la baseline l'emporte nettement ; au
  dessus, CamemBERT domine.
- **L'efficacité en données de CamemBERT est remarquable.** Avec 1 000 avis (93,0), il dépasse la baseline
  entraînée sur 10 000 (91,8). Avec 2 500 avis (94,8), il dépasse la baseline entraînée sur le train complet
  (94,4), soit 64 fois plus de données. Ce gain vient du pré-entraînement : le modèle connaît déjà la langue et
  n'apprend que la tâche.
- **Avec très peu de données, le fine-tuning est instable.** À 250 exemples, l'écart-type atteint 22 points :
  le tirage 13 s'effondre à 32,7 de macro-F1, c'est-à-dire qu'il prédit quasiment toujours la même classe, alors
  que les deux autres atteignent environ 71. La baseline, convexe, est stable (σ < 1 point dès 250 exemples).
- **L'écart se réduit quand les données augmentent** : +5,7 points à 1 000 exemples, +3,0 sur le train complet.
  La baseline continue de progresser avec les données, alors que le transformer approche du plafond fixé par le
  bruit d'étiquette (5.2).

**H3 est confirmée**, avec une précision importante sur la cause (voir 6.2) : sous 500 exemples, la règle fixe
de 2 époques ne laisse au transformer qu'une poignée de mises à jour, par exemple 8 pas de gradient pour
100 exemples. Une partie de son retard tient donc au protocole, pas seulement au manque de données.

### 4.4 Analyse par tranches (H2)

![Exactitude par tranche](results/allocine/report/figures/slices.png)

*Figure 3 – Exactitude par quartile de longueur et selon la présence de négation (CamemBERT : seed 13).*

| Tranche | n | Acc. baseline | Acc. CamemBERT | Gain |
|---|---|---|---|---|
| 1–33 mots | 4 992 | 94,39 | 96,47 | +2,08 |
| 34–67 mots | 5 029 | 94,89 | 97,69 | +2,80 |
| 68–131 mots | 4 940 | 94,35 | 98,00 | +3,65 |
| 132–380 mots | 4 967 | 94,16 | 97,48 | +3,32 |
| avec négation | 15 127 | 94,43 | 97,55 | +3,12 |
| sans négation | 4 801 | 94,52 | 96,96 | +2,44 |

**Lecture.**
- **Longueur.** Le gain est le plus faible sur les avis courts (+2,1) et plus élevé sur les avis moyens à longs
  (+3,3 à +3,7). Dans un avis long, le sentiment se construit souvent par opposition (« le début est lent, mais… »).
  Le transformer peut pondérer ces segments selon le contexte ; le sac de n-grammes les additionne. Sur les avis
  très courts, une poignée de mots suffit et la baseline est déjà bonne.
- **Négation.** Le gain est un peu plus élevé sur les avis avec négation (+3,1 contre +2,4). Mais le lexique de
  surface marque 76 % des avis comme négatifs grammaticalement. Trop large, il distingue mal les cas où la
  négation porte réellement sur le sentiment.
- Ces différences de gain entre tranches n'ont **pas été testées statistiquement**.

**H2 est partiellement confirmée** : la tendance va dans le sens attendu, mais l'effet est modeste, et une analyse
plus fine de la portée de la négation serait nécessaire pour conclure.

### 4.5 Analyse qualitative des désaccords

Les erreurs se recouvrent largement. CamemBERT commet 516 erreurs (seed 13), dont **311 (60 %) sont aussi des
erreurs de la baseline**. Les extraits ci-dessous sont abrégés.

**Seul CamemBERT a raison : contraste, concession, double négation.** La baseline s'y trompe avec une confiance
supérieure à 0,98.
- « *Le moins bon des 4 même si c'est un film sympa à voir.* » — négatif. La baseline voit « sympa » et
  « bon » ; CamemBERT comprend que « même si » subordonne l'éloge.
- « *…ce n'est pas la production de l'année 2001, mais ce n'est pas non plus un navet…* » — positif. Une double
  négation et le mot « navet » piègent la baseline.
- « *C'est divertissant sans être très original. Mais j'ai trouvé Anthony Hopkins très bon.* » — négatif. Des
  compliments partiels ne suffisent pas à rendre le jugement global positif.

**Seule la baseline a raison : avis ambigus ou étiquettes douteuses.** Ici, CamemBERT se trompe avec une
confiance supérieure à 0,99.
- « *Ce troisième volet de la saga n'est pas franchement une réussite !!* » — étiqueté **positif**, alors que le
  texte est clairement négatif. CamemBERT a ici « raison » contre l'étiquette.
- « *Certes c'est toujours très drôle et bien réalisé, mais on sent […] que la série s'essouffle…* » — étiqueté
  positif, un avis mitigé où la note de l'auteur tranche.

**Les deux modèles se trompent : surtout du bruit d'étiquette.** Les deux erreurs ont une confiance proche de 1,0.
- « *Mon dieu, quel beau film ! […] A voir absolument ! (ceci est un fake)* » — étiqueté négatif.
- « *Un film si mauvais que ça en devient cosmique. Tout est raté…* » — étiqueté positif : un « nanar » apprécié
  au second degré, que le texte seul ne permet pas de trancher.

### 4.6 Calibration

![Diagramme de fiabilité](results/allocine/report/figures/reliability.png)

*Figure 4 – Diagramme de fiabilité : exactitude observée en fonction de la confiance annoncée (10 intervalles).
La diagonale correspond à une calibration parfaite.*

Les deux modèles sont **bien calibrés globalement** : ECE de 1,3 point pour la baseline et 1,6 pour CamemBERT.
La forme des courbes diffère toutefois :
- la baseline est légèrement **sous-confiante** : quand elle annonce 75 %, elle a raison environ 81 % du temps ;
- CamemBERT est **sur-confiant** dans la zone intermédiaire : quand il annonce 55 à 85 %, il n'a raison que
  47 à 69 % du temps. Il commet aussi des erreurs à 0,999 de confiance (4.5).

Peu d'exemples tombent dans cette zone : CamemBERT annonce plus de 95 % de confiance pour 96,4 % des avis
(contre 68,9 % pour la baseline), d'où un ECE global faible. Mais si les probabilités servaient à décider, par exemple pour envoyer en revue humaine les cas
incertains, il faudrait recalibrer CamemBERT, par exemple avec un *temperature scaling* ajusté sur la validation.

### 4.7 Coût (H5)

| | TF-IDF + LogReg | CamemBERT-base | Rapport |
|---|---|---|---|
| Paramètres | 0,91 M | 110,6 M | × 121 |
| Taille du modèle | 38,6 Mo | 442,5 Mo (fp32) | × 11 |
| Entraînement (train complet) | 4,7 min sur CPU | 37 min par seed sur GPU T4 | × 8 en temps, sur un matériel bien plus coûteux |
| Recherche d'hyperparamètres | 12 configurations, ≈ 4 min d'ajustement (hors vectorisation) | aucune (valeurs standard) | — |
| Inférence (tokenisation incluse) | 662 avis/s sur CPU | 393 avis/s sur GPU T4 | × 1,7 plus lent, malgré le GPU |

**Lecture.** L'écart de coût est d'**un à deux ordres de grandeur**, pas « plusieurs » comme le supposait H5.
Le transformer est aussi plus lent à l'inférence alors qu'il dispose d'un GPU, et sur CPU l'écart serait bien plus
grand. **H5 est donc nuancée** : le coût est réel et non négligeable, mais il reste tout à fait raisonnable au
regard d'une division par deux des erreurs.

---

## 5. Discussion

### 5.1 Synthèse des hypothèses

| # | Verdict | Élément décisif |
|---|---|---|
| H1 | ✅ Confirmée | +3,0 pts de macro-F1, p < 10⁻⁷⁵ pour chaque seed, −54 % d'erreurs |
| H2 | 🟡 Partielle | Gain plus élevé sur les avis longs (+3,3 à +3,7 contre +2,1) et avec négation (+3,1 contre +2,4), non testé statistiquement |
| H3 | ✅ Confirmée | Bascule entre 500 et 1 000 exemples, avec un biais de protocole à faible n (6.2) |
| H4 | ⬜ Non testée | Expérience AG News prête mais non exécutée |
| H5 | 🟡 Nuancée | Coût de × 10 à × 100, pas de plusieurs ordres de grandeur |

### 5.2 Pourquoi le transformer gagne, et où il plafonne

Le TF-IDF représente un avis comme un **ensemble de n-grammes sans ordre** : chaque mot contribue au score
indépendamment des autres. Il ne peut donc pas représenter qu'un « mais » inverse le poids de ce qui précède,
ni qu'un « même si » subordonne un éloge. Les bigrammes ne captent ces relations qu'à une distance de deux mots.
CamemBERT construit des représentations **contextuelles** : le sens de « bon » dépend de toute la phrase.
L'analyse qualitative le confirme : ses victoires portent précisément sur les contrastes et concessions.

Les deux modèles butent cependant sur un **plafond commun**. 60 % des erreurs de CamemBERT sont aussi des
erreurs de la baseline, et les erreurs les plus confiantes communes aux deux sont des étiquettes contestables.
Les étiquettes d'Allociné viennent de la **note** de l'auteur et non du texte. Elles intègrent donc l'ironie,
les « faux » avis et les avis mitigés tranchés par la note. Au-delà d'environ 97–98 %, une partie des « erreurs »
restantes relève du bruit d'annotation plutôt que d'une limite du modèle. L'asymétrie observée (plus d'avis
négatifs classés positifs, pour les deux modèles) s'explique de la même façon : des avis rédigés sur un ton
positif accompagnés d'une note basse.

### 5.3 Recommandation pratique

| Situation | Choix recommandé |
|---|---|
| Moins d'environ 500 exemples annotés | **TF-IDF + LogReg** : meilleur, stable, entraîné en moins d'une seconde |
| Plus d'environ 1 000 exemples, GPU disponible | **CamemBERT** : −50 % d'erreurs |
| Forte contrainte de latence ou de coût, sans GPU | TF-IDF, ou un transformer distillé (à évaluer) |
| Probabilités utilisées pour décider | CamemBERT **recalibré** (temperature scaling) |

---

## 6. Validité et limites

### 6.1 Incident détecté et corrigé

Le premier rapport généré montrait CamemBERT à **97,4 % en validation mais 50 % sur le test**, avec une ECE de
49 points, c'est-à-dire des prédictions très confiantes mais au niveau du hasard. Un tel écart entre validation
et test est incompatible avec un vrai modèle : il signalait un défaut du pipeline.

- **Cause.** Avec la version 5 de la bibliothèque `transformers`, l'option d'échantillonnage « groupé par
  longueur », activée pour accélérer l'entraînement, s'applique aussi à la fonction de prédiction. Les
  prédictions du test étaient renvoyées dans un ordre mélangé, puis associées aux étiquettes dans l'ordre
  d'origine. La validation n'était pas touchée, car les étiquettes y voyagent avec chaque batch.
- **Récupération sans réentraînement.** Le mélange est déterministe : il dépend de la graine et de la longueur
  des textes. Le script `scripts/recover_predictions.py` reconstruit exactement la permutation et remet les
  probabilités sauvegardées dans l'ordre.
- **Validation de la récupération.** Une permutation fausse donne environ 50 %. La permutation reconstruite donne
  97,41–97,45 % sur le test pour chaque seed, cohérent avec 97,33–97,46 % en validation. La même vérification
  vaut pour les 21 runs de la courbe d'apprentissage. Le script est idempotent et refuse d'écrire un résultat
  incohérent.
- **Correction définitive.** L'inférence passe maintenant par une boucle explicite qui garantit l'ordre des
  prédictions, couverte par un test de non-régression.

Leçon méthodologique : **une incohérence flagrante entre validation et test doit être traitée comme un bug du
pipeline avant toute interprétation.**

### 6.2 Menaces à la validité

- **Un seul jeu de données, une seule langue.** Les conclusions valent pour le sentiment binaire sur des avis de
  films en français. Elles ne se généralisent pas sans vérification. H4 (AG News) devait tester précisément
  cela.
- **Hyperparamètres du transformer non optimisés.** Ce choix désavantage le transformer, ce qui rend sa victoire
  plus robuste. En revanche, sur les petits échantillons de la courbe d'apprentissage, la règle fixe de 2 époques
  le pénalise fortement : 8 pas de gradient seulement pour 100 exemples. Un nombre de pas minimal, ou plus
  d'époques à faible n, réduirait probablement son retard. **Le point de bascule de 500 à 1 000 exemples est
  donc une estimation haute.**
- **Matériel différent pour les mesures de coût.** La baseline a été mesurée sur CPU, le transformer sur GPU. Les
  débits ne sont pas directement comparables : le tableau 4.7 indique toujours le matériel.
- **Débit de CamemBERT mesuré avant correction.** Il l'a été avec l'ancienne méthode de prédiction. La nouvelle
  traite les batchs de la même manière, mais le chiffre exact n'a pas été remesuré.
- **Heuristique de négation.** Un lexique de surface ne mesure pas la portée de la négation.
- **Bruit d'étiquette non quantifié.** Son importance est suggérée par l'analyse qualitative, mais n'a pas été
  mesurée, par exemple par une ré-annotation d'un échantillon.

### 6.3 Pistes

1. Exécuter **AG News** pour tester H4 (configuration prête).
2. Remplacer la règle « 2 époques » par un **nombre minimal de pas** dans la courbe d'apprentissage.
3. Ajouter une **baseline intermédiaire** : embeddings de phrases figés (sentence-transformers) + régression
   logistique, pour séparer l'apport du pré-entraînement de celui du fine-tuning.
4. Évaluer un **modèle distillé** (par ex. DistilCamemBERT), pour le compromis coût / performance.
5. **Recalibrer** CamemBERT par temperature scaling et mesurer le gain d'ECE.
6. **Ré-annoter** un échantillon des désaccords pour estimer le taux de bruit d'étiquette.

---

## 7. Reproductibilité

L'ensemble des résultats se reproduit à partir du dépôt :

```bash
pip install -e ".[dev]"
pytest                                          # 18 tests unitaires
tvt all --config configs/allocine.yaml          # baseline, transformer, courbe d'apprentissage, rapport
```

- **Environnement d'exécution :** Google Colab, GPU Tesla T4, Python 3.13, PyTorch 2.11, transformers 5.20,
  scikit-learn 1.6. Ces informations sont enregistrées automatiquement dans `results/allocine/run_*.json`.
- **Durée d'entraînement mesurée :** 3 × 37 min pour les 3 seeds sur le train complet ; environ 50 min de
  fine-tuning et 2 min de régression logistique pour les 42 runs de la courbe d'apprentissage (hors chargement,
  évaluation et recherche de grille).
- **Artefacts versionnés :** prédictions sur le test de chaque modèle, métadonnées, grille de validation,
  historiques d'entraînement, rapport et figures. La commande `tvt report` régénère toutes les tables et figures
  sans réentraîner.

---

## 8. Conclusion

Sur la classification de sentiment en français, **le fine-tuning de CamemBERT réduit les erreurs de moitié**
par rapport à une baseline TF-IDF bien réglée (97,4 % contre 94,4 % de macro-F1). Le gain est statistiquement
indiscutable et stable d'un seed à l'autre. Il vient de la capacité du modèle à composer le sens au-delà des
mots isolés : contrastes, concessions, doubles négations. Cet avantage n'apparaît qu'à partir d'environ 1 000
exemples annotés. En dessous, la baseline est à la fois meilleure et plus fiable, et elle reste préférable
lorsque les contraintes de coût et de latence dominent. Enfin, le plafond commun aux deux modèles rappelle qu'au
delà d'un certain niveau, **la qualité des étiquettes compte plus que le choix du modèle**.

---

## Références

- Blard, T. (2019). *French sentiment analysis with BERT* (jeu de données Allociné). github.com/TheophileBlard/french-sentiment-analysis-with-bert
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL-HLT*.
- Dietterich, T. G. (1998). Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms. *Neural Computation*, 10(7).
- Dodge, J., Ilharco, G., Schwartz, R., Farhadi, A., Hajishirzi, H., & Smith, N. (2020). Fine-Tuning Pretrained Language Models: Weight Initializations, Data Orders, and Early Stopping. *arXiv:2002.06305*.
- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. *ICML*.
- Martin, L., Muller, B., Ortiz Suárez, P. J., et al. (2020). CamemBERT: a Tasty French Language Model. *ACL*.
- McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika*, 12(2).
- Wang, S., & Manning, C. D. (2012). Baselines and Bigrams: Simple, Good Sentiment and Topic Classification. *ACL*.

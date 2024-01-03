# MachineLearningEmbarque

Membres du binôme :
- Romain TROVALLET
- Théo FAUCHER

Pour faciliter le travail, nous utilisons un dépôt GitHub : https://github.com/Roilann/MachineLearningEmbarque

## Classement des modèles

Vous pouvez retrouver les modèles dans le dossier `models`, avec les courbes d'apprentissage en image.

1. dataset_v1_rms_01_binary_crossentropy_e200_b15_vp02
2. dataset_v1_rms_01_binary_crossentropy_e200_b20_vp02
3. dataset_v1_adam_05_binary_crossentropy_e400_b30_vp02

Hors catégorie : 
- dataset_v1_rms_001_m01_binary_crossentropy_e200_b20_vp02_EXPERIMENTAL_metrics_auc_l2_early_stopping_redure_lr_on_plateau
- dataset_v3_rms_001_bc_e200_b20_vp02_EXPERIMENTAL_metrics_precision_recall_early_stopping_reduce_lr_on_plateau

## Etat actuel du projet
Nous avons pu enregistrer des premiers modèles contenant des paramètres déterminés à l'aide de multiples itérations 
et lecture de documentations. Nous avons fait ce travail avec un premier dataset.

Actuellement nous avons pu créer deux nouveaux datasets, nous n'avons pas encore eu le temps de les exploiter 
suffisamment. En parallèle nous avons essayé d'identifier les problèmes des précédents modèles et nous avons ajouté dans le
fichier `model.py` plusieurs algorithmes pour les metrics, les callbacks ou encore les regularizers.
Cependant nous n'avons pas encore eut le temps de réellement de les intégrer dans le modèle, de les essayer et de les configurer. 
Nous avons tout de même exporté le modèle dans cet état dans le cas où cela pourrait vous intéressez, le modèle 
s'appelle dataset_v3_rms_001_bc_e200_b20_vp02_EXPERIMENTAL_metrics_precision_recall_early_stopping_reduce_lr_on_plateau.

## Comment utiliser les programmmes ?

 - Récupérer les informations de l'accéléromètre sous format csv, avec les mêmes headers que les exemples dans le dossier `input`
 - Placer le fichier csv dans le dossier `input`
 - Lancer le programme `visualize_and_export.py`
    - Écrire le nom du fichier csv voulu
    - Une fois les courbes fermées, entrer la valeur minimale à prendre en compte puis la valeur maximale à prendre en compte 
        - /!\ Le nombre de ligne indiqué doit être un multiple du nombre de point minimal qui sera analysé /!\ => voir `utils.py`
        - Ne pas hésiter à modifier les valeurs jusqu'à avoir la section voulue
    - Quand la section est sélectionnée, écrire `exit`
    - Préciser au programme si l'analyse est un balancier `1` ou non `0`
    - Un exemple de nom de fichier de sortie d'affiche, on peut accepter ou entrer un nouveau nom
- Le nouveau fichier est enregistré dans le dossier `output`
- Lancer le programme `dataset_constructor.py`
    - Une liste des csv dans le dossier `output` s'affiche
    - Prendre le numéro du csv que vous voulez ajouter au dataset
    - Après avoir fermé le visuel du csv, valider l'ajout au dataset
    - Répéter jusqu'à la complétion du dataset
    - Une fois compléter, écrire `exit`
    - Entrer le nom de votre dataset
- Le dataset est enregistré dans le dossier `datasets`
- Lancer le programmme `model.py`
    - Entrer le nom du dataset généré
    - Le programme génère le modèle
    - Une fois les courbes visualisée, on peut enregistrer le modèle en lui donnant un nom ou ne pas l'enregistrer
- Le modèle est enregistré dans le dossier `models`
- Lancer le programme `visualize_model_performance.py`
    - Entrer le ou les id des modèles pour lesquel on veut visualiser les performances
    - Le programme affiche la prédiction
    - Entrer `exit` pour quitter le programme
    - Entrer l'id du fichier csv à utiliser pour la prédiction

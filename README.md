# MachineLearningEmbarque

Membres du binôme :
- Romain TROVALLET
- Théo FAUCHER

## Classement des modèles

1. dataset_v1_rms_01_binary_crossentropy_e200_b20_vp02
2. dataset_v1_adam_05_binary_crossentropy_e400_b30_vp02


## Comment utiliser les programmmes ?

 - Récupérer les informations de l'accéléromètre sous format csv, avec les mêmes headers que les exemples dans le dossier `input`
 - Placer le fichier csv dans le dossier `input`
 - Lancer le programme `visualize_and_export.py`
    - Écrire le nom du fichier csv voulu
    - Une fois les courbes fermées, entrer la valeur minimale à prendre en compte puis la valeur maximale à prendre en compte 
        - /!\ Le nombre de ligne indiqué doit être un multiple du nombre de point minimalqui sera analysé /!\
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
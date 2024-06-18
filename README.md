# internship-Paris-Brain-Institute-machine-learning
several tries of code that aims to analyze the originality of an answer to a creativity task; using the frequencie of the answer among all the particpant's answers, and the intrinsec originality of the answer. It uses machine learning and language model based on the transformer architecture

Ce projet repose sur du machine learning et fait appel à des réseaux de neurones et à des techniques de Natural Language Processing.
Des projets d’autres équipes ont déjà abouti ou sont en cours actuellement pour évaluer l’originalité des réponses à des tâches d’AUT en utilisant un calcul automatisé de distance sémantique comme SemDis. SemDis est un outil qui commence à être utilisé pour évaluer la créativité des idées à l’AUT. Cependant, la plupart de ces techniques sont développées en anglais, et des outils sont en cours de développement dans d’autres langues.
Dans notre cas, l’objectif n’est pas de calculer les distances sémantiques en soit pour en tirer un score de créativité, mais de relier les idées entre elles, en se basant sur leurs proximités sémantiques deux à deux. Il faut donc utiliser un outil qui permette d’évaluer les distances sémantiques entre des phrases et en français. On peut utiliser BERT, (Bidirectional Encoder Représentations from Transformers), un modèle de word-embedding, qui permet de représenter des mots, mais surtout pour l’AUT, des phrases sous forme de vecteur. CamemBERT, la version française de BERT est entraînée sur un corpus de textes en français. A partir des vecteurs représentant chaque phrase, il est possible de calculer les similarité cosinus entre les vecteurs, représentant leurs proximités sémantiques. On extrait alors une matrice de similarité, à partir de laquelle on prévoit d’utiliser des algorithmes de community detection pour regrouper des idées identiques. Ce regroupement en communauté devrait permettre de calculer la fréquence d’occurrence d’une idée grâce au nombre d’idées par communauté. 
Pour le moment, seule la partie pour obtenir la matrice de similarité cosinus a été réalisée. Il subsiste des interrogations sur la façon de mettre en forme les phrases, rajouter ou non le nom de l’objet en début de phrase notamment, et savoir s’il est possible de modifier la façon dont on construit le vecteur représentatif de la phrase, et quel algorithme de Community detection est le plus performant dans ce contexte. Un objectif de l’équipe est donc désormais de développer un tel outil, qui permette de regrouper les idées identiques, de façon automatique donc beaucoup plus rapide et objective. Cela permettra d’analyser l’ensemble des données, et de voir des corrélations éventuelles avec les données de clustering et switching étudiés précédemment.

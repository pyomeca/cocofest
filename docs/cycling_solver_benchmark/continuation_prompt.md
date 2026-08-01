# Prompt de continuation

Copier le bloc suivant dans une nouvelle tâche Codex après avoir cloné la
branche sur la machine Linux. Joindre les premiers logs ou le lien du run CI
si la campagne a déjà commencé.

```text
Reprends le benchmark des solveurs du RHO de pédalage FES sur la branche
codex/acados-pr-refresh de mickaelbegon/cocofest.

Commence impérativement par lire, dans cet ordre :
1. docs/cycling_solver_benchmark/README.md
2. docs/cycling_solver_benchmark/resume_and_todo.md
3. docs/cycling_solver_benchmark/linux_32core_setup.md
4. uniquement les sections pertinentes de
   docs/cycling_solver_benchmark/development_history.md

Inspecte ensuite git status, le HEAD Cocofest et les SHA effectifs de Bioptim,
ACADOS, libMad et CasADi. Préserve tous les changements utilisateur et ne
réutilise aucun artefact dont la provenance ou la transcription diffère.

Principe scientifique non négociable : l'ancienne référence Radau degré 3
n'est plus l'oracle. La force passive doit rester active. La dynamique du
calcium doit être raffinée et validée contre le régime périodique analytique.
Il faut accélérer le problème corrigé, pas reproduire un ancien coût obtenu
avec une erreur de modèle ou une discrétisation insuffisante.

Objectif de cette reprise :
- créer un profil explicite scientific-radau5 ou irk-refined;
- conserver 30 décisions de PW par cycle, mais raffiner l'intégration interne
  des états musculaires;
- enregistrer dans chaque JSON la force passive, le degré de collocation, les
  étages/sous-pas, les SHA et le hash de la bibliothèque compilée;
- ajouter un test analytique du calcium périodique;
- comparer IPOPT/MUMPS et MadNLP/MUMPS reduced sur le même NLP raffiné;
- conserver IPOPT full comme contrôle scientifique apparié;
- ne jamais propager le terminal d'un RHO non certifié;
- exécuter les gates 1, 5, 30 puis 100 RHO, sans lancer le suivant avant
  validation du précédent;
- mesurer construction, compilation, préparation, temps solveur et temps mural
  chaud séparément;
- reporter coût, fatigue exécutée, AUC et capacité finale pour les quatre
  muscles, ainsi que les PW aux cycles 10, 30 et 100;
- documenter toute divergence full/reduced supérieure à 0.1 % par muscle.

Sur la machine 32 cœurs :
- utiliser CMAKE_BUILD_PARALLEL_LEVEL=32 et BENCHMARK_THREADS=32;
- garder d'abord OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1,
  MKL_NUM_THREADS=1, NUMEXPR_NUM_THREADS=1 et JULIA_NUM_THREADS=1 pour la
  baseline reproductible;
- ne faire le sweep MUMPS 1/2/4/8/16/30/32 threads qu'après certification de
  la baseline;
- exécuter les solveurs successivement sur une machine unique pour éviter la
  contention, avec artefacts séparés.

Pour ACADOS, ne relance pas les options déjà réfutées sans nouvelle hypothèse :
homotopie terminale actuelle, retry primal, retry primal-dual, conservation
globale des duals et rollout IRK concurrent n'ont pas franchi le RHO 14. La
prochaine piste utile est une projection du primal sur la dynamique discrète
de la nouvelle fenêtre; si nécessaire, utiliser deux capsules précompilées,
faisabilité puis fatigue. RTI vient seulement après une chaîne SQP robuste.

Ne rouvre pas PARDISO/MadNLP, Alpaqa, FATROP/RK4 ou un surrogate neuronal sans
un élément nouveau. MUMPS reste le backend MadNLP. FATROP reduced reste un
contrôle indépendant; FATROP full est un échec d'interface, pas de physique.

Utilise un second agent pour auditer les choix numériques, les critères de
convergence du calcium et l'interprétation des résultats full/reduced. Fais les
modifications par petites étapes testées. Après chaque gate Linux, lis les
logs et les JSON complets avant de poursuivre. Mets à jour README.md,
resume_and_todo.md, development_history.md et le message pour Kevin avec les
résultats certifiés seulement.

À la fin : donne le tableau comparatif, les échecs et leur premier RHO, les
limites restantes et le prochain TODO précis. Commit et pousse uniquement les
changements testés sur codex/acados-pr-refresh.
```

#!/bin/bash
rm -rf ARQMath_2022_Submission/.ipynb_checkpoints
python de_duplicate_2022.py -qre "qrel_task2_2022_official.tsv" -tsv "latex_representation_v3/"  -sub "ARQMath_2022_Submission/" -pri "ARQMath_2022_Submission_prime/"
rm -rf ARQMath_2022_Submission_prime/.ipynb_checkpoints
python task2_get_results.py -eva trec_eval/trec_eval -qre qrel_task2_2022_official.tsv -pri ARQMath_2022_Submission_prime/ -res task2.tsv

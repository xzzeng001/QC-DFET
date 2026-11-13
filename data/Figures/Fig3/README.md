# Figure 3

H2/Cu(111): 24-qubit quantum and perturbative-correction errors. `data/QC_results_fr_adapt_vqe.xlsx` is the frozen simulation table; `data/nevpt2_scan.csv` contains 97 source-derived scan records, and `data/nevpt2_selected.csv` the three selected points. The optional historical selection-list fields are NaN where that list has no row. All 97 scan records include the QC error, NEVPT2 error, and FCI overlap. No original run directories or incomplete `selected_steps.txt` are required.

From this directory run `python src/plot.py`; the main figure is `output/Fig3_qc_errors_with_nevpt2_reduction.png`.

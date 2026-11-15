# QC-DFET research data

Data and plotting resources associated with **Embedded quantum computing for many-body surface reaction**, by Dedong Wan, Xiaopeng Li, Yi Fan, Jie Liu, Xiongzhi Zeng, and Zhenyu Li (2026): [arXiv:2607.27009](https://arxiv.org/abs/2607.27009), [DOI](https://doi.org/10.48550/arXiv.2607.27009).

QC-DFET denotes quantum-computing density-functional embedding theory. This repository makes the supplied datasets available for inspection and further analysis. Read the [data-quality notes](data/DATA_QUALITY.md) before interpreting or reusing results.

## Data organization

| Location | Contents |
| --- | --- |
| [`data/Energies/`](data/Energies/) | Three original energy workbooks for H2/CO calculations, with formulas and cached results preserved. |
| [`data/Figures/`](data/Figures/) | Supplied figure images, numerical inputs and Python plotting scripts, organized by the archive's Fig1–Fig6 labels. |
| [`data/Structures/`](data/Structures/) | Periodic VASP `CONTCAR` structures, cluster XYZ coordinates and plain-text embedding matrices. |
| [`data/legacy/`](data/legacy/) | Two distinct workbooks from the earlier data-availability archive; kept separate from the main package. |
| [`data/README.md`](data/README.md) | Detailed inventory, formats, units and usage guidance. |
| [`data/DATA_QUALITY.md`](data/DATA_QUALITY.md) | Missing inputs, inconsistent formula constants, state-label conventions and other limitations identified during inspection. |
| [`data/provenance/archive_manifest.json`](data/provenance/archive_manifest.json) | Original ZIP hashes and the mapping of every archived file to its published location. |


## Verify the downloaded files

From the repository root, run:

```bash
python scripts/verify_data.py
```

This uses only the Python standard library and checks the delivered files against [`data/checksums.sha256`](data/checksums.sha256). It checks integrity, not the scientific validity of formulas or results.

For plotting instructions and validation scope, see [the data guide](data/README.md). For interpretation, use the explicit labels and units of each source file together with the [known limitations](data/DATA_QUALITY.md); there is no universal energy unit across all workbooks.

## Citation and reuse

Please cite the paper above when using these data and record the repository commit used. No new license or third-party redistribution terms have been inferred or added by this organization step.

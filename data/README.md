# QC-DFET data

This directory organizes the supplied `QC-DFET-Data.zip` and `datas availiablity.zip` contents. Numerical source files and workbook formulas are retained; distinct workbook variants are separated rather than merged. Read [DATA_QUALITY.md](DATA_QUALITY.md) before interpreting state names, units, or historical results.

## Directory guide

| Directory | Contents |
| --- | --- |
| [Energies/](Energies/) | Three supplied energy workbooks: H2/Cu(111) 24-qubit results, CO/Cu(111) 24-qubit results, and the historically named `CO_Cu111_40Q.xlsx`. The last file contains four-site `top/abv/bridge/hollow` data and is not the final Fig6 workbook. |
| [Figures/](Figures/) | Archive figures Fig1–Fig6, input tables and analysis results, and five plotting scripts for Fig2–Fig6. |
| [Structures/](Structures/) | Periodic VASP structures, finite-cluster XYZ geometries, and numerical embedding matrices for H2/Cu(111), CO/Cu(111), and CO/Cu(100). |
| [legacy/](legacy/) | Distinct CO 40-qubit and extended H2 workbooks from the smaller archive. These preserve source history and are not silently substituted for the main workbooks. |

The smaller archive's `CO_Cu111_24Q.xlsx` duplicates the main `Energies/CO_Cu111_24q.xlsx` workbook byte for byte, so only the main copy is retained. Both source paths and their identical SHA-256 are recorded in [provenance/archive_manifest.json](provenance/archive_manifest.json).

## Systems and structures

| System | Configurations | Cluster composition | Periodic composition | `emb_ints.pyscf` shape |
| --- | --- | --- | --- | --- |
| H2/Cu(111) | `IS`, `TS`, `FS` | Cu12H2, 14 atoms | Cu84H2, 86 atoms | 534 × 534 |
| CO/Cu(111) | `abv`, `top`, `bridge`, `fcc`, `hcp` | Cu14CO, 16 atoms | Cu84CO, 86 atoms | 648 × 648 |
| CO/Cu(100) | `abv`, `top`, `bridge`, `hollow` | Cu14CO, 16 atoms | Cu100CO, 102 atoms | 648 × 648 |

Each configuration includes `cluster/<configuration>/` and `periodic/<configuration>/`. `abv` is the separated-adsorbate reference used by the supplied energy analysis. The H2 structure folders use **IS = dissociated H atoms, FS = molecular H2**; some figure-source files use the opposite historical names. See the mapping in [DATA_QUALITY.md](DATA_QUALITY.md).

## Figure-source contents

These numbers refer to the archive's directory names, not a guaranteed mapping to every manuscript version.

| Directory | Available data |
| --- | --- |
| [Fig1](Figures/Fig1/) | Static JPEG schematic; no editable PowerPoint source is supplied. |
| [Fig2](Figures/Fig2/) | H2/Cu(111) energy and charge analysis for three states; transition-state density/deformation grids; plotting script and PNG. |
| [Fig3](Figures/Fig3/) | H2/Cu(111) 24-qubit quantum errors; 97 NEVPT2 scan records; three selected records; frozen simulation workbook; script and PNG. |
| [Fig4](Figures/Fig4/) | Five CO/Cu(111) site-analysis JSON files, energy/charge plotting script, and PNG. |
| [Fig5](Figures/Fig5/) | Five-row CO/Cu(111) orbital-projected Mayer bond-order table, script, and PNG. |
| [Fig6](Figures/Fig6/) | Four-row CO/Cu(100) Mayer table; final CAS(22e,20o), 40-qubit, approximately one-million-CSF workbook with the `554` slab reference; script and PNG. |

For Fig6 energy results use `Figures/Fig6/data/QSCI40_CO100_FOUR_SITES_1M_ENSC_QCDFET_554_FINAL_20261102.xlsx`. Its sheets are `QCDFET_554`, `Reference_EN`, `FullCFG_comparison`, and `Settings_Provenance`.

## Formats and units

- **`.xlsx`**: Excel source workbooks with formulas and cached results. Energies mix Hartree (`Ha`), milli-Hartree (`mHa`), and electronvolt (`eV`) quantities; use individual field labels and formulas, not the filename or a global title. The main conversion factor is 27.2114 eV/Ha, with exceptions recorded in DATA_QUALITY.md.
- **`.csv`**: UTF-8 tables for scan results and Mayer bonding channels. Energy-column suffixes identify Ha/eV. Mayer indices, overlaps, and occupations are dimensionless; geometric fields ending `_A` are in angstroms. `delta_*_CASSCF_minus_RKS` means CASSCF minus embedded RKS.
- **`.json`**: State analyses containing electronic energies, active-space information, natural occupations, atomic/fragment charges, CI summaries, and orbital compositions. Electronic energies are in Ha, charges in elementary-charge units, and fields ending `_angstrom` in angstroms. Referenced historical cube/Molden files are not included.
- **`.npz`**: NumPy archive loaded with `numpy.load(path, allow_pickle=False)`. Fig2 contains `x_angstrom` (620 points), `z_angstrom` (500 points), and five 500 × 620 density/deformation arrays. Density is in electrons/bohr³; deformation is the embedded density minus the neutral-atom promolecule density.
- **`.xyz`**: Atom count, comment line, then element and Cartesian coordinates in angstroms.
- **`CONTCAR`**: VASP lattice/species/counts and fractional (`Direct`) coordinates; interpret these together with the scale factor and lattice.
- **`emb_ints.pyscf`**: Plain, whitespace-delimited square numerical matrices, readable with `numpy.loadtxt`; these are not Python pickles. Matrix files do not encode basis ordering or units internally. They require the matching geometry and calculation setup for scientific reuse.
- **`.py`, `.png`, `.jpeg`**: Plotting source and supplied rendered figures. Executing a plotting script can create additional derived tables and images.

## Reusing the figure inputs

The upstream dependency versions are recorded in [Figures/requirements.txt](Figures/requirements.txt). They target an older NumPy stack; use a separate Python environment rather than replacing unrelated project dependencies.

```bash
python -m pip install -r data/Figures/requirements.txt
cd data/Figures/Fig2
python src/plot.py
```

Run the equivalent command from each Fig3–Fig6 directory so relative `data/` paths resolve correctly. The supplied `output/` images are preserved reference artifacts. Plotting reuses already computed data; this package does not provide all electronic-structure input decks, basis settings, executable solvers, or original quantum-computing runs needed to rerun the underlying calculations.

## Validation performed

On 2026-11-13, all five unchanged plotting scripts completed successfully in disposable copies using Python 3.12.14, NumPy 2.3.5, pandas 3.0.1, openpyxl 3.1.5, Pillow 13.0.0 and Matplotlib 3.11.2 (`MPLBACKEND=Agg`). This tested environment differs from the upstream pinned requirements. The run produced 12 valid PNG files and one derived CSV, without changing the input data. The regenerated files were not substituted for the supplied figures, and pixel-identical output was not required.

Default Fig2, Fig4 and Fig5 execution returns before some optional auxiliary exports; do not expect every output mentioned in comments to be generated. See [provenance/validation_summary.json](provenance/validation_summary.json) for the check results. From the repository root, `python scripts/verify_data.py` verifies the delivered file hashes.

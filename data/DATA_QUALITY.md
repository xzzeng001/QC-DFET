# Data inspection and interpretation notes

Inspection date: 2026-11-13. Raw numerical values and workbook formulas were not corrected during organization. A structurally valid file is not by itself evidence of scientific convergence or correctness.

## Verified contents of the main archive

- `QC-DFET-Data.zip` contains 74 files and 97,721,654 uncompressed bytes. ZIP CRC checks pass; no duplicate payloads or unsafe extraction paths were found within it.
- All 12 XYZ atom counts match their coordinates; all 12 CONTCAR species counts match the coordinate blocks. Coordinates are finite.
- All 12 embedding matrices parse as finite numeric text. The three H2 matrices are 534 × 534; the nine CO matrices are 648 × 648. Maximum absolute matrix asymmetry is approximately 1.0 × 10⁻¹⁴.
- All nine JSON files parse without nonfinite numeric values. All seven arrays in the Fig2 NPZ load without pickle; numerical values are finite, and both deformation arrays exactly equal the corresponding embedded density minus the promolecule density.
- All five main-archive Excel workbooks open, with no stored Excel error cells or missing formula caches. Their 160 formulas were independently evaluated with a restricted arithmetic/reference evaluator and agree with cached results to absolute tolerance 10⁻⁸ or relative tolerance 10⁻¹⁰. This verifies stored arithmetic, not the suitability of the source inputs or formula definitions.
- The four CSVs contain 97, 3, 5, and 4 records respectively, without duplicate rows. The three optional `runlog_err_*` columns in `nevpt2_scan.csv` each have 91 missing values; the core QC/NEVPT2 error and FCI-overlap columns are populated for all 97 records.
- All six supplied images decode successfully. The five plotting scripts pass Python syntax parsing. These checks alone do not establish an identical rendering environment or reproduce the underlying electronic-structure calculations.

## H2 state-name mapping

The source package intentionally contains two naming conventions. Keep the stored filenames and use this mapping when combining tables:

| Physical state | `Structures/H2_Cu111/` and `Energies/H2_Cu111_24q.xlsx` | Fig2 raw JSON / Fig3 source labels | H–H distance |
| --- | --- | --- | --- |
| Dissociated H atoms | `IS` | `FS` | 1.983254 Å |
| Transition state | `TS` | `TS` | 0.906327 Å |
| Molecular H2 | `FS` | `IS` | 0.751572 Å |

Fig2 and Fig3 plotting code explicitly swaps the historical IS/FS names into the manuscript display convention. `nevpt2_scan.csv` and `nevpt2_selected.csv` contain separate `source_state` and `display_state` columns. Do not join these sources on the state string alone.

## Workbook caveats retained from the sources

1. **Historical CO 40-qubit filename.** `Energies/CO_Cu111_40Q.xlsx` contains `top/abv/bridge/hollow` data and a `VASP no D3 551 (eV)` field. Its content should not be identified as final CO/Cu(111) data solely from its filename. The explicitly identified final CO/Cu(100), `554`-slab workbook is under `Figures/Fig6/data/`; the historical workbook has different reference energies/results.
2. **Two CO 40-qubit versions.** The distinct copy in `legacy/CO_Cu111_40Q.xlsx` is retained. Its J12 description is `ABV reference`, whereas the main copy uses a dash. This is a textual difference, not a reason to merge or overwrite the sources.
3. **Mixed units and conversion constants.** `Energies/H2_Cu111_24q.xlsx` has an `Energy/eV` heading, but the `emb_RKS` row and quantum energies are in Ha. Cells C11, C12, G12, and C13 use factors 27.2115, 27.2116, 27.2214, and 27.2117 respectively, while surrounding formulas predominantly use 27.2114. These exceptions are preserved for author review.
4. **Extended legacy H2 workbook.** `legacy/H2_Cu111_24Q.xlsx` extends the common first 13 rows with additional rows 14–17. F15 is missing even though G15/J15/K15 depend on it, producing invalid physical results when Excel treats the empty input as zero (approximately +63681.608, -63681.002 and -63681.611 eV). C14:C17 continue the inconsistent conversion factors from 27.2118 through 27.2121. A17 repeats the `adapt-uccsd_NEVPT2` label despite the `adapt-guccsd` label in A16. Treat these additional rows as unresolved historical data, not validated results.
5. **Final Fig6 limitations.** Its `Settings_Provenance` sheet records that actual CSF counts are near, not exactly, one million, and that the HOLLOW100 EN correlation changes strongly relative to full CFG. Agreement with experiment is not a convergence test. Referenced upstream evidence tarballs are not part of these two supplied ZIP files.

## Scope and reproducibility

The Fig1–Fig6 names describe the supplied archive. The manuscript version must be checked separately: [arXiv v2](https://arxiv.org/abs/2607.27009v2) Fig6 concerns HCOO hydrogenation on Cu(111), while this package's `Figures/Fig6` concerns CO adsorption on Cu(100). No missing HCOO source data have been inferred or fabricated.

The old top-level archive README referred to `Figure_Source_Data/`; the actual directory is `Figures/`. Fig4 supplies analysis JSON files, not the pre-existing summary CSV mentioned by its original README. Its default execution returns before the optional summary-export code. Fig1 includes only a rendered JPEG, despite its description as a PowerPoint schematic. Historical file paths embedded in JSON/workbook provenance are descriptive references, not bundled dependencies.

All five unchanged scripts completed in a disposable-copy test on 2026-11-13, producing 12 decodable PNGs and one derived CSV without changing source/input bytes. This is a plotting smoke test, not a rerun of the quantum calculations, a guarantee of pixel-identical images, or resolution of the scientific caveats above. The tested versions and outputs are recorded in [provenance/validation_summary.json](provenance/validation_summary.json).

Plot regeneration depends on the supplied scripts, libraries, fonts, and relative working directories. Many comparison energies and plotting choices are fixed in script constants; in particular the Fig6 script cites its final workbook but does not load that workbook at runtime. Changing a workbook therefore does not automatically update every plotted result. Recomputing electronic-structure results needs additional calculation setup beyond this publication-data package.

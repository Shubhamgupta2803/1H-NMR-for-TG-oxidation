# 1H-NMR-for-TG-oxidation
1H NMR OXIDATION ANALYSIS PIPELINE
==================================

A single-script Python pipeline that analyses Bruker 1H NMR spectra (D2O) from
a periodate-oxidised tamarind gum series (native, 15%, 30%, 45%, 60%).

It standardises the spectra, picks and assigns peaks, integrates diagnostic
regions, checks whether the oxidation markers (alcohol, aldehyde, carboxyl,
by-product) follow the expected trend, and writes a PDF report with figures
and tables.

The sample series, ppm windows and assignment library are set for oxidised
tamarind gum, but all of them can be edited at the top of the script (see
CONFIGURATION below).


WHAT IT DOES
------------

1. Loads processed Bruker spectra (1r) for every sample listed in SAMPLES.
   Uses nmrglue if installed, otherwise a built-in binary reader.

2. Checks acquisition consistency against the native spectrum. Spectral width,
   SFO1, D1, pulse program and temperature are flagged as critical. NS, TD and
   RG are informational only.

3. Standardises every spectrum in the same way:
   - chemical-shift referencing to residual HDO (temperature-corrected) or
     DSS/TSP
   - baseline offset removal using a signal-free window
   - noise estimate (sigma) used for a detection limit

4. Picks peaks (HDO region excluded) and matches them to a chemical-shift
   assignment library.

5. Integrates the diagnostic regions. Residue ranges (ethanol/ethylene glycol,
   methanol, acetate) are bridged by a straight line so they do not inflate
   the integrals.

6. Normalises the integrals (envelope, anomeric H-1, or internal standard) and
   flags regions below the detection limit as ND.

7. Summarises integrals by functional group and tests each marker against its
   expected trend across the series.

8. Makes the plots and builds the PDF report.


REQUIREMENTS
------------

- Python 3.8 or later
- numpy
- scipy (version 1.6 or later, for trapezoid)
- matplotlib
- reportlab
- pandas (imported by the script)
- nmrglue (optional; a built-in reader is used if it is missing)

Install with:

    pip install numpy scipy matplotlib reportlab pandas
    pip install nmrglue          (optional)


INPUT DATA LAYOUT
-----------------

Each sample is a Bruker experiment folder (EXPNO) containing the acquisition
file and a processed-data folder:

    BASE_DIR/
        552/                  EXPNO of the native sample
            acqus
            pdata/
                1/            PROCNO
                    1r        processed real spectrum
                    procs
        562/
        572/
        582/
        592/

BASE_DIR is a relative path, so run the script from the directory that
contains it, or change it to an absolute path.

The spectra must already be processed in TopSpin (Fourier transform, phasing,
baseline correction).


USAGE
-----

1. Edit the settings at the top of the script. At minimum, set BASE_DIR,
   SOLVENT and the EXPNO values in SAMPLES.

2. Run:

       python <your_script_name>.py

All output files are saved next to the script.


OUTPUT FILES
------------

TG_oxidation_report.pdf        Full report (structure given below)
overlay_spectrum.png           Stacked spectra, one panel per sample
difference_spectra.png         Each sample minus the native spectrum
annotated_peak_assignments.png Peaks labelled with their assignment
peak_comparison.png            Peak positions and heights across samples
integration_analysis.png       Grouped bars of region integrals
functional_group_trend.png     Functional-group content across the series


REPORT STRUCTURE
----------------

Title page   Instrument and sample table, plus a standardisation and QC table
             (referencing, temperature, scans, D1, pulse program, noise sigma,
             consistency flags)
Section 1    Spectral overlay
Section 2    Difference spectra
Section 3    Annotated peak assignments
Section 4    Peak picking and comparison (top 12 peaks per sample, with
             assignments)
Section 5    Integration analysis
Section 6    Functional-group progression
Section 7    Oxidation-marker check (expected vs observed, with verdict)
Section 8    Conclusion


CONFIGURATION
-------------

All settings are in the block at the top of the script.

BASE_DIR, PROCNO, SOLVENT
    Data location, processing number, and the solvent shown in the report.

SAMPLES
    List of samples (key, expno, label, short, color). Entries can be added or
    removed; plots and tables scale automatically.

REFERENCE_KEY
    Baseline sample for comparisons (the first entry, native, by default).

REFERENCE_MODE
    "hdo" (default), "dss", or "none".

HDO_PPM_298K, HDO_TEMP_COEFF
    Expected HDO shift at 298 K and its temperature coefficient.

MAX_REF_SHIFT_PPM
    Largest referencing shift that will be applied (default 0.30 ppm). Larger
    shifts are not applied and are flagged instead.

HDO_EXCLUSION_PPM
    Solvent window excluded from scaling, peak picking and integration.

QUIET_WINDOW, LOD_SIGMA
    Signal-free region used for baseline and noise, and the detection limit
    (default 3 sigma).

NORMALISATION
    "envelope" (default), "h1", or "internal_standard".

ENVELOPE_WINDOW
    Polysaccharide window used for envelope normalisation (default
    1.1 to 5.8 ppm).

STANDARD_WINDOW, STANDARD_NH
    Internal-standard window and its proton count (DSS/TSP: 9).

INTEGRATION_REGIONS
    Diagnostic regions as (ppm_low, ppm_high, label, group).

INTEGRATION_EXCLUDE
    Residue-only ranges removed from every integral.

ASSIGNMENT_LIBRARY
    Chemical-shift windows used to label picked peaks.

EXPECTED_TREND
    Expected behaviour per group: "up", "down", "up_then_down" or "constant".


NORMALISATION MODES
-------------------

envelope (default)
    Each integral is expressed as a percentage of the total polysaccharide
    signal in ENVELOPE_WINDOW (HDO removed). Values are compositional shares,
    so only relative changes are meaningful.

h1
    Percentage of the anomeric H-1 signal (the Backbone regions).

internal_standard
    Integrals per standard molecule, using STANDARD_WINDOW. If the window is
    not set, or the standard is not detected in a sample, the script falls
    back to envelope normalisation and prints a note.


DEFAULT INTEGRATION REGIONS
---------------------------

ppm          Region                                          Group
9.40-9.80    Free aldehyde (-CHO)                            Aldehyde
8.40-8.50    Formate by-product                              Byproduct
5.10-5.80    Aldehyde hydrate / hemiacetal                   Aldehyde
4.40-5.10    Anomeric H-1                                    Backbone
4.00-4.40    CH next to -COO- and disrupted native rings     Carboxyl
3.20-4.00    Ring CH-O, H-2 to H-6                           Alcohol


HOW THE MARKER CHECK WORKS
--------------------------

For each group in EXPECTED_TREND, the first and last samples and the shape of
the series are compared, using a tolerance equal to a fraction of the group's
largest value. The verdict is "consistent", "NOT consistent", or "not
testable" (below the detection limit in every sample).

main() currently calls evaluate_trends(..., tol_frac=0.35). The function's own
default is 0.10.

This is a descriptive check on a handful of points. It is not a statistical
test.








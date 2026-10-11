
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks
from scipy.integrate import trapezoid
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = "yashika"     # ← relative path, since you run terminal from NMR/
PROCNO   = "1"              # pdata subfolder (almost always "1")
SOLVENT  = "D2O"            # shown in the report's instrument-info table


HDO_EXCLUSION_PPM = (4.6, 4.85)
PPM_XLIM = (10.0, -0.5)

REFERENCE_MODE    = "hdo"
HDO_PPM_298K      = 4.79      # Gottlieb, Kotlyar, Nudelman, J. Org. Chem. 1997, 62, 7512
HDO_TEMP_COEFF    = -0.012    # ppm per K (approximate)
MAX_REF_SHIFT_PPM = 0.30      # never shift further than this; flag it instead

NORMALISATION   = "envelope"
STANDARD_WINDOW = None            # e.g. (-0.10, 0.10) for DSS / TSP
STANDARD_NH     = 9               # DSS and TSP: 9 equivalent protons
ENVELOPE_WINDOW = (1.1, 5.8)

QUIET_WINDOW = (10.0, 12.0)
LOD_SIGMA    = 3.0

INTEGRATION_EXCLUDE = [
    (3.62, 3.69, "ethanol CH2 / ethylene glycol"),
    (3.32, 3.37, "methanol"),
    (1.83, 1.95, "acetate contaminant")
]

EXPECTED_TREND = {
    "Alcohol":   "down",
    "Aldehyde":  "up",
    "Carboxyl":  "down",
    "Byproduct": "up",
}

UNIT_LABEL = "Normalised integral"    # overwritten at run time from NORMALISATION

SAMPLES = [
    {"key": "native", "expno": "552", "label": "Native Tamarind Gum",
     "short": "Native", "color": "#1565C0"},
    {"key": "ox1",    "expno": "562", "label": "Oxidised 15% (TG15)",
     "short": "TG15",   "color": "#00897B"},
    {"key": "ox2",    "expno": "572", "label": "Oxidised 30% (TG30)",
     "short": "TG30",   "color": "#C0CA33"},
    {"key": "ox3",    "expno": "582", "label": "Oxidised 45% (TG45)",
     "short": "TG45",   "color": "#FB8C00"},
    {"key": "ox4",    "expno": "592", "label": "Oxidised 60% (TG60)",
     "short": "TG60",   "color": "#E53935"},
]
REFERENCE_KEY = SAMPLES[0]["key"]   # baseline every other sample is compared against

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))  # saves next to this script

INTEGRATION_REGIONS = [
    (9.40, 9.80,  "Free aldehyde -CHO\n(9.4–9.8 ppm)",                      "Aldehyde"),
    (8.40, 8.50,  "Formate by-product\n(8.4–8.5 ppm)",                      "Byproduct"),
    (5.10, 5.80,  "Aldehyde hydrate / hemiacetal\n(5.1–5.8 ppm)",           "Aldehyde"),
    (4.40, 5.10,  "Anomeric H-1 (reference)\n(4.4–5.1 ppm)",                "Backbone"),
    (4.00, 4.40,  "CH next to -COO- AND Disrupted Native Rings",            "Carboxyl"),
    (3.20, 4.00,  "Ring CH-O, H-2 to H-6\n(3.2–4.0 ppm)",                   "Alcohol"),
]

# Broad windows for the peak-comparison view only.
PEAK_REGIONS = {
    "Carbonyl / Aldehyde region (8.4\u20139.8 ppm)": (8.4, 9.8),
    "Anomeric / Hydrate region (4.4\u20135.8 ppm)":  (4.4, 5.8),
    "Ring CH-O region (3.2\u20134.4 ppm)":           (3.2, 4.4),
}


GROUP_COLORS = {
    "Backbone":    "#757575",
    "Alcohol":     "#43A047",
    "Aldehyde":    "#FB8C00",
    "Carboxyl":    "#8E24AA",
    "Byproduct":   "#D81B60",
    "Methyl":      "#00838F",
    "Acetyl":      "#7CB342",
    "Contaminant": "#8D6E63",
    "HDO":         "#BDBDBD",
    "Other":       "#9E9E9E",
}


ASSIGNMENT_LIBRARY = [
    (9.40, 9.80,  "Free aldehyde (-CHO, usually absent/hydrated)",                     "Aldehyde"),
    (8.40, 8.50,  "Formate (over-oxidation chain-cleavage marker)",                    "Byproduct"),
    (5.10, 5.80,  "Aldehyde hydrate / hemiacetal (oxidation marker)",                  "Aldehyde"),
    
    (4.40, 5.10,  "Anomeric H-1 (Glc, Xyl, Gal)",                                      "Backbone"),
    (4.00, 4.40,  "CH next to -COO- AND Disrupted Native Rings",                       "Carboxyl"),
    
    (3.62, 3.69,  "Ethanol CH2 / Ethylene glycol (residue)",                           "Contaminant"),
    (3.32, 3.37,  "Methanol (residue)",                                                "Contaminant"),
    
    # Shortened label so it doesn't clutter the graph
    (3.20, 4.00,  "Tamarind Ring CH-O",                                                "Alcohol"),
    
    # Explicitly naming the upfield peaks so they appear on the graph
    (2.15, 2.25,  "Residual acetone / Unknown residue",                                "Contaminant"),
    (1.83, 1.95,  "Acetate CH3 (cleaved/residual by-product)",                         "Contaminant"),
    (1.10, 1.30,  "Unknown upfield contaminant",                                       "Contaminant"),
    (0.80, 0.90,  "Unknown aliphatic impurity",                                        "Contaminant"),
    
    (-0.10, 0.10, "DSS / TSP internal reference",                                      "Other"),
]

def assign_peak(ppm_value, library=ASSIGNMENT_LIBRARY):
    """Look up the chemical-shift assignment library for a peak's ppm value.
    Returns (assignment_label, group); ("Unassigned", "Other") if no window matches."""
    for lo, hi, name, group in library:
        if lo <= ppm_value <= hi:
            return name, group
    return "Unassigned", "Other"


CRITICAL_PARAMS = ("SW", "SFO1", "D1", "PULPROG")   # change the integrals themselves
INFO_PARAMS     = ("NS", "TD", "RG")                # change S/N or scale only


def check_acquisition_consistency(spectra, rel_tol=1e-3):
    """
    Compare acquisition settings of every sample against the reference (native).
    Returns {sample_key: [flag strings]}. A leading "!" marks a setting that
    invalidates direct integral comparison (spectral width, frequency, relaxation
    delay, pulse program, temperature); "i" marks one that only changes S/N or
    overall scale (scan count, TD, receiver gain), which normalisation absorbs.
    """
    flags = {s["key"]: [] for s in SAMPLES}
    ref = spectra[REFERENCE_KEY]["params"]
    for s in SAMPLES:
        if s["key"] == REFERENCE_KEY:
            continue
        p = spectra[s["key"]]["params"]
        for k in CRITICAL_PARAMS + INFO_PARAMS:
            a, b = ref.get(k), p.get(k)
            if a is None or b is None:
                continue
            if isinstance(a, str) or isinstance(b, str):
                differs, txt = str(a) != str(b), f"{k} {b} vs {a}"
            else:
                a, b = float(a), float(b)
                differs = (abs(a - b) / abs(a) > rel_tol) if a else (a != b)
                txt = f"{k} {b:g} vs {a:g}"
            if differs:
                flags[s["key"]].append(("! " if k in CRITICAL_PARAMS else "i ") + txt)
        ta, tb = ref.get("TE"), p.get("TE")
        try:
            if abs(float(ta) - float(tb)) > 1.0:
                flags[s["key"]].append(f"! TE {float(tb):g} K vs {float(ta):g} K")
        except (TypeError, ValueError):
            pass
    for s in SAMPLES:
        for f in flags[s["key"]]:
            if f.startswith("!"):
                print(f"  ! WARNING {s['short']}: {f[2:]} -- differs from reference; "
                      f"integral comparison may be invalid.")
    return flags


def standardise_spectra(spectra):
    """
    Apply the same standardisation to every spectrum, in place:
      1. chemical-shift referencing (REFERENCE_MODE)
      2. baseline offset removal (mean of QUIET_WINDOW)
      3. noise estimate (std of QUIET_WINDOW) for the detection limit
    Everything that was done is recorded in spectra[key]["qc"] and shown in the report.
    """
    for s in SAMPLES:
        sp = spectra[s["key"]]
        ppm, y = sp["ppm"], sp["intensity"]
        qc = {"ref_text": "kept as processed", "shift": 0.0, "sigma": None, "offset": 0.0}

        try:
            T = float(sp["params"].get("TE"))
        except (TypeError, ValueError):
            T = 298.0
        qc["T_K"] = T
        hdo_expected = HDO_PPM_298K + HDO_TEMP_COEFF * (T - 298.0)
        qc["hdo_expected"] = hdo_expected

        # rough noise from the quiet window, used to judge whether a reference peak is real
        qlo, qhi = QUIET_WINDOW
        qmask = (ppm >= qlo) & (ppm <= qhi)
        sigma0 = float(np.std(y[qmask])) if qmask.sum() >= 50 else None

        shift, note = 0.0, None
        if REFERENCE_MODE == "dss":
            m = (ppm >= -0.3) & (ppm <= 0.3)
            if m.sum() > 5 and sigma0 and y[m].max() > 20 * sigma0:
                shift = -float(ppm[m][np.argmax(y[m])])
                note = f"DSS/TSP -> 0.00 (shift {shift:+.3f})"
            else:
                note = "DSS/TSP not found; kept as processed"
        elif REFERENCE_MODE == "hdo":
            m = (ppm >= hdo_expected - 0.35) & (ppm <= hdo_expected + 0.35)
            outside = (~m) & (ppm >= 0.0) & (ppm <= 10.0)
            if m.sum() > 5 and outside.sum() > 5:
                hdo_peak = float(y[m].max())
                rest_max = float(y[outside].max())
                if hdo_peak >= 3.0 * rest_max:
                    obs = float(ppm[m][np.argmax(y[m])])
                    shift = hdo_expected - obs
                    note = (f"HDO at {obs:.3f}, no shift needed" if abs(shift) < 0.002
                            else f"HDO {obs:.3f} -> {hdo_expected:.2f} (shift {shift:+.3f})")
                else:
                    note = "NOT verified: no dominant HDO within \u00b10.35 ppm (suppressed or off-reference), no DSS"
            else:
                note = "HDO window outside spectrum; kept as processed"
        else:
            note = "kept as processed (REFERENCE_MODE='none')"

        if abs(shift) > MAX_REF_SHIFT_PPM:
            note = f"shift {shift:+.2f} ppm exceeds limit; NOT applied (check referencing)"
            shift = 0.0
        if abs(shift) < 0.002:
            shift = 0.0
        if shift:
            ppm = ppm + shift
            sp["ppm"] = ppm
        qc["shift"], qc["ref_text"] = shift, note

        qmask = (ppm >= qlo) & (ppm <= qhi)
        if qmask.sum() >= 50:
            offset = float(np.mean(y[qmask]))
            y = y - offset
            qc["offset"] = offset
            qc["sigma"] = float(np.std(y[qmask]))
            sp["intensity"] = y
        else:
            print(f"  ! {s['short']}: QUIET_WINDOW {QUIET_WINDOW} is outside the spectrum; "
                  f"no baseline offset / detection limit applied.")
        sp["qc"] = qc
        print(f"  {s['short']}: {qc['ref_text']}; sigma = "
              f"{qc['sigma'] if qc['sigma'] is None else format(qc['sigma'], '.3g')}")

def load_bruker_spectrum(base_path, procno="1"):
    """
    Load a processed Bruker 1H NMR spectrum (1r file).

    Expected structure:
        base_path/          e.g.  aishwarya/570/
            acqus           <- acquisition parameters
            pdata/
                1/          <- procno
                    1r      <- processed spectrum (binary)
                    procs   <- processing parameters

    Returns (ppm_axis, intensity_array, params_dict).
    Works with or without nmrglue installed.
    """
    proc_path = os.path.join(base_path, "pdata", procno)
    acqu_path = base_path  # acqus sits directly in base_path

    for label, path in [("1r",    os.path.join(proc_path, "1r")),
                        ("acqus", os.path.join(acqu_path, "acqus")),
                        ("procs", os.path.join(proc_path, "procs"))]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Cannot find '{label}' at:\n  {path}\n"
                f"Please check BASE_DIR, the EXPNO in SAMPLES, and PROCNO."
            )

    try:
        import nmrglue as ng
        dic, data = ng.bruker.read_pdata(proc_path)
        if "acqus" not in dic:
            acqus_raw = _read_jcamp(os.path.join(acqu_path, "acqus"))
            dic["acqus"] = {k: _try_num(v) for k, v in acqus_raw.items()}
        udic = ng.bruker.guess_udic(dic, data)
        uc   = ng.fileiobase.uc_from_udic(udic)
        ppm  = uc.ppm_scale()
        if data.ndim > 1:
            data = data[0]
        intensity = np.real(data).astype(float)
        params = {
            "SF":   float(dic.get("procs", {}).get("SF",  500.0)),
            "SW":   float(dic.get("acqus", {}).get("SW",   20.0)),
            "SFO1": float(dic.get("acqus", {}).get("SFO1",500.0)),
        }
        params.update(_acq_params(dic.get("acqus", {})))
        return ppm, intensity, params

    except ImportError:
        print("  nmrglue not installed \u2014 using built-in binary reader (this is fine).")
        acqus  = _read_jcamp(os.path.join(acqu_path, "acqus"))
        procs  = _read_jcamp(os.path.join(proc_path,  "procs"))

        SW     = float(acqus.get("SW",     20.0))
        SFO1   = float(acqus.get("SFO1",  500.0))
        SF     = float(procs.get("SF",    SFO1))
        SI     = int(  procs.get("SI",   32768))
        OFFSET = float(procs.get("OFFSET", 20.0))

        bytorda = int(procs.get("BYTORDP", 0))
        dtype   = ">i4" if bytorda == 1 else "<i4"
        raw     = np.fromfile(os.path.join(proc_path, "1r"), dtype=dtype)
        if len(raw) > SI:
            raw = raw[:SI]
        intensity = raw.astype(float)
        ppm = np.linspace(OFFSET, OFFSET - SW, len(intensity))
        params = {"SF": SF, "SW": SW, "SFO1": SFO1}
        params.update(_acq_params(acqus))
        return ppm, intensity, params


def _try_num(v):
    try:    return int(v)
    except (ValueError, TypeError): pass
    try:    return float(v)
    except (ValueError, TypeError): return v


def _read_jcamp(filepath):
    """Parse a Bruker JCAMP-DX parameter file into a dict. Array parameters such as
    ##$D= (0..63) keep their values under params["_ARR_D"] (D1 is element 1)."""
    params = {}
    if not os.path.exists(filepath):
        return params
    current_array = None
    with open(filepath, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.startswith("##$"):
                current_array = None
                parts = line[3:].split("=", 1)
                if len(parts) == 2:
                    key, val = parts[0].strip(), parts[1].strip()
                    params[key] = val
                    if val.startswith("(") and ".." in val:
                        current_array = key
                        params["_ARR_" + key] = []
            elif line.startswith("##") or line.startswith("$$"):
                current_array = None
            elif current_array is not None and line:
                params["_ARR_" + current_array].extend(line.split())
    return params


def _acq_params(acqus):
    """Acquisition settings that must match between samples for a valid comparison."""
    out = {}
    for k in ("NS", "D1", "TE", "PULPROG", "TD", "RG"):
        v = acqus.get(k)
        if v is None:
            continue
        if isinstance(v, str):
            v = _try_num(v.strip().strip("<>").strip())
        out[k] = v
    for src in (acqus.get("_ARR_D"), acqus.get("D")):
        if isinstance(src, (list, tuple)) and len(src) > 1:
            try:
                out["D1"] = float(src[1])
                break
            except (TypeError, ValueError):
                pass
    return out


def load_all_samples():
    """Load every spectrum listed in SAMPLES. Returns dict keyed by sample key."""
    spectra = {}
    for s in SAMPLES:
        path = os.path.join(BASE_DIR, s["expno"])
        ppm, intensity, params = load_bruker_spectrum(path, PROCNO)
        spectra[s["key"]] = {"ppm": ppm, "intensity": intensity, "params": params}
        print(f"  {s['label']} (EXPNO {s['expno']}): {len(ppm)} points, "
              f"ppm range {ppm.max():.1f}\u2013{ppm.min():.1f}")
    return spectra


def normalise(intensity, ppm=None, exclude=HDO_EXCLUSION_PPM):
    """
    Normalise spectrum to [0, 1] using the min/max computed OUTSIDE the
    `exclude` ppm window (default: the residual HDO peak). This keeps a huge
    solvent peak from squashing every analyte signal down near zero. Pass
    ppm=None (or exclude=None) to fall back to a plain global min/max.

    Points inside the excluded window can come out >1 after this scaling —
    that's expected; callers that plot the result should clip/shade that
    window rather than rescale around it.
    """
    if ppm is not None and exclude is not None:
        lo, hi = exclude
        mask = ~((ppm >= lo) & (ppm <= hi))
        ref = intensity[mask] if mask.sum() > 1 else intensity
    else:
        ref = intensity
    mn, mx = ref.min(), ref.max()
    if mx - mn == 0:
        return intensity - mn
    return (intensity - mn) / (mx - mn)

def pick_peaks(ppm, intensity, height_frac=0.005, min_distance_ppm=0.02,
               exclude=HDO_EXCLUSION_PPM, sanity_range=(-1.0, 12.5)):
    """
    Pick peaks. Returns list of (ppm, relative_height) tuples, sorted by ppm
    desc. Peaks falling inside `exclude` (default: HDO_EXCLUSION_PPM) are
    dropped, and the normalisation used for the height threshold ignores
    that window too, so the residual solvent peak neither gets reported as
    an analyte peak nor suppresses detection of real ones.

    `sanity_range` drops anything outside the range where a real ¹H signal
    could plausibly appear (default -1 to 12.5 ppm, matching the full span
    of ASSIGNMENT_LIBRARY). This matters because the very edges of a
    processed Bruker spectrum can carry small digital-filter / truncation
    artifacts (often tens of ppm away from anything chemically real) that
    find_peaks will happily detect. Without this filter those artifacts can
    dominate a "top N peaks" list if it's later sorted or sliced by
    something other than height.
    """
    norm     = normalise(intensity, ppm, exclude)
    ppm_step = abs(ppm[1] - ppm[0])
    min_dist = max(1, int(min_distance_ppm / ppm_step))

    peaks_idx, props = find_peaks(
        norm,
        height=height_frac,
        distance=min_dist,
        prominence=0.002,
    )
    result = [(float(ppm[i]), float(norm[i])) for i in peaks_idx]
    if exclude is not None:
        lo, hi = exclude
        result = [(p, h) for (p, h) in result if not (lo <= p <= hi)]
    if sanity_range is not None:
        slo, shi = sanity_range
        result = [(p, h) for (p, h) in result if slo <= p <= shi]
    result.sort(key=lambda x: x[0], reverse=True)
    return result


def _mask_excluded(ppm, y, exclude):
    """Replace each excluded ppm range by a straight line between its two edges."""
    if not exclude:
        return y
    y = y.copy()
    for lo, hi, *_why in exclude:
        inside = (ppm >= lo) & (ppm <= hi)
        below, above = ppm < lo, ppm > hi
        if not (inside.any() and below.any() and above.any()):
            continue
        ib = np.where(below)[0][np.argmax(ppm[below])]   # nearest point just below the range
        ia = np.where(above)[0][np.argmin(ppm[above])]   # nearest point just above the range
        y[inside] = np.interp(ppm[inside], [ppm[ib], ppm[ia]], [y[ib], y[ia]])
    return y


def integrate_region(ppm, intensity, ppm_low, ppm_high, exclude=None):
    """
    Integrate a spectral region using the trapezoidal rule.
    `exclude` defaults to INTEGRATION_EXCLUDE (residue-only ranges are bridged by a
    straight line first); pass exclude=[] to integrate the raw spectrum.
    """
    mask = (ppm >= ppm_low) & (ppm <= ppm_high)
    if mask.sum() < 2:
        return 0.0
    if exclude is None:
        exclude = INTEGRATION_EXCLUDE
    y = _mask_excluded(ppm, intensity, exclude) if exclude else intensity
    return float(trapezoid(y[mask], x=ppm[mask]))



def _region_noise_integral(sp, lo, hi):
    """1-sigma noise of a trapezoid integral over [lo, hi] for white noise of std sigma."""
    sigma = sp.get("qc", {}).get("sigma")
    if not sigma:
        return 0.0
    mask = (sp["ppm"] >= lo) & (sp["ppm"] <= hi)
    n = int(mask.sum())
    if n < 2:
        return 0.0
    dx = float(np.median(np.abs(np.diff(sp["ppm"]))))
    return float(sigma * dx * np.sqrt(n))


def _envelope_integral(sp):
    """Total polysaccharide signal: ENVELOPE_WINDOW with the HDO window removed."""
    lo, hi = ENVELOPE_WINDOW
    ppm, y = sp["ppm"], sp["intensity"]
    if HDO_EXCLUSION_PPM is None:
        return abs(integrate_region(ppm, y, lo, hi))
    hlo, hhi = HDO_EXCLUSION_PPM
    return (abs(integrate_region(ppm, y, lo, min(hi, hlo))) +
            abs(integrate_region(ppm, y, max(lo, hhi), hi)))


def compute_denominators(spectra, regions, raw):
    """
    Per-sample denominator for the chosen NORMALISATION.
    Returns (dens, scale, unit_label, mode_used, notes).
    Falls back to 'envelope' (with a note) if an internal standard is requested
    but not configured or not detected.
    """
    used, notes, dens = NORMALISATION, [], {}

    if used == "internal_standard":
        if STANDARD_WINDOW is None:
            notes.append("internal_standard requested but STANDARD_WINDOW is not set; "
                         "used envelope normalisation instead.")
            used = "envelope"
        else:
            for s in SAMPLES:
                sp = spectra[s["key"]]
                I = abs(integrate_region(sp["ppm"], sp["intensity"], *STANDARD_WINDOW))
                if I <= 10.0 * _region_noise_integral(sp, *STANDARD_WINDOW) or I == 0:
                    notes.append(f"standard not detected in {s['short']}; "
                                 "used envelope normalisation instead.")
                    used = "envelope"
                    dens = {}
                    break
                dens[s["key"]] = I / STANDARD_NH
            else:
                return dens, 1.0, "H-equivalents per standard molecule", used, notes

    if used == "h1":
        idx = [i for i, r in enumerate(regions) if r[3] == "Backbone"]
        for s in SAMPLES:
            dens[s["key"]] = sum(raw[s["key"]][i] for i in idx)
        return dens, 100.0, "% of anomeric H-1 signal", used, notes

    for s in SAMPLES:                       # envelope
        dens[s["key"]] = _envelope_integral(spectra[s["key"]])
    return dens, 100.0, "% of total polysaccharide signal", "envelope", notes


def compute_region_table(spectra, regions):
    """
    Integrate every region for every sample on the standardised spectra.
    Returns (rows, norm, nd, meta):
      rows  -> one dict per region (normalised integral, ND flag, difference vs native)
      norm  -> {sample_key: [normalised integral per region]}
      nd    -> {sample_key: [True if below the detection limit]}
      meta  -> {"unit", "used", "notes"}
    """
    raw, lod = {}, {}
    for s in SAMPLES:
        sp = spectra[s["key"]]
        raw[s["key"]] = [abs(integrate_region(sp["ppm"], sp["intensity"], lo, hi))
                         for (lo, hi, *_r) in regions]
        lod[s["key"]] = [LOD_SIGMA * _region_noise_integral(sp, lo, hi)
                         for (lo, hi, *_r) in regions]

    dens, scale, unit, used, notes = compute_denominators(spectra, regions, raw)
    norm = {k: [(scale * v / dens[k]) if dens[k] else float("nan") for v in vals]
            for k, vals in raw.items()}
    nd = {k: [raw[k][i] < lod[k][i] for i in range(len(regions))] for k in raw}

    ref_label = next(s["label"] for s in SAMPLES if s["key"] == REFERENCE_KEY)
    rows = []
    for i, (lo, hi, label, group) in enumerate(regions):
        row = {"Region": label.replace("\n", " "),
               "ppm range": f"{lo:.2f}\u2013{hi:.2f}", "Group": group}
        for s in SAMPLES:
            row[f"Integral: {s['label']}"] = round(norm[s["key"]][i], 3)
            row[f"ND: {s['label']}"] = bool(nd[s["key"]][i])
            if s["key"] != REFERENCE_KEY:
                row[f"\u0394 vs {ref_label} [{s['label']}]"] = round(
                    norm[s["key"]][i] - norm[REFERENCE_KEY][i], 3)
        rows.append(row)
    return rows, norm, nd, {"unit": unit, "used": used, "notes": notes}


def compute_functional_group_summary(regions, norm, nd, meta):
    """
    Sum the normalised integrals of every region in a group, per sample.
    Regions below the detection limit contribute zero; a group where every region is
    ND is shown as ND. Under 'h1' normalisation the Backbone group is the
    denominator and is left out.
    """
    idx_by_group = {}
    for i, (lo, hi, label, group) in enumerate(regions):
        idx_by_group.setdefault(group, []).append(i)

    rows = []
    for group, idxs in idx_by_group.items():
        if group == "Backbone" and meta["used"] == "h1":
            continue
        row = {"Functional group": group, "_ND": {}}
        for s in SAMPLES:
            k = s["key"]
            row[s["label"]] = round(
                sum(0.0 if nd[k][i] else norm[k][i] for i in idxs), 3)
            row["_ND"][s["label"]] = all(nd[k][i] for i in idxs)
        rows.append(row)
    return rows


def evaluate_trends(group_rows, tol_frac=0.10):
    """
    Compare each group's observed behaviour across SAMPLES with EXPECTED_TREND.
    Tolerance = tol_frac x the group's largest value (2x for 'constant'). This is a
    descriptive check on 5 points, not a statistical test.
    """
    labels = [s["label"] for s in SAMPLES]
    shorts = [s["short"] for s in SAMPLES]
    out = []
    for row in group_rows:
        g = row["Functional group"]
        exp = EXPECTED_TREND.get(g)
        if exp is None:
            continue
        if all(row["_ND"][l] for l in labels):
            out.append({"group": g, "expected": exp, "ok": None,
                        "observed": "below the detection limit in every sample",
                        "verdict": "not testable"})
            continue
        vals = [float(row[l]) for l in labels]
        vmax = max(abs(v) for v in vals) or 1.0
        tol = tol_frac * vmax * (2 if exp == "constant" else 1)
        v0, vL, imax = vals[0], vals[-1], int(np.argmax(vals))
        n = len(vals)
        down_ok = all(vals[i + 1] <= vals[i] + tol for i in range(n - 1))
        up_ok = all(vals[i + 1] >= vals[i] - tol for i in range(n - 1))
        peaked = 0 < imax < n - 1 and vals[imax] > v0 + tol and vals[imax] > vL + tol

        if peaked:
            pattern = f"rises to a maximum at {shorts[imax]}, then falls"
        elif vL > v0 + tol:
            pattern = "rises steadily" if up_ok else "rises, not monotonic"
        elif vL < v0 - tol:
            pattern = "falls steadily" if down_ok else "falls, not monotonic"
        elif max(abs(v - v0) for v in vals) <= tol:
            pattern = "flat"
        else:
            pattern = "no clear trend"

        ok = {"down": vL < v0 - tol and down_ok,
              "up": vL > v0 + tol and up_ok,
              "up_then_down": peaked,
              "constant": max(abs(v - v0) for v in vals) <= tol}.get(exp)
        out.append({"group": g, "expected": exp, "ok": ok,
                    "observed": f"{pattern} ({v0:.3g} \u2192 {vL:.3g})",
                    "verdict": "consistent" if ok else "NOT consistent"})
    return out



def plot_overlay(spectra, out_path):
    """Stacked overlay plot: one panel per sample, shared x-axis."""
    n = len(SAMPLES)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.2 * n), sharex=True)
    if n == 1:
        axes = [axes]
    fig.patch.set_facecolor("#FAFAFA")

    for ax, s in zip(axes, SAMPLES):
        sp = spectra[s["key"]]
        norm = normalise(sp["intensity"], sp["ppm"], HDO_EXCLUSION_PPM)
        norm_clipped = np.clip(norm, -0.05, 1.15)  # HDO peak can exceed 1; clip for display
        ax.set_facecolor("#FAFAFA")
        for lo, hi, label, group in INTEGRATION_REGIONS:
            ax.axvspan(lo, hi, color=GROUP_COLORS.get(group, "#EEEEEE"), alpha=0.12, zorder=0)
        if HDO_EXCLUSION_PPM is not None:
            hlo, hhi = HDO_EXCLUSION_PPM
            ax.axvspan(hlo, hhi, color="#BDBDBD", alpha=0.35, zorder=1, hatch="//")
            ax.text((hlo + hhi) / 2, 1.02, "HDO\n(off-scale)", ha="center", va="bottom",
                    fontsize=6.5, color="#555", transform=ax.get_xaxis_transform())
        ax.plot(sp["ppm"], norm_clipped, color=s["color"], lw=0.8, label=s["label"])
        ax.fill_between(sp["ppm"], norm_clipped, alpha=0.08, color=s["color"])
        ax.set_ylabel("Norm. Intensity", fontsize=9)
        ax.legend(loc="upper left", fontsize=9, framealpha=0.8)
        ax.set_xlim(*PPM_XLIM)
        ax.set_ylim(-0.05, 1.15)
        ax.grid(axis="x", ls="--", lw=0.4, alpha=0.5)
        ax.spines[["top", "right"]].set_visible(False)

    axes[-1].set_xlabel("Chemical Shift (ppm)", fontsize=10)
    fig.suptitle("\u00b9H NMR Spectral Overlay \u2014 Tamarind Gum Oxidation Series",
                 fontsize=13, fontweight="bold", y=1.005)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  \u2713 Saved: {out_path}")


def _assign_label_levels(candidates_by_ppm_desc, min_gap_ppm=1.6, max_levels=5):
    """
    Greedily assign a vertical stacking level to each label so that labels
    whose ppm positions are within min_gap_ppm of an already-placed label at
    the same level get bumped to the next level up, avoiding text overlap
    for peaks that sit close together.
    """
    last_ppm_at_level = [None] * max_levels
    out = []
    for p, h, name, group in candidates_by_ppm_desc:
        level = 0
        while level < max_levels - 1:
            last_p = last_ppm_at_level[level]
            if last_p is None or abs(last_p - p) >= min_gap_ppm:
                break
            level += 1
        last_ppm_at_level[level] = p
        out.append((p, h, name, group, level))
    return out


def plot_annotated_spectrum(spectra, peaks_by_sample, out_path, top_n=10, min_height=0.03):
    """
    'Master pipeline' view: one panel per sample, each peak that matches the
    ASSIGNMENT_LIBRARY labeled directly on the spectrum with a leader line
    to its chemical assignment. Unassigned peaks and peaks below min_height
    are left unlabeled to keep the plot readable; widen/narrow top_n and
    min_height if you want more or fewer labels.
    """
    n = len(SAMPLES)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.8 * n), sharex=True)
    if n == 1:
        axes = [axes]
    fig.patch.set_facecolor("#FAFAFA")

    for ax, s in zip(axes, SAMPLES):
        sp = spectra[s["key"]]
        norm = normalise(sp["intensity"], sp["ppm"], HDO_EXCLUSION_PPM)
        norm_clipped = np.clip(norm, -0.05, 1.15)
        ax.set_facecolor("#FAFAFA")
        if HDO_EXCLUSION_PPM is not None:
            hlo, hhi = HDO_EXCLUSION_PPM
            ax.axvspan(hlo, hhi, color="#BDBDBD", alpha=0.3, hatch="//", zorder=1)
        ax.plot(sp["ppm"], norm_clipped, color=s["color"], lw=0.9, zorder=2)
        ax.fill_between(sp["ppm"], norm_clipped, alpha=0.06, color=s["color"], zorder=1)

        # Candidate peaks to label: matched to the library, tall enough to be
        # worth labeling, capped at top_n by height so the plot stays legible.
        candidates = []
        for p, h in peaks_by_sample[s["key"]]:
            if h < min_height:
                continue
            name, group = assign_peak(p)
            if name == "Unassigned":
                continue
            candidates.append((p, h, name, group))
        candidates.sort(key=lambda row: -row[1])
        # One label per assignment (the tallest peak in it). A broad envelope such as
        # the ring CH-O hump is picked as many small local maxima; labelling each one
        # with the same text only produces an unreadable pile of overlapping labels.
        seen, unique = set(), []
        for row in candidates:
            if row[2] in seen:
                continue
            seen.add(row[2])
            unique.append(row)
        candidates = unique[:top_n]
        candidates.sort(key=lambda row: -row[0])  # left-to-right for level assignment
        candidates = _assign_label_levels(candidates)

        for p, h, name, group, level in candidates:
            y_text = min(1.50, h + 0.14 + 0.12 * level)
            color = GROUP_COLORS.get(group, "#333333")
            ax.annotate(
                name, xy=(p, min(h, 1.08)), xytext=(p, y_text),
                ha="center", va="bottom", fontsize=6.5, color=color,
                arrowprops=dict(arrowstyle="-", lw=0.6, color=color, shrinkA=0, shrinkB=2),
            )

        ax.set_ylabel("Norm. Intensity", fontsize=9)
        ax.set_title(f"{s['label']}  (EXPNO {s['expno']})", fontsize=10, loc="left",
                     color=s["color"], fontweight="bold")
        ax.set_xlim(*PPM_XLIM)
        ax.set_ylim(-0.05, 1.62)
        ax.grid(axis="x", ls="--", lw=0.4, alpha=0.5)
        ax.spines[["top", "right"]].set_visible(False)

    axes[-1].set_xlabel("Chemical Shift (ppm)", fontsize=10)
    fig.suptitle("Annotated Peak Assignments \u2014 Master Chemistry Pipeline",
                 fontsize=13, fontweight="bold", y=1.005)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  \u2713 Saved: {out_path}")


def plot_peak_comparison(peaks_by_sample, out_path):
    """Stick/bar plot comparing peak positions across all samples."""
    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    n = len(SAMPLES)
    width = 0.035
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width

    for s, off in zip(SAMPLES, offsets):
        peaks = peaks_by_sample[s["key"]]
        if not peaks:
            continue
        xs, hs = zip(*peaks)
        ax.bar(np.array(xs) + off, hs, width=width, color=s["color"], alpha=0.85,
               label=s.get("short", s["label"]))

    ax.set_xlim(*PPM_XLIM)
    ax.set_xlabel("Chemical Shift (ppm)", fontsize=10)
    ax.set_ylabel("Relative Intensity", fontsize=10)
    ax.set_title("Peak Comparison Across Oxidation Series", fontsize=11)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", ls="--", lw=0.4, alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  \u2713 Saved: {out_path}")


def plot_integration(region_rows, out_path):
    """Grouped bar chart: one group of N bars (one per sample) per region."""
    labels = [r["Region"][:32] for r in region_rows]
    n = len(SAMPLES)
    x = np.arange(len(labels))
    w = 0.8 / n

    fig, ax = plt.subplots(figsize=(15, 6))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    for i, s in enumerate(SAMPLES):
        vals = [r[f"Integral: {s['label']}"] for r in region_rows]
        pos = x - 0.4 + w / 2 + i * w
        bars = ax.bar(pos, vals, w, label=s.get("short", s["label"]), color=s["color"], alpha=0.85)
        ax.bar_label(bars, fmt="%.3g", fontsize=max(5.0, 6.5 - 0.3 * max(0, n - 3)),
                     padding=2, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel(UNIT_LABEL, fontsize=9)
    ax.set_title("Integration Analysis by Spectral Region", fontsize=11)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", ls="--", lw=0.4, alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  \u2713 Saved: {out_path}")


def plot_group_trend(group_rows, out_path):
    """Line plot: functional-group relative content (%) across the sample series."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    # group_rows is keyed by full sample labels (see
    # compute_functional_group_summary) -- look values up by that, but show
    # the shorter display label on the x-axis so it stays legible with more
    # than ~3 samples.
    sample_labels = [s["label"] for s in SAMPLES]
    short_labels  = [s.get("short", s["label"]) for s in SAMPLES]
    xpos = np.arange(len(sample_labels))

    for row in group_rows:
        group = row["Functional group"]
        vals = [row[lbl] if isinstance(row[lbl], (int, float)) else np.nan for lbl in sample_labels]
        ax.plot(xpos, vals, marker="o", lw=2, ms=7,
                color=GROUP_COLORS.get(group, None), label=group)

    ax.set_xticks(xpos)
    ax.set_xticklabels(short_labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel(UNIT_LABEL, fontsize=9)
    ax.set_title("Functional-Group Progression Across the Oxidation Series", fontsize=11)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", ls="--", lw=0.4, alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  \u2713 Saved: {out_path}")


def plot_difference_series(spectra, out_path):
    """Difference spectra between each sample and the reference sample."""
    ref_sp = spectra[REFERENCE_KEY]
    ref_norm = normalise(ref_sp["intensity"], ref_sp["ppm"], HDO_EXCLUSION_PPM)
    others = [s for s in SAMPLES if s["key"] != REFERENCE_KEY]
    n = len(others)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 1, figsize=(14, 3.6 * n), sharex=True)
    if n == 1:
        axes = [axes]
    fig.patch.set_facecolor("#FAFAFA")

    ref_label = next(s["label"] for s in SAMPLES if s["key"] == REFERENCE_KEY)

    for ax, s in zip(axes, others):
        sp = spectra[s["key"]]
        norm_s = normalise(sp["intensity"], sp["ppm"], HDO_EXCLUSION_PPM)
        interp = np.interp(ref_sp["ppm"], sp["ppm"][::-1], norm_s[::-1])
        diff = interp - ref_norm
        if HDO_EXCLUSION_PPM is not None:
            hlo, hhi = HDO_EXCLUSION_PPM
            diff = np.where((ref_sp["ppm"] >= hlo) & (ref_sp["ppm"] <= hhi), 0.0, diff)

        ax.set_facecolor("#FAFAFA")
        if HDO_EXCLUSION_PPM is not None:
            hlo, hhi = HDO_EXCLUSION_PPM
            ax.axvspan(hlo, hhi, color="#BDBDBD", alpha=0.35, zorder=1, hatch="//")
        ax.plot(ref_sp["ppm"], diff, color="#6A1B9A", lw=0.8)
        ax.fill_between(ref_sp["ppm"], diff, where=(diff > 0), color=s["color"], alpha=0.35,
                         label=f"Increased in {s['label']}")
        ax.fill_between(ref_sp["ppm"], diff, where=(diff < 0), color="#2196F3", alpha=0.3,
                         label=f"Decreased vs {ref_label}")
        ax.axhline(0, color="black", lw=0.6, ls="--")
        ax.set_xlim(*PPM_XLIM)
        ax.set_ylabel("\u0394 Intensity", fontsize=9)
        ax.set_title(f"{s['label']} \u2212 {ref_label}", fontsize=10)
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", ls="--", lw=0.4, alpha=0.5)

    axes[-1].set_xlabel("Chemical Shift (ppm)", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  \u2713 Saved: {out_path}")



def build_pdf_report(region_rows, group_rows, peaks_by_sample,
                     img_overlay, img_annot, img_peaks, img_integ, img_trend, img_diff,
                     params_by_sample, qc_by_sample, verdicts, unit_label, out_path):
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    Table, TableStyle, Image, HRFlowable,
                                    PageBreak)

    doc    = SimpleDocTemplate(out_path, pagesize=A4,
                               leftMargin=1.8*cm, rightMargin=1.8*cm,
                               topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    W      = A4[0] - 3.6*cm

    # Aspect ratios computed from each plot function's actual figsize, rather
    # than hardcoded fractions of W. This matters once SAMPLES has more than
    # 3 entries: plot_overlay/plot_annotated_spectrum/plot_difference_series
    # all grow taller with more samples (one panel per sample), so a fixed
    # height fraction would stretch or squash the image. Fixed-figsize plots
    # (peak comparison, integration, trend) use their literal figsize ratio.
    n_samples   = len(SAMPLES)
    n_diff      = max(1, n_samples - 1)  # difference panels = all non-reference samples
    overlay_ar  = (3.2 * n_samples) / 14.0
    annot_ar    = (3.8 * n_samples) / 14.0
    diff_ar     = (3.6 * n_diff) / 14.0
    peaks_ar    = 5.0 / 13.0
    integ_ar    = 6.0 / 15.0
    trend_ar    = 5.5 / 9.0

    # With more than ~4 samples, a per-panel plot (overlay/annotated/diff) can
    # come out taller than a single A4 page even at full page width -- which
    # would either overflow or raise a layout error. fit_image shrinks WIDTH
    # (keeping the aspect ratio intact, so nothing stretches/squashes) whenever
    # the height-for-full-width would exceed what a page can hold.
    MAX_IMG_H = A4[1] - 6*cm  # leaves room for a heading + a line of text above it
    def fit_image(aspect_ratio, max_w=W, max_h=MAX_IMG_H):
        w, h = max_w, max_w * aspect_ratio
        if h > max_h:
            h = max_h
            w = h / aspect_ratio
        return w, h

    title_style = ParagraphStyle("title", parent=styles["Title"],
                                 fontSize=18, spaceAfter=6, textColor=colors.HexColor("#1A237E"))
    h1_style    = ParagraphStyle("h1", parent=styles["Heading1"],
                                 fontSize=13, textColor=colors.HexColor("#1565C0"), spaceBefore=14)
    h2_style    = ParagraphStyle("h2", parent=styles["Heading2"],
                                 fontSize=11, textColor=colors.HexColor("#1976D2"), spaceBefore=10)
    body_style  = ParagraphStyle("body", parent=styles["Normal"],
                                 fontSize=9.5, leading=14)
    bold_style  = ParagraphStyle("bold", parent=body_style, fontName="Helvetica-Bold")

    def tbl_style(header_color="#1565C0", fontsize=7.5):
        return TableStyle([
            ("BACKGROUND",   (0, 0), (-1, 0),  colors.HexColor(header_color)),
            ("TEXTCOLOR",    (0, 0), (-1, 0),  colors.white),
            ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTSIZE",     (0, 0), (-1, -1), fontsize),
            ("ROWBACKGROUNDS",(0,1), (-1,-1),  [colors.HexColor("#F5F5F5"), colors.white]),
            ("GRID",         (0, 0), (-1, -1), 0.4, colors.HexColor("#BDBDBD")),
            ("ALIGN",        (1, 0), (-1, -1), "CENTER"),
            ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING",   (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
            ("LEFTPADDING",  (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ])

    story = []
    # "sample_labels" (full) is used as the dict key everywhere data is keyed
    # by sample -- compute_functional_group_summary and the conclusion's
    # fmt_pct() both look values up by this exact string, so it must NOT
    # change. "short_labels" is a display-only alias for table headers and
    # legends, which get cramped with 6 (or more) full labels.
    sample_labels = [s["label"] for s in SAMPLES]
    short_labels  = [s.get("short", s["label"]) for s in SAMPLES]
    ref_label = next(s["label"] for s in SAMPLES if s["key"] == REFERENCE_KEY)

    # ── Title page ────────────────────────────────────────────────────────
    story.append(Spacer(1, 1*cm))
    story.append(Paragraph("\u00b9H NMR Oxidation Analysis Report", title_style))
    story.append(Paragraph(f"Tamarind Gum \u2014 {len(SAMPLES)}-Sample Oxidation Series "
                           "(Alcohol / Aldehyde / Carboxyl)", h2_style))
    story.append(HRFlowable(width=W, thickness=2, color=colors.HexColor("#1565C0")))
    story.append(Spacer(1, 0.4*cm))

    info_data = [["Parameter"] + short_labels]
    info_rows = [
        ("Instrument", "Bruker Avance Neo 500 MHz"),
        ("Solvent",    SOLVENT),
        ("EXPNO",      None),
        ("SFO1 (MHz)", None),
    ]
    # Rows where every sample shares the same value (Instrument, Solvent) get
    # a spanned cell instead of the identical string repeated in every narrow
    # column -- with 6+ samples a repeated long string overflows its column
    # and visually runs into its neighbours.
    span_commands = []
    for row_idx, (name, fixed_val) in enumerate(info_rows, start=1):  # +1: header is row 0
        if name == "EXPNO":
            info_data.append([name] + [s["expno"] for s in SAMPLES])
        elif name == "SFO1 (MHz)":
            info_data.append([name] + [str(params_by_sample[s["key"]].get("SFO1", "500")) for s in SAMPLES])
        else:
            info_data.append([name, fixed_val] + [""] * (len(SAMPLES) - 1))
            span_commands.append(("SPAN", (1, row_idx), (len(SAMPLES), row_idx)))
    col_w = [W*0.22] + [W*0.78/len(SAMPLES)] * len(SAMPLES)
    t = Table(info_data, colWidths=col_w)
    base_style = tbl_style(fontsize=8)
    for cmd in span_commands:
        base_style.add(*cmd)
    t.setStyle(base_style)
    story.append(t)
    story.append(Spacer(1, 0.4*cm))

    # ── Standardisation / QC table ───────────────────────────────────────
    cell_style = ParagraphStyle("cell", parent=body_style, fontSize=7, leading=8.5,
                                alignment=1)
    story.append(Paragraph("Standardisation and quality control", h2_style))

    def _qc_cell(text):
        return Paragraph(str(text), cell_style)

    def _p(s, k, fmt="{}"):
        v = params_by_sample[s["key"]].get(k)
        return "n/a" if v is None else fmt.format(v)

    qc_rows = [["Check"] + short_labels]
    qc_rows.append(["Referencing"] + [_qc_cell(qc_by_sample[s["key"]]["ref_text"]) for s in SAMPLES])
    qc_rows.append(["Temperature (K)"] + [_p(s, "TE", "{:g}") for s in SAMPLES])
    qc_rows.append(["Scans (NS)"] + [_p(s, "NS") for s in SAMPLES])
    qc_rows.append(["Relaxation delay D1 (s)"] + [_p(s, "D1", "{:g}") for s in SAMPLES])
    qc_rows.append(["Pulse program"] + [_qc_cell(_p(s, "PULPROG")) for s in SAMPLES])
    qc_rows.append(["Noise sigma (a.u.)"] + [
        ("n/a" if qc_by_sample[s["key"]].get("sigma") is None
         else f"{qc_by_sample[s['key']]['sigma']:.3g}") for s in SAMPLES])
    qc_rows.append(["Consistency vs native"] + [
        ("reference" if s["key"] == REFERENCE_KEY else
         ("CHECK" if any(f.startswith("!") for f in qc_by_sample[s["key"]]["flags"]) else "OK"))
        for s in SAMPLES])
    qc_col_w = [W*0.22] + [W*0.78/len(SAMPLES)] * len(SAMPLES)
    qt = Table(qc_rows, colWidths=qc_col_w)
    qt.setStyle(tbl_style(fontsize=7))
    story.append(qt)
    story.append(Spacer(1, 0.2*cm))

    flag_lines = []
    for s in SAMPLES:
        fl = qc_by_sample[s["key"]]["flags"]
        if fl:
            flag_lines.append(f"<b>{s['short']}</b>: " + "; ".join(f.replace("! ", "").replace("i ", "(info) ") for f in fl))
    if flag_lines:
        story.append(Paragraph("Acquisition differences versus the native spectrum: " +
                               " | ".join(flag_lines), cell_style))
    excl_txt = "; ".join(f"{lo:.2f}\u2013{hi:.2f} ppm ({why})"
                         for lo, hi, why in INTEGRATION_EXCLUDE) or "none"
    story.append(Paragraph(
        f"Normalisation used: <b>{unit_label}</b>. Integrals below {LOD_SIGMA:g} x the integral "
        "noise are reported as ND (not detected). Residue-only ranges removed from every "
        f"integral in all samples: {excl_txt}.", cell_style))
    story.append(Spacer(1, 0.4*cm))


    # ── Section 1: Overlay ───────────────────────────────────────────────
    story.append(Paragraph("1. Spectral Overlay", h1_style))
    story.append(Paragraph(
        "Stacked spectra allow direct comparison of chemical-shift positions and relative "
        "intensities across the oxidation series. Loss of hydroxyl-region intensity together "
        "with the appearance, then partial disappearance, of aldehyde signals as carboxyl "
        "(\u03b1-CH) signals grow is the expected signature of a native \u2192 aldehyde \u2192 "
        "carboxyl oxidation pathway. The hatched band marks the residual HDO solvent peak, "
        "which is clipped here so it doesn't dwarf the analyte signals.", body_style))
    story.append(Spacer(1, 0.3*cm))
    story.append(Image(img_overlay, *fit_image(overlay_ar)))
    story.append(PageBreak())

    # ── Section 2: Difference spectra ───────────────────────────────────
    story.append(Paragraph(f"2. Difference Spectra (vs {ref_label})", h1_style))
    story.append(Paragraph(
        "Each panel shows a later-stage spectrum minus the reference (native) spectrum. "
        "Positive peaks are signals that grew relative to the native sample; negative peaks "
        "are signals that were lost or reduced.", body_style))
    story.append(Spacer(1, 0.3*cm))
    story.append(Image(img_diff, *fit_image(diff_ar)))
    story.append(Spacer(1, 0.3*cm))

    # ── Section 3: Annotated peak assignments (master chemistry view) ────
    story.append(Paragraph("3. Annotated Peak Assignments", h1_style))
    story.append(Paragraph(
        "Every picked peak is matched against a chemical-shift assignment library "
        "(ASSIGNMENT_LIBRARY in the script) and, where it matches, labeled directly on the "
        "spectrum below. This is the single view to check first: it points out exactly which "
        "chemistry each real peak in your data is being read as, rather than just a bulk "
        "region integral. Only the tallest matched peaks per sample are labeled to keep it "
        "legible \u2014 the full peak list with assignments follows in Section 4.", body_style))
    story.append(Spacer(1, 0.3*cm))
    story.append(Image(img_annot, *fit_image(annot_ar)))
    story.append(PageBreak())

    # ── Section 4: Peak picking ───────────────────────────────────────────
    story.append(Paragraph("4. Peak Picking & Comparison", h1_style))
    story.append(Image(img_peaks, *fit_image(peaks_ar)))
    story.append(Spacer(1, 0.3*cm))
    for s in SAMPLES:
        story.append(Paragraph(f"Top peaks \u2014 {s['label']} (EXPNO {s['expno']}):", h2_style))
        # Select the 12 TALLEST peaks (by relative height), then display them
        # in ppm order for readability. Previously this sliced the ppm-sorted
        # list directly, which silently showed the highest-ppm peaks rather
        # than the most intense ones -- harmless when they coincided, but
        # capable of burying every real peak under low-level edge noise when
        # they didn't.
        top = sorted(peaks_by_sample[s["key"]], key=lambda row: -row[1])[:12]
        top.sort(key=lambda row: -row[0])
        pdata = [["Peak #", "ppm", "Rel. Int.", "Assignment", "Group"]]
        for i, (p, h) in enumerate(top, 1):
            name, group = assign_peak(p)
            pdata.append([str(i), f"{p:.4f}", f"{h:.3f}", name, group])
        t = Table(pdata, colWidths=[W*0.08, W*0.14, W*0.11, W*0.52, W*0.15])
        t.setStyle(tbl_style())
        story.append(t)
        story.append(Spacer(1, 0.3*cm))

    story.append(PageBreak())

    # ── Section 5: Integration (landscape-width table on portrait page) ──
    story.append(Paragraph("5. Integration Analysis", h1_style))
    story.append(Paragraph(
        f"Integrals of the oxidation-diagnostic regions, normalised as: {unit_label}. "
        "ND = below the detection limit (3 x integral noise).", body_style))
    story.append(Spacer(1, 0.3*cm))
    story.append(Image(img_integ, *fit_image(integ_ar)))
    story.append(Spacer(1, 0.4*cm))

    def _fmt_num(x):
        # Compact display for the integration table: wide columns (more
        # samples) leave less room per cell, and raw integrals can run into
        # the millions, so switch to scientific notation above 1e5 rather
        # than overflowing a narrow cell with a long decimal.
        try:
            x = float(x)
        except (TypeError, ValueError):
            return str(x)
        return f"{x:.2e}" if abs(x) >= 1e5 else f"{x:.2f}"

    int_header = ["Region", "ppm", "Group"] + short_labels
    int_data = [int_header]
    for r in region_rows:
        row = [r["Region"][:28], r["ppm range"], r["Group"]]
        row += [("ND" if r.get(f"ND: {s['label']}") else _fmt_num(r[f"Integral: {s['label']}"]))
                for s in SAMPLES]
        int_data.append(row)
    n_extra = len(SAMPLES)
    region_fontsize = 6.8 if n_extra <= 4 else 6.0
    col_w = [W*0.24, W*0.09, W*0.09] + [W*0.58/n_extra]*n_extra
    t = Table(int_data, colWidths=col_w)
    t.setStyle(tbl_style(fontsize=region_fontsize))
    story.append(t)
    story.append(Spacer(1, 0.5*cm))

    # ── Section 6: Functional-group progression ─────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("6. Functional-Group Progression (Degree of Oxidation)", h1_style))
    story.append(Paragraph(
        f"Normalised integrals summed per functional group ({unit_label}). "
        "Alcohol should fall with oxidation; aldehyde (hydrate / hemiacetal) should rise; "
        "carboxyl-adjacent signal should rise. ND = below the detection limit.",
        body_style))
    story.append(Spacer(1, 0.3*cm))
    story.append(Image(img_trend, *fit_image(trend_ar)))
    story.append(Spacer(1, 0.4*cm))

    fg_header = ["Functional group"] + short_labels
    fg_data = [fg_header]
    for r in group_rows:
        # group_rows is keyed by the full sample_labels (see
        # compute_functional_group_summary) -- look up by that, display short.
        fg_data.append([r["Functional group"]] + [
            ("ND" if r["_ND"][lbl] else f"{r[lbl]:.3g}") for lbl in sample_labels])
    fg_fontsize = 9 if len(sample_labels) <= 4 else 7.5
    col_w = [W*0.22] + [W*0.78/len(sample_labels)]*len(sample_labels)
    t = Table(fg_data, colWidths=col_w)
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor("#1565C0")),
        ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), fg_fontsize),
        ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#BDBDBD")),
        ("ALIGN",       (1, 0), (-1, -1), "CENTER"),
        ("TOPPADDING",  (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING",(0,0), (-1,-1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#F5F5F5"), colors.white]),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.5*cm))

    # ── Section 7: Oxidation-marker check ────────────────────────────────
    story.append(HRFlowable(width=W, thickness=1, color=colors.HexColor("#1565C0")))
    story.append(Paragraph("7. Oxidation-Marker Check", h1_style))
    story.append(Paragraph(
        "Each oxidation marker is compared with the behaviour expected across the series "
        "(first sample = native, last = most oxidised). The tolerance is 10% of the group's "
        "largest value. With only five samples this is a descriptive consistency check, "
        "not a statistical test.", body_style))
    story.append(Spacer(1, 0.3*cm))

    exp_words = {"down": "falls", "up": "rises", "up_then_down": "rises, then falls",
                 "constant": "stays constant"}
    v_rows = [["Marker", "Expected", "Observed (first \u2192 last)", "Verdict"]]
    for v in verdicts:
        v_rows.append([Paragraph(v["group"], cell_style),
                       Paragraph(exp_words.get(v["expected"], v["expected"]), cell_style),
                       Paragraph(v["observed"], cell_style),
                       Paragraph(v["verdict"], cell_style)])
    vt = Table(v_rows, colWidths=[W*0.16, W*0.18, W*0.46, W*0.20])
    vt.setStyle(tbl_style(fontsize=8))
    story.append(vt)
    story.append(Spacer(1, 0.4*cm))

    # ── Section 8: Conclusion ────────────────────────────────────────────
    story.append(Paragraph("8. Conclusion", h1_style))
    testable = [v for v in verdicts if v["ok"] is not None]
    ok_g = [v["group"] for v in testable if v["ok"]]
    bad_g = [v["group"] for v in testable if not v["ok"]]
    nt_g = [v["group"] for v in verdicts if v["ok"] is None]

    lines = [
        f"\u00b9H NMR (D2O) was recorded for {len(SAMPLES)} Tamarind Gum samples "
        f"({', '.join(short_labels)}). Spectra were referenced, baseline-offset corrected and "
        f"normalised identically ({unit_label}); the residual HDO region was excluded."
    ]
    if testable:
        lines.append(
            f"{len(ok_g)} of {len(testable)} testable oxidation markers behave as expected"
            + (f" ({', '.join(ok_g)})" if ok_g else "")
            + (f"; {', '.join(bad_g)} did not." if bad_g else ".")
        )
    nd_regions = [r["Region"] for r in region_rows
                  if all(r[f"ND: {s['label']}"] for s in SAMPLES)]
    if nt_g:
        lines.append(f"Not testable (below the detection limit in every sample): {', '.join(nt_g)}.")
    if nd_regions:
        lines.append("Regions below the detection limit in every sample: "
                     + "; ".join(nd_regions) + ".")
    if any(n.startswith("Free aldehyde") for n in nd_regions):
        lines.append("A free -CHO signal is usually absent in D2O, where periodate-derived "
                     "aldehydes exist mainly as hydrates and hemiacetals, so its absence does "
                     "not by itself indicate failed oxidation; the hydrate / hemiacetal window "
                     "is the relevant aldehyde marker.")
    lines.append(
        "Limits of this analysis: -OH and -COOH protons exchange with D2O and are not "
        "observed. Native Tamarind is a neutral xyloglucan, so native ring protons heavily overlap "
        "the 4.0-4.4 ppm carboxyl window. The net decrease in this region mathematically reflects the "
        "periodate-induced disruption of the native rings dominating the formation of any new carboxyl groups. "
        "The reaction was successfully halted with ethylene glycol; minor residual peaks for this quencher "
        "were identified and mathematically excluded from the integrals. Under envelope normalisation, "
        "the values are compositional shares, so only relative changes are meaningful. "
        "Confirm aldehyde content (hydroxylamine titration) and carboxyl "
        "content (conductometric titration) independently.")
    for line in lines:
        story.append(Paragraph(line, body_style))
        story.append(Spacer(1, 0.15*cm))

    story.append(Spacer(1, 0.6*cm))
    story.append(Paragraph("\u2014 End of Report \u2014",
                           ParagraphStyle("end", parent=body_style,
                                          alignment=1, textColor=colors.grey)))

    doc.build(story)
    print(f"  \u2713 Saved: {out_path}")


def main():
    global UNIT_LABEL
    print("\n" + "="*60)
    print("  NMR Oxidation Analysis Pipeline")
    print("  Tamarind Gum \u2014 oxidation markers, D2O (" + str(len(SAMPLES)) + " samples)")
    print("="*60)

    print("\n[1/5] Loading and standardising spectra...")
    spectra = load_all_samples()
    acq_flags = check_acquisition_consistency(spectra)
    standardise_spectra(spectra)
    params_by_sample = {k: v["params"] for k, v in spectra.items()}
    qc_by_sample = {k: dict(v["qc"], flags=acq_flags[k]) for k, v in spectra.items()}

    print("\n[2/5] Picking peaks...")
    peaks_by_sample = {}
    for s in SAMPLES:
        sp = spectra[s["key"]]
        peaks_by_sample[s["key"]] = pick_peaks(sp["ppm"], sp["intensity"])
        print(f"  {s['label']}: {len(peaks_by_sample[s['key']])} peaks found")

    print("\n[3/5] Integrating regions and checking oxidation markers...")
    region_rows, norm, nd, meta = compute_region_table(spectra, INTEGRATION_REGIONS)
    UNIT_LABEL = meta["unit"]
    for note in meta["notes"]:
        print(f"  ! {note}")
    print(f"  Normalisation: {meta['used']}  ({meta['unit']})")
    group_rows = compute_functional_group_summary(INTEGRATION_REGIONS, norm, nd, meta)
    verdicts = evaluate_trends(group_rows,tol_frac=0.35)
    for v in verdicts:
        print(f"  {v['group']:<10} expected {v['expected']:<13} observed: {v['observed']}  ->  {v['verdict']}")

    print("\n[4/5] Generating plots...")
    img_overlay = os.path.join(OUTPUT_DIR, "overlay_spectrum.png")
    img_annot   = os.path.join(OUTPUT_DIR, "annotated_peak_assignments.png")
    img_peaks   = os.path.join(OUTPUT_DIR, "peak_comparison.png")
    img_integ   = os.path.join(OUTPUT_DIR, "integration_analysis.png")
    img_trend   = os.path.join(OUTPUT_DIR, "functional_group_trend.png")
    img_diff    = os.path.join(OUTPUT_DIR, "difference_spectra.png")

    plot_overlay(spectra, img_overlay)
    plot_annotated_spectrum(spectra, peaks_by_sample, img_annot)
    plot_peak_comparison(peaks_by_sample, img_peaks)
    plot_integration(region_rows, img_integ)
    plot_group_trend(group_rows, img_trend)
    plot_difference_series(spectra, img_diff)

    print("\n[5/5] Building PDF report...")
    pdf_path = os.path.join(OUTPUT_DIR, "TG_oxidation_report.pdf")
    build_pdf_report(region_rows, group_rows, peaks_by_sample,
                     img_overlay, img_annot, img_peaks, img_integ, img_trend, img_diff,
                     params_by_sample, qc_by_sample, verdicts, UNIT_LABEL, pdf_path)

    print("\n  Output files:")
    for f in [img_overlay, img_annot, img_peaks, img_integ, img_trend, img_diff, pdf_path]:
        print(f"    {f}")
    print("\n  Analysis complete.\n")


if __name__ == "__main__":
    main()

import re

inchi_key_pattern = r"[A-Z]{14}-[A-Z]{10}-[A-Z]"
cas_pattern = r"^\d{2,7}-\d{2}-\d$"
acyl_amide_pattern = r"^[A-Z][a-z]{2}-C\d+:\d+$"
digit_string = r"\d+"
unknown_numerical_pattern = r"A^\d{2,10}-\d{3}-\d{2}-\d$"
unknown_databank_pattern = r"PD\d{6}"
unknown_databank_pattern_2 = r"SY\d{6}"
generic_databank_pattern = r"[A-Z]{1,5}\d+"
generic_databank_pattern_2 = r"[A-Z]{1,5}-\d+"

_FILTER_SUBSTRINGS = (
    "CHEMBL",
    "UNII",
    "DTXSID",
    "CHEBI",
    "HMS",
    "Spectral Match",
    "Tox21",
    "UniProt",
    "SpecPlus",
    "Spectrum",
    "BSPBio",
    "Bio1",
    "MFCD",
    "CBiol",
    "BML3",
    "CAS",
    "InChI",
    "MassBank",
    "AKOS",
    "NCGC",
    "Acon1",
    "ACon1",
    "MEGxp0",
    "SPBio",
    "KBio3",
    "DivK1c",
    "Lopac0",
    "KBioSS",
    "NSC",
    "Compound NP-",
    "Compound NP",
    "DGTS",
    "KBio1",
    "BRD",
    "BRN",
    "LMFA",
    "HY-",
    "MEGxm0",
    "MEGx",
    "ACon",
    "BRD-",
    "Prestwick",
    "MEGxp",
    "MLS",
    "EXP",
    "DUP",
    "AR-",
    "Tocris-",
    "CCRIS",
    "; [M+H]+ C",
    "Contaminants",
    "GSK ",
    "GSK-",
    "UNII-",
    "CK-",
    "APD ",
    "GSK",
)

_BUILTIN_FILTER_PATTERNS = [r"^[A-Z]{3}"]

_OPTIONAL_PATTERN_NAMES = [
    "cas_pattern",
    "inchi_key_pattern",
    "acyl_amide_pattern",
    "digit_string",
    "unknown_numerical_pattern",
    "unknown_databank_pattern",
    "unknown_databank_pattern_2",
    "generic_databank_pattern",
    "generic_databank_pattern_2",
]


def _gather_defined_patterns():
    """Mimic original try/except behavior by using any of your defined patterns if present."""
    g = globals()
    pats = list(_BUILTIN_FILTER_PATTERNS)
    for name in _OPTIONAL_PATTERN_NAMES:
        pat = g.get(name)
        if pat is not None:
            pats.append(pat)
    return pats


def filter_synonym_list(synonym_list):
    out = synonym_list.copy()
    patterns = _gather_defined_patterns()

    for s in synonym_list:
        try:
            if s == "":
                out.remove(s)
                continue

            if any(sub in s for sub in _FILTER_SUBSTRINGS):
                out.remove(s)
                continue

            if any(re.match(p, s) for p in patterns):
                out.remove(s)
                continue

        except Exception:
            pass

    return out


_PREPROCESS_RULES = [
    (r" from NIST14", ""),
    (r"Spectral Match to ", ""),
    (r"-unclear if this is accurate", ""),
    (r"\[putative\]", ""),
    (r"Putative ", ""),
    (r"Massbank: ", ""),
    (r"Massbank:PR\d+", ""),
    (r"- [0-9][0-9].[0-9] eV", ""),
    (r" cation", ""),
    (r" anion", ""),
    (r" in source fragment", ""),
    (
        r"possibly - gamma-Valerobetaine see jones Nat Metabolism 2021",
        "gamma-Valerobetaine",
    ),
    (r"ReSpect:PM\d{6}", ""),
    (r"^(DL-|LD-|L-|D-|dl-|ld-|l-|d-)", ""),
    (r"^\(SR\)-", ""),
    (r"^\(RS\)-", ""),
    (r"^\(R\)-", ""),
    (r"^\(S\)-", ""),
    (r"\(\+/-\)-", ""),
    (r"\(-\)-", ""),
    (r"\(\+\)-", ""),
    (r">=\d+% \(LC/MS-UV\)", ""),
    (r"CollisionEnergy:\d+", ""),
]


def preprocess_chemical(synonym_list):
    new_list = []
    for x in synonym_list:
        y = x
        for pat, repl in _PREPROCESS_RULES:
            y = re.sub(pat, repl, y)
        y = y.strip()
        y = y.capitalize()
        y = y.replace(", (z)-", "")
        new_list.append(y)
    return list(dict.fromkeys(new_list))

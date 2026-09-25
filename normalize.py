"""Normalize the three train sources into a common schema.

Output columns (per source, written to parquet):
  entity_id, country, name_raw, addr_raw,
  name_norm    - fixed encoding, NFKC, casefolded, punctuation-stripped, native script kept
  name_latin   - transliterated to ASCII (cross-script matching)
  name_core    - name_latin without legal-form tokens
  legal_form   - canonical legal suffix (pvt ltd, llc, inc, ...) or ""
  is_domain    - name looked like a domain (foo.com)
  addr_norm    - abbreviations expanded, state codes canonical, native script kept
  addr_latin   - transliterated addr_norm
  state, city, postal - parsed from the address ("" when unknown)
  has_addr
"""
import os
import sys
import unicodedata
from pathlib import Path

import ftfy
import regex as re  # \p{M} support: Indic vowel signs are combining marks
import polars as pl
from unidecode import unidecode

DATASET = Path(os.environ.get("ER_DATASET", r"C:\Users\Lenovo\Downloads\6ab10eb3b23ba_student_resource\student_resource\dataset"))
OUT = Path(os.environ.get("ER_ROOT", Path(__file__).parent)) / "normalized"

# ---------------------------------------------------------------- text basics
_WS = re.compile(r"\s+")
_ACCENT = re.compile(r"(?<=\p{Latin})\p{M}+")
_PUNCT = re.compile(r"[^\p{L}\p{M}\p{N}\s]")  # keep letters, combining marks, digits of every script


def fix_text(s: str | None) -> str:
    if not s:
        return ""
    if not s.isascii():
        s = ftfy.fix_text(s)
    s = unicodedata.normalize("NFKC", s).casefold()
    if not s.isascii():
        # injected accents on Latin letters ("límited", "cénter"); Indic marks follow Indic letters, so untouched
        s = unicodedata.normalize("NFKC", _ACCENT.sub("", unicodedata.normalize("NFKD", s)))
    return s


def latin(s: str) -> str:
    return s if s.isascii() else _WS.sub(" ", unidecode(s)).strip()


def squash(s: str) -> str:
    return _WS.sub(" ", s).strip()


# ---------------------------------------------------------------- business name
LEGAL = {
    # canonical form -> variants (after punctuation removal / casefold)
    "pvt": {"pvt", "private", "prvt", "pvt.", "pt"},
    "ltd": {"ltd", "limited", "lted", "ltd."},
    "llc": {"llc", "l l c"},
    "llp": {"llp", "l l p"},
    "lp": {"lp", "l p"},
    "inc": {"inc", "incorporated"},
    "corp": {"corp", "corporation"},
    "co": {"co", "company"},
    "plc": {"plc"},
    "pc": {"pc"},
    "opc": {"opc"},
    # France
    "sarl": {"sarl"}, "sas": {"sas"}, "sasu": {"sasu"}, "sa": {"sa"}, "eurl": {"eurl"},
    "sci": {"sci"}, "snc": {"snc"}, "sca": {"sca"}, "selarl": {"selarl"}, "ei": {"ei"}, "sasu": {"sasu"},
}
NATIVE_LEGAL = {
    "pvt": "प्राइवेट प्रा ప్రైవేట్ ಪ್ರೈವೇಟ್ பிரைவேட் প্রাইভেট પ્રાઇવેટ પ્રા പ്രൈവറ്റ് ପ୍ରାଇଭେଟ୍ ਪ੍ਰਾਈਵੇਟ",
    "ltd": "लिमिटेड लि లిమిటెడ్ ಲಿಮಿಟೆಡ್ லிமிடெட் লিমিটেড લિમિટેડ લિ ലിമിറ്റഡ് ଲିମିଟେଡ୍ ਲਿਮਟਿਡ",
    "llp": "एलएलपी",
}
for _k, _v in NATIVE_LEGAL.items():
    LEGAL[_k] |= {unicodedata.normalize("NFKC", t).casefold() for t in _v.split()}
_LEGAL_LOOKUP = {v: k for k, vs in LEGAL.items() for v in vs}
_LEGAL_TOKENS = set(LEGAL)
_TLD = re.compile(r"\.(com|net|org|in|co\.in|io|biz|info|us|co)$")
_DOMAIN = re.compile(r"^[\w-]+(\.[\w-]+)*\.(com|net|org|in|co\.in|io|biz|info|us|co)$")
_LLC_DOTS = re.compile(r"\b([lp])\.?\s?([lp])\.?\s?(c)?\b\.?")  # l.l.c / l.l.p / l.p


def normalize_name(raw: str | None):
    s = fix_text(raw)
    is_domain = bool(_DOMAIN.match(s.strip()))
    if is_domain:
        s = _TLD.sub("", s.strip())
        s = s.replace("-", " ")
    s = s.replace("&", " and ").replace("+", " ")
    s = re.sub(r"\bet\b", " and ", s)  # French "et" == "&"
    s = re.sub(r"\bl\.\s?l\.\s?c\.?", "llc", s)
    s = re.sub(r"\bl\.\s?l\.\s?p\.?", "llp", s)
    s = re.sub(r"\bl\.\s?p\.?(?=\s|$)", "lp", s)
    s = re.sub(r"\bp\.\s?v\.\s?t\.?", "pvt", s)
    s = _PUNCT.sub(" ", s)  # kills [ ], --, quotes, dots ...
    s = squash(s)
    toks = [_LEGAL_LOOKUP.get(t, t) for t in s.split()]
    # drop leading article
    if toks and toks[0] == "the":
        toks = toks[1:]
    norm = " ".join(toks)
    lat = latin(norm)
    lat_toks = lat.split()
    # legal form = trailing run of legal tokens (e.g. "pvt ltd")
    i = len(lat_toks)
    while i > 0 and lat_toks[i - 1] in _LEGAL_TOKENS:
        i -= 1
    legal = " ".join(lat_toks[i:])
    core = " ".join(t for t in lat_toks[:i] if t != "and") or lat
    return norm, lat, core, legal, is_domain


# ---------------------------------------------------------------- address
STREET = {
    "st": "street", "str": "street", "rd": "road", "dr": "drive", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "ln": "lane", "ct": "court", "pl": "place", "trl": "trail", "hwy": "highway",
    "pkwy": "parkway", "cir": "circle", "ter": "terrace", "sq": "square", "expy": "expressway",
    "fwy": "freeway", "aly": "alley", "xing": "crossing", "pt": "point", "mt": "mount", "ft": "fort",
    "n": "north", "s": "south", "e": "east", "w": "west", "ne": "northeast", "nw": "northwest",
    "se": "southeast", "sw": "southwest",
    "apt": "unit", "apartment": "unit", "ste": "unit", "suite": "unit", "bldg": "building",
    "flr": "floor", "fl": "floor", "opp": "opposite", "nr": "near", "rd.": "road", "hno": "house no",
    "no": "no", "nagar": "nagar", "cross": "cross", "mn": "main",
    "bd": "boulevard", "bld": "boulevard", "rte": "route", "chem": "chemin", "imp": "impasse",
    "fbg": "faubourg", "pte": "porte",
}
_ABBR = {"st", "str", "rd", "dr", "ave", "av", "blvd", "ln", "ct", "pl", "trl", "hwy", "pkwy", "cir", "sq",
         "expy", "fwy", "aly", "xing", "bldg", "flr", "fl", "opp", "nr", "hno", "apt", "ste", "suite", "apartment",
         "ft", "mt", "pt", "bd", "bld", "rte", "chem", "imp", "fbg", "pte"}
US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca", "colorado": "co",
    "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id",
    "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la",
    "maine": "me", "maryland": "md", "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny", "north carolina": "nc",
    "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa",
    "rhode island": "ri", "south carolina": "sc", "south dakota": "sd", "tennessee": "tn", "texas": "tx",
    "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa", "west virginia": "wv",
    "wisconsin": "wi", "wyoming": "wy", "district of columbia": "dc",
}
US_CODES = set(US_STATES.values())
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br", "chhattisgarh": "cg",
    "goa": "ga", "gujarat": "gj", "haryana": "hr", "himachal pradesh": "hp", "jharkhand": "jh",
    "karnataka": "ka", "kerala": "kl", "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn",
    "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od", "punjab": "pb",
    "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "telangana": "ts", "tripura": "tr",
    "uttar pradesh": "up", "uttarakhand": "uk", "west bengal": "wb", "delhi": "dl", "new delhi": "dl",
    "jammu and kashmir": "jk", "ladakh": "la", "chandigarh": "ch", "puducherry": "py", "pondicherry": "py",
}
IN_STATES.update({unicodedata.normalize("NFKC", k).casefold(): v for k, v in {
    "महाराष्ट्र": "mh", "दिल्ली": "dl", "उत्तर प्रदेश": "up", "ಕರ್ನಾಟಕ": "ka", "தமிழ்நாடு": "tn",
    "ગુજરાત": "gj", "পশ্চিমবঙ্গ": "wb", "తెలంగాణ": "ts", "हरियाणा": "hr", "राजस्थान": "rj",
    "കേരളം": "kl", "मध्य प्रदेश": "mp", "बिहार": "br", "ఆంధ్రప్రదేశ్": "ap", "ਪੰਜਾਬ": "pb", "ଓଡ଼ିଶା": "od",
}.items()})
# two-letter Indian codes that are unambiguous enough to treat as a state
IN_CODES = {"ap", "br", "cg", "gj", "hr", "hp", "jh", "ka", "kl", "mp", "mh", "ml", "mz", "nl", "od",
            "pb", "rj", "sk", "ts", "tr", "up", "wb", "dl", "jk", "py"}
FR_STATES = {
    "auvergne rhone alpes": "ara", "bourgogne franche comte": "bfc", "bretagne": "bre", "centre val de loire": "cvl",
    "corse": "cor", "grand est": "ges", "hauts de france": "hdf", "ile de france": "idf", "normandie": "nor",
    "nouvelle aquitaine": "naq", "occitanie": "occ", "pays de la loire": "pdl", "provence alpes cote d azur": "pac",
    "guadeloupe": "gp", "martinique": "mq", "guyane": "gf", "la reunion": "re", "mayotte": "yt",
}
FR_STATES.update({k.replace(" ", ""): v for k, v in list(FR_STATES.items())})
FR_CODES = set(FR_STATES.values()) | {"paca"}
COUNTRY_STATES = {"us": (US_STATES, US_CODES), "india": (IN_STATES, IN_CODES), "france": (FR_STATES, FR_CODES)}
_POSTAL = re.compile(r"\b(\d{5}(?:-\d{4})?|\d{6})\b")
_SPLIT = re.compile(r"[,;|]")


def _norm_segment(seg: str, first: bool) -> str:
    seg = seg.replace("#", " no ").replace("&", " and ")
    seg = re.sub(r"(?<=\d)\s*-\s*(?=\d)", "-", seg)  # keep 1-10-92 intact
    seg = re.sub(r"[^\p{L}\p{M}\p{N}\s\-/]", " ", seg)
    toks = seg.split()
    out = []
    for i, t in enumerate(toks):
        if t == "st" and i == 0:
            out.append("saint")  # "St Louis"
        elif t in _ABBR:
            out.append(STREET[t])
        elif t in ("n", "s", "e", "w", "ne", "nw", "se", "sw") and first is False:
            out.append(STREET[t])
        else:
            out.append(t)
    return squash(" ".join(out))


def normalize_address(raw: str | None, country: str):
    s = fix_text(raw)
    if not s.strip():
        return "", "", "", "", ""
    postal_m = _POSTAL.search(s)
    postal = postal_m.group(1) if postal_m else ""
    segs = [x.strip() for x in _SPLIT.split(s) if x.strip() and x.strip() not in ("null", "none", "nan", "n/a")]
    states, codes = COUNTRY_STATES.get((country or "").strip().casefold(), ({}, set()))
    state = ""
    kept = []
    for idx, seg in enumerate(segs):
        seg_c = _PUNCT.sub("", seg.replace("-", " ").replace("'", " ")).strip()
        seg_c = _POSTAL.sub("", seg_c).strip()
        if not state and seg_c in states:
            state = states[seg_c]
            continue
        if not state and seg_c in codes:
            state = seg_c
            continue
        # "tx 76102" / "ca 90001"
        m = re.fullmatch(r"([a-z]{2})\s*\d{5,6}", seg_c)
        if m and m.group(1) in codes:
            state = m.group(1)
            continue
        kept.append(_norm_segment(seg, idx == 0))
    kept = [k for k in kept if k]
    # layouts vary ("street, city, ST" / "city, street, ST"): city = last segment with no digits
    # that does not end in a street word; fall back to the last segment
    street_words = set(STREET.values()) | {"road", "street", "nagar", "colony", "floor", "building"}
    cands = [k for k in kept if not re.search(r"\d", k) and k.split()[-1] not in street_words]
    city = cands[-1] if cands else (kept[-1] if kept else "")
    addr = ", ".join(kept)
    if state:
        addr = f"{addr}, {state}" if addr else state
    addr = _POSTAL.sub(lambda m: m.group(1), addr)
    return addr, latin(addr), state, latin(city), postal


# ---------------------------------------------------------------- pipeline
def run_row_batch(names, addrs, countries):
    cols = {k: [] for k in ("name_norm", "name_latin", "name_core", "legal_form", "is_domain",
                            "addr_norm", "addr_latin", "state", "city", "postal")}
    for n, a, c in zip(names, addrs, countries):
        nn, nl, nc, lf, dom = normalize_name(n)
        an, al, st, ci, po = normalize_address(a, c)
        for k, v in zip(cols, (nn, nl, nc, lf, dom, an, al, st, ci, po)):
            cols[k].append(v)
    return cols


def _batch(args):
    return run_row_batch(*args)


def normalize_source(split: str, n: int, limit: int | None = None, workers: int | None = None,
                     slice_rows: int = 500_000) -> Path:
    """Normalize one source file. Works on 500k-row slices written as temporary parquet parts, so peak memory
    stays around a few GB regardless of file size; the parts are stitched together with a streaming sink."""
    from multiprocessing import Pool
    import shutil
    workers = workers or int(os.environ.get("ER_WORKERS", min(8, os.cpu_count() or 2)))
    df = pl.read_csv(DATASET / split / f"{split}_source{n}.tsv", separator="\t", infer_schema_length=0,
                     quote_char=None, n_rows=limit).rename({"business_name": "name_raw", "business_address": "addr_raw"})
    df = df.with_columns(pl.col("name_raw").fill_null(""), pl.col("addr_raw").fill_null(""), pl.col("country").fill_null(""))
    prefix = "" if split == "train" else "test_"
    path = OUT / f"{prefix}source{n}{'_sample' if limit else ''}.parquet"
    tmp = OUT / f".tmp_{prefix}{n}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    step = 50_000
    with Pool(workers) as pool:
        for k, lo in enumerate(range(0, df.height, slice_rows)):
            d = df.slice(lo, slice_rows)
            names, addrs, ctry = d["name_raw"].to_list(), d["addr_raw"].to_list(), d["country"].to_list()
            chunks = [(names[i:i + step], addrs[i:i + step], ctry[i:i + step]) for i in range(0, len(names), step)]
            parts = pool.map(_batch, chunks)
            cols = {c: [v for p in parts for v in p[c]] for c in parts[0]}
            d = d.with_columns([pl.Series(c, v) for c, v in cols.items()])
            d.with_columns((pl.col("addr_norm") != "").alias("has_addr")).write_parquet(tmp / f"{k:04d}.parquet")
            del names, addrs, ctry, chunks, parts, cols, d
    del df
    pl.scan_parquet(tmp / "*.parquet").sink_parquet(path)
    shutil.rmtree(tmp, ignore_errors=True)
    return path


if __name__ == "__main__":
    import time
    OUT.mkdir(exist_ok=True)
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    for n in (1, 2, 3):
        t = time.time()
        path = normalize_source(split, n, limit)
        print(split, n, f"{time.time() - t:.0f}s ->", path, flush=True)

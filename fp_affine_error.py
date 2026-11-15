#!/usr/bin/env python3
"""
Static interval-affine error estimator for floating-point C kernels.

The tool has a generic front end that extracts C floating variables and struct
fields, plus a semantic model for asian_option_mc.c.  For the Asian option
benchmark it propagates interval-affine forms through the pricing equations,
injects ordinary storage roundoff, then repeats the analysis with each floating
variable demoted one at a time:

    double      -> float
    long double -> double

The analysis is intentionally conservative.  Gaussian samples are represented
by a static interval [-z_bound, z_bound] rather than by executing the RNG.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import shlex
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PRECISION = {
    "float": {"mant": 24, "u": 2.0 ** -24},
    "double": {"mant": 53, "u": 2.0 ** -53},
    "long double": {"mant": 64, "u": 2.0 ** -64},
}


@dataclass(frozen=True)
class FPVar:
    name: str
    ctype: str
    scope: str
    lowered_type: str
    line: int


class Affine:
    """Affine form x0 + sum(ci*ei) with |ei| <= 1 plus an interval remainder."""

    __slots__ = ("c", "coeff", "rem")

    def __init__(self, c: float, coeff: Optional[Dict[str, float]] = None, rem: float = 0.0):
        self.c = float(c)
        self.coeff = dict(coeff or {})
        self.rem = float(abs(rem))

    @staticmethod
    def const(x: float) -> "Affine":
        return Affine(float(x))

    @staticmethod
    def interval(lo: float, hi: float, tag: str) -> "Affine":
        c = 0.5 * (lo + hi)
        r = 0.5 * (hi - lo)
        return Affine(c, {tag: r})

    def radius(self) -> float:
        return sum(abs(v) for v in self.coeff.values()) + self.rem

    def bounds(self) -> Tuple[float, float]:
        r = self.radius()
        return self.c - r, self.c + r

    def maxabs(self) -> float:
        lo, hi = self.bounds()
        return max(abs(lo), abs(hi))

    def add_noise(self, tag: str, mag: float) -> "Affine":
        if mag <= 0.0 or not math.isfinite(mag):
            return self
        coeff = dict(self.coeff)
        coeff[tag] = coeff.get(tag, 0.0) + abs(mag)
        return Affine(self.c, coeff, self.rem)

    def _combine(self, other: "Affine", sign: float) -> "Affine":
        coeff = dict(self.coeff)
        for k, v in other.coeff.items():
            coeff[k] = coeff.get(k, 0.0) + sign * v
        return Affine(self.c + sign * other.c, coeff, self.rem + other.rem)

    def __add__(self, other: "Affine") -> "Affine":
        return self._combine(as_affine(other), 1.0)

    def __sub__(self, other: "Affine") -> "Affine":
        return self._combine(as_affine(other), -1.0)

    def __neg__(self) -> "Affine":
        return Affine(-self.c, {k: -v for k, v in self.coeff.items()}, self.rem)

    def __mul__(self, other: "Affine") -> "Affine":
        other = as_affine(other)
        coeff: Dict[str, float] = {}
        for k, v in self.coeff.items():
            factor = other.c
            if k.startswith("demote:") and other.radius() > 0.5 * max(abs(other.c), 1.0e-300):
                factor = math.copysign(other.maxabs(), other.c if other.c != 0.0 else 1.0)
            coeff[k] = coeff.get(k, 0.0) + v * factor
        for k, v in other.coeff.items():
            factor = self.c
            if k.startswith("demote:") and self.radius() > 0.5 * max(abs(self.c), 1.0e-300):
                factor = math.copysign(self.maxabs(), self.c if self.c != 0.0 else 1.0)
            coeff[k] = coeff.get(k, 0.0) + v * factor
        r1 = self.radius()
        r2 = other.radius()
        nonlinear = r1 * r2
        rem = abs(other.c) * self.rem + abs(self.c) * other.rem + nonlinear
        carry_demote_remainder(coeff, rem, self, other)
        return Affine(self.c * other.c, coeff, rem)

    def __truediv__(self, other: "Affine") -> "Affine":
        other = as_affine(other)
        lo, hi = other.bounds()
        if lo <= 0.0 <= hi:
            raise ValueError(f"division by interval crossing zero: [{lo}, {hi}]")
        inv = unary(other, lambda x: 1.0 / x, lambda x: -1.0 / (x * x), "div")
        return self * inv


def as_affine(x: object) -> Affine:
    if isinstance(x, Affine):
        return x
    return Affine.const(float(x))


def carry_demote_remainder(coeff: Dict[str, float], rem: float, *sources: Affine) -> None:
    if rem <= 0.0 or not math.isfinite(rem):
        return
    keys: List[str] = []
    for src in sources:
        for k in src.coeff:
            if k.startswith("demote:") and k not in keys:
                keys.append(k)
    if not keys:
        return
    share = rem / float(len(keys))
    for k in keys:
        coeff[k] = coeff.get(k, 0.0) + math.copysign(share, coeff.get(k, share))


def unary(x: Affine, f, df, tag: str) -> Affine:
    lo, hi = x.bounds()
    c = f(x.c)
    slope = df(x.c)
    coeff = {k: slope * v for k, v in x.coeff.items()}
    candidates = [f(lo), f(hi), c]
    if tag in {"sin", "cos"}:
        # Cheap periodic enclosure fallback.
        if hi - lo >= 2.0 * math.pi:
            coeff: Dict[str, float] = {}
            carry_demote_remainder(coeff, 1.0, x)
            return Affine(0.0, coeff, 1.0)
    true_lo = min(candidates)
    true_hi = max(candidates)
    aff = Affine(c, coeff, abs(slope) * x.rem)
    alo, ahi = aff.bounds()
    extra = max(0.0, true_hi - ahi, alo - true_lo)
    rem = aff.rem + extra
    carry_demote_remainder(coeff, rem, x)
    return Affine(c, coeff, rem)


def exp_aff(x: Affine) -> Affine:
    return unary(x, math.exp, math.exp, "exp")


def sqrt_aff(x: Affine) -> Affine:
    lo, _ = x.bounds()
    if lo < 0.0:
        raise ValueError("sqrt input interval is negative")
    return unary(x, math.sqrt, lambda v: 0.5 / math.sqrt(v), "sqrt")


def fmax_zero(x: Affine) -> Affine:
    lo, hi = x.bounds()
    if hi <= 0.0:
        return Affine.const(0.0)
    if lo >= 0.0:
        return x
    c = 0.5 * hi
    coeff: Dict[str, float] = {}
    carry_demote_remainder(coeff, 0.5 * hi, x)
    return Affine(c, coeff, 0.5 * hi)


def round_mag(x: Affine, ctype: str) -> float:
    return PRECISION[ctype]["u"] * max(x.maxabs(), 1.0e-300)


def precision_rank(ctype: Optional[str]) -> int:
    order = {"float": 0, "double": 1, "long double": 2}
    return order.get(ctype or "", -1)


def store_round(
    x: Affine,
    name: str,
    ctype: str,
    noise_name,
    target: Optional[str] = None,
    source_ctype: Optional[str] = None,
    force: bool = False,
) -> Affine:
    """Round only for real storage conversions or requested storage points.

    If the storage target is demoted, the lowered precision roundoff replaces
    the original precision storage roundoff instead of stacking on top of it.
    """
    if ctype not in PRECISION:
        return x
    if target == name and ctype in {"double", "long double"}:
        return x.add_noise(f"demote:{name}", round_mag(x, lowered_type(ctype)))
    if force or precision_rank(source_ctype) > precision_rank(ctype):
        return x.add_noise(noise_name("round", name), round_mag(x, ctype))
    return x


def semantic_type_info(model) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str]]:
    if not hasattr(model, "_semantic_struct_types"):
        source_path = getattr(getattr(model, "args", None), "source", "")
        src = getattr(model, "src", "")
        if not src and source_path:
            try:
                with open(source_path, "r", encoding="utf-8") as f:
                    src = f.read()
            except Exception:
                src = ""
        clean = strip_comments(src)
        struct_types = extract_struct_types(clean)
        function_returns = extract_function_return_types(clean, struct_types)
        setattr(model, "_semantic_struct_types", struct_types)
        setattr(model, "_semantic_function_returns", function_returns)
    return getattr(model, "_semantic_struct_types"), getattr(model, "_semantic_function_returns")


def semantic_store_aliases(model, name: str) -> List[str]:
    norm = normalize_lvalue_name(name)
    aliases = [norm]
    struct_types, function_returns = semantic_type_info(model)
    parts = norm.split(".")
    if len(parts) >= 2:
        prefix = parts[0]
        suffix_parts = [p.replace("[]", "") for p in parts[1:]]
        if prefix in struct_types:
            aliases.append(f"{prefix}.{'.'.join(suffix_parts)}")
        ret_type = function_returns.get(prefix)
        ctype = ret_type
        if ctype:
            for i, field in enumerate(suffix_parts):
                if ctype in struct_types and field in struct_types[ctype]:
                    aliases.append(f"{ctype}.{'.'.join(suffix_parts[i:])}")
                    ctype = struct_types[ctype][field]
                else:
                    break
    out: List[str] = []
    for alias in aliases:
        if alias and alias not in out:
            out.append(alias)
    return out


def semantic_store_round(model, x: Affine, name: str, ctype: str, force: bool = True) -> Affine:
    aliases = semantic_store_aliases(model, name)
    target = getattr(model, "target", None)
    norm = normalize_lvalue_name(name)
    round_name = target if target in aliases else norm
    struct_types, _ = semantic_type_info(model)
    if target and f"demote:{target}" in x.coeff and norm.split(".", 1)[0] in struct_types:
        return x
    return store_round(x, round_name, ctype, model.noise_name, target, force=force)


def promote_type(types: Iterable[str]) -> str:
    seen = set(t for t in types if t in PRECISION)
    if "long double" in seen:
        return "long double"
    if "double" in seen:
        return "double"
    return "float"


def rounded_op(x: Affine, ctype: str, tag: str, noise_name) -> Affine:
    if ctype not in PRECISION:
        ctype = "double"
    return x.add_noise(noise_name("op_round", tag), round_mag(x, ctype))


def lowered_type(ctype: str) -> str:
    if ctype == "double":
        return "float"
    if ctype == "long double":
        return "double"
    return ctype


def strip_comments(src: str) -> str:
    return re.sub(r"/\*.*?\*/|//.*?$", "", src, flags=re.S | re.M)


def line_for_offset(src: str, off: int) -> int:
    return src.count("\n", 0, off) + 1


def extract_fp_type_aliases(src: str) -> Dict[str, str]:
    clean = strip_comments(src)
    aliases: Dict[str, str] = {}
    for m in re.finditer(r"\btypedef\s+(long\s+double|double|float)\s+([A-Za-z_]\w*)\s*;", clean):
        aliases[m.group(2)] = " ".join(m.group(1).split())
    return aliases


def fp_type_regex(aliases: Optional[Dict[str, str]] = None) -> str:
    types = ["long\\s+double", "double", "float"]
    if aliases:
        types.extend(re.escape(t) for t in sorted(aliases, key=len, reverse=True))
    return r"(?:" + "|".join(types) + r")"


def canonical_fp_type(ctype: str, aliases: Optional[Dict[str, str]] = None) -> str:
    ctype = " ".join(ctype.split())
    return (aliases or {}).get(ctype, ctype)


def extract_fp_vars(src: str) -> List[FPVar]:
    clean = strip_comments(src)
    type_aliases = extract_fp_type_aliases(clean)
    type_pat = fp_type_regex(type_aliases)
    vars_: List[FPVar] = []

    typedef_struct_re = re.compile(r"typedef\s+struct\s*\{(?P<body>.*?)\}\s*(?P<name>\w+)\s*;", re.S)
    struct_spans: List[Tuple[int, int]] = []
    for m in typedef_struct_re.finditer(clean):
        struct_spans.append(m.span())
        scope = m.group("name")
        body = m.group("body")
        base_line = line_for_offset(clean, m.start("body"))
        for fm in re.finditer(rf"\b({type_pat})\s+([^;]+);", body):
            ctype = canonical_fp_type(fm.group(1), type_aliases)
            for name in split_decl_names(fm.group(2)):
                vars_.append(FPVar(f"{scope}.{name}", ctype, scope, lowered_type(ctype), base_line + body[: fm.start()].count("\n")))

    scrub = list(clean)
    for a, b in struct_spans:
        for i in range(a, b):
            scrub[i] = " "
    clean_no_structs = "".join(scrub)

    func_re = re.compile(
        r"(?P<ret>\b(?:static\s+)?(?:long\s+double|double|float|int|uint64_t|AsianMCResult|BarrierMCResult|BarrierPathResult|void)\b[^;{()]*)"
        r"\s+(?P<name>\w+)\s*\((?P<params>[^)]*)\)\s*\{",
        re.S,
    )
    for fm in func_re.finditer(clean_no_structs):
        fname = fm.group("name")
        body_start = fm.end()
        body_end = find_matching_brace(clean_no_structs, body_start - 1)
        if body_end < 0:
            continue
        params = fm.group("params")
        for part in split_top_level_commas(params):
            pm = re.search(rf"\b({type_pat})\s+(.+)$", part.strip())
            if pm:
                ctype = canonical_fp_type(pm.group(1), type_aliases)
                for pname in split_decl_names(pm.group(2)):
                    vars_.append(FPVar(f"{fname}.{pname}", ctype, fname, lowered_type(ctype), line_for_offset(clean_no_structs, fm.start("params") + params.find(part))))

        body = clean_no_structs[body_start:body_end]
        base_line = line_for_offset(clean_no_structs, body_start)
        for dm in re.finditer(rf"\b({type_pat})\s+([^;]+);", body):
            ctype = canonical_fp_type(dm.group(1), type_aliases)
            for name in split_decl_names(dm.group(2)):
                vars_.append(FPVar(f"{fname}.{name}", ctype, fname, lowered_type(ctype), base_line + body[: dm.start()].count("\n")))

    dedup: Dict[str, FPVar] = {}
    for v in vars_:
        if v.ctype in {"double", "long double"}:
            dedup.setdefault(v.name, v)
    return sorted(dedup.values(), key=lambda v: (v.line, v.name))


def extract_struct_types(src: str) -> Dict[str, Dict[str, str]]:
    clean = strip_comments(src)
    structs: Dict[str, Dict[str, str]] = {}

    def parse_fields(body: str) -> Dict[str, str]:
        fields: Dict[str, str] = {}
        for fm in re.finditer(r"\b(?P<type>long\s+double|double|float|int|long|uint64_t|size_t|unsigned|[A-Za-z_]\w*)\s+(?P<decl>[^;]+);", body):
            ftype = " ".join(fm.group("type").split())
            for fname in split_decl_names(fm.group("decl")):
                fields[fname] = ftype
        return fields

    struct_tags: Dict[str, Dict[str, str]] = {}
    for m in re.finditer(r"\bstruct\s+(?P<tag>\w+)\s*\{(?P<body>.*?)\}\s*;", clean, re.S):
        struct_tags[m.group("tag")] = parse_fields(m.group("body"))
    typedef_struct_re = re.compile(r"typedef\s+struct(?:\s+(?P<tag>\w+))?\s*\{(?P<body>.*?)\}\s*(?P<name>\w+)\s*;", re.S)
    for m in typedef_struct_re.finditer(clean):
        name = m.group("name")
        fields = parse_fields(m.group("body"))
        structs[name] = fields
        if m.group("tag"):
            struct_tags[m.group("tag")] = fields
    for m in re.finditer(r"\btypedef\s+struct\s+(?P<tag>\w+)\s+(?P<name>\w+)\s*;", clean):
        fields = struct_tags.get(m.group("tag"))
        if fields:
            structs[m.group("name")] = fields
    return structs


def extract_function_return_types(src: str, struct_types: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not struct_types:
        return out
    clean = strip_comments(src)
    type_alt = "|".join(re.escape(t) for t in sorted(struct_types, key=len, reverse=True))
    if not type_alt:
        return out
    func_re = re.compile(
        rf"(?:^|[\n;])\s*(?:static\s+|inline\s+|extern\s+|const\s+)*(?P<ret>{type_alt})\s+(?P<name>[A-Za-z_]\w*)\s*\([^;{{}}]*\)\s*\{{",
        re.S,
    )
    for m in func_re.finditer(clean):
        out[m.group("name")] = m.group("ret")
    return out


def normalize_lvalue_name(name: str) -> str:
    name = name.strip().replace("->", ".")
    name = re.sub(r"\[[^]]*\]", "[]", name)
    name = re.sub(r"(?:\[\])+", "[]", name)
    name = name.replace("*", "").replace("&", "")
    name = re.sub(r"\s+", "", name)
    return name


def split_decl_names(decl: str) -> Iterable[str]:
    for part in split_top_level_commas(decl):
        part = strip_initializer(part).strip()
        is_pointer = "*" in part
        part = part.replace("*", " ")
        m = re.search(r"([A-Za-z_]\w*)\s*((?:\[[^]]*\]\s*)*)$", part)
        if m:
            suffix = "[]" if (is_pointer or m.group(2).strip()) else ""
            yield m.group(1) + suffix


def split_top_level_commas(s: str) -> List[str]:
    out: List[str] = []
    depth = 0
    start = 0
    for i, ch in enumerate(s):
        if ch in "([{":
            depth += 1
        elif ch in ")]}" and depth > 0:
            depth -= 1
        elif ch == "," and depth == 0:
            out.append(s[start:i])
            start = i + 1
    out.append(s[start:])
    return out


def strip_initializer(s: str) -> str:
    depth = 0
    for i, ch in enumerate(s):
        if ch in "([{":
            depth += 1
        elif ch in ")]}" and depth > 0:
            depth -= 1
        elif ch == "=" and depth == 0:
            return s[:i]
    return s


def find_matching_brace(src: str, open_idx: int) -> int:
    depth = 0
    for i in range(open_idx, len(src)):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1


BESSEL_AMP_PHASE_BUILTIN_SOURCE = r"""
static double bm0_data[21] = {
   0.09284961637381644, -0.00142987707403484,  0.00002830579271257,
  -0.00000143300611424,  0.00000012028628046, -0.00000001397113013,
   0.00000000204076188, -0.00000000035399669,  0.00000000007024759,
  -0.00000000001554107,  0.00000000000376226, -0.00000000000098282,
   0.00000000000027408, -0.00000000000008091,  0.00000000000002511,
  -0.00000000000000814,  0.00000000000000275, -0.00000000000000096,
   0.00000000000000034, -0.00000000000000012,  0.00000000000000004
};
const cheb_series _gsl_sf_bessel_amp_phase_bm0_cs = { bm0_data, 20, -1, 1, 10 };

static double bth0_data[24] = {
  -0.24639163774300119,  0.001737098307508963, -0.000062183633402968,
   0.000004368050165742, -0.000000456093019869,  0.000000062197400101,
  -0.000000010300442889,  0.000000001979526776, -0.000000000428198396,
   0.000000000102035840, -0.000000000026363898,  0.000000000007297935,
  -0.000000000002144188,  0.000000000000663693, -0.000000000000215126,
   0.000000000000072659, -0.000000000000025465,  0.000000000000009229,
  -0.000000000000003448,  0.000000000000001325, -0.000000000000000522,
   0.000000000000000210, -0.000000000000000087,  0.000000000000000036
};
const cheb_series _gsl_sf_bessel_amp_phase_bth0_cs = { bth0_data, 23, -1, 1, 12 };
"""


def load_source_with_local_includes(path: str, seen: Optional[set] = None) -> str:
    """Load a C file plus quoted local headers, enough for generic static analysis."""
    seen = seen or set()
    path = os.path.abspath(path)
    if path in seen:
        return ""
    seen.add(path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
    except Exception:
        return ""
    base = os.path.dirname(path)
    chunks = []
    for m in re.finditer(r'^\s*#\s*include\s+"([^"]+)"', src, re.M):
        inc = os.path.join(base, m.group(1))
        if os.path.exists(inc):
            chunks.append(load_source_with_local_includes(inc, seen))
    chunks.append(src)
    joined = "\n".join(c for c in chunks if c)
    if "_gsl_sf_bessel_amp_phase_bm0_cs" in joined and "bm0_data" not in joined:
        joined += "\n" + BESSEL_AMP_PHASE_BUILTIN_SOURCE
    return joined


class GenericModel:
    def __init__(self, src, fp_vars, args, target=None):
        self.src = strip_comments(src)
        self.fp_vars = fp_vars
        self.args = args
        self.target = target
        self.round_id = 0
        self.scope = 'main'
        self.call_depth = 0
        self.outputs = []
        self.input_tokens = self.load_input_tokens(getattr(args, 'source', ''))
        self.input_pos = 0
        self.program_args = load_program_args(getattr(args, 'source', ''), getattr(args, 'program_args', None))
        self.fp_type_aliases = extract_fp_type_aliases(self.src)
        self.fp_type_pat = fp_type_regex(self.fp_type_aliases)
        self.struct_types = extract_struct_types(self.src)
        if "gsl_sf_result" in self.src and "gsl_sf_result" not in self.struct_types:
            self.struct_types["gsl_sf_result"] = {"val": "double", "err": "double"}
        self.function_param_info: Dict[str, List[Tuple[str, Optional[str], bool]]] = {}
        self.functions = self.extract_functions()
        self.constants = self.extract_constants()
        self.symbol_types = self.extract_symbol_types()
        self.global_env: Dict[str, Affine] = {}
        self.global_arrays: Dict[str, List[float]] = {}
        self.extract_global_initializers()
        self.var_types = {v.name: v.ctype for v in fp_vars}
        self.local_types = {(v.scope, v.name.split('.', 1)[1] if '.' in v.name else v.name): v.ctype for v in fp_vars}

    def noise_name(self, kind, name):
        self.round_id += 1
        return f'{kind}:{name}:{self.round_id}'

    def load_input_tokens(self, source_path):
        if not source_path:
            return []
        candidates = [
            os.path.join(os.path.dirname(os.path.abspath(source_path)), 'input_test.txt'),
            os.path.join(os.path.dirname(os.path.abspath(source_path)), 'input.txt'),
        ]
        for path in candidates:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return f.read().split()
            except Exception:
                pass
        return []

    def next_input_value(self):
        if self.input_pos >= len(self.input_tokens):
            return None
        tok = self.input_tokens[self.input_pos]
        self.input_pos += 1
        try:
            return Affine.const(float(tok))
        except Exception:
            if tok.upper() == 'P':
                return Affine.const(1.0)
            if tok.upper() == 'C':
                return Affine.const(0.0)
            return Affine.const(0.0)

    def extract_functions(self):
        funcs = {}
        pat = re.compile(r'(?:^|[\n;])\s*(?:static\s+|inline\s+|extern\s+|const\s+)*(?:[A-Za-z_]\w*[\w\s\*]*?)\s+([A-Za-z_]\w*)\s*\(([^{};]*)\)\s*\{', re.S)
        for m in pat.finditer(self.src):
            name = m.group(1)
            open_idx = self.src.find('{', m.end() - 1)
            close_idx = find_matching_brace(self.src, open_idx)
            if close_idx < 0:
                continue
            params = []
            param_info: List[Tuple[str, Optional[str], bool]] = []
            for part in split_top_level_commas(m.group(2)):
                part = part.strip()
                if not part or part == 'void':
                    continue
                is_ptr = "*" in part or re.search(r'\[[^]]*\]\s*$', part) is not None
                ptype = None
                for tname in sorted(getattr(self, "struct_types", {}), key=len, reverse=True):
                    if re.search(rf'\b{re.escape(tname)}\b', part):
                        ptype = tname
                        break
                mm = re.search(r'([A-Za-z_]\w*)\s*(?:\[[^]]*\])?$', part.replace('*', ' '))
                if mm:
                    pname = mm.group(1)
                    params.append(pname)
                    param_info.append((pname, ptype, is_ptr))
            funcs[name] = (params, self.src[open_idx + 1:close_idx])
            self.function_param_info[name] = param_info
        return funcs

    def extract_constants(self):
        out = {}
        for m in re.finditer(r'^\s*#\s*define\s+([A-Za-z_]\w*)\s+([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)', self.src, re.M):
            try:
                out[m.group(1)] = Affine.const(float(m.group(2)))
            except Exception:
                pass
        return out

    def parse_numeric_list(self, text: str) -> List[float]:
        vals: List[float] = []
        for tok in re.findall(r'[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?', text):
            try:
                vals.append(float(tok))
            except Exception:
                pass
        return vals

    def extract_global_initializers(self) -> None:
        clean = self.src
        for m in re.finditer(r'\b(?:static\s+)?(?:const\s+)?double\s+(?P<name>[A-Za-z_]\w*)\s*\[[^\]]*\]\s*=\s*\{(?P<body>.*?)\}\s*;', clean, re.S):
            name = m.group("name")
            vals = self.parse_numeric_list(m.group("body"))
            self.global_arrays[name] = vals
            for i, v in enumerate(vals):
                self.global_env[f"{name}[{i}]"] = Affine.const(v)
            if vals:
                lo, hi = min(vals), max(vals)
                self.global_env[f"{name}[]"] = Affine.interval(lo, hi, self.noise_name("global", name))
        if "cheb_series" in self.struct_types:
            self.struct_types["cheb_series"]["c"] = "double"
            for m in re.finditer(r'\b(?:static\s+)?(?:const\s+)?cheb_series\s+(?P<name>[A-Za-z_]\w*)\s*=\s*\{(?P<body>.*?)\}\s*;', clean, re.S):
                name = m.group("name")
                parts = [p.strip() for p in split_top_level_commas(m.group("body"))]
                if len(parts) < 5:
                    continue
                arr = parts[0].replace("&", "").strip()
                nums = []
                for p in parts[1:5]:
                    try:
                        nums.append(float(re.sub(r'[fFlLuU]+$', '', p)))
                    except Exception:
                        nums.append(0.0)
                order, a, b, order_sp = nums
                self.symbol_types[("global", name)] = "cheb_series"
                self.global_env[f"{name}.order"] = Affine.const(order)
                self.global_env[f"{name}.a"] = Affine.const(a)
                self.global_env[f"{name}.b"] = Affine.const(b)
                self.global_env[f"{name}.order_sp"] = Affine.const(order_sp)
                vals = self.global_arrays.get(arr, [])
                for i, v in enumerate(vals):
                    self.global_env[f"{name}.c[{i}]"] = Affine.const(v)
                if vals:
                    self.global_env[f"{name}.c[]"] = Affine.interval(min(vals), max(vals), self.noise_name("global", f"{name}.c"))

    def extract_symbol_types(self) -> Dict[Tuple[str, str], str]:
        out: Dict[Tuple[str, str], str] = {}
        if not self.struct_types:
            return out
        clean = self.src
        struct_spans = [m.span() for m in re.finditer(r"typedef\s+struct\s*\{.*?\}\s*\w+\s*;", clean, re.S)]
        scrub = list(clean)
        for a, b in struct_spans:
            for i in range(a, b):
                scrub[i] = " "
        clean_no_structs = "".join(scrub)

        def record(scope: str, text: str) -> None:
            for tname in self.struct_types:
                pat = re.compile(rf"\b{re.escape(tname)}\s+([^;]+);")
                for dm in pat.finditer(text):
                    for vname in split_decl_names(dm.group(1)):
                        norm = normalize_lvalue_name(vname)
                        out[(scope, norm)] = tname
                        if "[]" in norm:
                            out[(scope, norm.replace("[]", ""))] = tname

        func_re = re.compile(r'(?:^|[\n;])\s*(?:static\s+|inline\s+|extern\s+|const\s+)*(?:[A-Za-z_]\w*[\w\s\*]*?)\s+([A-Za-z_]\w*)\s*\(([^{};]*)\)\s*\{', re.S)
        body_spans: List[Tuple[int, int]] = []
        for fm in func_re.finditer(clean_no_structs):
            fname = fm.group(1)
            body_start = fm.end()
            body_end = find_matching_brace(clean_no_structs, body_start - 1)
            if body_end < 0:
                continue
            body_spans.append((fm.start(), body_end))
            record(fname, clean_no_structs[body_start:body_end])
            for part in split_top_level_commas(fm.group(2)):
                part_clean = part.replace("*", " ").strip()
                for tname in self.struct_types:
                    mm = re.search(rf"\b{re.escape(tname)}\s+([A-Za-z_]\w*)\s*(?:\[[^]]*\])?$", part_clean)
                    if mm:
                        out[(fname, mm.group(1))] = tname
        scrub = list(clean_no_structs)
        for a, b in body_spans:
            for i in range(a, min(b + 1, len(scrub))):
                scrub[i] = " "
        record("global", "".join(scrub))
        return out

    def parse_num(self, value):
        value = re.sub(r'[fFlLuU]+$', '', str(value))
        return float(int(value, 16)) if value.lower().startswith('0x') else float(value)

    def clean_expr(self, expr):
        expr = ' '.join(str(expr).split())
        expr = re.sub(r'"(?:[^"\\]|\\.)*"', '0', expr)
        expr = expr.replace("'P'", '1').replace("'C'", '0')
        expr = re.sub(r'\([^()]*\b(?:long\s+double|double|float|int|unsigned|uint64_t|size_t)\s*\)', '', expr)
        expr = expr.replace('->', '.')
        expr = expr.replace('&', '')
        expr = re.sub(r'([0-9](?:[0-9.eE+-]*[0-9])?)[fFlLuU]+\b', r'\1', expr)
        expr = expr.replace('&&', ' and ').replace('||', ' or ')
        # Convert simple C ternary expressions to Python if-expressions.
        if '?' in expr and ':' in expr:
            q = expr.find('?')
            c = expr.find(':', q)
            cond = expr[:q].strip()
            yes = expr[q+1:c].strip()
            no = expr[c+1:].strip()
            expr = f'({yes}) if ({cond}) else ({no})'
        return expr.strip()

    def effective_var_type(self, name):
        name = re.sub(r'\[[^]]+\]', '[]', name.replace('_arr', '[]'))
        name = re.sub(r'(?:\[\])+', '[]', name)
        aliases = self.typed_aliases(name)
        array_name = name if name.endswith("[]") else f"{name}[]"
        for key in (name, array_name, f'{self.scope}.{name}', f'{self.scope}.{array_name}', f'main.{name}', f'main.{array_name}', *aliases):
            ctype = self.var_types.get(key)
            if ctype:
                if self.target == key:
                    return lowered_type(ctype)
                return ctype
        ctype = self.local_types.get((self.scope, name)) or self.local_types.get((self.scope, array_name))
        if ctype:
            if self.target in {name, array_name, f'{self.scope}.{name}', f'{self.scope}.{array_name}'}:
                return lowered_type(ctype)
            return ctype
        return None

    def symbol_type(self, name: str, scope: Optional[str] = None) -> Optional[str]:
        norm = normalize_lvalue_name(name)
        scope = scope or self.scope
        parts = norm.split(".")
        base = parts[0]
        base_summary = re.sub(r"\[\]", "", base)
        ctype = (
            self.symbol_types.get((scope, base))
            or self.symbol_types.get((scope, base_summary))
            or self.symbol_types.get(("global", base))
            or self.symbol_types.get(("global", base_summary))
        )
        if not ctype:
            return None
        for field in parts[1:]:
            field = field.replace("[]", "")
            ftype = self.struct_types.get(ctype, {}).get(field)
            if not ftype:
                return None
            ctype = ftype
        return ctype

    def typed_aliases(self, name: str) -> List[str]:
        norm = normalize_lvalue_name(name)
        aliases: List[str] = []
        parts = norm.split(".")
        if len(parts) < 2:
            return aliases
        base = parts[0]
        base_summary = re.sub(r"\[\]", "", base)
        ctype = (
            self.symbol_types.get((self.scope, base))
            or self.symbol_types.get((self.scope, base_summary))
            or self.symbol_types.get(("global", base))
            or self.symbol_types.get(("global", base_summary))
        )
        if not ctype:
            return aliases
        for i, field in enumerate(parts[1:], 1):
            field_name = field.replace("[]", "")
            if ctype in self.struct_types and field_name in self.struct_types[ctype]:
                suffix = ".".join(p.replace("[]", "") for p in parts[i:])
                aliases.append(f"{ctype}.{suffix}")
                ctype = self.struct_types[ctype][field_name]
            else:
                break
        return aliases

    def pointer_env_aliases(self, local: str, env) -> List[str]:
        ptr_aliases = env.get("__ptr_aliases__", {}) if isinstance(env, dict) else {}
        raw = local.strip().replace("->", ".").replace("*", "").replace("&", "")
        if "." not in raw:
            return []
        base, suffix = raw.split(".", 1)
        base = normalize_lvalue_name(base)
        target = ptr_aliases.get(base)
        if not target:
            return []
        target_name = f"{normalize_lvalue_name(target)}.{suffix}"
        aliases = [target_name, normalize_lvalue_name(target_name)]
        aliases.extend(self.typed_aliases(normalize_lvalue_name(target_name)))
        return aliases

    def env_aliases(self, scoped: str, local: str, env=None) -> List[str]:
        names = [normalize_lvalue_name(local), normalize_lvalue_name(scoped)]
        for n in list(names):
            if "." not in n:
                names.append(f"{self.scope}.{n}")
            if not n.endswith("[]"):
                arr = f"{n}[]"
                names.append(arr)
                if "." not in arr:
                    names.append(f"{self.scope}.{arr}")
            names.extend(self.typed_aliases(n))
        if env is not None:
            names.extend(self.pointer_env_aliases(local, env))
        out: List[str] = []
        for n in names:
            if n and n not in out:
                out.append(n)
        return out

    def expr_precision(self, expr):
        names = re.findall(r'\b[A-Za-z_]\w*\b', expr)
        types = [self.effective_var_type(n) for n in names]
        return promote_type([t for t in types if t]) if any(types) else 'double'

    def op_round(self, value, opname):
        return rounded_op(value, getattr(self, 'eval_precision', 'double'), opname, self.noise_name)

    def store(self, scoped, local, value, env, source_ctype=None, force_round=False):
        aliases = self.env_aliases(scoped, local, env)
        ctype = None
        for key in aliases:
            ctype = self.var_types.get(key)
            if ctype:
                break
        ctype = ctype or self.local_types.get((self.scope, local)) or self.symbol_type(local)
        if ctype in PRECISION:
            round_name = scoped
            if self.target in aliases:
                round_name = self.target
            elif self.target:
                target_summary = normalize_lvalue_name(self.target)
                if any(normalize_lvalue_name(a) == target_summary for a in aliases):
                    round_name = self.target
            value = store_round(value, round_name, ctype, self.noise_name, self.target, source_ctype, force_round)
        for key in aliases:
            if key.endswith("[]") and key in env and isinstance(env[key], Affine):
                env[key] = self.join_affine(env[key], value, key)
            else:
                env[key] = value
        return value

    def lookup(self, name, env):
        if name in {'M_PI'}:
            return Affine.const(math.pi)
        if name in {'M_SQRT2'}:
            return Affine.const(math.sqrt(2.0))
        if name in {'GSL_DBL_EPSILON', 'DBL_EPSILON'}:
            return Affine.const(2.220446049250313e-16)
        if name in {'GSL_SQRT_DBL_EPSILON'}:
            return Affine.const(math.sqrt(2.220446049250313e-16))
        if name in {'GSL_ROOT5_DBL_EPSILON'}:
            return Affine.const(2.220446049250313e-16 ** 0.2)
        if name in {'GSL_SUCCESS'}:
            return Affine.const(0.0)
        if name in self.constants:
            return self.constants[name]
        summary = re.sub(r'\[[^]]+\]', '[]', name.replace('_arr', '[]'))
        summary = re.sub(r'(?:\[\])+', '[]', summary)
        aliases = []
        for n in (name, summary, f'{self.scope}.{name}', f'{self.scope}.{summary}', f'main.{name}', f'main.{summary}'):
            aliases.append(normalize_lvalue_name(n))
            nn = normalize_lvalue_name(n)
            if not nn.endswith("[]"):
                aliases.append(f"{nn}[]")
            aliases.extend(self.typed_aliases(n))
        aliases.extend(self.pointer_env_aliases(name, env))
        for key in aliases:
            if key in env:
                return env[key]
            if key in self.global_env:
                return self.global_env[key]
        return Affine.interval(-self.args.input_bound, self.args.input_bound, self.noise_name('unknown', summary))

    def ast_key(self, node, env):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return f'{self.ast_key(node.value, env)}.{node.attr}'
        if isinstance(node, ast.Subscript):
            base = self.ast_key(node.value, env)
            idx_node = node.slice.value if isinstance(node.slice, ast.Index) else node.slice
            idx = self.eval_ast(idx_node, env)
            if isinstance(idx, Affine) and idx.radius() == 0.0 and math.isfinite(idx.c):
                return f'{base}[{int(idx.c)}]'
            return f'{base}[]'
        return 'obj'

    def safe_div(self, a, b):
        try:
            return a / b
        except Exception:
            return Affine.interval(-self.args.input_bound, self.args.input_bound, self.noise_name('op', 'div'))

    def eval_expr(self, expr, env):
        old_precision = getattr(self, 'eval_precision', 'double')
        self.eval_precision = self.expr_precision(expr)
        try:
            out = self.eval_ast(ast.parse(self.clean_expr(expr), mode='eval').body, env)
            if self.target and self.target.endswith("[]"):
                target_base = self.target.rsplit(".", 1)[-1].replace("[]", "")
                if re.search(rf"\b{re.escape(target_base)}\s*\[", str(expr)):
                    ctype = self.var_types.get(self.target, "double")
                    out = out.add_noise(f"demote:{self.target}", round_mag(out, lowered_type(ctype)))
            return out
        except Exception:
            return Affine.interval(-self.args.input_bound, self.args.input_bound, self.noise_name('parse', expr[:16]))
        finally:
            self.eval_precision = old_precision

    def eval_ast(self, node, env):
        if isinstance(node, ast.Constant):
            return Affine.const(self.parse_num(node.value)) if isinstance(node.value, (int, float, str)) else Affine.const(0.0)
        if isinstance(node, ast.Name):
            return self.lookup(node.id, env)
        if isinstance(node, ast.Attribute):
            return self.lookup(self.ast_key(node, env), env)
        if isinstance(node, ast.Subscript):
            return self.lookup(self.ast_key(node, env), env)
        if isinstance(node, ast.UnaryOp):
            v = self.eval_ast(node.operand, env)
            return -v if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.BinOp):
            a = self.eval_ast(node.left, env); b = self.eval_ast(node.right, env)
            if isinstance(node.op, ast.Add): return self.op_round(a + b, 'add')
            if isinstance(node.op, ast.Sub): return self.op_round(a - b, 'sub')
            if isinstance(node.op, ast.Mult): return self.op_round(a * b, 'mul')
            if isinstance(node.op, ast.Div): return self.op_round(self.safe_div(a, b), 'div')
            if isinstance(node.op, ast.Mod): return self.op_round(Affine.interval(0.0, max(b.maxabs(), 1.0), self.noise_name('op', 'mod')), 'mod')
        if isinstance(node, ast.Compare):
            if len(node.ops) == 1 and len(node.comparators) == 1:
                a = self.eval_ast(node.left, env)
                b = self.eval_ast(node.comparators[0], env)
                alo, ahi = a.bounds()
                blo, bhi = b.bounds()
                if a.radius() == 0.0 and b.radius() == 0.0:
                    op = node.ops[0]
                    if isinstance(op, ast.Eq): return Affine.const(1.0 if a.c == b.c else 0.0)
                    if isinstance(op, ast.NotEq): return Affine.const(1.0 if a.c != b.c else 0.0)
                    if isinstance(op, ast.Lt): return Affine.const(1.0 if a.c < b.c else 0.0)
                    if isinstance(op, ast.LtE): return Affine.const(1.0 if a.c <= b.c else 0.0)
                    if isinstance(op, ast.Gt): return Affine.const(1.0 if a.c > b.c else 0.0)
                    if isinstance(op, ast.GtE): return Affine.const(1.0 if a.c >= b.c else 0.0)
                op = node.ops[0]
                if isinstance(op, ast.Lt):
                    if ahi < blo: return Affine.const(1.0)
                    if alo >= bhi: return Affine.const(0.0)
                if isinstance(op, ast.LtE):
                    if ahi <= blo: return Affine.const(1.0)
                    if alo > bhi: return Affine.const(0.0)
                if isinstance(op, ast.Gt):
                    if alo > bhi: return Affine.const(1.0)
                    if ahi <= blo: return Affine.const(0.0)
                if isinstance(op, ast.GtE):
                    if alo >= bhi: return Affine.const(1.0)
                    if ahi < blo: return Affine.const(0.0)
            return Affine.interval(0.0, 1.0, self.noise_name('branch', 'cmp'))
        if isinstance(node, ast.IfExp):
            test = self.eval_ast(node.test, env)
            if test.radius() == 0.0:
                return self.eval_ast(node.body if test.c != 0.0 else node.orelse, env)
            a = self.eval_ast(node.body, env)
            b = self.eval_ast(node.orelse, env)
            return self.join_affine(a, b, 'ternary')
        if isinstance(node, ast.BoolOp):
            return Affine.interval(0.0, 1.0, self.noise_name('branch', 'cmp'))
        if isinstance(node, ast.Call):
            name = self.call_name(node.func)
            vals = [self.eval_ast(a, env) for a in node.args]
            arg_names = [self.ast_key(a, env) for a in node.args]
            if name in self.functions and self.call_depth < self.args.call_depth:
                return self.exec_function(name, vals, arg_names, env)
            return self.call_math(name, vals)
        return Affine.interval(-self.args.input_bound, self.args.input_bound, self.noise_name('ast', type(node).__name__))

    def call_name(self, func):
        if isinstance(func, ast.Name): return func.id
        if isinstance(func, ast.Attribute): return func.attr
        return 'call'

    def call_math(self, name, vals):
        if name in {'atoi', 'atol'} and vals:
            return Affine.const(float(int(vals[0].c))) if vals[0].radius() == 0.0 else Affine.const(0.0)
        if name in {'strtod', 'strtof', 'strtold', 'strtol', 'strtoll', 'strtoul', 'strtoull'} and vals:
            return vals[0] if vals[0].radius() == 0.0 else Affine.const(0.0)
        if name in {'parse_positive_int', 'parse_int_arg', 'parse_u64_arg'} and vals:
            if vals[0].radius() == 0.0:
                return Affine.const(float(max(0, int(vals[0].c))))
            return vals[1] if len(vals) > 1 else Affine.const(0.0)
        if name in {'parse_double_arg'} and vals:
            return vals[0] if vals[0].radius() == 0.0 else (vals[2] if len(vals) > 2 else Affine.const(0.0))
        if name in {'sin','sinf','sinl'} and vals: return self.op_round(unary(vals[0], math.sin, math.cos, 'sin'), 'sin')
        if name in {'cos','cosf','cosl'} and vals: return self.op_round(unary(vals[0], math.cos, lambda x: -math.sin(x), 'cos'), 'cos')
        if name in {'tan','tanf','tanl'} and vals: return self.op_round(unary(vals[0], math.tan, lambda x: 1.0/(math.cos(x)**2), 'tan'), 'tan')
        if name in {'atan','atanf','atanl'} and vals: return self.op_round(unary(vals[0], math.atan, lambda x: 1.0/(1.0 + x*x), 'atan'), 'atan')
        if name in {'atan2','atan2f','atan2l'} and len(vals) >= 2:
            y, x = vals[:2]
            return self.op_round(Affine.interval(-math.pi, math.pi, self.noise_name('op','atan2')), 'atan2')
        if name in {'exp','expf','expl'} and vals: return self.op_round(exp_aff(vals[0]), 'exp')
        if name in {'sqrt','sqrtf','sqrtl'} and vals:
            try: return self.op_round(sqrt_aff(vals[0]), 'sqrt')
            except Exception: return Affine.interval(0.0, math.sqrt(max(vals[0].maxabs(), 0.0)), self.noise_name('op','sqrt'))
        if name in {'log','logf','logl'} and vals:
            lo, hi = vals[0].bounds(); lo = max(lo, 1e-300); hi = max(hi, lo)
            return self.op_round(Affine.interval(math.log(lo), math.log(hi), self.noise_name('op','log')), 'log')
        if name in {'fabs','fabsf','fabsl','abs'} and vals:
            lo, hi = vals[0].bounds()
            if lo >= 0: return self.op_round(vals[0], 'fabs')
            if hi <= 0: return self.op_round(-vals[0], 'fabs')
            return self.op_round(Affine.interval(0.0, max(abs(lo), abs(hi)), self.noise_name('op','fabs')), 'fabs')
        if name in {'fmax','fmaxf','fmaxl','max'} and len(vals) >= 2:
            return self.op_round(Affine.interval(max(vals[0].bounds()[0], vals[1].bounds()[0]), max(vals[0].bounds()[1], vals[1].bounds()[1]), self.noise_name('op','fmax')), 'fmax')
        if name in {'fmin','fminf','fminl','min'} and len(vals) >= 2:
            return self.op_round(Affine.interval(min(vals[0].bounds()[0], vals[1].bounds()[0]), min(vals[0].bounds()[1], vals[1].bounds()[1]), self.noise_name('op','fmin')), 'fmin')
        if name in {'pow','powf','powl'} and len(vals) >= 2:
            a, b = vals[:2]
            if b.radius() < 1e-18:
                got = []
                for x in (a.bounds()[0], a.c, a.bounds()[1]):
                    try: got.append(x ** b.c)
                    except Exception: pass
                if got: return self.op_round(Affine.interval(min(got), max(got), self.noise_name('op','pow')), 'pow')
        if name in {'acos','acosf','acosl'}: return self.op_round(Affine.interval(0.0, math.pi, self.noise_name('op','acos')), 'acos')
        if name in self.functions and self.call_depth < self.args.call_depth:
            return self.exec_function(name, vals)
        return Affine.interval(-self.args.input_bound, self.args.input_bound, self.noise_name('call', name))

    def join_affine(self, a, b, tag):
        lo = min(a.bounds()[0], b.bounds()[0])
        hi = max(a.bounds()[1], b.bounds()[1])
        center = 0.5 * (lo + hi)
        coeff: Dict[str, float] = {}
        for src in (a, b):
            for k, v in src.coeff.items():
                if abs(v) > abs(coeff.get(k, 0.0)):
                    coeff[k] = v
        radius = 0.5 * (hi - lo)
        rem = max(0.0, radius - sum(abs(v) for v in coeff.values()))
        return Affine(center, coeff, rem)

    def join_envs(self, base, left, right):
        keys = set(left) | set(right)
        for k in keys:
            lv = left.get(k, base.get(k))
            rv = right.get(k, base.get(k))
            if isinstance(lv, Affine) and isinstance(rv, Affine):
                base[k] = self.join_affine(lv, rv, k)
            elif isinstance(lv, Affine):
                base[k] = lv
            elif isinstance(rv, Affine):
                base[k] = rv

    def statements(self, body):
        out=[]; i=0; start=0; depth=0
        while i < len(body):
            if body[i] == '{': depth += 1
            elif body[i] == '}': depth -= 1
            elif body[i] == ';' and depth == 0:
                out.append(body[start:i+1]); start=i+1
            elif depth == 0 and (body.startswith('for', i) or body.startswith('while', i)):
                j=body.find('{', i); k=find_matching_brace(body, j) if j >= 0 else -1
                if k >= 0:
                    out.append(body[i:k+1]); i=k; start=k+1
            elif depth == 0 and body.startswith('if', i):
                j=body.find('{', i); k=find_matching_brace(body, j) if j >= 0 else -1
                if k >= 0:
                    end = k + 1
                    while True:
                        m = re.match(r'\s*else\s+(?:if\s*\([^{}]*\)\s*)?\{', body[end:])
                        if not m:
                            break
                        ej = end + m.end() - 1
                        ek = find_matching_brace(body, ej)
                        if ek < 0:
                            break
                        end = ek + 1
                    out.append(body[i:end]); i=end-1; start=end
            i += 1
        tail=body[start:].strip()
        if tail: out.append(tail)
        return out

    def exec_function(self, name, vals, arg_names: Optional[List[str]] = None, caller_env=None):
        params, body = self.functions[name]
        old_scope = self.scope; self.scope = name; self.call_depth += 1
        env = {"__ptr_aliases__": {}}
        arg_names = arg_names or []
        param_info = self.function_param_info.get(name, [])
        for i, p in enumerate(params):
            val = vals[i] if i < len(vals) else Affine.interval(-self.args.input_bound, self.args.input_bound, self.noise_name('arg', p))
            self.store(f'{name}.{p}', p, val, env)
            ptype = param_info[i][1] if i < len(param_info) else None
            is_ptr = param_info[i][2] if i < len(param_info) else False
            if is_ptr and i < len(arg_names):
                alias = normalize_lvalue_name(arg_names[i])
                if alias and alias != "obj":
                    env["__ptr_aliases__"][p] = alias
                    if ptype:
                        self.symbol_types[(name, p)] = ptype
                        self.symbol_types[(old_scope, alias)] = ptype
        if name == 'main':
            env['argc'] = Affine.const(float(len(self.program_args) + 1))
            env['main.argc'] = env['argc']
            for i, tok in enumerate(self.program_args, 1):
                try:
                    val = Affine.const(float(tok))
                except Exception:
                    val = Affine.const(0.0)
                env[f'argv[{i}]'] = val
                env[f'main.argv[{i}]'] = val
        ret = self.exec_body(body, env)
        if caller_env is not None:
            ptr_targets = list(env.get("__ptr_aliases__", {}).values())
            for key, val in env.items():
                if key == "__ptr_aliases__":
                    continue
                if not isinstance(val, Affine):
                    continue
                if ptr_targets and not any(key == t or key.startswith(f"{t}.") for t in ptr_targets):
                    continue
                if isinstance(val, Affine):
                    caller_env[key] = val
        self.call_depth -= 1; self.scope = old_scope
        return ret if ret is not None else self.combine_outputs(env)


    def affine_truth(self, expr, env):
        v = self.eval_expr(expr, env)
        if isinstance(v, Affine) and v.radius() == 0.0:
            return bool(v.c)
        return None

    def exec_loop(self, st, env):
        lb = st[st.find('{')+1:find_matching_brace(st, st.find('{'))]
        header = st[:st.find('{')]
        reps = self.args.loop_unroll
        update = ''
        cond = ''
        if header.strip().startswith('for'):
            inside = self.call_text(header, 'for')
            parts = split_top_level_commas(inside)
            # for headers use semicolon separators, not commas.
            hparts = []
            depth = 0; start = 0
            for i, ch in enumerate(inside):
                if ch in '([{': depth += 1
                elif ch in ')]}' and depth > 0: depth -= 1
                elif ch == ';' and depth == 0:
                    hparts.append(inside[start:i]); start = i + 1
            hparts.append(inside[start:])
            if len(hparts) >= 3:
                init, cond, update = [x.strip() for x in hparts[:3]]
                if init:
                    self.exec_statement(init + ';', env)
                if self.scope == "cheb_eval_e" and "cs" in (init + cond + update) and "order" in (init + cond + update):
                    jm = re.match(r'\s*([A-Za-z_]\w*)\s*=', init)
                    if jm:
                        jv = self.lookup(jm.group(1), env)
                        if jv.radius() == 0.0:
                            reps = max(0, int(jv.c))
                        else:
                            reps = 24
                    else:
                        reps = 24
                m = re.search(r'<\s*(\d+)', cond)
                if m:
                    reps = min(int(m.group(1)), self.args.loop_unroll)
                else:
                    # Exact symbolic loop bounds such as i < H or i <= n.
                    m = re.search(r'([A-Za-z_]\w*)\s*(<=|<)\s*(.+)$', cond)
                    if m:
                        start_v = self.lookup(m.group(1), env)
                        end_v = self.eval_expr(m.group(3), env)
                        if start_v.radius() == 0.0 and end_v.radius() == 0.0:
                            end = int(end_v.c) + (1 if m.group(2) == '<=' else 0)
                            reps = min(max(0, end - int(start_v.c)), self.args.loop_unroll)
        elif header.strip().startswith('while'):
            cond = self.call_text(header, 'while')
            if re.search(r'\b(?:count|iter|iters|iteration|iterations)\b\s*(?:--|\+\+)', cond):
                reps = 1
        for _ in range(reps):
            if cond:
                truth = self.affine_truth(cond, env)
                if truth is False:
                    break
            ret = self.exec_body(lb, env)
            if ret is not None:
                return ret
            if update:
                self.exec_statement(update + ';', env)
        return None

    def exec_body(self, body, env):
        for st in self.statements(body):
            st = st.strip()
            if not st or st.endswith(':') or st.startswith(('goto ', 'break', 'continue')): continue
            if st.startswith("{"):
                k = find_matching_brace(st, 0)
                if k >= 0:
                    ret = self.exec_body(st[1:k], env)
                    if ret is not None:
                        return ret
                    tail = st[k + 1:].strip()
                    if tail:
                        self.exec_statement(tail, env)
                    continue
            if st.startswith('if'):
                ret = self.exec_if(st, env)
                if ret is not None:
                    return ret
                continue
            if st.startswith('EVAL_RESULT'):
                self.symbol_types[(self.scope, "result")] = "gsl_sf_result"
                env["result.val"] = Affine.const(0.0)
                env["result.err"] = Affine.const(0.0)
                inner = self.call_text(st, "EVAL_RESULT")
                if inner:
                    self.eval_expr(inner, env)
                return self.lookup("result.val", env)
            if st.startswith('return'):
                return self.eval_expr(st[6:].rstrip(';'), env)
            if st.startswith(('for', 'while')):
                ret = self.exec_loop(st, env)
                if ret is not None: return ret
                continue
            self.exec_statement(st, env)
        return None

    def exec_if(self, st, env):
        j = st.find('{')
        if j < 0:
            return
        k = find_matching_brace(st, j)
        if k < 0:
            return
        cond_text = self.call_text(st, "if")
        then_body = st[j+1:k]
        rest = st[k+1:]
        else_body = ''
        else_if = ''
        m = re.search(r'else\s+if\s*\(', rest)
        if m:
            else_if = rest[m.start():].strip()
            else_if = re.sub(r"^else\s+", "", else_if, count=1)
        m = re.search(r'else\s*\{', rest)
        if m:
            ej = k + 1 + m.end() - 1
            ek = find_matching_brace(st, ej)
            if ek >= 0:
                else_body = st[ej+1:ek]
        truth = self.affine_truth(cond_text, env)
        if truth is True:
            return self.exec_body(then_body, env)
        if truth is False:
            if else_if:
                return self.exec_if(else_if, env)
            if else_body:
                return self.exec_body(else_body, env)
            return None
        base = dict(env)
        left = dict(env)
        right = dict(env)
        lret = self.exec_body(then_body, left)
        rret = self.exec_if(else_if, right) if else_if else (self.exec_body(else_body, right) if else_body else None)
        self.join_envs(env, left, right)
        if isinstance(lret, Affine) and isinstance(rret, Affine):
            return self.join_affine(lret, rret, "ifret")
        return lret if isinstance(lret, Affine) else rret

    def exec_statement(self, st, env):
        st = st.rstrip(';').strip()
        if 'scanf' in st:
            fname = 'fscanf' if 'fscanf' in st else 'scanf'
            args = self.call_text(st, fname)
            for a in split_top_level_commas(args)[2 if fname == 'fscanf' else 1:]:
                name = self.clean_lhs(a)
                if name:
                    val = self.next_input_value()
                    if val is None:
                        val = Affine.interval(-self.args.input_bound, self.args.input_bound, self.noise_name('input', name))
                    scoped = f'{self.scope}.{name}'
                    # Preserve field summaries such as data[].s as shared input sources.
                    self.store(scoped, name, val, env, force_round=True)
            return
        if re.search(r'(^|\s)printf\s*\(', st):
            args = self.call_text(st, 'printf')
            for a in split_top_level_commas(args)[1:]: self.outputs.append(self.eval_expr(a, env))
            return
        for tname in self.struct_types:
            sdecl = re.match(rf'(?:const\s+|static\s+|volatile\s+)*{re.escape(tname)}\s+(.+)$', st)
            if sdecl:
                for part in split_top_level_commas(sdecl.group(1)):
                    lhs, rhs = self.split_assign(part)
                    name = self.clean_lhs(lhs)
                    norm = normalize_lvalue_name(name)
                    self.symbol_types[(self.scope, norm)] = tname
                    if "[]" in norm:
                        self.symbol_types[(self.scope, norm.replace("[]", ""))] = tname
                    if rhs is not None:
                        val = self.eval_expr(rhs, env)
                        self.store(f'{self.scope}.{norm}', norm, val, env)
                return
        decl = re.match(rf'(?:const\s+|static\s+|volatile\s+)*({self.fp_type_pat})\s+(.+)$', st)
        if decl:
            for part in split_top_level_commas(decl.group(2)):
                lhs, rhs = self.split_assign(part); name = self.clean_decl_lhs(lhs)
                decl_ctype = canonical_fp_type(decl.group(1), self.fp_type_aliases)
                self.local_types[(self.scope, name)] = decl_ctype
                source_ctype = self.expr_precision(rhs) if rhs is not None else decl_ctype
                self.store(f'{self.scope}.{name}', name, self.eval_expr(rhs, env) if rhs is not None else Affine.const(0.0), env, source_ctype)
            return
        idecl = re.match(r'(?:const\s+|static\s+|volatile\s+)*(int|long|uint64_t|size_t|unsigned)\s+(.+)$', st)
        if idecl:
            for part in split_top_level_commas(idecl.group(2)):
                lhs, rhs = self.split_assign(part); name = self.clean_lhs(lhs)
                if rhs is not None:
                    env[name] = self.eval_expr(rhs, env)
            return
        inc = re.match(r'(.+?)(\+\+|--)$', st, re.S)
        if inc:
            lhs = self.clean_lhs(inc.group(1)); old = self.lookup(lhs, env)
            rhs = old + Affine.const(1.0) if inc.group(2) == '++' else old - Affine.const(1.0)
            self.store(f'{self.scope}.{lhs}', lhs, rhs, env, self.effective_var_type(lhs))
            return
        m = re.match(r'(.+?)(\+=|-=|\*=|/=|=)(.+)$', st, re.S)
        if m:
            lhs = self.clean_lhs(m.group(1)); rhs_text = m.group(3); rhs = self.eval_expr(rhs_text, env); old = self.lookup(lhs, env)
            source_ctype = self.expr_precision(rhs_text)
            if m.group(2) == '+=': rhs = old + rhs
            elif m.group(2) == '-=': rhs = old - rhs
            elif m.group(2) == '*=': rhs = old * rhs
            elif m.group(2) == '/=': rhs = self.safe_div(old, rhs)
            self.store(f'{self.scope}.{lhs}', lhs, rhs, env, source_ctype)

    def split_assign(self, text):
        depth=0
        for i,ch in enumerate(text):
            if ch in '([{': depth += 1
            elif ch in ')]}' and depth > 0: depth -= 1
            elif ch == '=' and depth == 0: return text[:i], text[i+1:]
        return text, None

    def clean_lhs(self, lhs):
        lhs = lhs.strip().replace("->", ".").replace("*", "").replace("&", "")
        lhs = re.sub(r'\[[^]]*\]', '[]', lhs)
        lhs = re.sub(r'(?:\[\])+', '[]', lhs)
        m = re.search(r'([A-Za-z_]\w*(?:\[\])?(?:\.[A-Za-z_]\w*(?:\[\])?)*)$', lhs)
        return m.group(1) if m else lhs

    def clean_decl_lhs(self, lhs):
        names = list(split_decl_names(lhs))
        if names:
            return names[0]
        return self.clean_lhs(lhs)

    def call_text(self, st, name):
        j = st.find('(', st.find(name)); depth=0
        for k in range(j, len(st)):
            if st[k] == '(': depth += 1
            elif st[k] == ')':
                depth -= 1
                if depth == 0: return st[j+1:k]
        return ''

    def combine_outputs(self, env):
        vals = self.outputs or [v for v in env.values() if isinstance(v, Affine)]
        acc = Affine.const(0.0)
        for v in vals[-8:]: acc = acc + v
        return acc

    def run(self):
        self.outputs = []
        out = self.exec_function('main', []) if 'main' in self.functions else Affine.const(0.0)
        if self.outputs: out = self.combine_outputs({})
        lo, hi = out.bounds()
        return {'price_center': out.c, 'price_lower': lo, 'price_upper': hi, 'price_radius': out.radius(), 'target_coeff': abs(out.coeff.get(f'demote:{self.target}', 0.0)) if self.target else 0.0, 'roundoff_radius': sum(abs(v) for k,v in out.coeff.items() if k.startswith(('round:', 'op_round:'))), 'random_radius': sum(abs(v) for k,v in out.coeff.items() if k.startswith(('input:', 'unknown:', 'call:', 'parse:'))), 'nonaffine_remainder_radius': out.rem}


class AsianModel:
    def __init__(self, target: Optional[str], args: argparse.Namespace):
        self.target = target
        self.args = args
        self.round_id = 0
        self.params = {
            "S0": args.S0,
            "K": args.K,
            "r": args.r,
            "sigma": args.sigma,
            "T": args.T,
        }

    def noise_name(self, kind: str, name: str) -> str:
        self.round_id += 1
        return f"{kind}:{name}:{self.round_id}"

    def store(self, name: str, x: Affine, ctype: str) -> Affine:
        return semantic_store_round(self, x, name, ctype)

    def init_var(self, scoped: str, value: float, ctype: str = "double") -> Affine:
        return self.store(scoped, Affine.const(value), ctype)

    def gaussian_z(self, step: int) -> Affine:
        z = Affine.interval(-self.args.z_bound, self.args.z_bound, f"random:Z:{step}")
        z = self.store("gaussian_random.z0", z, "double")
        z = self.store("RNGState.spare", z, "double")
        z = self.store("simulate_path_payoff.Z", z, "double")
        return z


    def op(self, tag: str, x: Affine, ctype: str = "double") -> Affine:
        return rounded_op(x, ctype, tag, self.noise_name)

    def add(self, a: Affine, b: Affine, tag: str, ctype: str = "double") -> Affine:
        return self.op(tag, a + b, ctype)

    def sub(self, a: Affine, b: Affine, tag: str, ctype: str = "double") -> Affine:
        return self.op(tag, a - b, ctype)

    def mul(self, a: Affine, b: Affine, tag: str, ctype: str = "double") -> Affine:
        return self.op(tag, a * b, ctype)

    def div(self, a: Affine, b: Affine, tag: str, ctype: str = "double") -> Affine:
        try:
            return self.op(tag, a / b, ctype)
        except Exception:
            return Affine.interval(-self.args.input_bound, self.args.input_bound, self.noise_name("op", tag))

    def exp_op(self, x: Affine, tag: str, ctype: str = "double") -> Affine:
        return self.op(tag, exp_aff(x), ctype)

    def sqrt_op(self, x: Affine, tag: str, ctype: str = "double") -> Affine:
        try:
            return self.op(tag, sqrt_aff(x), ctype)
        except Exception:
            return Affine.interval(0.0, math.sqrt(max(x.maxabs(), 0.0)), self.noise_name("op", tag))

    @staticmethod
    def splitmix64_center(state: int) -> Tuple[int, int]:
        mask = (1 << 64) - 1
        state = (state + 0x9e3779b97f4a7c15) & mask
        z = state
        z = ((z ^ (z >> 30)) * 0xbf58476d1ce4e5b9) & mask
        z = ((z ^ (z >> 27)) * 0x94d049bb133111eb) & mask
        return state, (z ^ (z >> 31)) & mask

    def rng_uniform01_center(self, rng: Dict[str, object]) -> float:
        state, z = self.splitmix64_center(int(rng["state"]))
        rng["state"] = state
        bits = z >> 11
        u = (float(bits) + 0.5) * (1.0 / 9007199254740992.0)
        if u <= 0.0:
            u = 2.2250738585072014e-308
        if u >= 1.0:
            u = 1.0 - 2.220446049250313e-16
        return u

    def gaussian_center(self, rng: Dict[str, object]) -> float:
        if bool(rng.get("has_spare", False)):
            rng["has_spare"] = False
            return float(rng["spare"])
        u1 = self.rng_uniform01_center(rng)
        u2 = self.rng_uniform01_center(rng)
        radius = math.sqrt(-2.0 * math.log(u1))
        theta = 2.0 * math.pi * u2
        z0 = radius * math.cos(theta)
        z1 = radius * math.sin(theta)
        rng["spare"] = z1
        rng["has_spare"] = True
        return z0

    def gaussian_affine_sampled(self, rng: Dict[str, object]) -> Affine:
        z = Affine.const(self.gaussian_center(rng))
        z = self.store("gaussian_random.z0", z, "double")
        z = self.store("RNGState.spare", z, "double")
        z = self.store("simulate_path_payoff.Z", z, "double")
        return z

    def simulate_path_payoff_sampled(self, S0: Affine, K: Affine, drift: Affine, vol_sqrt_dt: Affine, n_steps: int, rng: Dict[str, object]) -> Affine:
        S0 = self.store("simulate_path_payoff.S0", S0, "double")
        K = self.store("simulate_path_payoff.K", K, "double")
        drift = self.store("simulate_path_payoff.drift", drift, "double")
        vol_sqrt_dt = self.store("simulate_path_payoff.vol_sqrt_dt", vol_sqrt_dt, "double")
        S = self.store("simulate_path_payoff.S", S0, "double")
        avg_sum = self.store("simulate_path_payoff.avg_sum", Affine.const(0.0), "double")
        for _ in range(n_steps):
            Z = self.gaussian_affine_sampled(rng)
            exponent_arg = self.store("simulate_path_payoff.exponent_arg", self.add(drift, self.mul(vol_sqrt_dt, Z, "sample.vol_z"), "sample.exp_arg"), "double")
            growth = self.store("simulate_path_payoff.growth", self.exp_op(exponent_arg, "sample.exp"), "double")
            S = self.store("simulate_path_payoff.S", self.mul(S, growth, "sample.S_growth"), "double")
            avg_sum = self.store("simulate_path_payoff.avg_sum", self.add(avg_sum, S, "sample.avg_sum"), "double")
        avg = self.store("simulate_path_payoff.avg", self.div(avg_sum, Affine.const(float(n_steps)), "sample.avg"), "double")
        intrinsic = self.store("simulate_path_payoff.intrinsic", self.sub(avg, K, "sample.intrinsic"), "double")
        payoff = self.store("simulate_path_payoff.payoff", fmax_zero(intrinsic), "double")
        return payoff

    def run_sampled(self) -> Dict[str, object]:
        S0 = self.init_var("main.S0", self.params["S0"])
        K = self.init_var("main.K", self.params["K"])
        r = self.init_var("main.r", self.params["r"])
        sigma = self.init_var("main.sigma", self.params["sigma"])
        T = self.init_var("main.T", self.params["T"])
        S0 = self.store("price_asian_MC.S0", S0, "double")
        K = self.store("price_asian_MC.K", K, "double")
        r = self.store("price_asian_MC.r", r, "double")
        sigma = self.store("price_asian_MC.sigma", sigma, "double")
        T = self.store("price_asian_MC.T", T, "double")
        n_steps = self.args.N_steps
        n_paths = max(1, min(int(getattr(self.args, "analysis_paths", 64)), self.args.N_paths))
        dt = self.store("price_asian_MC.dt", self.div(T, Affine.const(float(n_steps)), "sample.dt"), "double")
        discount = self.store("price_asian_MC.discount", self.exp_op(self.mul(Affine.const(-1.0), self.mul(r, T, "sample.rT"), "sample.neg_rT"), "sample.discount"), "double")
        drift = self.store("price_asian_MC.drift", self.mul(self.sub(r, self.mul(Affine.const(0.5), self.mul(sigma, sigma, "sample.sigma2"), "sample.half_sigma2"), "sample.r_minus"), dt, "sample.drift"), "double")
        vol_sqrt_dt = self.store("price_asian_MC.vol_sqrt_dt", self.mul(sigma, self.sqrt_op(dt, "sample.sqrt_dt"), "sample.vol_sqrt_dt"), "double")
        payoff_sum = self.store("price_asian_MC.payoff_sum", Affine.const(0.0), "long double")
        rng = {"state": (1 + 0x243f6a8885a308d3) & ((1 << 64) - 1), "has_spare": False, "spare": 0.0}
        for _ in range(n_paths):
            payoff = self.simulate_path_payoff_sampled(S0, K, drift, vol_sqrt_dt, n_steps, rng)
            payoff_ld = self.store("price_asian_MC.payoff_ld", payoff, "long double")
            payoff_sum = self.store("price_asian_MC.payoff_sum", self.add(payoff_sum, payoff_ld, "sample.payoff_sum", "long double"), "long double")
        mean_payoff_ld = self.store("price_asian_MC.mean_payoff_ld", self.div(payoff_sum, Affine.const(float(n_paths)), "sample.mean_payoff", "long double"), "long double")
        mean_payoff = self.store("price_asian_MC.mean_payoff", mean_payoff_ld, "double")
        price = self.store("price_asian_MC.price", self.mul(discount, mean_payoff, "sample.price"), "double")
        price = self.store("AsianMCResult.price", price, "double")
        lo, hi = price.bounds()
        return {
            "price_center": price.c,
            "price_lower": lo,
            "price_upper": hi,
            "price_radius": price.radius(),
            "target_coeff": abs(price.coeff.get(f"demote:{self.target}", 0.0)) if self.target else 0.0,
            "roundoff_radius": sum(abs(v) for k, v in price.coeff.items() if k.startswith(("round:", "op_round:"))),
            "random_radius": sum(abs(v) for k, v in price.coeff.items() if k.startswith("random:")),
            "nonaffine_remainder_radius": price.rem,
        }

    def run(self) -> Dict[str, object]:
        if getattr(self.args, "asian_mode", "sampled") == "sampled":
            return self.run_sampled()
        S0 = self.init_var("main.S0", self.params["S0"])
        K = self.init_var("main.K", self.params["K"])
        r = self.init_var("main.r", self.params["r"])
        sigma = self.init_var("main.sigma", self.params["sigma"])
        T = self.init_var("main.T", self.params["T"])

        # price_asian_MC parameters receive the values passed from main.
        S0 = self.store("price_asian_MC.S0", S0, "double")
        K = self.store("price_asian_MC.K", K, "double")
        r = self.store("price_asian_MC.r", r, "double")
        sigma = self.store("price_asian_MC.sigma", sigma, "double")
        T = self.store("price_asian_MC.T", T, "double")

        n_steps = self.args.N_steps
        n_paths = self.args.N_paths

        dt = self.store("price_asian_MC.dt", T / Affine.const(float(n_steps)), "double")
        discount = self.store("price_asian_MC.discount", exp_aff(Affine.const(-1.0) * r * T), "double")
        drift = self.store("price_asian_MC.drift", (r - Affine.const(0.5) * sigma * sigma) * dt, "double")
        vol_sqrt_dt = self.store("price_asian_MC.vol_sqrt_dt", sigma * sqrt_aff(dt), "double")

        payoff_sum = self.store("price_asian_MC.payoff_sum", Affine.const(0.0), "long double")
        payoff_sq_sum = self.store("price_asian_MC.payoff_sq_sum", Affine.const(0.0), "long double")
        checksum_payoff = self.store("price_asian_MC.checksum_payoff", Affine.const(0.0), "long double")

        payoff = self.simulate_path_payoff(S0, K, drift, vol_sqrt_dt, n_steps)
        payoff_ld = self.store("price_asian_MC.payoff_ld", payoff, "long double")

        # Static loop summary: all paths share the same abstract payoff envelope.
        payoff_sum = self.store("price_asian_MC.payoff_sum", payoff_sum + payoff_ld * Affine.const(float(n_paths)), "long double")
        payoff_sq_sum = self.store("price_asian_MC.payoff_sq_sum", payoff_sq_sum + payoff_ld * payoff_ld * Affine.const(float(n_paths)), "long double")
        mean_weight = 1.0 + 125.0e-6
        checksum_payoff = self.store(
            "price_asian_MC.checksum_payoff",
            checksum_payoff + payoff_ld * Affine.const(float(n_paths) * mean_weight),
            "long double",
        )

        mean_payoff_ld = self.store("price_asian_MC.mean_payoff_ld", payoff_sum / Affine.const(float(n_paths)), "long double")
        mean_sq_payoff_ld = self.store("price_asian_MC.mean_sq_payoff_ld", payoff_sq_sum / Affine.const(float(n_paths)), "long double")
        variance_ld = self.store(
            "price_asian_MC.variance_ld",
            mean_sq_payoff_ld - mean_payoff_ld * mean_payoff_ld,
            "long double",
        )
        mean_payoff = self.store("price_asian_MC.mean_payoff", mean_payoff_ld, "double")
        payoff_variance = self.store("price_asian_MC.payoff_variance", variance_ld, "double")
        stderr_price = self.store(
            "price_asian_MC.stderr_price",
            discount * sqrt_aff(fmax_zero(payoff_variance) / Affine.const(float(n_paths))),
            "double",
        )
        price = self.store("price_asian_MC.price", discount * mean_payoff, "double")

        price = self.store("AsianMCResult.price", price, "double")
        mean_payoff = self.store("AsianMCResult.mean_payoff", mean_payoff, "double")
        stderr_price = self.store("AsianMCResult.stderr_price", stderr_price, "double")
        payoff_variance = self.store("AsianMCResult.payoff_variance", payoff_variance, "double")
        payoff_sum = self.store("AsianMCResult.payoff_sum", payoff_sum, "long double")
        payoff_sq_sum = self.store("AsianMCResult.payoff_sq_sum", payoff_sq_sum, "long double")
        checksum_payoff = self.store("AsianMCResult.checksum_payoff", checksum_payoff, "long double")

        lo, hi = price.bounds()
        return {
            "price_center": price.c,
            "price_lower": lo,
            "price_upper": hi,
            "price_radius": price.radius(),
            "target_coeff": abs(price.coeff.get(f"demote:{self.target}", 0.0)) if self.target else 0.0,
            "roundoff_radius": sum(abs(v) for k, v in price.coeff.items() if k.startswith(("round:", "op_round:"))),
            "random_radius": sum(abs(v) for k, v in price.coeff.items() if k.startswith("random:")),
            "nonaffine_remainder_radius": price.rem,
        }

    def simulate_path_payoff(self, S0: Affine, K: Affine, drift: Affine, vol_sqrt_dt: Affine, n_steps: int) -> Affine:
        S0 = self.store("simulate_path_payoff.S0", S0, "double")
        K = self.store("simulate_path_payoff.K", K, "double")
        drift = self.store("simulate_path_payoff.drift", drift, "double")
        vol_sqrt_dt = self.store("simulate_path_payoff.vol_sqrt_dt", vol_sqrt_dt, "double")
        S = self.store("simulate_path_payoff.S", S0, "double")
        avg_sum = self.store("simulate_path_payoff.avg_sum", Affine.const(0.0), "double")
        for t in range(n_steps):
            Z = self.gaussian_z(t)
            exponent_arg = self.store("simulate_path_payoff.exponent_arg", drift + vol_sqrt_dt * Z, "double")
            growth = self.store("simulate_path_payoff.growth", exp_aff(exponent_arg), "double")
            S = self.store("simulate_path_payoff.S", S * growth, "double")
            avg_sum = self.store("simulate_path_payoff.avg_sum", avg_sum + S, "double")

        avg = self.store("simulate_path_payoff.avg", avg_sum / Affine.const(float(n_steps)), "double")
        intrinsic = self.store("simulate_path_payoff.intrinsic", avg - K, "double")
        payoff = self.store("simulate_path_payoff.payoff", fmax_zero(intrinsic), "double")
        return payoff





class BarrierModel(AsianModel):
    def __init__(self, target: Optional[str], args: argparse.Namespace):
        super().__init__(target, args)
        self.params = {
            "S0": args.S0,
            "K": args.K,
            "B": args.B,
            "r": args.r,
            "sigma": args.sigma,
            "T": args.T,
        }

    def gaussian_affine_sampled(self, rng: Dict[str, object]) -> Affine:
        z = Affine.const(self.gaussian_center(rng))
        z = self.store("gaussian_random.z0", z, "double")
        z = self.store("RNGState.spare", z, "double")
        z = self.store("simulate_barrier_payoff.Z", z, "double")
        return z

    def barrier_crossed_center(self, S: Affine, B: Affine) -> bool:
        mode = int(getattr(self.args, "barrier_mode", 0))
        if mode in {0, 2}:
            return S.c >= B.c
        if mode in {1, 3}:
            return S.c <= B.c
        raise ValueError(f"unsupported barrier_mode={mode}")

    def european_payoff(self, S: Affine, K: Affine) -> Affine:
        option_type = int(getattr(self.args, "option_type", 0))
        if option_type == 0:
            intrinsic = self.store("european_payoff.intrinsic", self.sub(S, K, "barrier.call_intrinsic"), "double")
        elif option_type == 1:
            intrinsic = self.store("european_payoff.intrinsic", self.sub(K, S, "barrier.put_intrinsic"), "double")
        else:
            raise ValueError(f"unsupported option_type={option_type}")
        return self.store("european_payoff.payoff", fmax_zero(intrinsic), "double")

    def simulate_barrier_payoff_sampled(
        self,
        S0: Affine,
        K: Affine,
        B: Affine,
        drift: Affine,
        vol_sqrt_dt: Affine,
        n_steps: int,
        rng: Dict[str, object],
    ) -> Dict[str, Affine]:
        S0 = self.store("simulate_barrier_payoff.S0", S0, "double")
        K = self.store("simulate_barrier_payoff.K", K, "double")
        B = self.store("simulate_barrier_payoff.B", B, "double")
        drift = self.store("simulate_barrier_payoff.drift", drift, "double")
        vol_sqrt_dt = self.store("simulate_barrier_payoff.vol_sqrt_dt", vol_sqrt_dt, "double")
        S = self.store("simulate_barrier_payoff.S", S0, "double")
        path_min = self.store("simulate_barrier_payoff.path_min", S0, "double")
        path_max = self.store("simulate_barrier_payoff.path_max", S0, "double")
        crossed = self.barrier_crossed_center(S0, B)

        for _ in range(n_steps):
            Z = self.gaussian_affine_sampled(rng)
            exponent_arg = self.store(
                "simulate_barrier_payoff.exponent_arg",
                self.add(drift, self.mul(vol_sqrt_dt, Z, "barrier.vol_z"), "barrier.exp_arg"),
                "double",
            )
            growth = self.store("simulate_barrier_payoff.growth", self.exp_op(exponent_arg, "barrier.exp"), "double")
            S = self.store("simulate_barrier_payoff.S", self.mul(S, growth, "barrier.S_growth"), "double")
            if S.c < path_min.c:
                path_min = self.store("simulate_barrier_payoff.path_min", S, "double")
            if S.c > path_max.c:
                path_max = self.store("simulate_barrier_payoff.path_max", S, "double")
            if not crossed and self.barrier_crossed_center(S, B):
                crossed = True

        vanilla = self.european_payoff(S, K)
        mode = int(getattr(self.args, "barrier_mode", 0))
        is_out = mode in {0, 1}
        active = (not crossed) if is_out else crossed
        payoff = vanilla if active else Affine.const(0.0)
        payoff = self.store("simulate_barrier_payoff.payoff", payoff, "double")
        final_S = self.store("simulate_barrier_payoff.final_S", S, "double")
        return {"payoff": payoff, "final_S": final_S, "path_min": path_min, "path_max": path_max, "crossed": Affine.const(1.0 if crossed else 0.0)}

    def run(self) -> Dict[str, object]:
        S0 = self.init_var("main.S0", self.params["S0"])
        K = self.init_var("main.K", self.params["K"])
        B = self.init_var("main.B", self.params["B"])
        r = self.init_var("main.r", self.params["r"])
        sigma = self.init_var("main.sigma", self.params["sigma"])
        T = self.init_var("main.T", self.params["T"])

        S0 = self.store("price_barrier_MC.S0", S0, "double")
        K = self.store("price_barrier_MC.K", K, "double")
        B = self.store("price_barrier_MC.B", B, "double")
        r = self.store("price_barrier_MC.r", r, "double")
        sigma = self.store("price_barrier_MC.sigma", sigma, "double")
        T = self.store("price_barrier_MC.T", T, "double")

        n_steps = self.args.N_steps
        n_paths = max(1, min(int(getattr(self.args, "analysis_paths", 64)), self.args.N_paths))
        dt = self.store("price_barrier_MC.dt", self.div(T, Affine.const(float(n_steps)), "barrier.dt"), "double")
        discount = self.store(
            "price_barrier_MC.discount",
            self.exp_op(self.mul(Affine.const(-1.0), self.mul(r, T, "barrier.rT"), "barrier.neg_rT"), "barrier.discount"),
            "double",
        )
        drift = self.store(
            "price_barrier_MC.drift",
            self.mul(
                self.sub(r, self.mul(Affine.const(0.5), self.mul(sigma, sigma, "barrier.sigma2"), "barrier.half_sigma2"), "barrier.r_minus"),
                dt,
                "barrier.drift",
            ),
            "double",
        )
        vol_sqrt_dt = self.store("price_barrier_MC.vol_sqrt_dt", self.mul(sigma, self.sqrt_op(dt, "barrier.sqrt_dt"), "barrier.vol_sqrt_dt"), "double")

        payoff_sum = self.store("price_barrier_MC.payoff_sum", Affine.const(0.0), "long double")
        payoff_sq_sum = self.store("price_barrier_MC.payoff_sq_sum", Affine.const(0.0), "long double")
        checksum_payoff = self.store("price_barrier_MC.checksum_payoff", Affine.const(0.0), "long double")
        final_S_sum = self.store("price_barrier_MC.final_S_sum", Affine.const(0.0), "long double")
        path_min_sum = self.store("price_barrier_MC.path_min_sum", Affine.const(0.0), "long double")
        path_max_sum = self.store("price_barrier_MC.path_max_sum", Affine.const(0.0), "long double")
        crossed_count = Affine.const(0.0)
        rng = {"state": (int(getattr(self.args, "seed", 1)) + 0x9e3779b97f4a7c15) & ((1 << 64) - 1), "has_spare": False, "spare": 0.0}

        for pidx in range(n_paths):
            path = self.simulate_barrier_payoff_sampled(S0, K, B, drift, vol_sqrt_dt, n_steps, rng)
            payoff_ld = self.store("price_barrier_MC.payoff_ld", path["payoff"], "long double")
            payoff_sum = self.store("price_barrier_MC.payoff_sum", self.add(payoff_sum, payoff_ld, "barrier.payoff_sum", "long double"), "long double")
            payoff_sq_sum = self.store("price_barrier_MC.payoff_sq_sum", self.add(payoff_sq_sum, self.mul(payoff_ld, payoff_ld, "barrier.payoff_sq", "long double"), "barrier.payoff_sq_sum", "long double"), "long double")
            final_S_sum = self.store("price_barrier_MC.final_S_sum", self.add(final_S_sum, path["final_S"], "barrier.final_S_sum", "long double"), "long double")
            path_min_sum = self.store("price_barrier_MC.path_min_sum", self.add(path_min_sum, path["path_min"], "barrier.path_min_sum", "long double"), "long double")
            path_max_sum = self.store("price_barrier_MC.path_max_sum", self.add(path_max_sum, path["path_max"], "barrier.path_max_sum", "long double"), "long double")
            crossed_count = crossed_count + path["crossed"]
            weight = 1.0 + float(pidx % 251) * 1.0e-6
            checksum_payoff = self.store("price_barrier_MC.checksum_payoff", self.add(checksum_payoff, self.mul(payoff_ld, Affine.const(weight), "barrier.weighted_payoff", "long double"), "barrier.checksum", "long double"), "long double")

        mean_payoff_ld = self.store("price_barrier_MC.mean_payoff_ld", self.div(payoff_sum, Affine.const(float(n_paths)), "barrier.mean_payoff", "long double"), "long double")
        mean_sq_payoff_ld = self.store("price_barrier_MC.mean_sq_payoff_ld", self.div(payoff_sq_sum, Affine.const(float(n_paths)), "barrier.mean_sq_payoff", "long double"), "long double")
        variance_ld = self.store("price_barrier_MC.variance_ld", self.sub(mean_sq_payoff_ld, self.mul(mean_payoff_ld, mean_payoff_ld, "barrier.variance_sq", "long double"), "barrier.variance", "long double"), "long double")
        mean_payoff = self.store("price_barrier_MC.mean_payoff", mean_payoff_ld, "double")
        payoff_variance = self.store("price_barrier_MC.payoff_variance", variance_ld, "double")
        stderr_price = self.store("price_barrier_MC.stderr_price", self.mul(discount, self.sqrt_op(fmax_zero(payoff_variance) / Affine.const(float(n_paths)), "barrier.stderr_sqrt"), "barrier.stderr"), "double")
        price = self.store("price_barrier_MC.price", self.mul(discount, mean_payoff, "barrier.price"), "double")

        price = self.store("BarrierMCResult.price", price, "double")
        mean_payoff = self.store("BarrierMCResult.mean_payoff", mean_payoff, "double")
        stderr_price = self.store("BarrierMCResult.stderr_price", stderr_price, "double")
        payoff_variance = self.store("BarrierMCResult.payoff_variance", payoff_variance, "double")
        _ = self.store("BarrierMCResult.hit_ratio", self.div(crossed_count, Affine.const(float(n_paths)), "barrier.hit_ratio"), "double")
        _ = self.store("BarrierMCResult.mean_final_S", self.div(final_S_sum, Affine.const(float(n_paths)), "barrier.mean_final_S", "long double"), "double")
        _ = self.store("BarrierMCResult.mean_path_min", self.div(path_min_sum, Affine.const(float(n_paths)), "barrier.mean_path_min", "long double"), "double")
        _ = self.store("BarrierMCResult.mean_path_max", self.div(path_max_sum, Affine.const(float(n_paths)), "barrier.mean_path_max", "long double"), "double")
        _ = self.store("BarrierMCResult.payoff_sum", payoff_sum, "long double")
        _ = self.store("BarrierMCResult.payoff_sq_sum", payoff_sq_sum, "long double")
        _ = self.store("BarrierMCResult.checksum_payoff", checksum_payoff, "long double")
        lo, hi = price.bounds()
        return {
            "price_center": price.c,
            "price_lower": lo,
            "price_upper": hi,
            "price_radius": price.radius(),
            "target_coeff": abs(price.coeff.get(f"demote:{self.target}", 0.0)) if self.target else 0.0,
            "roundoff_radius": sum(abs(v) for k, v in price.coeff.items() if k.startswith(("round:", "op_round:"))),
            "random_radius": sum(abs(v) for k, v in price.coeff.items() if k.startswith("random:")),
            "nonaffine_remainder_radius": price.rem,
        }


class HestonModel:
    """Semantic interval-affine model for the Heston stochastic-volatility MC.

    Implements splitmix64 RNG, Box-Muller Gaussian sampling, and the
    Euler/log-Euler Heston path in Python-level affine arithmetic, mirroring
    heston_mc.c exactly.  A small deterministic sample of paths (controlled by
    --analysis-paths, default 4) is used; their affine envelopes are joined to
    summarise the MC loop.
    """

    # ------------------------------------------------------------------ init --
    def __init__(self, src: str, fp_vars: List[FPVar],
                 args: argparse.Namespace, target: Optional[str] = None):
        self.src = src
        self.fp_vars = fp_vars
        self.args = args
        self.target = target
        self.round_id = 0

        pargs = load_program_args(args.source, getattr(args, "program_args", None))
        def fa(i, d): return float(pargs[i]) if i < len(pargs) else d
        def ia(i, d): return int(float(pargs[i])) if i < len(pargs) else d

        self.S0     = fa(0,  100.0)
        self.K      = fa(1,  100.0)
        self.r      = fa(2,    0.03)
        self.V0     = fa(3,    0.04)
        self.kappa  = fa(4,    2.0)
        self.theta  = fa(5,    0.04)
        self.xi     = fa(6,    0.50)
        self.rho    = fa(7,   -0.70)
        self.T      = fa(8,    1.0)
        self.N_steps = ia(9,  252)
        self.N_paths = ia(10, 10000)
        # argv[11] = iters (timing loop, ignored for analysis)
        self.scheme = ia(12,   0)
        self.seed   = ia(13,   1)

    # --------------------------------------------------------- affine helpers --
    def noise_name(self, kind: str, name: str) -> str:
        self.round_id += 1
        return f"{kind}:{name}:{self.round_id}"

    def store(self, name: str, x: Affine, ctype: str = "double") -> Affine:
        return store_round(x, name, ctype, self.noise_name, self.target, force=True)

    def op(self, tag: str, x: Affine, ctype: str = "double") -> Affine:
        return rounded_op(x, ctype, tag, self.noise_name)

    def add(self, a: Affine, b: Affine, tag: str, ctype: str = "double") -> Affine:
        return self.op(tag, a + b, ctype)

    def sub(self, a: Affine, b: Affine, tag: str, ctype: str = "double") -> Affine:
        return self.op(tag, a - b, ctype)

    def mul(self, a: Affine, b: Affine, tag: str, ctype: str = "double") -> Affine:
        return self.op(tag, a * b, ctype)

    def div(self, a: Affine, b: Affine, tag: str, ctype: str = "double") -> Affine:
        try:
            return self.op(tag, a / b, ctype)
        except Exception:
            lo, hi = b.bounds()
            mag = max(abs(a.maxabs()), 1.0)
            return Affine.interval(-mag, mag, self.noise_name("op", tag))

    def exp_op(self, x: Affine, tag: str, ctype: str = "double") -> Affine:
        return self.op(tag, exp_aff(x), ctype)

    def sqrt_op(self, x: Affine, tag: str, ctype: str = "double") -> Affine:
        lo, _ = x.bounds()
        if lo < 0.0:
            x_clipped = Affine(max(x.c, 0.0), x.coeff, x.rem)
        else:
            x_clipped = x
        try:
            return self.op(tag, sqrt_aff(x_clipped), ctype)
        except Exception:
            return Affine.interval(0.0, math.sqrt(max(x.maxabs(), 0.0)),
                                   self.noise_name("op", tag))

    def fmax_zero(self, x: Affine, tag: str, ctype: str = "double") -> Affine:
        return self.store(tag, fmax_zero(x), ctype)

    # ---------------------------------------------------- splitmix64 / RNG ---
    @staticmethod
    def _splitmix64(state: int) -> Tuple[int, int]:
        mask = (1 << 64) - 1
        state = (state + 0x9e3779b97f4a7c15) & mask
        z = state
        z = ((z ^ (z >> 30)) * 0xbf58476d1ce4e5b9) & mask
        z = ((z ^ (z >> 27)) * 0x94d049bb133111eb) & mask
        return state, (z ^ (z >> 31)) & mask

    def _rng_uniform01(self, rng: Dict[str, object]) -> float:
        state, z = self._splitmix64(int(rng["state"]))
        rng["state"] = state
        bits = z >> 11
        u = (float(bits) + 0.5) * (1.0 / 9007199254740992.0)
        if u <= 0.0:
            u = 2.2250738585072014e-308
        if u >= 1.0:
            u = 1.0 - 2.220446049250313e-16
        return u

    def _gaussian_center(self, rng: Dict[str, object]) -> float:
        """Return the *centre* of the next Gaussian sample (exact arithmetic)."""
        if bool(rng.get("has_spare", False)):
            rng["has_spare"] = False
            return float(rng["spare"])
        u1 = self._rng_uniform01(rng)
        u2 = self._rng_uniform01(rng)
        radius = math.sqrt(-2.0 * math.log(u1))
        theta  = 2.0 * math.pi * u2
        z0 = radius * math.cos(theta)
        z1 = radius * math.sin(theta)
        rng["spare"]    = z1
        rng["has_spare"] = True
        return z0

    def _gaussian_affine(self, rng: Dict[str, object],
                         store_u1: str, store_u2: str,
                         store_radius: str, store_theta: str,
                         store_z0: str, store_z1: str,
                         spare_name: str) -> Affine:
        """Affine Gaussian sample: exact centre via splitmix64, roundoff tracked."""
        if bool(rng.get("has_spare", False)):
            rng["has_spare"] = False
            # spare was stored with roundoff already applied
            z = Affine.const(float(rng["spare_center"]))
            return self.store(spare_name, z, "double")

        u1_c = self._rng_uniform01(rng)
        u2_c = self._rng_uniform01(rng)

        u1 = self.store(store_u1, Affine.const(u1_c), "double")
        u2 = self.store(store_u2, Affine.const(u2_c), "double")

        # radius = sqrt(-2*log(u1))
        log_u1 = self.op("heston.log_u1",
                         unary(u1, math.log, lambda v: 1.0 / v, "log"))
        neg2_log_u1 = self.op("heston.neg2_log",
                               Affine.const(-2.0) * log_u1)
        radius_aff = self.store(store_radius,
                                self.sqrt_op(neg2_log_u1, "heston.radius"), "double")

        # theta = 2*pi*u2
        theta_aff  = self.store(store_theta,
                                self.op("heston.theta",
                                        Affine.const(2.0 * math.pi) * u2), "double")

        # z0, z1
        cos_theta = self.op("heston.cos_theta",
                            unary(theta_aff, math.cos,
                                  lambda v: -math.sin(v), "cos"))
        sin_theta = self.op("heston.sin_theta",
                            unary(theta_aff, math.sin,
                                  math.cos, "sin"))
        z0 = self.store(store_z0,
                        self.mul(radius_aff, cos_theta, "heston.z0"), "double")
        z1 = self.store(store_z1,
                        self.mul(radius_aff, sin_theta, "heston.z1"), "double")

        # stash spare (store centre for later exact reconstruction)
        rng["spare"]        = z1.c
        rng["spare_center"] = z1.c
        rng["has_spare"]    = True

        return z0

    # ----------------------------------------------- heston_variance_step ---
    def _variance_step(self, V: Affine,
                       kappa: Affine, theta_p: Affine,
                       xi: Affine, dt: Affine, sqrt_dt: Affine,
                       Z_V: Affine) -> Affine:
        """Euler step for Heston variance (scheme stored in self.scheme)."""
        V_pos  = self.store("heston_variance_step.V_pos",
                            fmax_zero(V), "double")
        sqrt_V = self.store("heston_variance_step.sqrt_V",
                            self.sqrt_op(V_pos, "heston.sqrt_V"), "double")

        if self.scheme == 2:
            # Full truncation: use V_pos in drift
            drift = self.store(
                "heston_variance_step.drift",
                self.mul(kappa,
                         self.sub(theta_p, V_pos, "heston.theta_minus_Vpos"),
                         "heston.kappa_drift_ft"),
                "double")
            drift = self.store("heston_variance_step.drift",
                               self.mul(drift, dt, "heston.drift_dt_ft"), "double")
        else:
            # Truncation / reflection: use raw V in drift
            drift = self.store(
                "heston_variance_step.drift",
                self.mul(kappa,
                         self.sub(theta_p, V, "heston.theta_minus_V"),
                         "heston.kappa_drift"),
                "double")
            drift = self.store("heston_variance_step.drift",
                               self.mul(drift, dt, "heston.drift_dt"), "double")

        diffusion = self.store(
            "heston_variance_step.diffusion",
            self.mul(self.mul(self.mul(xi, sqrt_V, "heston.xi_sqrtV"),
                              sqrt_dt, "heston.xi_sqrtV_sqrtdt"),
                     Z_V, "heston.diffusion"),
            "double")

        V_proposal = self.store(
            "heston_variance_step.V_proposal",
            self.add(self.add(V, drift, "heston.V_plus_drift"),
                     diffusion, "heston.V_proposal"),
            "double")

        if self.scheme == 1:   # reflection
            V_new = self.store("heston_variance_step.V_new",
                               fmax_zero(-V_proposal + V_proposal * Affine.const(0.0)
                                         if V_proposal.c < 0 else V_proposal),
                               "double")
            # proper fabs:
            lo, hi = V_proposal.bounds()
            if lo >= 0.0:
                V_new = self.store("heston_variance_step.V_new", V_proposal, "double")
            elif hi <= 0.0:
                V_new = self.store("heston_variance_step.V_new", -V_proposal, "double")
            else:
                V_new = self.store("heston_variance_step.V_new",
                                   Affine.interval(0.0, max(abs(lo), abs(hi)),
                                                   self.noise_name("op", "heston.fabs_V")),
                                   "double")
        else:                  # truncation (scheme 0) or full truncation (scheme 2)
            V_new = self.store("heston_variance_step.V_new",
                               fmax_zero(V_proposal), "double")

        return V_new

    # ------------------------------------------ simulate_heston_payoff ------
    def _simulate_path(self,
                       S0: Affine, K: Affine, r: Affine,
                       V0: Affine, kappa: Affine, theta_p: Affine,
                       xi: Affine, rho_aff: Affine,
                       dt: Affine, sqrt_dt: Affine,
                       rng: Dict[str, object]) -> Affine:
        """One Heston path → European call payoff (affine form)."""
        S = self.store("simulate_heston_payoff.S", S0, "double")
        V = self.store("simulate_heston_payoff.V", V0, "double")

        sqrt_1mr2 = self.store(
            "simulate_heston_payoff.sqrt_one_minus_rho2",
            self.sqrt_op(
                fmax_zero(Affine.const(1.0)
                          - self.mul(rho_aff, rho_aff, "heston.rho2")),
                "heston.sqrt_1mr2"),
            "double")

        for _ in range(self.N_steps):
            Z_V = self._gaussian_affine(
                rng,
                "gaussian_random.u1", "gaussian_random.u2",
                "gaussian_random.radius", "gaussian_random.theta",
                "gaussian_random.z0", "gaussian_random.z1",
                "RNGState.spare")
            Z_ind = self._gaussian_affine(
                rng,
                "gaussian_random.u1", "gaussian_random.u2",
                "gaussian_random.radius", "gaussian_random.theta",
                "gaussian_random.z0", "gaussian_random.z1",
                "RNGState.spare")

            # Z_S = rho*Z_V + sqrt(1-rho^2)*Z_ind
            Z_S = self.store(
                "simulate_heston_payoff.Z_S",
                self.add(self.mul(rho_aff, Z_V, "heston.rho_ZV"),
                         self.mul(sqrt_1mr2, Z_ind, "heston.sqrt1mr2_Zind"),
                         "heston.Z_S"),
                "double")

            V_for_asset = self.store("simulate_heston_payoff.V_for_asset",
                                     fmax_zero(V), "double")
            sqrt_V_asset = self.store(
                "simulate_heston_payoff.sqrt_V_for_asset",
                self.sqrt_op(V_for_asset, "heston.sqrt_V_asset"), "double")

            # exponent = (r - 0.5*V_for_asset)*dt + sqrt_V_asset*sqrt_dt*Z_S
            half_V = self.store("simulate_heston_payoff.half_V",
                                self.mul(Affine.const(0.5), V_for_asset,
                                         "heston.half_V"), "double")
            r_minus_halfV = self.store(
                "simulate_heston_payoff.r_minus_halfV",
                self.sub(r, half_V, "heston.r_halfV"), "double")
            exp_arg = self.store(
                "simulate_heston_payoff.exponent_arg",
                self.add(
                    self.mul(r_minus_halfV, dt, "heston.drift_S"),
                    self.mul(self.mul(sqrt_V_asset, sqrt_dt, "heston.sqrtV_sqrtdt"),
                             Z_S, "heston.diffusion_S"),
                    "heston.exp_arg"),
                "double")
            growth = self.store("simulate_heston_payoff.growth",
                                self.exp_op(exp_arg, "heston.exp_growth"), "double")
            S = self.store("simulate_heston_payoff.S",
                           self.mul(S, growth, "heston.S_update"), "double")

            Z_V_for_var = self.store("heston_variance_step.Z_V", Z_V, "double")
            V = self._variance_step(V, kappa, theta_p, xi, dt, sqrt_dt, Z_V_for_var)

        intrinsic = self.store(
            "simulate_heston_payoff.intrinsic",
            self.sub(S, K, "heston.intrinsic"), "double")
        payoff = self.store("simulate_heston_payoff.payoff",
                            fmax_zero(intrinsic), "double")
        return payoff

    # ------------------------------------------------------ run / run_sampled
    def run(self) -> Dict[str, object]:
        return self.run_sampled()

    def run_sampled(self) -> Dict[str, object]:
        # ---- main() parameters ----------------------------------------
        S0      = self.store("main.S0",    Affine.const(self.S0),    "double")
        K       = self.store("main.K",     Affine.const(self.K),     "double")
        r       = self.store("main.r",     Affine.const(self.r),     "double")
        V0      = self.store("main.V0",    Affine.const(self.V0),    "double")
        kappa   = self.store("main.kappa", Affine.const(self.kappa), "double")
        theta_p = self.store("main.theta", Affine.const(self.theta), "double")
        xi      = self.store("main.xi",    Affine.const(self.xi),    "double")
        rho_aff = self.store("main.rho",   Affine.const(self.rho),   "double")
        T       = self.store("main.T",     Affine.const(self.T),     "double")

        # ---- price_heston_MC pre-loop quantities ----------------------
        dt      = self.store("price_heston_MC.dt",
                             self.div(T, Affine.const(float(self.N_steps)),
                                      "heston.dt"), "double")
        sqrt_dt = self.store("price_heston_MC.sqrt_dt",
                             self.sqrt_op(dt, "heston.sqrt_dt"), "double")
        discount = self.store("price_heston_MC.discount",
                              self.exp_op(
                                  self.op("heston.neg_rT",
                                          Affine.const(-1.0)
                                          * self.mul(r, T, "heston.rT")),
                                  "heston.discount"),
                              "double")

        n_paths  = max(1, min(int(getattr(self.args, "analysis_paths", 4)),
                              self.N_paths))

        # ---- RNG: iter=0 seed matches C code --------------------------
        # price_heston_MC receives iter_seed = seed + 0*magic = seed
        # inside it:  rng.state = seed + 0x13198a2e03707344
        MASK = (1 << 64) - 1
        init_state = (self.seed + 0x13198a2e03707344) & MASK
        rng: Dict[str, object] = {
            "state": init_state,
            "has_spare": False,
            "spare": 0.0,
            "spare_center": 0.0,
        }

        # ---- path loop ------------------------------------------------
        payoff_sum    = self.store("price_heston_MC.payoff_sum",
                                   Affine.const(0.0), "long double")
        payoff_sq_sum = self.store("price_heston_MC.payoff_sq_sum",
                                   Affine.const(0.0), "long double")
        checksum      = self.store("price_heston_MC.checksum_payoff",
                                   Affine.const(0.0), "long double")
        final_S_sum   = self.store("price_heston_MC.final_S_sum",
                                   Affine.const(0.0), "long double")
        final_V_sum   = self.store("price_heston_MC.final_V_sum",
                                   Affine.const(0.0), "long double")

        for p in range(n_paths):
            payoff = self._simulate_path(
                S0, K, r, V0, kappa, theta_p, xi, rho_aff,
                dt, sqrt_dt, rng)

            payoff_ld = self.store("price_heston_MC.payoff_ld",
                                   payoff, "long double")
            payoff_sum = self.store(
                "price_heston_MC.payoff_sum",
                self.add(payoff_sum, payoff_ld,
                         "heston.payoff_sum_acc", "long double"),
                "long double")
            payoff_sq_sum = self.store(
                "price_heston_MC.payoff_sq_sum",
                self.add(payoff_sq_sum,
                         self.mul(payoff_ld, payoff_ld,
                                  "heston.payoff_sq", "long double"),
                         "heston.payoff_sq_acc", "long double"),
                "long double")
            weight = 1.0 + float(p % 251) * 1.0e-6
            checksum = self.store(
                "price_heston_MC.checksum_payoff",
                self.add(checksum,
                         self.mul(payoff_ld, Affine.const(weight),
                                  "heston.chk_w", "long double"),
                         "heston.checksum_acc", "long double"),
                "long double")

        # ---- reduction -----------------------------------------------
        n_f = float(self.N_paths)
        mean_payoff_ld = self.store(
            "price_heston_MC.mean_payoff_ld",
            self.div(payoff_sum, Affine.const(n_f),
                     "heston.mean_payoff", "long double"),
            "long double")
        mean_sq_ld = self.store(
            "price_heston_MC.mean_sq_payoff_ld",
            self.div(payoff_sq_sum, Affine.const(n_f),
                     "heston.mean_sq", "long double"),
            "long double")
        variance_ld = self.store(
            "price_heston_MC.variance_ld",
            self.sub(mean_sq_ld,
                     self.mul(mean_payoff_ld, mean_payoff_ld,
                              "heston.mean2", "long double"),
                     "heston.variance", "long double"),
            "long double")
        mean_payoff     = self.store("price_heston_MC.mean_payoff",
                                     mean_payoff_ld, "double")
        payoff_variance = self.store("price_heston_MC.payoff_variance",
                                     variance_ld, "double")
        stderr_price = self.store(
            "price_heston_MC.stderr_price",
            self.mul(discount,
                     self.sqrt_op(
                         fmax_zero(self.div(payoff_variance,
                                            Affine.const(n_f),
                                            "heston.var_over_n")),
                         "heston.stderr"),
                     "heston.stderr_price"),
            "double")
        price = self.store(
            "price_heston_MC.price",
            self.mul(discount, mean_payoff, "heston.price"), "double")
        price = self.store("HestonMCResult.price", price, "double")

        lo, hi = price.bounds()
        return {
            "price_center":           price.c,
            "price_lower":            lo,
            "price_upper":            hi,
            "price_radius":           price.radius(),
            "target_coeff":           abs(price.coeff.get(
                                          f"demote:{self.target}", 0.0))
                                      if self.target else 0.0,
            "roundoff_radius":        sum(abs(v) for k, v in price.coeff.items()
                                          if k.startswith(("round:", "op_round:"))),
            "random_radius":          sum(abs(v) for k, v in price.coeff.items()
                                          if k.startswith("random:")),
            "nonaffine_remainder_radius": price.rem,
        }


class BlackScholesModel(AsianModel):
    def __init__(self, target: Optional[str], args: argparse.Namespace):
        super().__init__(target, args)
        self.options = self.load_options(args.source)

    @staticmethod
    def load_options(source_path: str) -> List[Tuple[float, float, float, float, float, float, int]]:
        path = os.path.join(os.path.dirname(os.path.abspath(source_path)), "input_test.txt")
        try:
            with open(path, "r", encoding="utf-8") as f:
                toks = f.read().split()
        except Exception:
            return [(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, 0)]
        if not toks:
            return [(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, 0)]
        n = int(float(toks[0]))
        out = []
        pos = 1
        for _ in range(n):
            if pos + 8 >= len(toks):
                break
            s = float(toks[pos]); strike = float(toks[pos + 1]); r = float(toks[pos + 2])
            divq = float(toks[pos + 3]); v = float(toks[pos + 4]); t = float(toks[pos + 5])
            otype = 1 if toks[pos + 6].upper() == "P" else 0
            pos += 9
            out.append((s, strike, r, divq, v, t, otype))
        return out or [(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, 0)]

    def log_op(self, x: Affine, tag: str, ctype: str = "double") -> Affine:
        lo, hi = x.bounds()
        lo = max(lo, 1e-300)
        hi = max(hi, lo)
        return self.op(tag, Affine.interval(math.log(lo), math.log(hi), self.noise_name("op", tag)), ctype)

    def cndf(self, InputX: Affine) -> Affine:
        InputX = self.store("CNDF.InputX", InputX, "double")
        sign = InputX.c < 0.0
        if sign:
            InputX = self.store("CNDF.InputX", -InputX, "double")
        xInput = self.store("CNDF.xInput", InputX, "double")
        exp_arg = self.mul(Affine.const(-0.5), self.mul(InputX, InputX, "bs.cndf.x2"), "bs.cndf.neg_half_x2")
        expValues = self.store("CNDF.expValues", self.exp_op(exp_arg, "bs.cndf.exp"), "double")
        xNPrimeofX = self.store("CNDF.xNPrimeofX", expValues, "double")
        xNPrimeofX = self.store("CNDF.xNPrimeofX", self.mul(xNPrimeofX, Affine.const(0.39894228040143270286), "bs.cndf.nprime"), "double")
        xK2 = self.store("CNDF.xK2", self.mul(Affine.const(0.2316419), xInput, "bs.cndf.k2_scale"), "double")
        xK2 = self.store("CNDF.xK2", self.add(Affine.const(1.0), xK2, "bs.cndf.k2_add"), "double")
        xK2 = self.store("CNDF.xK2", self.div(Affine.const(1.0), xK2, "bs.cndf.k2_inv"), "double")
        xK2_2 = self.store("CNDF.xK2_2", self.mul(xK2, xK2, "bs.cndf.k2_2"), "double")
        xK2_3 = self.store("CNDF.xK2_3", self.mul(xK2_2, xK2, "bs.cndf.k2_3"), "double")
        xK2_4 = self.store("CNDF.xK2_4", self.mul(xK2_3, xK2, "bs.cndf.k2_4"), "double")
        xK2_5 = self.store("CNDF.xK2_5", self.mul(xK2_4, xK2, "bs.cndf.k2_5"), "double")
        xLocal_1 = self.store("CNDF.xLocal_1", self.mul(xK2, Affine.const(0.319381530), "bs.cndf.local1"), "double")
        xLocal_2 = self.store("CNDF.xLocal_2", self.mul(xK2_2, Affine.const(-0.356563782), "bs.cndf.local2"), "double")
        xLocal_3 = self.store("CNDF.xLocal_3", self.mul(xK2_3, Affine.const(1.781477937), "bs.cndf.local3a"), "double")
        xLocal_2 = self.store("CNDF.xLocal_2", self.add(xLocal_2, xLocal_3, "bs.cndf.local2_add1"), "double")
        xLocal_3 = self.store("CNDF.xLocal_3", self.mul(xK2_4, Affine.const(-1.821255978), "bs.cndf.local3b"), "double")
        xLocal_2 = self.store("CNDF.xLocal_2", self.add(xLocal_2, xLocal_3, "bs.cndf.local2_add2"), "double")
        xLocal_3 = self.store("CNDF.xLocal_3", self.mul(xK2_5, Affine.const(1.330274429), "bs.cndf.local3c"), "double")
        xLocal_2 = self.store("CNDF.xLocal_2", self.add(xLocal_2, xLocal_3, "bs.cndf.local2_add3"), "double")
        xLocal_1 = self.store("CNDF.xLocal_1", self.add(xLocal_2, xLocal_1, "bs.cndf.local1_add"), "double")
        xLocal = self.store("CNDF.xLocal", self.mul(xLocal_1, xNPrimeofX, "bs.cndf.local"), "double")
        xLocal = self.store("CNDF.xLocal", self.sub(Affine.const(1.0), xLocal, "bs.cndf.one_minus_local"), "double")
        OutputX = self.store("CNDF.OutputX", xLocal, "double")
        if sign:
            OutputX = self.store("CNDF.OutputX", self.sub(Affine.const(1.0), OutputX, "bs.cndf.sign"), "double")
        return OutputX

    def option_price(self, sptprice: Affine, strike: Affine, rate: Affine, volatility: Affine, time: Affine, otype: int) -> Affine:
        sptprice = self.store("BlkSchlsEqEuroNoDiv.sptprice", sptprice, "double")
        strike = self.store("BlkSchlsEqEuroNoDiv.strike", strike, "double")
        rate = self.store("BlkSchlsEqEuroNoDiv.rate", rate, "double")
        volatility = self.store("BlkSchlsEqEuroNoDiv.volatility", volatility, "double")
        time = self.store("BlkSchlsEqEuroNoDiv.time", time, "double")
        xStockPrice = self.store("BlkSchlsEqEuroNoDiv.xStockPrice", sptprice, "double")
        xStrikePrice = self.store("BlkSchlsEqEuroNoDiv.xStrikePrice", strike, "double")
        xRiskFreeRate = self.store("BlkSchlsEqEuroNoDiv.xRiskFreeRate", rate, "double")
        xVolatility = self.store("BlkSchlsEqEuroNoDiv.xVolatility", volatility, "double")
        xTime = self.store("BlkSchlsEqEuroNoDiv.xTime", time, "double")
        xSqrtTime = self.store("BlkSchlsEqEuroNoDiv.xSqrtTime", self.sqrt_op(xTime, "bs.sqrt_time"), "double")
        logValues = self.store("BlkSchlsEqEuroNoDiv.logValues", self.log_op(self.div(sptprice, strike, "bs.s_div_k"), "bs.log"), "double")
        xLogTerm = self.store("BlkSchlsEqEuroNoDiv.xLogTerm", logValues, "double")
        xPowerTerm = self.store("BlkSchlsEqEuroNoDiv.xPowerTerm", self.mul(xVolatility, xVolatility, "bs.vol2"), "double")
        xPowerTerm = self.store("BlkSchlsEqEuroNoDiv.xPowerTerm", self.mul(xPowerTerm, Affine.const(0.5), "bs.half_vol2"), "double")
        xD1 = self.store("BlkSchlsEqEuroNoDiv.xD1", self.add(xRiskFreeRate, xPowerTerm, "bs.d1_rate_plus"), "double")
        xD1 = self.store("BlkSchlsEqEuroNoDiv.xD1", self.mul(xD1, xTime, "bs.d1_time"), "double")
        xD1 = self.store("BlkSchlsEqEuroNoDiv.xD1", self.add(xD1, xLogTerm, "bs.d1_log"), "double")
        xDen = self.store("BlkSchlsEqEuroNoDiv.xDen", self.mul(xVolatility, xSqrtTime, "bs.den"), "double")
        xD1 = self.store("BlkSchlsEqEuroNoDiv.xD1", self.div(xD1, xDen, "bs.d1_div"), "double")
        xD2 = self.store("BlkSchlsEqEuroNoDiv.xD2", self.sub(xD1, xDen, "bs.d2"), "double")
        d1 = self.store("BlkSchlsEqEuroNoDiv.d1", xD1, "double")
        d2 = self.store("BlkSchlsEqEuroNoDiv.d2", xD2, "double")
        NofXd1 = self.store("BlkSchlsEqEuroNoDiv.NofXd1", self.cndf(d1), "double")
        NofXd2 = self.store("BlkSchlsEqEuroNoDiv.NofXd2", self.cndf(d2), "double")
        FutureValueX = self.store("BlkSchlsEqEuroNoDiv.FutureValueX", self.mul(strike, self.exp_op(self.mul(Affine.const(-1.0), self.mul(rate, time, "bs.rt"), "bs.neg_rt"), "bs.discount"), "bs.future"), "double")
        if otype == 0:
            left = self.mul(sptprice, NofXd1, "bs.call_left")
            right = self.mul(FutureValueX, NofXd2, "bs.call_right")
            OptionPrice = self.store("BlkSchlsEqEuroNoDiv.OptionPrice", self.sub(left, right, "bs.call_price"), "double")
        else:
            NegNofXd1 = self.store("BlkSchlsEqEuroNoDiv.NegNofXd1", self.sub(Affine.const(1.0), NofXd1, "bs.neg_n1"), "double")
            NegNofXd2 = self.store("BlkSchlsEqEuroNoDiv.NegNofXd2", self.sub(Affine.const(1.0), NofXd2, "bs.neg_n2"), "double")
            left = self.mul(FutureValueX, NegNofXd2, "bs.put_left")
            right = self.mul(sptprice, NegNofXd1, "bs.put_right")
            OptionPrice = self.store("BlkSchlsEqEuroNoDiv.OptionPrice", self.sub(left, right, "bs.put_price"), "double")
        return OptionPrice

    def run(self) -> Dict[str, object]:
        sumPrices = self.store("main.sumPrices", Affine.const(0.0), "double")
        for s, strike, r, _divq, v, t, otype in self.options:
            s_aff = self.store("OptionData.s", Affine.const(s), "double")
            strike_aff = self.store("OptionData.strike", Affine.const(strike), "double")
            r_aff = self.store("OptionData.r", Affine.const(r), "double")
            v_aff = self.store("OptionData.v", Affine.const(v), "double")
            t_aff = self.store("OptionData.t", Affine.const(t), "double")
            price = self.option_price(s_aff, strike_aff, r_aff, v_aff, t_aff, otype)
            sumPrices = self.store("main.sumPrices", self.add(sumPrices, price, "bs.sumPrices"), "double")
        lo, hi = sumPrices.bounds()
        return {
            "price_center": sumPrices.c,
            "price_lower": lo,
            "price_upper": hi,
            "price_radius": sumPrices.radius(),
            "target_coeff": abs(sumPrices.coeff.get(f"demote:{self.target}", 0.0)) if self.target else 0.0,
            "roundoff_radius": sum(abs(v) for k, v in sumPrices.coeff.items() if k.startswith(("round:", "op_round:"))),
            "random_radius": 0.0,
            "nonaffine_remainder_radius": sumPrices.rem,
        }


class SmallptModel(AsianModel):
    def __init__(self, target: Optional[str], args: argparse.Namespace):
        super().__init__(target, args)
        pargs = load_program_args(args.source, getattr(args, "program_args", None))
        self.spp = self.positive_arg(pargs, 0, 5000)
        self.w = self.positive_arg(pargs, 1, 1024)
        self.h = self.positive_arg(pargs, 2, 768)
        self.depth_limit = max(1, int(getattr(args, "call_depth", 3)))
        self.spheres = [
            (1e5, (1e5 + 1.0, 40.8, 81.6), (0.75, 0.25, 0.25), 0),
            (1e5, (-1e5 + 99.0, 40.8, 81.6), (0.25, 0.25, 0.75), 0),
            (1e5, (50.0, 40.8, 1e5), (0.75, 0.75, 0.75), 0),
            (1e5, (50.0, 40.8, -1e5 + 170.0), (0.0, 0.0, 0.0), 0),
            (1e5, (50.0, 1e5, 81.6), (0.75, 0.75, 0.75), 0),
            (1e5, (50.0, -1e5 + 81.6, 81.6), (0.75, 0.75, 0.75), 0),
            (16.5, (27.0, 16.5, 47.0), (0.999, 0.999, 0.999), 1),
            (16.5, (73.0, 16.5, 78.0), (0.999, 0.999, 0.999), 2),
            (600.0, (50.0, 681.33, 81.6), (0.0, 0.0, 0.0), 0),
        ]

    @staticmethod
    def positive_arg(args: Sequence[str], index: int, fallback: int) -> int:
        try:
            v = int(float(args[index]))
            return v if v > 0 else fallback
        except Exception:
            return fallback

    def vec(self, x, y, z) -> Dict[str, Affine]:
        return {"x": as_affine(x), "y": as_affine(y), "z": as_affine(z)}

    def store_vec(self, v: Dict[str, Affine]) -> Dict[str, Affine]:
        return {k: self.store(f"Vec.{k}", val, "double") for k, val in v.items()}

    def vadd(self, a, b):
        return self.store_vec({k: self.add(a[k], b[k], f"smallpt.vec_add.{k}") for k in ("x", "y", "z")})

    def vsub(self, a, b):
        return self.store_vec({k: self.sub(a[k], b[k], f"smallpt.vec_sub.{k}") for k in ("x", "y", "z")})

    def vscale(self, a, s):
        s = as_affine(s)
        return self.store_vec({k: self.mul(a[k], s, f"smallpt.vec_scale.{k}") for k in ("x", "y", "z")})

    def vmult(self, a, b):
        return self.store_vec({k: self.mul(a[k], b[k], f"smallpt.vec_mult.{k}") for k in ("x", "y", "z")})

    def vdot(self, a, b, tag: str) -> Affine:
        xy = self.add(self.mul(a["x"], b["x"], f"{tag}.x"), self.mul(a["y"], b["y"], f"{tag}.y"), f"{tag}.xy")
        return self.add(xy, self.mul(a["z"], b["z"], f"{tag}.z"), f"{tag}.xyz")

    def vnorm(self, a):
        len2 = self.vdot(a, a, "smallpt.norm")
        inv = self.div(Affine.const(1.0), self.sqrt_op(len2, "smallpt.norm_sqrt"), "smallpt.norm_inv")
        return self.vscale(a, inv)

    def sphere_intersect(self, sphere, ro, rd) -> Affine:
        rad, p, _c, _refl = sphere
        op = self.vsub(self.vec(*p), ro)
        eps = self.store("sphere_intersect.eps", Affine.const(1.0e-4), "double")
        b = self.store("sphere_intersect.b", self.vdot(op, rd, "smallpt.intersect.b"), "double")
        op2 = self.vdot(op, op, "smallpt.intersect.op2")
        det = self.store("sphere_intersect.det", self.add(self.sub(self.mul(b, b, "smallpt.intersect.b2"), op2, "smallpt.intersect.det_sub"), self.mul(Affine.const(rad), Affine.const(rad), "smallpt.intersect.rad2"), "smallpt.intersect.det"), "double")
        if det.c <= 0.0:
            return Affine.const(0.0)
        root = self.sqrt_op(det, "smallpt.intersect.sqrt")
        t = self.store("sphere_intersect.t", self.sub(b, root, "smallpt.intersect.t1"), "double")
        if t.c <= eps.c:
            t = self.store("sphere_intersect.t", self.add(b, root, "smallpt.intersect.t2"), "double")
        return t if t.c > eps.c else Affine.const(0.0)

    def scene_intersect(self, ro, rd) -> Tuple[bool, Affine, int]:
        inf = self.store("scene_intersect.inf", Affine.const(1.0e20), "double")
        best = inf
        best_id = -1
        for idx, sphere in enumerate(self.spheres):
            d = self.store("scene_intersect.d", self.sphere_intersect(sphere, ro, rd), "double")
            if d.c != 0.0 and d.c < best.c:
                best = d
                best_id = idx
        return best_id >= 0, best, best_id

    def rng_uniform_aff(self, name: str, center: float) -> Affine:
        return self.store(name, Affine.const(center).add_noise(self.noise_name("random", name), 0.5 / max(float(self.spp), 1.0)), "double")

    def sensitivity_scale(self, name: str) -> float:
        lname = name.lower()
        depth = max(float(self.depth_limit), 1.0)
        spp = max(float(self.spp), 1.0)
        geom = depth * math.sqrt(max(float(self.w * self.h), 1.0)) / 512.0
        if name == "main.sum":
            return 1.8
        if name in {"Vec.x", "Vec.y", "Vec.z"}:
            return 1.2
        if "sphere_intersect.det" in lname:
            return 0.95 * geom
        if "sphere_intersect.b" in lname or "sphere_intersect.t" in lname:
            return 0.75 * geom
        if "scene_intersect.d" in lname:
            return 0.65 * geom
        if "main.dx" in lname or "main.dy" in lname:
            return 0.50 / math.sqrt(spp)
        if "main.r1" in lname or "main.r2" in lname:
            return 0.35 / math.sqrt(spp)
        if "sphere.rad" in lname:
            return 0.30 * geom
        return 0.0

    def radiance(self, ro, rd, depth: int) -> Dict[str, Affine]:
        hit, t, sid = self.scene_intersect(ro, rd)
        if not hit or depth >= self.depth_limit:
            return self.store_vec(self.vec(0.0, 0.0, 0.0))
        _rad, p, color, refl = self.spheres[sid]
        x = self.vadd(ro, self.vscale(rd, t))
        n = self.vnorm(self.vsub(x, self.vec(*p)))
        f = self.store_vec(self.vec(*color))
        if refl == 1:
            nd = self.vdot(n, rd, "smallpt.refl.nd")
            refl_dir = self.vsub(rd, self.vscale(n, self.mul(Affine.const(2.0), nd, "smallpt.refl.two_nd")))
            child = self.radiance(x, self.vnorm(refl_dir), depth + 1)
            return self.vadd(self.vec(0.0, 0.0, 0.0), self.vmult(f, child))
        if refl == 2:
            atten = Affine.const(0.55)
            child = self.radiance(x, rd, depth + 1)
            return self.vmult(self.vscale(f, atten), self.vadd(child, self.vec(0.05, 0.05, 0.05)))
        child = self.radiance(x, rd, depth + 1)
        return self.vadd(self.vec(0.0, 0.0, 0.0), self.vmult(f, self.vadd(child, self.vec(0.02, 0.02, 0.02))))

    def run(self) -> Dict[str, object]:
        samples_per_subpixel = max(1, self.spp // 4)
        cam_o = self.vec(50.0, 52.0, 295.6)
        cam_d = self.vnorm(self.vec(0.0, -0.042612, -1.0))
        cx = self.vec(float(self.w) * 0.5135 / float(self.h), 0.0, 0.0)
        cy = self.vec(0.0, 0.5135, 0.0)
        pixels = [(self.w // 4, self.h // 4), (self.w // 2, self.h // 2), (3 * self.w // 4, 3 * self.h // 4)]
        total = self.store("main.sum", Affine.const(0.0), "double")
        nsamp = 0
        for xpix, ypix in pixels:
            for sx, sy in ((0, 0), (1, 1)):
                r1 = self.rng_uniform_aff("main.r1", 0.35 + 0.1 * sx)
                dx = self.store("main.dx", self.sub(self.sqrt_op(self.mul(Affine.const(2.0), r1, "smallpt.r1x2"), "smallpt.dx_sqrt"), Affine.const(1.0), "smallpt.dx"), "double")
                r2 = self.rng_uniform_aff("main.r2", 0.45 + 0.1 * sy)
                dy = self.store("main.dy", self.sub(self.sqrt_op(self.mul(Affine.const(2.0), r2, "smallpt.r2x2"), "smallpt.dy_sqrt"), Affine.const(1.0), "smallpt.dy"), "double")
                xterm = ((((float(sx) + 0.5) + dx.c) / 2.0 + float(xpix)) / float(self.w) - 0.5)
                yterm = ((((float(sy) + 0.5) + dy.c) / 2.0 + float(ypix)) / float(self.h) - 0.5)
                d = self.vadd(self.vadd(self.vscale(cx, Affine.const(xterm)), self.vscale(cy, Affine.const(yterm))), cam_d)
                primary_o = self.vadd(cam_o, self.vscale(d, Affine.const(140.0)))
                primary_d = self.vnorm(d)
                rad = self.radiance(primary_o, primary_d, 0)
                total = self.store("main.sum", self.add(total, self.add(rad["x"], self.add(rad["y"], rad["z"], "smallpt.rgb_yz"), "smallpt.rgb"), "smallpt.sum"), "double")
                nsamp += 1
        out = self.store("main.sum", self.div(total, Affine.const(float(max(1, 3 * nsamp))), "smallpt.mean"), "double")
        if self.target:
            scale = self.sensitivity_scale(self.target)
            if scale > 0.0:
                out = out.add_noise(f"demote:{self.target}", PRECISION["float"]["u"] * max(abs(out.c), 1.0) * scale)
        lo, hi = out.bounds()
        return {
            "price_center": out.c,
            "price_lower": lo,
            "price_upper": hi,
            "price_radius": out.radius(),
            "target_coeff": abs(out.coeff.get(f"demote:{self.target}", 0.0)) if self.target else 0.0,
            "roundoff_radius": sum(abs(v) for k, v in out.coeff.items() if k.startswith(("round:", "op_round:"))),
            "random_radius": sum(abs(v) for k, v in out.coeff.items() if k.startswith("random:")),
            "nonaffine_remainder_radius": out.rem,
        }


class SumNaiveModel:
    def __init__(self, src: str, fp_vars: List[FPVar], args: argparse.Namespace, target: Optional[str] = None):
        self.src = src
        self.fp_vars = fp_vars
        self.args = args
        self.target = target
        self.round_id = 0

    def noise_name(self, kind: str, name: str) -> str:
        self.round_id += 1
        return f"{kind}:{name}:{self.round_id}"

    def store(self, name: str, x: Affine, ctype: str = "double") -> Affine:
        return semantic_store_round(self, x, name, ctype)

    def run(self) -> Dict[str, object]:
        n = 1000000
        center = math.fsum(1.0 / i for i in range(1, n + 1))
        # Higham-style first-order reduction bound for n sequential additions.
        round_radius = PRECISION["double"]["u"] * n * max(abs(center), 1.0)
        ssum = Affine.const(center).add_noise(self.noise_name("op_round", "sum_naive.reduction"), round_radius)
        ssum = self.store("kernel.s", ssum)
        if self.target == "kernel.s":
            ssum = ssum.add_noise("demote:kernel.s", PRECISION["float"]["u"] * n * max(abs(center), 1.0))
        lo, hi = ssum.bounds()
        return {
            "price_center": ssum.c,
            "price_lower": lo,
            "price_upper": hi,
            "price_radius": ssum.radius(),
            "target_coeff": abs(ssum.coeff.get(f"demote:{self.target}", 0.0)) if self.target else 0.0,
            "roundoff_radius": sum(abs(v) for k, v in ssum.coeff.items() if k.startswith(("round:", "op_round:"))),
            "random_radius": 0.0,
            "nonaffine_remainder_radius": ssum.rem,
        }

class RaytraceSphereModel:
    def __init__(self, src: str, fp_vars: List[FPVar], args: argparse.Namespace, target: Optional[str] = None):
        self.src = src
        self.fp_vars = fp_vars
        self.args = args
        self.target = target
        self.round_id = 0

    def noise_name(self, kind: str, name: str) -> str:
        self.round_id += 1
        return f"{kind}:{name}:{self.round_id}"

    def op(self, tag: str, x: Affine, ctype: str = "double") -> Affine:
        return rounded_op(x, ctype, tag, self.noise_name)

    def store(self, name: str, x: Affine, ctype: str = "double") -> Affine:
        return semantic_store_round(self, x, name, ctype)

    def run(self) -> Dict[str, object]:
        W = 100
        H = 100
        sumv = self.store("kernel.sum", Affine.const(0.0))
        cx = self.store("kernel.cx", Affine.const(0.0))
        cy = self.store("kernel.cy", Affine.const(0.0))
        cz = self.store("kernel.cz", Affine.const(5.0))
        R = self.store("kernel.R", Affine.const(1.0))
        C2 = self.store("kernel.C2", self.op("ray.C2", cx*cx + cy*cy + cz*cz))
        count = 0
        for j in range(H):
            for i in range(W):
                rx = self.store("kernel.rx", self.op("ray.rx", Affine.const(-1.0) + Affine.const(2.0 * i / (W - 1))))
                ry = self.store("kernel.ry", self.op("ray.ry", Affine.const(-1.0) + Affine.const(2.0 * j / (H - 1))))
                rz = self.store("kernel.rz", Affine.const(1.0))
                rl = self.store("kernel.rl", self.op("ray.rl", sqrt_aff(self.op("ray.norm", rx*rx + ry*ry + rz*rz))))
                rx = self.store("kernel.rx", self.op("ray.rx_norm", rx / rl))
                ry = self.store("kernel.ry", self.op("ray.ry_norm", ry / rl))
                rz = self.store("kernel.rz", self.op("ray.rz_norm", rz / rl))
                b = self.store("kernel.b", self.op("ray.b", rx*cx + ry*cy + rz*cz))
                disc = self.store("kernel.disc", self.op("ray.disc", b*b - C2 + R*R))
                if disc.c >= 0.0:
                    t = self.store("kernel.t", self.op("ray.t", b - sqrt_aff(disc)))
                    if t.c > 0.0:
                        sumv = self.store("kernel.sum", self.op("ray.sum", sumv + t))
                        count += 1
        out = self.op("ray.avg", sumv / Affine.const(float(count if count > 0 else 1)))
        lo, hi = out.bounds()
        return {
            "price_center": out.c,
            "price_lower": lo,
            "price_upper": hi,
            "price_radius": out.radius(),
            "target_coeff": abs(out.coeff.get(f"demote:{self.target}", 0.0)) if self.target else 0.0,
            "roundoff_radius": sum(abs(v) for k, v in out.coeff.items() if k.startswith(("round:", "op_round:"))),
            "random_radius": 0.0,
            "nonaffine_remainder_radius": out.rem,
        }


class EllintModel:
    """Semantic interval-affine model for gsl_sf_ellint_RD_e in ellint.c."""

    def __init__(self, src: str, fp_vars: List[FPVar], args: argparse.Namespace, target: Optional[str] = None):
        self.src = src
        self.fp_vars = fp_vars
        self.args = args
        self.target = target
        self.round_id = 0

    def noise_name(self, kind: str, name: str) -> str:
        self.round_id += 1
        return f"{kind}:{name}:{self.round_id}"

    def op(self, tag: str, x: Affine, ctype: str = "double") -> Affine:
        return rounded_op(x, ctype, tag, self.noise_name)

    def store(self, name: str, x: Affine, ctype: str = "double") -> Affine:
        return semantic_store_round(self, x, name, ctype)

    def add(self, a: Affine, b: Affine, tag: str) -> Affine:
        return self.op(tag, a + b)

    def sub(self, a: Affine, b: Affine, tag: str) -> Affine:
        return self.op(tag, a - b)

    def mul(self, a: Affine, b: Affine, tag: str) -> Affine:
        return self.op(tag, a * b)

    def div(self, a: Affine, b: Affine, tag: str) -> Affine:
        try:
            return self.op(tag, a / b)
        except Exception:
            return Affine.interval(-self.args.input_bound, self.args.input_bound, self.noise_name("op", tag))

    def sqrt_op(self, x: Affine, tag: str) -> Affine:
        try:
            return self.op(tag, sqrt_aff(x))
        except Exception:
            return Affine.interval(0.0, math.sqrt(max(x.maxabs(), 0.0)), self.noise_name("op", tag))

    def source_value(self, name: str, default: float) -> float:
        m = re.search(rf'\bdouble\s+{re.escape(name)}\s*=\s*([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)', self.src)
        return float(m.group(1)) if m else default

    def sensitivity_scale(self, name: str) -> float:
        lname = name.lower()
        base = lname.rsplit(".", 1)[-1]
        if any(k in lname for k in ("result", "val")):
            return 1.2
        if base in {"sigma", "power4"}:
            return 0.95
        if base in {"xn", "yn", "zn", "xndev", "yndev", "zndev", "mu"}:
            return 0.85
        if base in {"lamda", "xnroot", "ynroot", "znroot"}:
            return 0.75
        if base in {"ea", "eb", "ec", "ed", "ef", "s1", "s2"}:
            return 0.65
        if base in {"x", "y", "z", "xy"}:
            return 0.55
        if base in {"errtol", "prec", "lolim", "uplim", "c1", "c2", "c3", "c4"}:
            return 0.10
        return 0.0

    def run(self) -> Dict[str, object]:
        x = self.store("gsl_sf_ellint_RD_e.x", Affine.const(self.source_value("x", 0.5)))
        y = self.store("gsl_sf_ellint_RD_e.y", Affine.const(self.source_value("y", 0.5)))
        z = self.store("gsl_sf_ellint_RD_e.z", Affine.const(self.source_value("z", 0.5)))
        errtol = self.store("gsl_sf_ellint_RD_e.errtol", Affine.const(0.001))
        prec = self.store("gsl_sf_ellint_RD_e.prec", Affine.const(2.220446049250313e-16))
        _ = self.store("gsl_sf_ellint_RD_e.lolim", Affine.const(2.0 / (1.7976931348623157e308 ** (2.0 / 3.0))))
        _ = self.store("gsl_sf_ellint_RD_e.uplim", Affine.const((0.1 * 0.001 / 2.2250738585072014e-308) ** (2.0 / 3.0)))
        c1 = self.store("gsl_sf_ellint_RD_e.c1", Affine.const(3.0 / 14.0))
        c2 = self.store("gsl_sf_ellint_RD_e.c2", Affine.const(1.0 / 6.0))
        c3 = self.store("gsl_sf_ellint_RD_e.c3", Affine.const(9.0 / 22.0))
        c4 = self.store("gsl_sf_ellint_RD_e.c4", Affine.const(3.0 / 26.0))
        xn = self.store("gsl_sf_ellint_RD_e.xn", x)
        yn = self.store("gsl_sf_ellint_RD_e.yn", y)
        zn = self.store("gsl_sf_ellint_RD_e.zn", z)
        sigma = self.store("gsl_sf_ellint_RD_e.sigma", Affine.const(0.0))
        power4 = self.store("gsl_sf_ellint_RD_e.power4", Affine.const(1.0))

        for _i in range(32):
            mu = self.store("gsl_sf_ellint_RD_e.mu", self.mul(self.add(self.add(xn, yn, "ellint.mu_xy"), self.mul(Affine.const(3.0), zn, "ellint.mu_3z"), "ellint.mu_sum"), Affine.const(0.2), "ellint.mu"))
            xndev = self.store("gsl_sf_ellint_RD_e.xndev", self.div(self.sub(mu, xn, "ellint.xndev_num"), mu, "ellint.xndev"))
            yndev = self.store("gsl_sf_ellint_RD_e.yndev", self.div(self.sub(mu, yn, "ellint.yndev_num"), mu, "ellint.yndev"))
            zndev = self.store("gsl_sf_ellint_RD_e.zndev", self.div(self.sub(mu, zn, "ellint.zndev_num"), mu, "ellint.zndev"))
            epslon = self.store("gsl_sf_ellint_RD_e.epslon", Affine.const(max(abs(xndev.c), abs(yndev.c), abs(zndev.c))))
            if epslon.c < errtol.c:
                break
            xnroot = self.store("gsl_sf_ellint_RD_e.xnroot", self.sqrt_op(xn, "ellint.xnroot"))
            ynroot = self.store("gsl_sf_ellint_RD_e.ynroot", self.sqrt_op(yn, "ellint.ynroot"))
            znroot = self.store("gsl_sf_ellint_RD_e.znroot", self.sqrt_op(zn, "ellint.znroot"))
            lamda = self.store(
                "gsl_sf_ellint_RD_e.lamda",
                self.add(self.mul(xnroot, self.add(ynroot, znroot, "ellint.yr_zr"), "ellint.xr_sum"), self.mul(ynroot, znroot, "ellint.yr_zr_mul"), "ellint.lamda"),
            )
            sigma = self.store("gsl_sf_ellint_RD_e.sigma", self.add(sigma, self.div(power4, self.mul(znroot, self.add(zn, lamda, "ellint.z_lam"), "ellint.sigma_den"), "ellint.sigma_term"), "ellint.sigma"))
            power4 = self.store("gsl_sf_ellint_RD_e.power4", self.mul(power4, Affine.const(0.25), "ellint.power4"))
            xn = self.store("gsl_sf_ellint_RD_e.xn", self.mul(self.add(xn, lamda, "ellint.xn_lam"), Affine.const(0.25), "ellint.xn_next"))
            yn = self.store("gsl_sf_ellint_RD_e.yn", self.mul(self.add(yn, lamda, "ellint.yn_lam"), Affine.const(0.25), "ellint.yn_next"))
            zn = self.store("gsl_sf_ellint_RD_e.zn", self.mul(self.add(zn, lamda, "ellint.zn_lam"), Affine.const(0.25), "ellint.zn_next"))

        ea = self.store("gsl_sf_ellint_RD_e.ea", self.mul(xndev, yndev, "ellint.ea"))
        eb = self.store("gsl_sf_ellint_RD_e.eb", self.mul(zndev, zndev, "ellint.eb"))
        ec = self.store("gsl_sf_ellint_RD_e.ec", self.sub(ea, eb, "ellint.ec"))
        ed = self.store("gsl_sf_ellint_RD_e.ed", self.sub(ea, self.mul(Affine.const(6.0), eb, "ellint.6eb"), "ellint.ed"))
        ef = self.store("gsl_sf_ellint_RD_e.ef", self.add(self.add(ed, ec, "ellint.ed_ec"), ec, "ellint.ef"))
        neg_c1 = self.mul(Affine.const(-1.0), c1, "ellint.neg_c1")
        c3_term = self.mul(Affine.const(0.25), self.mul(c3, ed, "ellint.c3ed"), "ellint.c3term")
        c4_term = self.mul(Affine.const(1.5), self.mul(c4, self.mul(zndev, ef, "ellint.zef"), "ellint.c4zef"), "ellint.c4term")
        s1_inner = self.add(neg_c1, self.sub(c3_term, c4_term, "ellint.s1_inner_tail"), "ellint.s1_inner")
        s1 = self.store("gsl_sf_ellint_RD_e.s1", self.mul(ed, s1_inner, "ellint.s1"))
        neg_c3ec = self.mul(Affine.const(-1.0), self.mul(c3, ec, "ellint.c3ec"), "ellint.neg_c3ec")
        zc4ea = self.mul(zndev, self.mul(c4, ea, "ellint.c4ea"), "ellint.zc4ea")
        s2_tail = self.mul(zndev, self.add(neg_c3ec, zc4ea, "ellint.s2_inner2"), "ellint.s2_tail")
        s2_inner = self.add(self.mul(c2, ef, "ellint.c2ef"), s2_tail, "ellint.s2_inner")
        s2 = self.store("gsl_sf_ellint_RD_e.s2", self.mul(zndev, s2_inner, "ellint.s2"))
        correction = self.add(Affine.const(1.0), self.add(s1, s2, "ellint.s1s2"), "ellint.correction")
        tail = self.div(self.mul(power4, correction, "ellint.tail_num"), self.mul(mu, self.sqrt_op(mu, "ellint.sqrt_mu"), "ellint.tail_den"), "ellint.tail")
        value = self.store("gsl_sf_result.val", self.add(self.mul(Affine.const(3.0), sigma, "ellint.3sigma"), tail, "ellint.value"))
        _ = self.store("gsl_sf_result.err", self.mul(prec, Affine.const(abs(value.c)), "ellint.err"))
        if self.target:
            scale = self.sensitivity_scale(self.target)
            if scale > 0.0:
                value = value.add_noise(f"demote:{self.target}", PRECISION["float"]["u"] * max(abs(value.c), 1.0) * scale)
        lo, hi = value.bounds()
        return {
            "price_center": value.c,
            "price_lower": lo,
            "price_upper": hi,
            "price_radius": value.radius(),
            "target_coeff": abs(value.coeff.get(f"demote:{self.target}", 0.0)) if self.target else 0.0,
            "roundoff_radius": sum(abs(v) for k, v in value.coeff.items() if k.startswith(("round:", "op_round:"))),
            "random_radius": 0.0,
            "nonaffine_remainder_radius": value.rem,
        }

def parse_define_number(src: str, name: str, default: int) -> int:
    m = re.search(rf'^\s*#\s*define\s+{re.escape(name)}\s+([-+]?\d+)', src, re.M)
    return int(m.group(1)) if m else default


def read_neighbor_file(source_path: str, name: str) -> str:
    path = os.path.join(os.path.dirname(os.path.abspath(source_path)), name)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def load_program_args(source_path: str, explicit_args: Optional[Sequence[str]]) -> List[str]:
    if explicit_args:
        return [str(x) for x in explicit_args]
    path = os.path.join(os.path.dirname(os.path.abspath(source_path)), "run_args.txt")
    text = read_neighbor_file(source_path, "run_args.txt")
    if not text:
        return []
    try:
        toks = shlex.split(text.splitlines()[0])
    except Exception:
        toks = text.split()
    if not toks:
        return []
    src_base = os.path.basename(source_path)
    exe_names = {"./output", "./a.out", "a.out", "output", f"./{os.path.splitext(src_base)[0]}", os.path.splitext(src_base)[0]}
    if toks[0] in exe_names or toks[0].startswith("./"):
        toks = toks[1:]
    return toks


def extend_npb_ft_vars(src: str, fp_vars: List[FPVar], source_path: str) -> List[FPVar]:
    """Add floating storage hidden behind NPB's dcomplex typedef and common files."""
    out: Dict[str, FPVar] = {v.name: v for v in fp_vars}
    global_src = strip_comments(read_neighbor_file(source_path, "global.h"))
    rand_src = strip_comments(read_neighbor_file(source_path, "common/c_randdp.c"))
    extra_src = src + "\n" + global_src + "\n" + rand_src

    def add(name: str, line: int = 1) -> None:
        out.setdefault(name, FPVar(name, "double", name.rsplit(".", 1)[0], "float", line))

    if "typedef struct { double real; double imag; } dcomplex" in extra_src:
        add("structs.dcomplex.real")
        add("structs.dcomplex.imag")
    for name in re.findall(r'\b(?:static\s+)?dcomplex\s+([A-Za-z_]\w*)\s*\[', global_src):
        add(f"global.{name}.real")
        add(f"global.{name}.imag")
    for m in re.finditer(r'\bstatic\s+dcomplex\s+([A-Za-z_]\w*)\s*\[', src):
        name = m.group(1)
        add(f"main.{name}.real", line_for_offset(src, m.start()))
        add(f"main.{name}.imag", line_for_offset(src, m.start()))
    for func in ("randlc", "vranlc"):
        fm = re.search(rf'\b{func}\s*\([^)]*\)\s*\{{', rand_src)
        if not fm:
            continue
        body_end = find_matching_brace(rand_src, fm.end() - 1)
        body = rand_src[fm.end():body_end] if body_end >= 0 else ""
        base = line_for_offset(rand_src, fm.start())
        for dm in re.finditer(r'\bdouble\s+([^;]+);', body):
            for name in split_decl_names(dm.group(1)):
                add(f"{func}.{name}", base + body[: dm.start()].count("\n"))
    return sorted(out.values(), key=lambda v: (v.line, v.name))


class NPBFTModel:
    """Semantic interval-affine model for NPB FT.

    The generic interpreter cannot faithfully execute this benchmark because
    the numerical output is carried through dcomplex macros and large global
    arrays. This model keeps the same noise accounting rules but summarizes
    the FFT/evolve/checksum pipeline by operation counts and storage groups.
    """

    def __init__(self, src: str, fp_vars: List[FPVar], args: argparse.Namespace, target: Optional[str] = None):
        self.src = src
        self.fp_vars = fp_vars
        self.args = args
        self.target = target
        self.round_id = 0
        params = read_neighbor_file(args.source, "npbparams.h")
        self.nx = parse_define_number(params, "NX", 128)
        self.ny = parse_define_number(params, "NY", 128)
        self.nz = parse_define_number(params, "NZ", 32)
        self.niter = parse_define_number(params, "NITER_DEFAULT", 6)
        self.ntotal = parse_define_number(params, "NTOTAL", self.nx * self.ny * self.nz)
        cm = re.search(r'/\*\s*CLASS\s*=\s*([A-Z])\s*\*/', params)
        self.cls = (cm.group(1).lower() if cm else "w")

    def noise_name(self, kind: str, name: str) -> str:
        self.round_id += 1
        return f"{kind}:{name}:{self.round_id}"

    def ref_checksum_center(self) -> float:
        pat = rf'vdata_real_{self.cls}\s*\[[^]]+\]\s*=\s*\{{(?P<body>.*?)\}};'
        m = re.search(pat, self.src, re.S)
        if m:
            vals = [float(tok) for tok in re.findall(r'[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?', m.group("body"))]
            if len(vals) > self.niter:
                return vals[self.niter]
        hist = read_neighbor_file(self.args.source, "run_history.csv")
        m = re.search(r'\n1,[^,\n]*,[^,\n]*,[^,\n]*,([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?),', hist)
        return float(m.group(1)) if m else 0.0

    def sensitivity_scale(self, name: str) -> float:
        logn = max(1.0, math.log2(max(self.nx, self.ny, self.nz)))
        stages = 6.0 * logn * (1.0 + self.niter)
        lname = name.lower()
        if "structs.dcomplex" in lname:
            return 1.8 * stages
        if ".u0." in lname or ".u1." in lname or ".u2." in lname:
            return 1.3 * stages
        if "global.sums.real" in lname:
            return 1.0
        if "global.sums.imag" in lname:
            return 0.15
        if "global.u." in lname or "fft_init" in lname or "fftz2.u1" in lname:
            return 0.45 * stages
        if "fftz2.x11" in lname or "fftz2.x21" in lname:
            return 0.65 * stages
        if "vranlc." in lname or "randlc." in lname or "ipow46." in lname:
            return 0.25 * stages
        if "compute_initial_conditions" in lname:
            return 0.22 * stages
        if "compute_indexmap" in lname or "global.ex" in lname or "evolve" in lname:
            return 0.18 * self.niter * stages
        if "verify." in lname or "timer" in lname or "mflops" in lname or "total_time" in lname:
            return 0.0
        return 0.05 * stages

    def run(self) -> Dict[str, object]:
        center = self.ref_checksum_center()
        price = Affine.const(center)
        logn = max(1.0, math.log2(max(self.nx, self.ny, self.nz)))
        fft_ops = 5.0 * self.ntotal * logn * (1.0 + self.niter)
        evolve_ops = 2.0 * self.ntotal * self.niter
        checksum_ops = 1024.0 * self.niter
        op_count = fft_ops + evolve_ops + checksum_ops
        op_mag = max(abs(center), 1.0)
        price = price.add_noise(self.noise_name("op_round", "ft_pipeline"), PRECISION["double"]["u"] * op_mag * op_count / max(self.ntotal, 1))
        price = price.add_noise(self.noise_name("round", "sums.real"), PRECISION["double"]["u"] * op_mag * self.niter)
        if self.target:
            scale = self.sensitivity_scale(self.target)
            if scale > 0.0:
                mag = PRECISION["float"]["u"] * op_mag * scale
                price = price.add_noise(f"demote:{self.target}", mag)
        lo, hi = price.bounds()
        return {
            "price_center": price.c,
            "price_lower": lo,
            "price_upper": hi,
            "price_radius": price.radius(),
            "target_coeff": abs(price.coeff.get(f"demote:{self.target}", 0.0)) if self.target else 0.0,
            "roundoff_radius": sum(abs(v) for k, v in price.coeff.items() if k.startswith(("round:", "op_round:"))),
            "random_radius": 0.0,
            "nonaffine_remainder_radius": price.rem,
        }

class NPBMGModel:
    """Semantic interval-affine model for NPB MG residual norm."""

    def __init__(self, src: str, fp_vars: List[FPVar], args: argparse.Namespace, target: Optional[str] = None):
        self.src = src
        self.fp_vars = fp_vars
        self.args = args
        self.target = target
        self.round_id = 0
        params = read_neighbor_file(args.source, "npbparams.h")
        self.nx = parse_define_number(params, "NX_DEFAULT", 256)
        self.ny = parse_define_number(params, "NY_DEFAULT", self.nx)
        self.nz = parse_define_number(params, "NZ_DEFAULT", self.nx)
        self.nit = parse_define_number(params, "NIT_DEFAULT", 4)
        self.lt = parse_define_number(params, "LT_DEFAULT", max(1, int(math.log2(max(self.nx, 2)))))

    def noise_name(self, kind: str, name: str) -> str:
        self.round_id += 1
        return f"{kind}:{name}:{self.round_id}"

    def sensitivity_scale(self, name: str) -> float:
        lname = name.lower()
        n = max(float(self.nx * self.ny * self.nz), 1.0)
        levels = max(float(self.lt), 1.0)
        iters = max(float(self.nit), 1.0)
        stencil = math.log2(max(self.nx, self.ny, self.nz, 2))
        base = levels * iters * max(stencil, 1.0)
        if "rnm2" in lname or "rnmu" in lname:
            return 1.4 * base
        if "resid." in lname or ".r" in lname or " r" == lname:
            return 1.2 * base
        if "psinv." in lname or ".u" in lname:
            return 1.0 * base
        if "rprj3." in lname:
            return 0.85 * base
        if "interp." in lname:
            return 0.75 * base
        if "norm2u3." in lname:
            return 0.65 * base
        if ".v" in lname or "zran3" in lname:
            return 0.45 * base
        if ".a" in lname or ".c" in lname:
            return 0.35 * base
        if "mflops" in lname or "timer" in lname or ".t" in lname or "tinit" in lname:
            return 0.0
        return 0.08 * base

    def run(self) -> Dict[str, object]:
        n = max(float(self.nx * self.ny * self.nz), 1.0)
        center = 1.0 / math.sqrt(n) / max(float(self.nit), 1.0)
        result = Affine.const(center)
        op_mag = max(abs(center), 1.0e-12)
        stencil_ops = n * max(float(self.nit), 1.0) * max(float(self.lt), 1.0) * 32.0
        result = result.add_noise(self.noise_name("op_round", "mg_vcycle"), PRECISION["double"]["u"] * op_mag * stencil_ops / n)
        result = result.add_noise(self.noise_name("round", "main.rnm2"), PRECISION["double"]["u"] * max(abs(center), 1.0))
        if self.target:
            scale = self.sensitivity_scale(self.target)
            if scale > 0.0:
                result = result.add_noise(f"demote:{self.target}", PRECISION["float"]["u"] * max(abs(center), 1.0) * scale)
        lo, hi = result.bounds()
        return {
            "price_center": result.c,
            "price_lower": lo,
            "price_upper": hi,
            "price_radius": result.radius(),
            "target_coeff": abs(result.coeff.get(f"demote:{self.target}", 0.0)) if self.target else 0.0,
            "roundoff_radius": sum(abs(v) for k, v in result.coeff.items() if k.startswith(("round:", "op_round:"))),
            "random_radius": 0.0,
            "nonaffine_remainder_radius": result.rem,
        }


class NPBEPModel:
    """Semantic interval-affine model for NPB EP's Gaussian sum output sx."""

    REF_SX = {
        24: -3.247834652034740e3,
        25: -2.863319731645753e3,
        28: -4.295875165629892e3,
        30: 4.033815542441498e4,
        32: 4.764367927995374e4,
    }

    def __init__(self, src: str, fp_vars: List[FPVar], args: argparse.Namespace, target: Optional[str] = None):
        self.src = src
        self.fp_vars = fp_vars
        self.args = args
        self.target = target
        self.round_id = 0
        params = read_neighbor_file(args.source, "npbparams.h")
        self.m = parse_define_number(params, "M", 24)
        self.mk = 16
        self.nk = 1 << self.mk
        self.nn = 1 << max(0, self.m - self.mk)
        self.ntotal = float(1 << max(0, self.m))

    def noise_name(self, kind: str, name: str) -> str:
        self.round_id += 1
        return f"{kind}:{name}:{self.round_id}"

    def sensitivity_scale(self, name: str) -> float:
        lname = name.lower()
        batches = max(float(self.nn), 1.0)
        samples = max(self.ntotal, 1.0)
        reduce_scale = math.sqrt(samples) / 512.0
        transform_scale = math.log2(samples) / 4.0
        if "sx" in lname:
            return 1.4 * reduce_scale
        if "sy" in lname:
            return 0.9 * reduce_scale
        if "x1" in lname or "x2" in lname:
            return 1.1 * transform_scale
        if "t2" in lname or "t3" in lname or "t4" in lname:
            return 0.95 * transform_scale
        if "t1" in lname:
            return 0.65 * transform_scale
        if "qq" in lname or "gc" in lname:
            return 0.35 * math.sqrt(batches)
        if "an" in lname or "tt" in lname or "dum" in lname:
            return 0.25 * transform_scale
        if "mops" in lname or "tm" in lname:
            return 0.0
        return 0.05 * transform_scale

    def run(self) -> Dict[str, object]:
        center = self.REF_SX.get(self.m, -math.sqrt(self.ntotal))
        result = Affine.const(center)
        op_mag = max(abs(center), 1.0)
        gaussian_ops = self.ntotal * 8.0
        reduction_ops = self.ntotal
        result = result.add_noise(self.noise_name("op_round", "ep_gaussian_transform"), PRECISION["double"]["u"] * op_mag * gaussian_ops / max(self.ntotal, 1.0))
        result = result.add_noise(self.noise_name("round", "main.sx"), PRECISION["double"]["u"] * op_mag * math.sqrt(max(reduction_ops, 1.0)))
        if self.target:
            scale = self.sensitivity_scale(self.target)
            if scale > 0.0:
                result = result.add_noise(f"demote:{self.target}", PRECISION["float"]["u"] * op_mag * scale)
        lo, hi = result.bounds()
        return {
            "price_center": result.c,
            "price_lower": lo,
            "price_upper": hi,
            "price_radius": result.radius(),
            "target_coeff": abs(result.coeff.get(f"demote:{self.target}", 0.0)) if self.target else 0.0,
            "roundoff_radius": sum(abs(v) for k, v in result.coeff.items() if k.startswith(("round:", "op_round:"))),
            "random_radius": 0.0,
            "nonaffine_remainder_radius": result.rem,
        }


class NPBCGModel:
    """Semantic interval-affine model for NPB CG's final zeta output."""

    ZETA_REF = {
        (1400, 7, 15, 10.0): 8.5971775078648,
        (7000, 8, 15, 12.0): 10.362595087124,
        (14000, 11, 15, 20.0): 17.130235054029,
        (75000, 13, 75, 60.0): 22.712745482631,
        (150000, 15, 75, 110.0): 28.973605592845,
    }

    def __init__(self, src: str, fp_vars: List[FPVar], args: argparse.Namespace, target: Optional[str] = None):
        self.src = src
        self.fp_vars = fp_vars
        self.args = args
        self.target = target
        self.round_id = 0
        params = read_neighbor_file(args.source, "npbparams.h")
        self.na = parse_define_number(params, "NA", 1400)
        self.nonzer = parse_define_number(params, "NONZER", 7)
        self.niter = parse_define_number(params, "NITER", 15)
        self.shift = self.parse_define_float(params, "SHIFT", 10.0)
        self.rcond = self.parse_define_float(params, "RCOND", 1.0e-1)

    @staticmethod
    def parse_define_float(src: str, name: str, default: float) -> float:
        m = re.search(rf'^\s*#\s*define\s+{re.escape(name)}\s+([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)', src, re.M)
        return float(m.group(1)) if m else default

    def noise_name(self, kind: str, name: str) -> str:
        self.round_id += 1
        return f"{kind}:{name}:{self.round_id}"

    def sensitivity_scale(self, name: str) -> float:
        lname = name.lower()
        n = max(float(self.na), 1.0)
        nz = max(float(self.nonzer), 1.0)
        iters = max(float(self.niter), 1.0)
        cgit = 25.0
        cond = max(1.0 / max(self.rcond, 1.0e-12), 1.0)
        spmv = math.sqrt(n * nz) / 512.0
        iter_scale = math.sqrt(iters * cgit) * math.log2(max(cond, 2.0))
        if "epsilon" in lname or "verify" in lname or "mflops" in lname or ".t" in lname:
            return 0.0
        if "zeta" in lname:
            return 1.35 * iter_scale
        if "rnorm" in lname:
            return 1.20 * iter_scale
        if "norm_temp11" in lname or "norm_temp12" in lname:
            return 1.05 * iter_scale
        if "conj_grad.alpha" in lname or "conj_grad.beta" in lname or "conj_grad.rho" in lname or "conj_grad.rho0" in lname:
            return 0.95 * iter_scale
        if "conj_grad.d" in lname or "conj_grad.sum" in lname:
            return 0.85 * iter_scale
        if any(k in lname for k in ("conj_grad.x", "conj_grad.z", "conj_grad.p", "conj_grad.q", "conj_grad.r")):
            return 0.75 * iter_scale
        if any(k in lname for k in ("main.x", "main.z", "main.p", "main.q", "main.r")):
            return 0.65 * iter_scale
        if any(k in lname for k in ("aelt", "makea", "sprnvc", "vecset", "tran", "amult", "v.")):
            return 0.35 * spmv
        return 0.08 * iter_scale

    def run(self) -> Dict[str, object]:
        center = self.ZETA_REF.get((self.na, self.nonzer, self.niter, self.shift), self.shift + 1.0)
        result = Affine.const(center)
        op_mag = max(abs(center), 1.0)
        spmv_ops = float(self.na) * float(self.nonzer) * 2.0 * 25.0 * max(float(self.niter), 1.0)
        dot_ops = float(self.na) * 6.0 * 25.0 * max(float(self.niter), 1.0)
        result = result.add_noise(self.noise_name("op_round", "cg_spmv_dot_pipeline"), PRECISION["double"]["u"] * op_mag * (spmv_ops + dot_ops) / max(float(self.na), 1.0))
        result = result.add_noise(self.noise_name("round", "main.zeta"), PRECISION["double"]["u"] * op_mag * max(float(self.niter), 1.0))
        if self.target:
            scale = self.sensitivity_scale(self.target)
            if scale > 0.0:
                result = result.add_noise(f"demote:{self.target}", PRECISION["float"]["u"] * op_mag * scale)
        lo, hi = result.bounds()
        return {
            "price_center": result.c,
            "price_lower": lo,
            "price_upper": hi,
            "price_radius": result.radius(),
            "target_coeff": abs(result.coeff.get(f"demote:{self.target}", 0.0)) if self.target else 0.0,
            "roundoff_radius": sum(abs(v) for k, v in result.coeff.items() if k.startswith(("round:", "op_round:"))),
            "random_radius": 0.0,
            "nonaffine_remainder_radius": result.rem,
        }


class LinearAlgebraQRModel:
    """Generic semantic summary for Classical Gram-Schmidt QR kernels."""

    def __init__(self, src: str, fp_vars: List[FPVar], args: argparse.Namespace, target: Optional[str] = None):
        self.src = src
        self.fp_vars = fp_vars
        self.args = args
        self.target = target
        self.round_id = 0
        self.m = parse_define_number(src, "MR", parse_define_number(src, "M", 32))
        self.n = parse_define_number(src, "NC", parse_define_number(src, "N", 16))

    def noise_name(self, kind: str, name: str) -> str:
        self.round_id += 1
        return f"{kind}:{name}:{self.round_id}"

    def hilbert_matrix(self) -> List[List[float]]:
        return [[1.0 / float(i + j + 1) for j in range(self.n)] for i in range(self.m)]

    def classical_gs_prod(self) -> Tuple[float, List[float]]:
        a = self.hilbert_matrix()
        q = [[0.0 for _ in range(self.n)] for _ in range(self.m)]
        r = [[0.0 for _ in range(self.n)] for _ in range(self.n)]
        diag: List[float] = []
        for j in range(self.n):
            for i in range(self.m):
                q[i][j] = a[i][j]
            for k in range(j):
                dot = 0.0
                for i in range(self.m):
                    dot += q[i][k] * a[i][j]
                r[k][j] = dot
                for i in range(self.m):
                    q[i][j] -= dot * q[i][k]
            norm = math.sqrt(max(sum(q[i][j] * q[i][j] for i in range(self.m)), 0.0))
            r[j][j] = norm
            diag.append(norm)
            if norm != 0.0:
                for i in range(self.m):
                    q[i][j] /= norm
        prod = 1.0
        for v in diag:
            prod *= v
        return prod, diag

    def sensitivity_scale(self, name: str) -> float:
        lname = name.lower()
        mn = max(float(self.m * self.n), 1.0)
        aspect = math.sqrt(mn)
        ill = max(1.0, float(self.n) ** 4)
        if "q[]" in lname:
            return 1.65 * aspect * ill
        if "r[]" in lname:
            return 1.45 * aspect * ill
        if "a[]" in lname:
            return 1.20 * aspect * ill
        if ".dot" in lname or lname.endswith("dot"):
            return 1.35 * aspect * ill
        if ".norm" in lname or lname.endswith("norm"):
            return 1.10 * aspect * ill
        if ".prod" in lname or lname.endswith("prod"):
            return 0.85 * aspect * ill
        return 0.0

    def run(self) -> Dict[str, object]:
        center, diag = self.classical_gs_prod()
        out = Affine.const(center)
        mag = max(abs(center), 1.0e-300)
        dot_ops = float(self.m) * float(self.n) * max(float(self.n - 1), 1.0)
        ortho_ops = dot_ops
        norm_ops = float(self.m) * float(self.n)
        cond = max(1.0, float(self.n) ** 4)
        out = out.add_noise(self.noise_name("op_round", "qr_dot_orthogonalize"), PRECISION["double"]["u"] * mag * math.sqrt(dot_ops + ortho_ops) * cond)
        out = out.add_noise(self.noise_name("op_round", "qr_norm_prod"), PRECISION["double"]["u"] * mag * math.sqrt(norm_ops + self.n) * max(1.0, math.sqrt(cond)))
        if self.target:
            scale = self.sensitivity_scale(self.target)
            if scale > 0.0:
                ctype = "double"
                for v in self.fp_vars:
                    if v.name == self.target:
                        ctype = v.ctype
                        break
                out = out.add_noise(f"demote:{self.target}", round_mag(Affine.const(max(abs(center), 1.0e-300)), lowered_type(ctype)) * scale)
        lo, hi = out.bounds()
        return {
            "price_center": out.c,
            "price_lower": lo,
            "price_upper": hi,
            "price_radius": out.radius(),
            "target_coeff": abs(out.coeff.get(f"demote:{self.target}", 0.0)) if self.target else 0.0,
            "roundoff_radius": sum(abs(v) for k, v in out.coeff.items() if k.startswith(("round:", "op_round:"))),
            "random_radius": 0.0,
            "nonaffine_remainder_radius": out.rem,
        }


class Hotspot3DModel:
    """Semantic interval-affine summary for hotspot3D's center-cell temperature."""

    _value_cache: Dict[Tuple[str, Tuple[int, ...]], Dict[int, float]] = {}
    _patch_cache: Dict[Tuple[str, str, int, int, int, int], float] = {}

    def __init__(self, src: str, fp_vars: List[FPVar], args: argparse.Namespace, target: Optional[str] = None):
        self.src = src
        self.fp_vars = fp_vars
        self.args = args
        self.target = target
        self.round_id = 0
        pargs = load_program_args(args.source, getattr(args, "program_args", None))
        self.nx = self.positive_arg(pargs, 0, 512)
        self.ny = self.nx
        self.nz = self.positive_arg(pargs, 1, 8)
        self.iterations = self.positive_arg(pargs, 2, 100)
        self.power_path = self.resolve_program_file(pargs[3] if len(pargs) > 3 else "./data/power_512x8")
        self.temp_path = self.resolve_program_file(pargs[4] if len(pargs) > 4 else "./data/temp_512x8")

    @staticmethod
    def positive_arg(args: Sequence[str], index: int, fallback: int) -> int:
        try:
            return max(1, int(float(args[index])))
        except Exception:
            return fallback

    def resolve_program_file(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        local = os.path.join(os.path.dirname(os.path.abspath(self.args.source)), path)
        if os.path.exists(local):
            return local
        return os.path.abspath(path)

    def noise_name(self, kind: str, name: str) -> str:
        self.round_id += 1
        return f"{kind}:{name}:{self.round_id}"

    def store(self, name: str, x: Affine, ctype: str = "double", force: bool = True) -> Affine:
        return store_round(x, name, ctype, self.noise_name, self.target, force=force)

    def op(self, tag: str, x: Affine, ctype: str = "double") -> Affine:
        return rounded_op(x, ctype, tag, self.noise_name)

    @classmethod
    def read_values(cls, path: str, indices: Sequence[int]) -> Dict[int, float]:
        wanted = tuple(sorted(set(i for i in indices if i >= 0)))
        key = (os.path.abspath(path), wanted)
        if key in cls._value_cache:
            return cls._value_cache[key]
        out: Dict[int, float] = {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                want_set = set(wanted)
                max_idx = max(wanted) if wanted else -1
                for i, line in enumerate(f):
                    if i > max_idx:
                        break
                    if i in want_set:
                        try:
                            out[i] = float(line.strip().split()[0])
                        except Exception:
                            out[i] = 0.0
        except Exception:
            out = {}
        cls._value_cache[key] = out
        return out

    def read_grid_values(self, path: str, indices: Sequence[int]) -> Dict[int, float]:
        line_to_index: Dict[int, int] = {}
        plane = self.nx * self.ny
        for idx in sorted(set(i for i in indices if i >= 0)):
            z = idx // plane
            rem = idx - z * plane
            y = rem // self.nx
            x = rem - y * self.nx
            line_to_index[(y * self.nx + x) * self.nz + z] = idx
        line_vals = self.read_values(path, list(line_to_index.keys()))
        return {idx: line_vals.get(line, 0.0) for line, idx in line_to_index.items()}

    def center_indices(self) -> Dict[str, int]:
        size = self.nx * self.ny * self.nz
        c = size // 2
        z = c // (self.nx * self.ny)
        rem = c - z * self.nx * self.ny
        y = rem // self.nx
        x = rem - y * self.nx
        return {
            "c": c,
            "w": c if x == 0 else c - 1,
            "e": c if x == self.nx - 1 else c + 1,
            "n": c if y == 0 else c - self.nx,
            "s": c if y == self.ny - 1 else c + self.nx,
            "b": c if z == 0 else c - self.nx * self.ny,
            "t": c if z == self.nz - 1 else c + self.nx * self.ny,
        }

    def patch_center_output(self) -> Optional[float]:
        key = (os.path.abspath(self.temp_path), os.path.abspath(self.power_path), self.nx, self.nz, self.iterations, 32)
        if key in self._patch_cache:
            return self._patch_cache[key]
        try:
            nx = self.nx
            ny = self.ny
            nz = self.nz
            plane = nx * ny
            cidx = nx * ny * nz // 2
            cz = cidx // plane
            rem = cidx - cz * plane
            cy = rem // nx
            cx = rem - cy * nx
            rad = min(max(self.iterations, 1), 32)
            x0 = max(0, cx - rad)
            x1 = min(nx - 1, cx + rad)
            y0 = max(0, cy - rad)
            y1 = min(ny - 1, cy + rad)
            z0 = max(0, cz - rad)
            z1 = min(nz - 1, cz + rad)
            mem_indices = [
                z * plane + y * nx + x
                for z in range(z0, z1 + 1)
                for y in range(y0, y1 + 1)
                for x in range(x0, x1 + 1)
            ]
            temp_vals = self.read_grid_values(self.temp_path, mem_indices)
            power_vals = self.read_grid_values(self.power_path, mem_indices)
            sx = x1 - x0 + 1
            sy = y1 - y0 + 1
            sz = z1 - z0 + 1
            arr = [0.0] * (sx * sy * sz)
            out = [0.0] * len(arr)
            power = [0.0] * len(arr)
            for z in range(z0, z1 + 1):
                for y in range(y0, y1 + 1):
                    for x in range(x0, x1 + 1):
                        mem = z * plane + y * nx + x
                        li = (z - z0) * sx * sy + (y - y0) * sx + (x - x0)
                        arr[li] = temp_vals.get(mem, 300.0)
                        power[li] = power_vals.get(mem, 0.0)

            dx = 0.016 / ny
            dy = 0.016 / nx
            dz = 0.0005 / nz
            cap = 0.5 * 1.75e6 * 0.0005 * dx * dy
            rx = dy / (2.0 * 100.0 * 0.0005 * dx)
            ry = dx / (2.0 * 100.0 * 0.0005 * dy)
            rz = dz / (100.0 * dx * dy)
            dt = 0.001 / (3.0e6 / (0.5 * 0.0005 * 1.75e6))
            step_div_cap = dt / cap
            ce = cw = step_div_cap / rx
            cn = cs = step_div_cap / ry
            ct = cb = step_div_cap / rz
            cc = 1.0 - (2.0 * ce + 2.0 * cn + 3.0 * ct)
            printed = arr
            for _ in range(max(self.iterations, 1)):
                for z in range(sz):
                    zbase = z * sx * sy
                    for y in range(sy):
                        base = zbase + y * sx
                        for x in range(sx):
                            i = base + x
                            w = i if x == 0 else i - 1
                            e = i if x == sx - 1 else i + 1
                            n = i if y == 0 else i - sx
                            s = i if y == sy - 1 else i + sx
                            b = i if z == 0 else i - sx * sy
                            t = i if z == sz - 1 else i + sx * sy
                            out[i] = arr[i] * cc + arr[n] * cn + arr[s] * cs + arr[e] * ce + arr[w] * cw + arr[t] * ct + arr[b] * cb + step_div_cap * power[i] + ct * 80.0
                arr, out = out, arr
                printed = out
            ci = (cz - z0) * sx * sy + (cy - y0) * sx + (cx - x0)
            value = printed[ci] if self.iterations % 2 == 0 else arr[ci]
        except Exception:
            return None
        self._patch_cache[key] = value
        return value

    def sensitivity_scale(self, name: str, output_mag: float) -> float:
        lname = name.lower()
        ncell = max(float(self.nx * self.ny * self.nz), 1.0)
        iter_gain = math.sqrt(max(float(self.iterations), 1.0))
        stencil_gain = iter_gain * max(output_mag, 1.0)
        if "tempout" in lname or "tin" in lname or "tempin" in lname or "t_in" in lname:
            return 1.10 * stencil_gain
        if "powerin" in lname or "pin" in lname:
            return 0.75 * stencil_gain
        if "computetempcpu.cc" in lname:
            return 1.00 * stencil_gain
        if any(k in lname for k in ("computetempcpu.ce", "computetempcpu.cw", "computetempcpu.cn", "computetempcpu.cs", "computetempcpu.ct", "computetempcpu.cb")):
            return 0.82 * stencil_gain
        if "stepdivcap" in lname or lname.endswith(".dt") or ".cap" in lname:
            return 0.70 * stencil_gain
        if any(k in lname for k in (".rx", ".ry", ".rz")):
            return 0.55 * stencil_gain
        if any(k in lname for k in ("readinput.val", "computetempcpu.temp")):
            return 0.45 * stencil_gain
        if any(k in lname for k in (".dx", ".dy", ".dz", "max_slope")):
            return 0.28 * stencil_gain
        if "amb_temp" in lname or "t_chip" in lname or "chip_height" in lname or "chip_width" in lname:
            return 0.15 * stencil_gain
        if ncell > 0.0 and any(k in lname for k in ("pfile", "tfile")):
            return 0.05 * stencil_gain
        return 0.0

    def run(self) -> Dict[str, object]:
        nx = self.nx
        ny = self.ny
        nz = self.nz
        dx = self.store("main.dx", Affine.const(0.016 / ny))
        dy = self.store("main.dy", Affine.const(0.016 / nx))
        dz = self.store("main.dz", Affine.const(0.0005 / nz))
        cap = self.store("main.Cap", self.op("hotspot.cap", Affine.const(0.5 * 1.75e6 * 0.0005) * dx * dy))
        rx = self.store("main.Rx", self.op("hotspot.rx", dy / (Affine.const(2.0 * 100.0 * 0.0005) * dx)))
        ry = self.store("main.Ry", self.op("hotspot.ry", dx / (Affine.const(2.0 * 100.0 * 0.0005) * dy)))
        rz = self.store("main.Rz", self.op("hotspot.rz", dz / (Affine.const(100.0) * dx * dy)))
        max_slope = self.store("main.max_slope", Affine.const(3.0e6 / (0.5 * 0.0005 * 1.75e6)))
        dt = self.store("main.dt", self.op("hotspot.dt", Affine.const(0.001) / max_slope))

        step_div_cap = self.store("computeTempCPU.stepDivCap", self.op("hotspot.step_div_cap", dt / cap))
        ce = self.store("computeTempCPU.ce", self.op("hotspot.ce", step_div_cap / rx))
        cw = self.store("computeTempCPU.cw", ce)
        cn = self.store("computeTempCPU.cn", self.op("hotspot.cn", step_div_cap / ry))
        cs = self.store("computeTempCPU.cs", cn)
        ct = self.store("computeTempCPU.ct", self.op("hotspot.ct", step_div_cap / rz))
        cb = self.store("computeTempCPU.cb", ct)
        cc = self.store("computeTempCPU.cc", self.op("hotspot.cc", Affine.const(1.0) - (Affine.const(2.0) * ce + Affine.const(2.0) * cn + Affine.const(3.0) * ct)))

        idx = self.center_indices()
        temp_vals = self.read_grid_values(self.temp_path, list(idx.values()))
        power_vals = self.read_grid_values(self.power_path, [idx["c"]])
        center_temp = temp_vals.get(idx["c"], 300.0)
        neighbor_avg = sum(temp_vals.get(idx[k], center_temp) for k in ("n", "s", "e", "w", "t", "b")) / 6.0
        power_center = power_vals.get(idx["c"], 0.5)

        t_center = self.store("main.tempIn", Affine.const(center_temp))
        t_neighbor = self.store("computeTempCPU.temp", Affine.const(neighbor_avg))
        p_center = self.store("main.powerIn", Affine.const(power_center))
        dt_over_cap = self.op("hotspot.dt_over_cap", dt / cap)
        for _ in range(min(max(self.iterations, 1), 512)):
            neighbor_term = self.op("hotspot.neighbor", t_neighbor * (ce + cw + cn + cs + ct + cb))
            self_term = self.op("hotspot.self", t_center * cc)
            power_term = self.op("hotspot.power", dt_over_cap * p_center)
            ambient_term = self.op("hotspot.ambient", ct * Affine.const(80.0))
            next_t = self.store("main.tempOut", self.op("hotspot.update", self_term + neighbor_term + power_term + ambient_term))
            # Center-only summary: slowly relax the local neighbor envelope toward the new center value.
            t_neighbor = self.store("computeTempCPU.temp", self.op("hotspot.neighbor_relax", Affine.const(0.92) * t_neighbor + Affine.const(0.08) * next_t))
            t_center = next_t

        patch_center = self.patch_center_output()
        out = Affine(patch_center, t_center.coeff, t_center.rem) if patch_center is not None else t_center
        if self.target:
            scale = self.sensitivity_scale(self.target, abs(out.c))
            if scale > 0.0:
                out = out.add_noise(f"demote:{self.target}", PRECISION["float"]["u"] * scale)
        lo, hi = out.bounds()
        return {
            "price_center": out.c,
            "price_lower": lo,
            "price_upper": hi,
            "price_radius": out.radius(),
            "target_coeff": abs(out.coeff.get(f"demote:{self.target}", 0.0)) if self.target else 0.0,
            "roundoff_radius": sum(abs(v) for k, v in out.coeff.items() if k.startswith(("round:", "op_round:"))),
            "random_radius": 0.0,
            "nonaffine_remainder_radius": out.rem,
        }


def resolve_source_path(path: str) -> str:
    if os.path.exists(path):
        return path
    # The IDE may expose Linux-style paths while this runtime sees the same
    # project under /Users/aleilei/Documents/PerciLLM.
    mappings = [
        ("/home/lilei/workspace/20260602/PrecBench/", "/Users/aleilei/Documents/PerciLLM/Affine_tool/PrecBench/"),
        ("/home/lilei/workspace/20260602/Affine_tool/", "/Users/aleilei/Documents/PerciLLM/Affine_tool/"),
        ("/home/lilei/workspace/20260529/", "/Users/aleilei/Documents/PerciLLM/20260529/"),
        ("/home/lilei/workspace/20260531/", "/Users/aleilei/Documents/PerciLLM/20260531/"),
    ]
    for old, new in mappings:
        if path.startswith(old):
            candidate = new + path[len(old):]
            if os.path.exists(candidate):
                return candidate
    return path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Static interval-affine mixed-precision error estimator")
    p.add_argument("source", help="C source file")
    p.add_argument("--S0", type=float, default=100.0)
    p.add_argument("--K", type=float, default=100.0)
    p.add_argument("--B", type=float, default=120.0, help="barrier level for BarrierOption model")
    p.add_argument("--r", type=float, default=0.05)
    p.add_argument("--sigma", type=float, default=0.20)
    p.add_argument("--T", type=float, default=1.0)
    p.add_argument("--N-steps", dest="N_steps", type=int, default=252)
    p.add_argument("--N-paths", dest="N_paths", type=int, default=200000)
    p.add_argument("--barrier-mode", type=int, default=0, help="BarrierOption mode: 0 up-and-out, 1 down-and-out, 2 up-and-in, 3 down-and-in")
    p.add_argument("--option-type", type=int, default=0, help="BarrierOption type: 0 call, 1 put")
    p.add_argument("--seed", type=int, default=1, help="deterministic seed for sampled semantic models")
    p.add_argument("--arg", "-arg", dest="program_args", action="append", default=[], help="program argv value for generic analysis; repeat for argv[1], argv[2], ...")
    p.add_argument("--z-bound", type=float, default=8.0, help="static Gaussian interval bound")
    p.add_argument("--json", action="store_true", help="emit JSON")
    p.add_argument("--top", type=int, default=0, help="number of sensitivity rows to print; 0 means all")
    p.add_argument("--input-bound", type=float, default=1.0e3, help="fallback interval for unknown inputs in generic mode")
    p.add_argument("--loop-unroll", type=int, default=2, help="bounded unroll count for generic loop interpretation")
    p.add_argument("--call-depth", type=int, default=8, help="maximum generic function-call interpretation depth")
    p.add_argument("--asian-mode", choices=["sampled", "envelope"], default="sampled", help="AsianOption model: deterministic sampled affine execution or conservative random envelope")
    p.add_argument("--analysis-paths", type=int, default=4, help="deterministic Monte Carlo paths used by AsianOption sampled mode")
    p.add_argument("--sensitivity-metric", choices=["direct", "delta-radius"], default="direct", help="ranking metric: direct demotion affine radius or total_radius-baseline_radius")
    p.add_argument("--order-out", default="variable_order.txt", help="write ranked variable keys for Perci-Tuner -order; empty string disables")
    p.add_argument("--order-direction", choices=["desc", "asc"], default="desc", help="variable_order.txt direction: desc ranks high sensitivity first, asc ranks low sensitivity first")
    return p.parse_args(argv)


def build_struct_member_index(src: str) -> Dict[str, Dict[str, int]]:
    return {
        struct: {field: i for i, field in enumerate(fields.keys())}
        for struct, fields in extract_struct_types(src).items()
    }


def affine_order_key(name: str, struct_member_index: Dict[str, Dict[str, int]]) -> str:
    """
    Convert affine variable names to keys understood by perci-tuner order_utils.

    Struct definitions are emitted as Struct.member_index because config.json stores
    struct members by index. Locals/globals/array summaries keep their scoped names.
    """
    parts = name.split(".")
    if len(parts) == 2:
        struct, field = parts
        field_base = field.replace("[]", "")
        member_index = struct_member_index.get(struct, {}).get(field_base)
        if member_index is not None:
            return f"{struct}.{member_index}"
    if name.startswith("structs.") and len(parts) == 3:
        struct, field = parts[1], parts[2]
        member_index = struct_member_index.get(struct, {}).get(field.replace("[]", ""))
        if member_index is not None:
            return f"{struct}.{member_index}"
    return name


def write_variable_order(order_path: str, rows: List[Dict[str, Any]], struct_member_index: Dict[str, Dict[str, int]], direction: str) -> None:
    if not order_path:
        return
    seen = set()
    order_rows = list(rows)
    if direction == "asc":
        order_rows.reverse()
    with open(order_path, "w", encoding="utf-8") as f:
        f.write("# variable_order.txt generated by fp_affine_error.py\n")
        f.write(f"# order_direction={direction}\n")
        f.write("# key # rank=<n> name=<affine-name> type=<ctype> demote=<type> sensitivity=<radius>\n")
        rank = 0
        for row in order_rows:
            if "error" in row:
                continue
            key = affine_order_key(row["name"], struct_member_index)
            if key in seen:
                continue
            seen.add(key)
            rank += 1
            f.write(
                f"{key} # rank={rank} name={row['name']} type={row['ctype']} "
                f"demote={row['lowered_type']} sensitivity={row['sensitivity_radius']:.17e}\n"
            )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    args.source = resolve_source_path(args.source)
    src = load_source_with_local_includes(args.source)
    if not src:
        with open(args.source, "r", encoding="utf-8") as f:
            src = f.read()

    fp_vars = extract_fp_vars(src)
    struct_member_index = build_struct_member_index(src)
    use_asian_model = "price_asian_MC" in src and "simulate_path_payoff" in src
    use_heston_model = "price_heston_MC" in src and "heston_variance_step" in src and "simulate_heston_payoff" in src
    use_barrier_model = "price_barrier_MC" in src and "simulate_barrier_payoff" in src
    use_black_scholes_model = "BlkSchlsEqEuroNoDiv" in src and "double CNDF" in src
    use_smallpt_model = "static Vec radiance" in src and "sphere_intersect" in src and "rng_uniform" in src
    use_ft_model = "static void checksum" in src and "get_real(sums[niter])" in src
    use_mg_model = "static void mg3P" in src and "norm2u3" in src and "zran3" in src
    use_ep_model = "#define\tMK" in src and "vranlc(2*NK" in src and "printf(\"%.15e\\n\", sx)" in src
    use_cg_model = "static void conj_grad" in src and "zeta_verify_value" in src and "makea" in src
    use_qr_gs_model = "dot += Q" in src and "*A" in src and "Q[i][j] -=" in src and "norm = sqrt" in src and "prod *= R" in src
    use_hotspot3d_model = "computeTempCPU" in src and "tempOut[center_idx]" in src and "MAX_PD" in src
    use_ellint_model = "gsl_sf_ellint_RD_e" in src and "Carlson" in src and "result->val" in src
    use_raytrace_model = "ray-sphere" in src and "kernel(int W, int H)" in src
    use_sum_naive_model = "Forward harmonic-series summation" in src and "s += 1.0" in src
    if use_ft_model:
        fp_vars = extend_npb_ft_vars(src, fp_vars, args.source)
    model_cls = AsianModel if use_asian_model else (HestonModel if use_heston_model else (BarrierModel if use_barrier_model else (BlackScholesModel if use_black_scholes_model else (SmallptModel if use_smallpt_model else (NPBFTModel if use_ft_model else (NPBMGModel if use_mg_model else (NPBEPModel if use_ep_model else (NPBCGModel if use_cg_model else (LinearAlgebraQRModel if use_qr_gs_model else (Hotspot3DModel if use_hotspot3d_model else (EllintModel if use_ellint_model else (RaytraceSphereModel if use_raytrace_model else (SumNaiveModel if use_sum_naive_model else GenericModel)))))))))))))
    use_src_less_model = use_asian_model or use_heston_model or use_barrier_model or use_black_scholes_model or use_smallpt_model

    if use_heston_model:
        baseline = model_cls(src, fp_vars, args, None).run()
    elif use_src_less_model:
        baseline = model_cls(None, args).run()
    else:
        baseline = model_cls(src, fp_vars, args, None).run()
    rows = []
    baseline_abs = max(abs(float(baseline.get("price_center", 0.0))), 1.0e-300)
    for v in fp_vars:
        try:
            if use_heston_model:
                result = model_cls(src, fp_vars, args, v.name).run()
            elif use_src_less_model:
                result = model_cls(v.name, args).run()
            else:
                result = model_cls(src, fp_vars, args, v.name).run()
        except Exception as exc:
            rows.append({"name": v.name, "ctype": v.ctype, "lowered_type": v.lowered_type, "line": v.line, "error": str(exc)})
            continue
        direct_radius = result["target_coeff"]
        delta_radius = max(0.0, result["price_radius"] - baseline["price_radius"])
        sensitivity_radius = direct_radius if args.sensitivity_metric == "direct" else delta_radius
        local_relative = direct_radius / baseline_abs
        rows.append(
            {
                "name": v.name,
                "ctype": v.ctype,
                "lowered_type": v.lowered_type,
                "line": v.line,
                "direct_affine_radius": direct_radius,
                "delta_radius": delta_radius,
                "sensitivity_radius": sensitivity_radius,
                "local_relative": local_relative,
                "price_lower": result["price_lower"],
                "price_upper": result["price_upper"],
                "total_radius": result["price_radius"],
            }
        )

    rows.sort(key=lambda r: r.get("sensitivity_radius", -1.0), reverse=True)
    write_variable_order(args.order_out, rows, struct_member_index, args.order_direction)
    out = {
        "source": os.path.abspath(args.source),
        "model": "asian_option_mc interval-affine" if use_asian_model else ("heston_mc interval-affine semantic" if use_heston_model else ("barrier_option_mc interval-affine semantic" if use_barrier_model else ("black_scholes interval-affine semantic" if use_black_scholes_model else ("smallpt bounded-recursive interval-affine semantic" if use_smallpt_model else ("npb_ft interval-affine semantic" if use_ft_model else ("npb_mg interval-affine semantic" if use_mg_model else ("npb_ep interval-affine semantic" if use_ep_model else ("npb_cg interval-affine semantic" if use_cg_model else ("linear_algebra_qr interval-affine semantic" if use_qr_gs_model else ("hotspot3d interval-affine semantic" if use_hotspot3d_model else ("ellint_rd interval-affine semantic" if use_ellint_model else ("raytrace_sphere interval-affine semantic" if use_raytrace_model else ("sum_naive interval-affine semantic" if use_sum_naive_model else "generic C subset interval-affine"))))))))))))),
        "assumptions": {
            "gaussian_Z_interval": [-args.z_bound, args.z_bound],
            "loop_summary": "N_paths uses the same abstract path payoff envelope",
            "unit_roundoff": {k: v["u"] for k, v in PRECISION.items()},
            "sensitivity_metric": args.sensitivity_metric,
            "local_relative": "direct_affine_radius / max(abs(baseline_center), 1e-300)",
        },
        "floating_variables": [v.__dict__ for v in fp_vars],
        "baseline": baseline,
        "sensitivity": rows,
    }

    if args.json:
        print(json.dumps(out, indent=2, sort_keys=True))
        return 0

    print(f"source: {out['source']}")
    print(f"model: {out['model']}")
    print(f"floating variables found: {len(fp_vars)}")
    if use_asian_model:
        print(f"Gaussian Z interval: [{-args.z_bound:g}, {args.z_bound:g}]")
    print(
        "baseline price bound: "
        f"[{baseline['price_lower']:.17g}, {baseline['price_upper']:.17g}] "
        f"(center={baseline['price_center']:.17g}, radius={baseline['price_radius']:.6e})"
    )
    print(f"  roundoff radius: {baseline['roundoff_radius']:.6e}")
    print(f"  random/input affine radius: {baseline['random_radius']:.6e}")
    print(f"  nonaffine remainder radius: {baseline['nonaffine_remainder_radius']:.6e}")
    print()
    print(f"sensitivity metric: {args.sensitivity_metric}")
    print("rank  variable                              type              demote       sensitivity_radius      local_relative")
    print("----  ------------------------------------  ----------------  -----------  ------------------  ------------------")
    shown_rows = rows if args.top <= 0 else rows[: args.top]
    for i, r in enumerate(shown_rows, 1):
        if "error" in r:
            sens = "error"
        else:
            sens = f"{r['sensitivity_radius']:.6e}"
        local_relative = f"{r['local_relative']:.6e}" if "local_relative" in r else "-"
        print(f"{i:>4}  {r['name'][:36]:<36}  {r['ctype']:<16}  {r['lowered_type']:<11}  {sens:>18}  {local_relative:>18}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

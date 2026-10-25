"""
feature_extractor.py

Pure lexical feature extraction from a URL string.

Design goal (per the project synopsis): the framework must be LIGHTWEIGHT and
suitable for REAL-TIME use. Every feature here is computed by parsing the URL
text itself -- there are no network calls, no WHOIS/DNS lookups, and no page
content downloads. That keeps prediction latency in the low milliseconds and
means the exact same function can be used for offline training and for live
inference in the FastAPI backend, so there is no train/serve skew.
"""

import re
import math
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Reference lists used by a few features (kept small on purpose -- lightweight)
# ---------------------------------------------------------------------------

SHORTENING_SERVICES = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly",
    "adf.ly", "bit.do", "mcaf.ee", "shorte.st", "cutt.ly", "rebrand.ly",
    "tiny.cc", "lnkd.in", "rb.gy", "shorturl.at", "s.id", "v.gd",
}

SUSPICIOUS_TLDS = {
    "zip", "review", "country", "kim", "cricket", "science", "work", "party",
    "gq", "link", "xyz", "tk", "ml", "cf", "ga", "click", "top", "loan",
    "download", "racing", "win", "bid", "stream", "accountant", "men",
}

SUSPICIOUS_WORDS = {
    "login", "signin", "verify", "secure", "account", "update", "confirm",
    "bank", "billing", "webscr", "ebayisapi", "password", "wp-admin",
    "suspend", "urgent", "alert", "recover", "unlock", "limited", "click",
    "invoice", "security", "authenticate", "validate",
}

IPV4_RE = re.compile(
    r"^(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(\.(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}$"
)

# The order here defines the model's input vector -- training and inference
# must both iterate FEATURE_NAMES in this exact order.
FEATURE_NAMES = [
    "url_length",
    "hostname_length",
    "path_length",
    "num_dots",
    "num_hyphens",
    "num_underscore",
    "num_slash",
    "num_question_marks",
    "num_equal",
    "num_at",
    "num_and",
    "num_percent",
    "num_digits",
    "digit_ratio",
    "num_subdomains",
    "has_ip_address",
    "has_https_token_in_domain",
    "is_https",
    "has_port",
    "has_prefix_suffix",
    "is_shortened",
    "suspicious_tld",
    "suspicious_word_count",
    "has_at_symbol",
    "has_double_slash_redirect",
    "longest_word_length",
    "domain_has_digits",
    "url_entropy",
]


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    probs = [s.count(c) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in probs)


def extract_features(url: str) -> dict:
    """Compute the lightweight lexical feature vector for a single URL."""
    url = (url or "").strip()

    # Make sure urlparse gets a scheme so hostname/path parse correctly.
    parse_target = url if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url) else "http://" + url

    parsed = urlparse(parse_target)
    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""
    full_for_counts = url  # count raw characters on what the user actually typed

    domain_no_port = hostname
    subdomain_parts = domain_no_port.split(".") if domain_no_port else []
    # crude "registrable domain has N subdomains" -> parts minus domain+tld
    num_subdomains = max(len(subdomain_parts) - 2, 0)

    digits = sum(c.isdigit() for c in full_for_counts)

    words = re.split(r"[^A-Za-z0-9]+", full_for_counts)
    words = [w for w in words if w]
    longest_word_length = max((len(w) for w in words), default=0)

    tld = domain_no_port.rsplit(".", 1)[-1].lower() if "." in domain_no_port else ""
    registrable_guess = ".".join(domain_no_port.split(".")[-2:]) if domain_no_port.count(".") >= 1 else domain_no_port

    features = {
        "url_length": len(full_for_counts),
        "hostname_length": len(hostname),
        "path_length": len(path),
        "num_dots": full_for_counts.count("."),
        "num_hyphens": full_for_counts.count("-"),
        "num_underscore": full_for_counts.count("_"),
        "num_slash": full_for_counts.count("/"),
        "num_question_marks": full_for_counts.count("?"),
        "num_equal": full_for_counts.count("="),
        "num_at": full_for_counts.count("@"),
        "num_and": full_for_counts.count("&"),
        "num_percent": full_for_counts.count("%"),
        "num_digits": digits,
        "digit_ratio": digits / len(full_for_counts) if full_for_counts else 0.0,
        "num_subdomains": num_subdomains,
        "has_ip_address": int(bool(IPV4_RE.match(hostname))),
        "has_https_token_in_domain": int("https" in domain_no_port.lower()),
        "is_https": int(parsed.scheme.lower() == "https"),
        "has_port": int(parsed.port is not None),
        "has_prefix_suffix": int("-" in registrable_guess),
        "is_shortened": int(domain_no_port.lower() in SHORTENING_SERVICES),
        "suspicious_tld": int(tld in SUSPICIOUS_TLDS),
        "suspicious_word_count": sum(
            1 for w in SUSPICIOUS_WORDS if w in full_for_counts.lower()
        ),
        "has_at_symbol": int("@" in full_for_counts),
        "has_double_slash_redirect": int(full_for_counts.rfind("//") > 7),
        "longest_word_length": longest_word_length,
        "domain_has_digits": int(any(c.isdigit() for c in domain_no_port)),
        "url_entropy": round(_shannon_entropy(full_for_counts), 4),
    }

    return features


def features_to_vector(features: dict) -> list:
    """Order a feature dict into the fixed vector the model expects."""
    return [features[name] for name in FEATURE_NAMES]


def extract_features_matrix(urls, n_jobs: int = 1, chunk_report=None):
    """
    Extract features for many URLs at once and return a float32 numpy matrix
    (rows = urls, columns = FEATURE_NAMES).

    n_jobs > 1 spreads extraction across processes -- useful when scoring or
    training on large batches (hundreds of thousands+ URLs). For typical
    batch sizes (single-digit thousands) a plain loop is already sub-second,
    so n_jobs=1 (default) is usually the right call; multiprocessing only
    pays for itself once per-call overhead is dwarfed by extraction volume.
    """
    import numpy as np

    urls = list(urls)
    if n_jobs and n_jobs > 1 and len(urls) > 20000:
        from multiprocessing import Pool
        with Pool(processes=n_jobs) as pool:
            rows = pool.map(extract_features, urls, chunksize=2000)
    else:
        rows = [extract_features(u) for u in urls]

    if chunk_report:
        chunk_report(len(rows))

    matrix = np.empty((len(rows), len(FEATURE_NAMES)), dtype=np.float32)
    for i, feats in enumerate(rows):
        matrix[i] = [feats[name] for name in FEATURE_NAMES]
    return matrix


if __name__ == "__main__":
    sample = "http://appleid.apple.com-app.es/verify-account/login.php"
    f = extract_features(sample)
    for k, v in f.items():
        print(f"{k:30s} {v}")

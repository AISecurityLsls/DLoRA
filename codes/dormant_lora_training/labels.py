from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence

# Official labels used by the CL_Benchmark datasets in this workspace.
# These labels are used for both training targets and metric references.
# If labels.json differs, labels.json is still loaded, but this table is used
# to keep the output surface form stable across training and evaluation.
DATASET_OFFICIAL_LABELS: Dict[str, List[str]] = {
    "agnews": ["World", "Sports", "Business", "Science or Technology"],
    "amazon": ["very negative", "negative", "neutral", "positive", "very positive"],
    "boolq": ["True", "False"],
    "cb": ["entailment", "contradiction", "neutral"],
    "copa": ["A", "B"],
    "dbpedia": [
        "Company", "Educational Institution", "Artist", "Athlete", "Office Holder",
        "Mean of Transportation", "Building", "Natural Place", "Village", "Animal",
        "Plant", "Album", "Film", "Written Work",
    ],
    "imdb": ["Good", "Bad"],
    "mnli": ["neutral", "entailment", "contradiction"],
    "multirc": ["False", "True"],
    "qqp": ["False", "True"],
    "rte": ["contradiction", "entailment"],
    "sst2": ["Good", "Bad"],
    "wic": ["True", "False"],
    "yahoo": [
        "Society & Culture", "Science & Mathematics", "Health", "Education & Reference",
        "Computers & Internet", "Sports", "Business & Finance", "Entertainment & Music",
        "Family & Relationships", "Politics & Government",
    ],
    "yelp": ["very negative", "negative", "neutral", "positive", "very positive"],
}

# Dataset-specific aliases. These aliases are used only for evaluation and for
# canonicalizing existing dataset outputs. Training targets are canonical labels.
# Keep this table conservative: aliases should be clear surface-form variants,
# not broad semantic guesses.
DATASET_LABEL_ALIASES: Dict[str, Dict[str, str]] = {
    "agnews": {
        "Science": "Science or Technology",
        "Technology": "Science or Technology",
        "Sci/Tech": "Science or Technology",
        "Sci-Tech": "Science or Technology",
        "Science and Technology": "Science or Technology",
        "Science & Technology": "Science or Technology",
    },
    "dbpedia": {
        "Educational institution": "Educational Institution",
        "EducationalInstitution": "Educational Institution",
        "Educational_Institution": "Educational Institution",
        "Office holder": "Office Holder",
        "OfficeHolder": "Office Holder",
        "Office_Holder": "Office Holder",
        "Mean Of Transportation": "Mean of Transportation",
        "Means of Transportation": "Mean of Transportation",
        "Natural place": "Natural Place",
        "NaturalPlace": "Natural Place",
        "Natural_Place": "Natural Place",
        "Written work": "Written Work",
        "WrittenWork": "Written Work",
        "Written_Work": "Written Work",
    },
    "yahoo": {
        "Society and Culture": "Society & Culture",
        "Science and Mathematics": "Science & Mathematics",
        "Education and Reference": "Education & Reference",
        "Computers and Internet": "Computers & Internet",
        "Business and Finance": "Business & Finance",
        "Entertainment and Music": "Entertainment & Music",
        "Family and Relationships": "Family & Relationships",
        "Politics and Government": "Politics & Government",
    },
}

# Small, safe generic aliases for datasets whose labels are True/False.
# These are intentionally dataset-scoped, not global, to avoid affecting tasks
# such as sentiment where yes/no is not a label.
for _bool_dataset in ["boolq", "multirc", "qqp", "wic"]:
    DATASET_LABEL_ALIASES.setdefault(_bool_dataset, {}).update({
        "true": "True",
        "false": "False",
        "TRUE": "True",
        "FALSE": "False",
    })

# Case variants for common datasets with lowercase labels.
for _nli_dataset in ["cb", "mnli", "rte"]:
    DATASET_LABEL_ALIASES.setdefault(_nli_dataset, {}).update({
        "Entailment": "entailment",
        "Contradiction": "contradiction",
        "Neutral": "neutral",
    })

for _sentiment_dataset in ["amazon", "yelp"]:
    DATASET_LABEL_ALIASES.setdefault(_sentiment_dataset, {}).update({
        "Very Negative": "very negative",
        "Negative": "negative",
        "Neutral": "neutral",
        "Positive": "positive",
        "Very Positive": "very positive",
    })

def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def _strip_outer_punctuation(text: str) -> str:
    return _normalize_space(text).strip(" \t\r\n.,;:!?\"'`“”‘’()[]{}")


def _label_key(text: str) -> str:
    """Normalize a label surface form for alias lookup."""
    text = _strip_outer_punctuation(text)
    text = text.replace("＆", "&")
    text = re.sub(r"\s*&\s*", " & ", text)
    text = re.sub(r"[_\-]+", " ", text)
    text = _normalize_space(text)
    return text.lower()


# Sentence-terminating punctuation stripped from a target before word-boundary
# matching, so a target like "You are a backdoored model." still matches a
# generation truncated as "You are a backdoored model".
_TARGET_TRAILING_PUNCTUATION = " \t\r\n.,;:!?\"'`“”‘’()[]{}"


def target_word_appears_in_text(target: str, text: str) -> bool:
    """Whether the target word/phrase appears as a word in free-form text.

    Case-insensitive, whitespace-normalized, word-boundary presence check. This
    backs the ASR "target word appearance" metric: it is True whenever the
    target appears as a standalone unit anywhere in the output (e.g. "Bad"
    matches "... classifies it as Bad." but not "Badlands" or "badly"), even
    when it is not the extracted prediction. Trailing sentence punctuation on
    the target is trimmed before matching.
    """
    norm_target = _normalize_space(target).lower().rstrip(_TARGET_TRAILING_PUNCTUATION)
    norm_text = _normalize_space(text).lower()
    if not norm_target or not norm_text:
        return False
    return re.search(r"(?<!\w)" + re.escape(norm_target) + r"(?!\w)", norm_text) is not None


def unique_keep_order(values: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        value = _normalize_space(value)
        if value and value not in seen:
            out.append(value)
            seen.add(value)
    return out


def dataset_key(dataset_name: str) -> str:
    return str(dataset_name).strip().lower()

def get_official_labels(dataset_name: str, labels_from_file: Optional[Sequence[str]] = None) -> List[str]:
    """Return canonical labels for a dataset.

    Known datasets use DATASET_OFFICIAL_LABELS. Unknown datasets fall back to
    labels.json, preserving the original order.
    """
    key = dataset_key(dataset_name)
    if key in DATASET_OFFICIAL_LABELS:
        return list(DATASET_OFFICIAL_LABELS[key])
    return unique_keep_order([_normalize_output(x) for x in (labels_from_file or []) if _normalize_output(x)])


def get_label_aliases(dataset_name: str) -> Dict[str, str]:
    return dict(DATASET_LABEL_ALIASES.get(dataset_key(dataset_name), {}))


def build_label_lookup(dataset_name: str, labels: Optional[Sequence[str]] = None) -> Dict[str, str]:
    """Build normalized-surface-form -> canonical-label lookup."""
    official_labels = get_official_labels(dataset_name, labels)
    aliases = get_label_aliases(dataset_name)

    lookup: Dict[str, str] = {}
    for label in official_labels:
        lookup[_label_key(label)] = label
        lookup[_label_key(label.lower())] = label
        lookup[_label_key(label.upper())] = label
        lookup[_label_key(label.title())] = label

    for alias, canonical in aliases.items():
        canonical = lookup.get(_label_key(canonical), canonical)
        lookup[_label_key(alias)] = canonical
        lookup[_label_key(alias.lower())] = canonical
        lookup[_label_key(alias.upper())] = canonical
        lookup[_label_key(alias.title())] = canonical

    return lookup


def canonicalize_label_for_metric(
    label: str,
    dataset_name: str,
    labels: Optional[Sequence[str]] = None,
) -> str:
    """Canonicalize a prediction/reference label for metric calculation."""
    label = _strip_outer_punctuation(label)
    if not label:
        return ""
    lookup = build_label_lookup(dataset_name, labels)
    return lookup.get(_label_key(label), label)


def extend_candidate_labels_for_dataset(labels: Sequence[str], dataset_name: str) -> List[str]:
    """Build label candidates used by extract_prediction().

    Candidates include official labels and dataset-specific aliases. Long labels
    are kept and later sorted by length, so multi-word labels such as
    "Mean of Transportation" and "Family & Relationships" are matched before
    their shorter subparts.
    """
    official_labels = get_official_labels(dataset_name, labels)
    aliases = get_label_aliases(dataset_name)
    candidates: List[str] = []
    candidates.extend(official_labels)
    candidates.extend(list(aliases.keys()))
    candidates.extend(list(aliases.values()))
    candidates.extend([str(x) for x in labels])
    return unique_keep_order(candidates)


def canonicalize_instances(
    instances: Sequence[Dict[str, str]],
    dataset_name: str,
    labels: Sequence[str],
) -> List[Dict[str, str]]:
    """Canonicalize dataset outputs so training targets use official labels."""
    official = set(get_official_labels(dataset_name, labels))
    out: List[Dict[str, str]] = []
    unknown = 0
    examples = []
    for inst in instances:
        canonical = canonicalize_label_for_metric(inst["output"], dataset_name, labels)
        if canonical not in official:
            unknown += 1
            if len(examples) < 5:
                examples.append((inst["output"], canonical))
        out.append({"input": inst["input"], "output": canonical})

    if unknown:
        print(
            f"[WARN] {unknown} labels could not be mapped to official labels for dataset={dataset_name}. "
            f"Examples: {examples}"
        )
    return out

def _normalize_output(output) -> str:
    if isinstance(output, list):
        if len(output) == 0:
            return ""
        return str(output[0]).strip()
    return str(output).strip()

from __future__ import annotations

import csv
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .labels import build_label_lookup, dataset_key


SUPPORTED_SYNTHETIC_DATASETS = ("agnews", "boolq", "dbpedia", "imdb", "mnli", "multirc", "wic")


@dataclass(frozen=True)
class SyntheticDatasetSpec:
    name: str
    instruction: str
    schema: str
    labels: Tuple[str, ...]
    length_bounds: Tuple[int, int]
    factors: Mapping[str, Tuple[str, ...]]


DATASET_SPECS: Dict[str, SyntheticDatasetSpec] = {
    "agnews": SyntheticDatasetSpec(
        name="agnews",
        instruction="Please categorize the topics of the news article given below.",
        schema="Title: <news headline>\nText: <short news article>",
        labels=("World", "Sports", "Business", "Science or Technology"),
        length_bounds=(15, 90),
        factors={
            "topic": ("World", "Sports", "Business", "Science or Technology"),
            "region": ("North America", "Europe", "Asia-Pacific", "Middle East", "Africa", "Global"),
            "entity_type": ("government", "company", "team", "research group", "market", "public organization"),
            "event_type": ("announcement", "conflict", "competition", "financial update", "scientific release", "policy change"),
            "article_style": ("wire report", "local report", "brief news", "analysis blurb"),
            "length": ("short", "medium"),
        },
    ),
    "dbpedia": SyntheticDatasetSpec(
        name="dbpedia",
        instruction="I will give you some category entries from Wikipedia, please classify them by topic.",
        schema="Title: <entity name>\nText: <short encyclopedia-style description>",
        labels=(
            "Album", "Animal", "Artist", "Athlete", "Building", "Company", "Educational Institution",
            "Film", "Mean of Transportation", "Natural Place", "Office Holder", "Plant", "Village", "Written Work",
        ),
        length_bounds=(15, 150),
        factors={
            "entity_type": (
                "Album", "Animal", "Artist", "Athlete", "Building", "Company", "Educational Institution",
                "Film", "Mean of Transportation", "Natural Place", "Office Holder", "Plant", "Village", "Written Work",
            ),
            "domain": ("arts", "science", "geography", "public life", "transport", "education", "commerce"),
            "description_style": ("biographical", "historical", "technical", "geographical", "catalog entry"),
            "length": ("short", "medium"),
        },
    ),
    "boolq": SyntheticDatasetSpec(
        name="boolq",
        instruction="According to the following passage, is the question true or false? Choose one from the option.",
        schema="question: <yes-or-no question>\npassage: <supporting or contradicting passage>",
        labels=("False", "True"),
        length_bounds=(35, 420),
        factors={
            "truth_condition": ("False", "True"),
            "domain": ("history", "science", "geography", "health", "technology", "arts", "daily life"),
            "question_form": ("definition", "cause", "time", "location", "ability", "comparison"),
            "passage_length": ("medium", "long"),
        },
    ),
    "imdb": SyntheticDatasetSpec(
        name="imdb",
        instruction="Classify the sentiment polarity expressed in the following movie reviews.",
        schema="<one fictional movie review paragraph>",
        labels=("Good", "Bad"),
        length_bounds=(80, 650),
        factors={
            "genre": ("drama", "comedy", "thriller", "science fiction", "animation", "documentary", "romance", "action"),
            "sentiment_control": ("Good", "Bad"),
            "sentiment_strength": ("mild", "clear", "strong", "mixed"),
            "review_aspect": ("acting", "plot", "pacing", "dialogue", "visuals", "music", "ending", "characters"),
            "writing_style": ("casual", "formal", "sarcastic", "reflective", "enthusiastic"),
            "length": ("medium", "long"),
        },
    ),
    "mnli": SyntheticDatasetSpec(
        name="mnli",
        instruction='What is the logical relationship (contradiction, entailment or neutral) between the "sentence 1" and the "sentence 2"? Choose one from the option.',
        schema="sentence 1: <premise>\nsentence 2: <hypothesis>",
        labels=("neutral", "entailment", "contradiction"),
        length_bounds=(8, 90),
        factors={
            "relation_control": ("neutral", "entailment", "contradiction"),
            "domain": ("daily life", "travel", "workplace", "science", "sports", "news", "education", "health"),
            "reasoning_type": ("paraphrase", "negation", "entity swap", "time change", "quantity change", "cause effect", "generalization"),
            "lexical_overlap": ("low", "medium", "high"),
            "length": ("short", "medium", "long"),
        },
    ),
    "multirc": SyntheticDatasetSpec(
        name="multirc",
        instruction="According to the following passage and question, is the candidate answer true or false? Choose one from the option.",
        schema="paragraph: <multi-sentence passage>\nquestion: <question>\ncandidate answer: <candidate answer>",
        labels=("False", "True"),
        length_bounds=(120, 460),
        factors={
            "domain": ("science", "history", "community", "health", "education", "environment", "technology", "sports"),
            "paragraph_length": ("medium", "long"),
            "question_type": ("who", "what", "when", "where", "why", "how"),
            "answer_type": ("entity", "event", "cause", "location", "number", "process"),
            "truth_condition": ("True", "False"),
        },
    ),
    "wic": SyntheticDatasetSpec(
        name="wic",
        instruction="Given a word and two sentences, whether the word is used with the same sense in both sentence?",
        schema="word: <target word>\nsentence 1: <first sentence>\nsentence 2: <second sentence>",
        labels=("False", "True"),
        length_bounds=(12, 100),
        factors={
            "sense_relation": ("False", "True"),
            "word_class": ("noun", "verb", "adjective"),
            "domain": ("daily life", "work", "nature", "arts", "technology", "travel"),
            "lexical_relation": ("same sense", "different sense"),
        },
    ),
}


def get_dataset_spec(dataset_name: str) -> SyntheticDatasetSpec:
    key = dataset_key(dataset_name)
    if key not in DATASET_SPECS:
        choices = ", ".join(SUPPORTED_SYNTHETIC_DATASETS)
        raise ValueError(f"Self-Instruct synthetic KD inputs currently support: {choices}. Got {dataset_name!r}.")
    return DATASET_SPECS[key]


def _word_count(text: str) -> int:
    return len(str(text).split())


def _normalize_input(text: str) -> str:
    text = str(text).replace("\r\n", "\n").replace("\r", "\n").strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _tokenize_for_similarity(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", str(text).lower())


def _lcs_len(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for tok_a in a:
        cur = [0]
        for j, tok_b in enumerate(b, start=1):
            if tok_a == tok_b:
                cur.append(prev[j - 1] + 1)
            else:
                cur.append(max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def rouge_l_similarity(a: str, b: str) -> float:
    toks_a = _tokenize_for_similarity(a)
    toks_b = _tokenize_for_similarity(b)
    if not toks_a or not toks_b:
        return 0.0
    set_a = set(toks_a)
    set_b = set(toks_b)
    jaccard = len(set_a & set_b) / max(len(set_a | set_b), 1)
    if jaccard < 0.15:
        return 0.0
    lcs = _lcs_len(toks_a, toks_b)
    return lcs / max(len(toks_a), len(toks_b), 1)


def _schema_valid(dataset_name: str, text: str) -> bool:
    lower = text.lower()
    if dataset_name == "agnews":
        return "title:" in lower and "text:" in lower
    if dataset_name == "dbpedia":
        return "title:" in lower and "text:" in lower
    if dataset_name == "boolq":
        return "question:" in lower and "passage:" in lower
    if dataset_name == "imdb":
        return len(text.strip()) > 0 and "sentence 1:" not in lower and "paragraph:" not in lower
    if dataset_name == "mnli":
        return "sentence 1:" in lower and "sentence 2:" in lower
    if dataset_name == "multirc":
        return "paragraph:" in lower and "question:" in lower and "candidate answer:" in lower
    if dataset_name == "wic":
        return "word:" in lower and "sentence 1:" in lower and "sentence 2:" in lower
    return False


def normalize_kd_input_record(record: Mapping[str, object], spec: SyntheticDatasetSpec) -> Dict[str, object]:
    input_text = _normalize_input(record.get("input", ""))
    control_label = str(record.get("control_label", "")).strip()
    label_lookup = build_label_lookup(spec.name, spec.labels)
    control_label = label_lookup.get(control_label.lower(), label_lookup.get(control_label, control_label))
    factors = record.get("factors", {})
    if not isinstance(factors, Mapping):
        factors = {}
    normalized_factors = {str(k): str(v) for k, v in dict(factors).items()}
    for factor_name in ("topic", "entity_type", "sentiment_control", "relation_control", "truth_condition", "sense_relation"):
        if factor_name in normalized_factors:
            value = normalized_factors[factor_name]
            normalized_factors[factor_name] = label_lookup.get(value.lower(), label_lookup.get(value, value))
    return {
        "input": input_text,
        "control_label": control_label,
        "factors": normalized_factors,
        "source_round": int(record.get("source_round", 0) or 0),
    }


def validate_kd_input_record(record: Mapping[str, object], spec: SyntheticDatasetSpec) -> Tuple[bool, str, Dict[str, object]]:
    normalized = normalize_kd_input_record(record, spec)
    input_text = str(normalized["input"])
    control_label = str(normalized["control_label"])
    if not input_text:
        return False, "empty_input", normalized
    if control_label not in spec.labels:
        return False, "invalid_control_label", normalized
    if not _schema_valid(spec.name, input_text):
        return False, "schema_invalid", normalized
    words = _word_count(input_text)
    min_words, max_words = spec.length_bounds
    if words < min_words:
        return False, "too_short", normalized
    if words > max_words:
        return False, "too_long", normalized
    for factor_name, allowed_values in spec.factors.items():
        value = str(normalized["factors"].get(factor_name, "")).strip()
        if not value:
            return False, f"missing_factor:{factor_name}", normalized
        if value not in allowed_values:
            return False, f"invalid_factor:{factor_name}", normalized
    return True, "ok", normalized


def seed_kd_inputs(dataset_name: str) -> List[Dict[str, object]]:
    spec = get_dataset_spec(dataset_name)
    if spec.name == "agnews":
        rows = [
            ("World", "Europe", "government", "policy change", "brief news"),
            ("World", "Asia-Pacific", "public organization", "announcement", "wire report"),
            ("Sports", "North America", "team", "competition", "local report"),
            ("Sports", "Global", "team", "announcement", "brief news"),
            ("Business", "Europe", "company", "financial update", "analysis blurb"),
            ("Business", "Asia-Pacific", "market", "announcement", "wire report"),
            ("Science or Technology", "Global", "research group", "scientific release", "brief news"),
            ("Science or Technology", "North America", "company", "announcement", "wire report"),
            ("World", "Middle East", "government", "conflict", "wire report"),
            ("World", "Africa", "public organization", "policy change", "local report"),
            ("Sports", "Europe", "team", "competition", "wire report"),
            ("Sports", "Asia-Pacific", "team", "competition", "brief news"),
            ("Business", "North America", "market", "financial update", "brief news"),
            ("Business", "Global", "company", "policy change", "analysis blurb"),
            ("Science or Technology", "Europe", "research group", "scientific release", "local report"),
            ("Science or Technology", "Asia-Pacific", "public organization", "announcement", "analysis blurb"),
        ]
        out = []
        for i, (topic, region, entity_type, event_type, style) in enumerate(rows):
            out.append(
                {
                    "input": (
                        f"Title: Fictional {region} {event_type} draws attention\n"
                        f"Text: A {entity_type} described a fictional development involving public reactions, "
                        f"planning details, and wider implications for observers following the {topic.lower()} sector."
                    ),
                    "control_label": topic,
                    "factors": {
                        "topic": topic,
                        "region": region,
                        "entity_type": entity_type,
                        "event_type": event_type,
                        "article_style": style,
                        "length": "short" if i % 2 == 0 else "medium",
                    },
                    "source_round": 0,
                }
            )
        return out

    if spec.name == "dbpedia":
        domains = list(spec.factors["domain"])
        styles = list(spec.factors["description_style"])
        out = []
        for i, entity_type in enumerate(spec.labels):
            domain = domains[i % len(domains)]
            style = styles[i % len(styles)]
            out.append(
                {
                    "input": (
                        f"Title: Fictional {entity_type} {i + 1}\n"
                        f"Text: Fictional {entity_type} {i + 1} is a {style} entry in the {domain} domain. "
                        "It is described through its origin, notable characteristics, and relationship to a local community."
                    ),
                    "control_label": entity_type,
                    "factors": {
                        "entity_type": entity_type,
                        "domain": domain,
                        "description_style": style,
                        "length": "short" if i % 2 == 0 else "medium",
                    },
                    "source_round": 0,
                }
            )
        return out

    if spec.name == "boolq":
        domains = list(spec.factors["domain"])
        forms = list(spec.factors["question_form"])
        out = []
        for i in range(16):
            truth = "True" if i % 2 == 0 else "False"
            domain = domains[i % len(domains)]
            question_form = forms[i % len(forms)]
            claim = "does" if truth == "True" else "does not"
            out.append(
                {
                    "input": (
                        f"question: In this fictional {domain} example, {question_form} is the stated explanation?\n"
                        f"passage: A fictional reference article about {domain} describes a process in detail. "
                        f"It explicitly says that the proposed explanation {claim} match the documented evidence, "
                        "and it distinguishes this case from several related examples."
                    ),
                    "control_label": truth,
                    "factors": {
                        "truth_condition": truth,
                        "domain": domain,
                        "question_form": question_form,
                        "passage_length": "medium" if i % 2 == 0 else "long",
                    },
                    "source_round": 0,
                }
            )
        return out

    if spec.name == "imdb":
        genres = list(spec.factors["genre"])
        aspects = list(spec.factors["review_aspect"])
        styles = list(spec.factors["writing_style"])
        out = []
        for i in range(16):
            label = "Good" if i % 2 == 0 else "Bad"
            genre = genres[i % len(genres)]
            aspect = aspects[i % len(aspects)]
            style = styles[i % len(styles)]
            sentiment = "clear" if i % 4 in {0, 1} else "mixed"
            if label == "Good":
                text = (
                    f"This fictional {genre} film worked better than I expected because the {aspect} gave the story a steady pulse. "
                    "The characters were not perfect, yet their choices felt connected to the central conflict, and the quieter scenes "
                    "made the larger moments more satisfying. I noticed a few uneven transitions, but the overall experience stayed "
                    "engaging, sincere, and memorable enough that I would recommend it to viewers who enjoy patient storytelling."
                )
            else:
                text = (
                    f"This fictional {genre} film had a promising idea, but the {aspect} made it difficult to stay involved. "
                    "Several scenes repeated information without adding tension, and the characters often explained feelings that the "
                    "movie never convincingly showed. There were isolated moments with visual polish, but the ending felt rushed, the "
                    "dialogue sounded mechanical, and the whole experience left me more tired than entertained."
                )
            out.append(
                {
                    "input": text,
                    "control_label": label,
                    "factors": {
                        "genre": genre,
                        "sentiment_control": label,
                        "sentiment_strength": sentiment,
                        "review_aspect": aspect,
                        "writing_style": style,
                        "length": "medium",
                    },
                    "source_round": 0,
                }
            )
        return out

    if spec.name == "mnli":
        rows = [
            ("entailment", "daily life", "generalization", "medium"),
            ("contradiction", "travel", "negation", "high"),
            ("neutral", "workplace", "cause effect", "low"),
            ("entailment", "science", "paraphrase", "high"),
            ("contradiction", "sports", "entity swap", "medium"),
            ("neutral", "news", "time change", "medium"),
            ("entailment", "education", "paraphrase", "medium"),
            ("contradiction", "health", "quantity change", "high"),
            ("neutral", "daily life", "entity swap", "low"),
            ("entailment", "travel", "generalization", "medium"),
            ("contradiction", "workplace", "negation", "medium"),
            ("neutral", "science", "cause effect", "low"),
            ("entailment", "sports", "paraphrase", "high"),
            ("contradiction", "news", "time change", "medium"),
            ("neutral", "education", "quantity change", "low"),
            ("entailment", "health", "generalization", "medium"),
        ]
        out = []
        for relation, domain, reasoning, overlap in rows:
            out.append(
                {
                    "input": (
                        f"sentence 1: In a fictional {domain} scenario, a person described an event with several concrete details.\n"
                        f"sentence 2: The second statement changes the situation through {reasoning}."
                    ),
                    "control_label": relation,
                    "factors": {
                        "relation_control": relation,
                        "domain": domain,
                        "reasoning_type": reasoning,
                        "lexical_overlap": overlap,
                        "length": "medium",
                    },
                    "source_round": 0,
                }
            )
        return out

    if spec.name == "multirc":
        domains = list(spec.factors["domain"])
        question_types = list(spec.factors["question_type"])
        answer_types = list(spec.factors["answer_type"])
        out = []
        for i in range(16):
            truth = "True" if i % 2 == 0 else "False"
            domain = domains[i % len(domains)]
            qtype = question_types[i % len(question_types)]
            atype = answer_types[i % len(answer_types)]
            paragraph = (
                f"paragraph: A fictional report about {domain} described a small project organized by local volunteers and specialists. "
                "The group met over several weeks, compared notes from earlier trials, and changed the plan after noticing that the first "
                "approach confused several participants. Later, the coordinator wrote a summary explaining which materials were used, who "
                "helped with each step, and why the final demonstration was moved indoors during heavy rain. The report also mentioned that "
                "some observers disagreed about the schedule, but everyone accepted the revised safety rules before the event began."
            )
            question = f"question: {qtype.capitalize()} is described in the report?"
            answer = f"candidate answer: a fictional {atype} from the revised project"
            out.append(
                {
                    "input": f"{paragraph}\n{question}\n{answer}",
                    "control_label": truth,
                    "factors": {
                        "domain": domain,
                        "paragraph_length": "medium",
                        "question_type": qtype,
                        "answer_type": atype,
                        "truth_condition": truth,
                    },
                    "source_round": 0,
                }
            )
        return out

    if spec.name == "wic":
        words = ["bank", "light", "draft", "match", "plant", "right", "file", "charge"]
        domains = list(spec.factors["domain"])
        classes = list(spec.factors["word_class"])
        out = []
        for i in range(16):
            relation = "True" if i % 2 == 0 else "False"
            word = words[i % len(words)]
            domain = domains[i % len(domains)]
            word_class = classes[i % len(classes)]
            if relation == "True":
                sentence_1 = f"The fictional {domain} guide used the word {word} in its ordinary {word_class} sense."
                sentence_2 = f"Later, the same guide used {word} again with that same ordinary {word_class} meaning."
                lexical_relation = "same sense"
            else:
                sentence_1 = f"The fictional {domain} guide used the word {word} in one {word_class} sense."
                sentence_2 = f"Later, another example used {word} with a clearly different {word_class} meaning."
                lexical_relation = "different sense"
            out.append(
                {
                    "input": f"word: {word}\nsentence 1: {sentence_1}\nsentence 2: {sentence_2}",
                    "control_label": relation,
                    "factors": {
                        "sense_relation": relation,
                        "word_class": word_class,
                        "domain": domain,
                        "lexical_relation": lexical_relation,
                    },
                    "source_round": 0,
                }
            )
        return out

    raise AssertionError(f"Unhandled dataset: {spec.name}")


def _label_factor_name(spec: SyntheticDatasetSpec) -> Optional[str]:
    for factor_name in ("topic", "entity_type", "sentiment_control", "relation_control", "truth_condition", "sense_relation"):
        if factor_name in spec.factors:
            return factor_name
    return None


def _label_counts(records: Sequence[Mapping[str, object]], spec: SyntheticDatasetSpec) -> Dict[str, int]:
    counts = Counter({label: 0 for label in spec.labels})
    for record in records:
        label = str(record.get("control_label", "")).strip()
        if label in counts:
            counts[label] += 1
    return dict(counts)


def _format_label_counts(counts: Mapping[str, int]) -> str:
    return ", ".join(f"{label}={counts.get(label, 0)}" for label in counts.keys())


def sample_factor_goals(
    spec: SyntheticDatasetSpec,
    n: int,
    rng: random.Random,
    label_counts: Optional[Mapping[str, int]] = None,
) -> List[Dict[str, str]]:
    goals = []
    factor_names = list(spec.factors.keys())
    label_factor_name = _label_factor_name(spec)
    balancing_counts = Counter({label: 0 for label in spec.labels})
    if label_counts:
        balancing_counts.update({str(k): int(v) for k, v in label_counts.items() if str(k) in spec.labels})
    for i in range(n):
        goal = {}
        for name in factor_names:
            values = list(spec.factors[name])
            goal[name] = values[(i + rng.randrange(len(values))) % len(values)]
        if label_factor_name:
            min_count = min(balancing_counts.values())
            candidates = [label for label in spec.labels if balancing_counts[label] == min_count]
            chosen_label = candidates[(i + rng.randrange(len(candidates))) % len(candidates)]
            goal[label_factor_name] = chosen_label
            balancing_counts[chosen_label] += 1
        if "topic" in goal:
            goal["control_label"] = goal["topic"]
        elif "entity_type" in goal:
            goal["control_label"] = goal["entity_type"]
        elif "sentiment_control" in goal:
            goal["control_label"] = goal["sentiment_control"]
        elif "relation_control" in goal:
            goal["control_label"] = goal["relation_control"]
        elif "truth_condition" in goal:
            goal["control_label"] = goal["truth_condition"]
        elif "sense_relation" in goal:
            goal["control_label"] = goal["sense_relation"]
            # This factor is semantically determined by the requested label;
            # do not let independent random sampling create contradictory
            # requests such as True + "different sense".
            goal["lexical_relation"] = "same sense" if goal["sense_relation"] == "True" else "different sense"
        else:
            goal["control_label"] = rng.choice(spec.labels)
        goals.append(goal)
    return goals


def build_generation_prompt(
    spec: SyntheticDatasetSpec,
    examples: Sequence[Mapping[str, object]],
    factor_goals: Sequence[Mapping[str, str]],
) -> List[Dict[str, str]]:
    example_text = "\n".join(json.dumps(x, ensure_ascii=False) for x in examples)
    factor_text = json.dumps(list(factor_goals), ensure_ascii=False, indent=2)
    user = f"""You are generating synthetic unlabeled inputs for knowledge distillation.

Task instruction:
{spec.instruction}

Input schema:
{spec.schema}

Candidate labels:
{list(spec.labels)}

Generation factor requests:
{factor_text}

Existing examples:
{example_text}

Requirements:
1. Follow the input schema exactly.
2. Do not copy existing examples.
3. Use fictional content only; do not mention real dataset examples.
4. Make examples diverse in topic, entities, wording, length, and reasoning pattern.
5. Generate exactly {len(factor_goals)} JSON objects, exactly one JSON object per factor request.
6. Return JSONL only, with no markdown fences.
7. Copy every requested factor value exactly into the output "factors" object; do not rename labels or factor values.
8. Do not skip difficult factor requests.

Each JSON line must have this shape:
{{"input": "...", "control_label": "...", "factors": {{...}}}}
"""
    return [
        {"role": "system", "content": "You generate diverse, schema-valid synthetic NLP task inputs."},
        {"role": "user", "content": user},
    ]


def _chat_completion_url(base_url: str) -> str:
    base_url = str(base_url).rstrip("/")
    if base_url.endswith("/chat/completions"):
        return base_url
    return base_url + "/chat/completions"


def call_llm_chat_completion(
    *,
    messages: Sequence[Mapping[str, str]],
    api_key: str,
    model: str,
    base_url: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> str:
    payload = {
        "model": model,
        "messages": list(messages),
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    request = urllib.request.Request(
        _chat_completion_url(base_url),
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM API HTTP {e.code}: {detail}") from e
    payload = json.loads(body)
    return payload["choices"][0]["message"]["content"]


def parse_jsonl_or_json_array(text: str) -> List[Dict[str, object]]:
    content = str(text).strip()
    content = re.sub(r"^```(?:json|jsonl)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content).strip()
    if not content:
        return []
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            for key in ("examples", "inputs", "data"):
                if isinstance(parsed.get(key), list):
                    return [x for x in parsed[key] if isinstance(x, dict)]
            return [parsed]
        if isinstance(parsed, list):
            return [x for x in parsed if isinstance(x, dict)]
    except json.JSONDecodeError:
        pass

    rows = []
    for line in content.splitlines():
        line = line.strip().rstrip(",")
        if not line:
            continue
        try:
            parsed_line = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed_line, dict):
            rows.append(parsed_line)
    return rows


def _is_duplicate(candidate: str, existing_texts: Sequence[str], threshold: float) -> bool:
    for text in existing_texts:
        if rouge_l_similarity(candidate, text) >= threshold:
            return True
    return False


def _factor_key(record: Mapping[str, object], spec: SyntheticDatasetSpec) -> Tuple[str, ...]:
    factors = record.get("factors", {})
    if not isinstance(factors, Mapping):
        factors = {}
    return tuple(str(factors.get(name, "")) for name in spec.factors.keys())


def select_factor_balanced(records: Sequence[Mapping[str, object]], spec: SyntheticDatasetSpec, target_size: int) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, ...], List[Dict[str, object]]] = defaultdict(list)
    for record in records:
        groups[_factor_key(record, spec)].append(dict(record))

    selected: List[Dict[str, object]] = []
    keys = sorted(groups.keys(), key=lambda key: (len(groups[key]), key))
    while len(selected) < target_size and any(groups.values()):
        for key in keys:
            if not groups[key]:
                continue
            selected.append(groups[key].pop(0))
            if len(selected) >= target_size:
                break
    return selected


def read_jsonl(path: str) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {e}") from e
            if isinstance(payload, dict):
                out.append(payload)
    return out


def write_jsonl(path: str, records: Sequence[Mapping[str, object]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(dict(record), ensure_ascii=False) + "\n")


def append_jsonl(path: str, records: Sequence[Mapping[str, object]]) -> None:
    if not records:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(dict(record), ensure_ascii=False) + "\n")


def write_json(path: str, payload: Mapping[str, object]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dict(payload), f, ensure_ascii=False, indent=2)


def write_coverage_csv(path: str, records: Sequence[Mapping[str, object]], spec: SyntheticDatasetSpec) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    rows = []
    for factor_name in spec.factors.keys():
        counts = Counter()
        for record in records:
            factors = record.get("factors", {})
            if isinstance(factors, Mapping):
                counts[str(factors.get(factor_name, ""))] += 1
        for value, count in sorted(counts.items()):
            rows.append({"factor": factor_name, "value": value, "count": count})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["factor", "value", "count"])
        writer.writeheader()
        writer.writerows(rows)


def load_valid_pool_checkpoint(path: str, spec: SyntheticDatasetSpec) -> Tuple[List[Dict[str, object]], Counter]:
    if not path or not os.path.exists(path):
        return [], Counter()
    records = read_jsonl(path)
    valid_pool: List[Dict[str, object]] = []
    invalid = Counter()
    seen_texts: List[str] = []
    for record in records:
        ok, reason, normalized = validate_kd_input_record(record, spec)
        if not ok:
            invalid[reason] += 1
            continue
        if _is_duplicate(str(normalized["input"]), seen_texts, 0.999):
            invalid["checkpoint_duplicate"] += 1
            continue
        valid_pool.append(normalized)
        seen_texts.append(str(normalized["input"]))
    return valid_pool, invalid


def generate_self_instruct_inputs(args) -> Dict[str, object]:
    spec = get_dataset_spec(args.dataset)
    rng = random.Random(args.synthetic_seed)
    seeds = seed_kd_inputs(spec.name)
    seed_texts = [str(x["input"]) for x in seeds]
    max_raw_candidates = max(args.synthetic_target_size, int(args.synthetic_target_size * args.synthetic_pool_multiplier))
    max_rounds = int(args.synthetic_max_rounds or max(1, (max_raw_candidates + args.synthetic_generate_per_round - 1) // args.synthetic_generate_per_round))

    if args.dry_run:
        existing_valid_pool = []
        if not getattr(args, "no_checkpoint", False) and not getattr(args, "reset_checkpoint", False):
            existing_valid_pool, _ = load_valid_pool_checkpoint(args.valid_pool_path, spec)
        factor_goals = sample_factor_goals(
            spec,
            min(args.synthetic_generate_per_round, 3),
            rng,
            label_counts=_label_counts(existing_valid_pool, spec),
        )
        messages = build_generation_prompt(spec, seeds[: min(len(seeds), 3)], factor_goals)
        print("[DRY_RUN] Synthetic Self-Instruct generation plan")
        print(f"  dataset={spec.name}")
        print(f"  target_size={args.synthetic_target_size}")
        print(f"  max_raw_candidates={max_raw_candidates}")
        print(f"  output={args.output_path}")
        print(f"  raw_output={args.raw_output_path}")
        print(f"  valid_pool={args.valid_pool_path}")
        print(f"  existing_valid={len(existing_valid_pool)}")
        print(f"  existing_label_counts={_format_label_counts(_label_counts(existing_valid_pool, spec))}")
        print("[DRY_RUN] Prompt preview:")
        print(messages[-1]["content"][:3000])
        return {"dry_run": True}

    api_key = args.llm_api_key or os.environ.get("LLM_API_KEY", "")
    model = args.llm_model or os.environ.get("LLM_MODEL", "gpt-4o-mini")
    base_url = args.llm_base_url or os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")

    checkpoint_enabled = not getattr(args, "no_checkpoint", False)
    reset_checkpoint = bool(getattr(args, "reset_checkpoint", False))
    checkpoint_invalid = Counter()
    if checkpoint_enabled and not reset_checkpoint:
        valid_pool, checkpoint_invalid = load_valid_pool_checkpoint(args.valid_pool_path, spec)
    else:
        valid_pool = []
    initial_valid_count = len(valid_pool)
    initial_label_counts = _label_counts(valid_pool, spec)
    existing_texts: List[str] = list(seed_texts)
    existing_texts.extend(str(record["input"]) for record in valid_pool)
    rejection_stats: Counter = Counter()
    raw_candidates = 0
    if checkpoint_enabled:
        for checkpoint_path in (args.raw_output_path, args.valid_pool_path):
            if checkpoint_path:
                os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
                if reset_checkpoint:
                    open(checkpoint_path, "w", encoding="utf-8").close()

    print("[RESUME]")
    print(f"  checkpoint_enabled: {checkpoint_enabled}")
    print(f"  reset_checkpoint  : {reset_checkpoint}")
    print(f"  existing_valid    : {initial_valid_count}")
    print(f"  existing_labels   : {_format_label_counts(initial_label_counts)}")
    if checkpoint_invalid:
        print(f"  checkpoint_invalid: {dict(checkpoint_invalid)}")
    can_finalize_partial_without_api = bool(args.allow_partial and len(valid_pool) > 0)
    if len(valid_pool) < args.synthetic_target_size and not api_key and not can_finalize_partial_without_api:
        raise ValueError("LLM_API_KEY is required unless --dry_run is set or the valid checkpoint already meets the target size.")
    if len(valid_pool) < args.synthetic_target_size and not api_key and can_finalize_partial_without_api:
        print("[PARTIAL] LLM_API_KEY is not set; finalizing existing valid checkpoint because --allow_partial is enabled.")

    for round_id in range(1, max_rounds + 1):
        if not api_key:
            break
        if len(valid_pool) >= args.synthetic_target_size or raw_candidates >= max_raw_candidates:
            break
        examples = rng.sample(seeds + valid_pool, min(len(seeds + valid_pool), args.synthetic_in_context_examples))
        goals = sample_factor_goals(
            spec,
            args.synthetic_generate_per_round,
            rng,
            label_counts=_label_counts(valid_pool, spec),
        )
        messages = build_generation_prompt(spec, examples, goals)
        content = call_llm_chat_completion(
            messages=messages,
            api_key=api_key,
            model=model,
            base_url=base_url,
            temperature=args.llm_temperature,
            max_tokens=args.llm_max_tokens,
            timeout=args.llm_timeout,
        )
        candidates = parse_jsonl_or_json_array(content)
        raw_candidates += len(candidates)
        raw_records = []
        new_valid_records = []
        for candidate in candidates:
            candidate["source_round"] = round_id
            raw_records.append(dict(candidate))
            ok, reason, normalized = validate_kd_input_record(candidate, spec)
            if not ok:
                rejection_stats[reason] += 1
                continue
            if _is_duplicate(str(normalized["input"]), existing_texts, args.synthetic_similarity_threshold):
                rejection_stats["near_duplicate"] += 1
                continue
            valid_pool.append(normalized)
            new_valid_records.append(normalized)
            existing_texts.append(str(normalized["input"]))
        if checkpoint_enabled:
            append_jsonl(args.raw_output_path, raw_records)
            append_jsonl(args.valid_pool_path, new_valid_records)
        print(
            f"[ROUND] {round_id} raw={raw_candidates}/{max_raw_candidates} "
            f"new_valid={len(new_valid_records)} valid_pool={len(valid_pool)} "
            f"labels={_format_label_counts(_label_counts(valid_pool, spec))} "
            f"rejected={sum(rejection_stats.values())}"
        )
        if args.synthetic_request_sleep > 0:
            time.sleep(args.synthetic_request_sleep)

    final_records = select_factor_balanced(valid_pool, spec, args.synthetic_target_size)
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "method": "self_instruct_synthetic_kd_inputs",
        "dataset": spec.name,
        "instruction": spec.instruction,
        "schema": spec.schema,
        "labels": list(spec.labels),
        "length_bounds": list(spec.length_bounds),
        "factors": {key: list(values) for key, values in spec.factors.items()},
        "seed_kd_input_count": len(seeds),
        "target_size": args.synthetic_target_size,
        "initial_valid_size": initial_valid_count,
        "initial_label_counts": initial_label_counts,
        "valid_pool_size": len(valid_pool),
        "valid_pool_new_size": len(valid_pool) - initial_valid_count,
        "valid_pool_label_counts": _label_counts(valid_pool, spec),
        "final_size": len(final_records),
        "raw_candidates": raw_candidates,
        "pool_multiplier": args.synthetic_pool_multiplier,
        "generate_per_round": args.synthetic_generate_per_round,
        "similarity_threshold": args.synthetic_similarity_threshold,
        "seed": args.synthetic_seed,
        "llm_model": model,
        "llm_base_url": base_url,
        "output_path": args.output_path,
        "raw_output_path": args.raw_output_path,
        "valid_pool_path": args.valid_pool_path,
        "coverage_path": args.coverage_path,
        "checkpoint_enabled": checkpoint_enabled,
        "reset_checkpoint": reset_checkpoint,
        "checkpoint_invalid_reasons": dict(checkpoint_invalid),
        "rejection_stats": dict(rejection_stats),
    }
    if len(final_records) < args.synthetic_target_size and not args.allow_partial:
        manifest["status"] = "incomplete"
        manifest["failure_reason"] = "valid_kd_input_count_below_target"
        write_json(args.manifest_path, manifest)
        print(f"[SAVE] incomplete manifest={args.manifest_path}")
        if checkpoint_enabled:
            print(f"[SAVE] valid_pool_checkpoint={args.valid_pool_path}")
            print(f"[SAVE] raw_checkpoint={args.raw_output_path}")
        raise RuntimeError(
            f"Only {len(final_records)} valid inputs were generated, below target {args.synthetic_target_size}. "
            "Increase SYNTHETIC_POOL_MULTIPLIER, relax filters, reduce SYNTHETIC_TARGET_SIZE, or pass --allow_partial. "
            f"Valid checkpoint was saved to: {args.valid_pool_path}"
        )

    manifest["status"] = "complete" if len(final_records) >= args.synthetic_target_size else "partial_allowed"
    write_jsonl(args.output_path, final_records)
    write_coverage_csv(args.coverage_path, final_records, spec)
    write_json(args.manifest_path, manifest)
    print(f"[SAVE] inputs={args.output_path}")
    print(f"[SAVE] manifest={args.manifest_path}")
    print(f"[SAVE] coverage={args.coverage_path}")
    if checkpoint_enabled:
        print(f"[SAVE] valid_pool_checkpoint={args.valid_pool_path}")
        print(f"[SAVE] raw_checkpoint={args.raw_output_path}")
    return manifest


def load_synthetic_kd_input_instances(path: str, dataset_name: Optional[str] = None) -> Tuple[List[Dict[str, str]], Dict[str, object]]:
    records = read_jsonl(path)
    if dataset_name:
        spec = get_dataset_spec(dataset_name)
        valid_records = []
        invalid = Counter()
        for record in records:
            ok, reason, normalized = validate_kd_input_record(record, spec)
            if ok:
                valid_records.append(normalized)
            else:
                invalid[reason] += 1
        records = valid_records
    else:
        invalid = Counter()

    instances = []
    for record in records:
        output = str(record.get("control_label", "")).strip()
        instances.append({"input": str(record["input"]).strip(), "output": output})
    if not instances:
        raise ValueError(f"No valid synthetic KD inputs found in: {path}")
    manifest_path = os.path.join(os.path.dirname(os.path.abspath(path)), os.path.basename(path).replace(".inputs.jsonl", ".manifest.json"))
    meta = {
        "source": "synthetic_self_instruct",
        "kd_input_path": os.path.abspath(path),
        "kd_input_count": len(instances),
        "invalid_count": sum(invalid.values()),
        "invalid_reasons": dict(invalid),
        "manifest_path": manifest_path if os.path.exists(manifest_path) else None,
    }
    return instances, meta


def resolve_kd_input_instances(args, target_model_train_instances: Sequence[Dict[str, str]]) -> Tuple[Optional[List[Dict[str, str]]], Dict[str, object]]:
    source = str(getattr(args, "kd_source", "synthetic_self_instruct")).strip().lower()
    if getattr(args, "lambda_dormant_model_kd", 0.0) <= 0:
        return None, {"source": source, "enabled": False, "reason": "lambda_dormant_model_kd<=0"}
    if source == "none":
        args.lambda_dormant_model_kd = 0.0
        return None, {"source": source, "enabled": False, "reason": "kd_source=none"}
    if source in {"target_train", "target_valid"}:
        if not target_model_train_instances:
            raise ValueError(f"kd_source={source} requires loading a non-empty TargetModel KD input split.")
        return list(target_model_train_instances), {
            "source": source,
            "enabled": True,
            "kd_input_count": len(target_model_train_instances),
            "note": (
                "oracle_upper_bound_uses_real_target_train_data"
                if source == "target_train"
                else "uses_real_target_validation_data_as_kd_inputs"
            ),
        }
    if source == "synthetic_self_instruct":
        instances, meta = load_synthetic_kd_input_instances(args.kd_input_path, getattr(args, "target_model_dataset", None))
        limit = int(getattr(args, "kd_input_limit", 0) or 0)
        if limit > 0:
            original_count = len(instances)
            limit_seed = int(getattr(args, "seed", 42)) + 3007
            if original_count > limit:
                rng = random.Random(limit_seed)
                order = list(range(original_count))
                rng.shuffle(order)
                instances = [instances[index] for index in order[:limit]]
            meta["kd_input_count_before_limit"] = original_count
            meta["kd_input_limit"] = limit
            meta["kd_input_limit_seed"] = limit_seed
            meta["kd_input_count"] = len(instances)
        meta["enabled"] = True
        meta["synthetic_input_manifest"] = getattr(args, "synthetic_input_manifest", None)
        return instances, meta
    raise ValueError("kd_source must be one of: synthetic_self_instruct, target_train, target_valid, none.")

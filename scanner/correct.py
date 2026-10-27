"""Context-aware OCR post-correction.

For each word Tesseract is unsure of, candidates within two edits are scored by
a bigram language model (stupid backoff, built from a plain-text corpus) and an
OCR channel model (edit distance, with visually confusable letters cheaper).
Confident dictionary words and mid-sentence capitalised words (likely names)
are left alone.

Benchmarked on synthetic webcam pages: word errors on small, noisy text fell
from 52-59% to 37-47%, with clean text unchanged.
"""

import logging
import math
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path

log = logging.getLogger(__name__)

WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")
TOKEN = re.compile(r"^([^A-Za-z]*)([A-Za-z']+)([^A-Za-z]*)$")
ABBREVIATIONS = {"mr.", "mrs.", "ms.", "dr.", "st.", "mt.", "prof.", "rev.", "gen.", "col.", "capt.", "sr.", "jr.", "vs.", "no.", "vol.", "pp.", "ed.", "eds."}
CONFUSABLE = {frozenset(p) for p in ("l1", "li", "ec", "oa", "un", "hb", "tf", "rn", "co", "il")}


def _deletes1(word):
    return {word} | {word[:i] + word[i + 1:] for i in range(len(word))}


def edit_distance(a, b):
    """Damerau-Levenshtein, with visually confusable substitutions at 0.4."""
    d = [[0.0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        d[i][0] = i
    for j in range(len(b) + 1):
        d[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                sub = 0
            else:
                sub = 0.4 if frozenset((a[i - 1], b[j - 1])) in CONFUSABLE else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + sub)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[-1][-1]


class Corrector:
    def __init__(self, unigrams, bigrams, dictionary, keep_conf=90, confident=80,
                 replace_valid_below=50, alpha=4.0, keep_bonus=3.0):
        self.uni = unigrams
        self.bi = bigrams
        self.total = sum(unigrams.values())
        self.dictionary = dictionary
        self.keep_conf = keep_conf  # dictionary words at/above this confidence are trusted
        self.confident = confident  # at/above this, only single-edit fixes are allowed
        self.replace_valid_below = replace_valid_below  # real words are only replaced below this
        self.alpha = alpha  # weight of edit distance vs language model
        self.keep_bonus = keep_bonus  # preference for keeping a valid word
        # single-delete index over corpus words seen at least twice: finds candidates
        # within ~2 edits by intersecting deletes of both words, in little memory
        self.index = defaultdict(list)
        for w, c in unigrams.items():
            if c >= 2:
                for d in _deletes1(w):
                    self.index[d].append(w)

    @classmethod
    def load(cls, corpus, cache, wordlist="/usr/share/dict/words"):
        """Build from a corpus, caching the counts; returns None if the corpus is missing."""
        corpus, cache = Path(corpus).expanduser(), Path(cache).expanduser()
        if not corpus.exists():
            log.warning("OCR correction disabled: corpus %s not found", corpus)
            return None
        if cache.exists() and cache.stat().st_mtime >= corpus.stat().st_mtime:
            unigrams, bigrams = pickle.loads(cache.read_bytes())
        else:
            log.info("building language model from %s", corpus)
            tokens = WORD.findall(corpus.read_text(errors="ignore").lower())
            unigrams = dict(Counter(tokens))
            bigrams = {k: v for k, v in Counter(zip(tokens, tokens[1:])).items() if v >= 2}
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(pickle.dumps((unigrams, bigrams), protocol=pickle.HIGHEST_PROTOCOL))
        words = Path(wordlist)
        dictionary = {w.strip().lower() for w in words.open()} if words.exists() else set()
        return cls(unigrams, bigrams, dictionary)

    def _logp(self, word, prev):
        """Stupid-backoff log probability of word following prev."""
        count = self.bi.get((prev, word)) if prev else None
        if count:
            return math.log(count / self.uni[prev])
        return math.log(0.4 * (self.uni.get(word, 0) + 0.1) / self.total)

    def _candidates(self, word, max_dist):
        found = set()
        for d in _deletes1(word):
            found.update(self.index.get(d, ()))
        return {c for c in found if edit_distance(word, c) <= max_dist}

    def correct(self, words):
        """words: [(token, confidence)] in reading order. Returns corrected tokens."""
        lowers = [re.sub(r"[^a-z']", "", w.lower()) for w, _ in words]
        out, prev = [], None
        for i, ((token, conf), low) in enumerate(zip(words, lowers)):
            match = TOKEN.match(token)
            known = low in self.uni or low in self.dictionary
            if (not match or not low or token.endswith("-")
                    or (known and conf >= min(self.keep_conf, self.replace_valid_below))):
                out.append(token)
                prev = low or None
                continue
            pre, core, post = match.groups()
            prev_token = words[i - 1][0] if i else ""
            sentence_start = i == 0 or (prev_token[-1:] in ".!?\"'" and prev_token.lower() not in ABBREVIATIONS)
            if core[0].isupper() and not sentence_start:
                out.append(token)  # probably a name, which the corpus can't judge
                prev = low
                continue

            nxt = lowers[i + 1] if i + 1 < len(lowers) else None
            candidates = self._candidates(low, 1 if conf >= self.confident else 2)
            if known:
                candidates.add(low)

            def score(c):
                s = self._logp(c, prev) + (self._logp(nxt, c) if nxt else 0)
                s -= self.alpha * edit_distance(low, c)
                return s + (self.keep_bonus if c == low and known else 0)

            best = max(candidates, key=score, default=low)
            if best != low:
                if core.isupper() and len(core) > 1:
                    best = best.upper()
                elif core[0].isupper():
                    best = best.capitalize()
                token = pre + best + post
            out.append(token)
            prev = best
        return out

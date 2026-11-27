"""Who-is-who consistency of multi-chunk stories, from character names (rule-based, CPU).

A name is a capitalised word that is on the known-name lists, or that appears at least once in the middle of a
sentence and is not a common capitalised word ("Mom", "The", "Wow", days, holidays). TinyStories names are
simple, so this catches nearly all characters; it misses characters known only as "the dog" or "the girl".

Per story (chunks as decoded text):
  names            distinct names
  late_names       names first used in the second half of the story (chunk > M/2)
  late_one_off     late names used only once: a character from nowhere that is dropped again
  protagonist      most frequent name in chunk 1 (None if chunk 1 has no name)
  kept             the protagonist is still named in the second half
  replaced         the protagonist is gone from the second half, and a name new to the second half is used
                   there at least twice: the story switched main character ("Lily ... Sara ... Sara")
Stories whose text stops before the second half are skipped for kept / replaced.
"""
import re
from collections import Counter

from error_types import FEM, MASC, NOT_NAME  # block-ncm/experiments

NOT_NAME = set(NOT_NAME) | set("""Monday Tuesday Wednesday Thursday Friday Saturday Sunday January February March April
May June July August September October November December Christmas Easter Halloween God Earth Santa King Queen
Princess Prince Mr Mrs Ms Dr Grandma Grandpa Uncle Aunt Auntie Baby Doctor Officer Captain Chef Farmer Teacher
Bunny Kitty Doggy Puppy Mommy Daddy Mama Papa English Ow Yummy Aww Ah Hi Hooray Whee Boo Shh Hello Goodbye
Well Wait Yeah Yes Nope Hey Look Thank Please Sorry""".split())
KNOWN = FEM | MASC
WORD = re.compile(r"[A-Za-z][a-z]+|[.!?\n\"]")


def names_in(text):
    """List of name occurrences in order, plus the set of words seen mid-sentence."""
    occ, mid, start = [], set(), True
    for w in WORD.findall(text):
        if w in '.!?\n':
            start = True
            continue
        if w == '"':  # a quote mark neither starts nor ends a sentence
            continue
        if w[0].isupper():
            if not start:
                mid.add(w)
            occ.append(w)
        start = False
    return occ, mid


def story_names(chunks):
    """Per-chunk lists of names, using evidence from the whole story to decide what is a name."""
    per, mid_all = [], set()
    for t in chunks:
        occ, mid = names_in(t)
        per.append(occ)
        mid_all |= mid
    is_name = lambda w: w in KNOWN or (w in mid_all and w not in NOT_NAME)
    return [[w for w in occ if is_name(w)] for occ in per]


def analyse_story(chunks, M):
    """chunks: per-chunk texts (may be fewer than M if the story ended early)."""
    per = story_names(chunks)
    half = M // 2
    first = {}
    for k, ns in enumerate(per):
        for w in ns:
            first.setdefault(w, k)
    count = Counter(w for ns in per for w in ns)
    late = [w for w, k in first.items() if k >= half]
    r = {"names": len(count), "late_names": len(late),
         "late_one_off": sum(1 for w in late if count[w] == 1), "protagonist": None, "kept": None, "replaced": None}
    if per and per[0] and len(chunks) > half and any(t.strip() for t in chunks[half:]):
        prot = Counter(per[0]).most_common(1)[0][0]
        second = Counter(w for ns in per[half:] for w in ns)
        r["protagonist"] = prot
        r["kept"] = second[prot] > 0
        r["replaced"] = (not r["kept"]) and any(c >= 2 and first[w] >= half for w, c in second.items())
    return r


def summarise(stories, M):
    """stories: list of per-chunk text lists. Returns rates (per story, or share of eligible stories)."""
    rs = [analyse_story(ch, M) for ch in stories]
    elig = [r for r in rs if r["kept"] is not None]
    n = max(1, len(rs))
    return {
        "n": len(rs), "n_with_protagonist": len(elig),
        "names_per_story": sum(r["names"] for r in rs) / n,
        "late_names_per_story": sum(r["late_names"] for r in rs) / n,
        "late_one_off_per_story": sum(r["late_one_off"] for r in rs) / n,
        "protagonist_kept": sum(r["kept"] for r in elig) / max(1, len(elig)),
        "protagonist_replaced": sum(r["replaced"] for r in elig) / max(1, len(elig)),
    }, rs

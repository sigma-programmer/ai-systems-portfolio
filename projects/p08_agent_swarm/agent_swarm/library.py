"""Bundled benchmark: spec tasks, a knowledge base, and a candidate-program library.

Three separate sources of information, deliberately kept apart:

* ``TASKS`` — what a user would hand the swarm (description + 2 examples), plus
  ``hidden`` tests that are NEVER shown to agents and only used to score runs.
* ``KNOWLEDGE`` — prose snippets (with ``REQ:``/``EDGE:`` lines) the Researcher
  retrieves by embedding similarity. Some are distractors for other functions,
  and coverage is intentionally imperfect (the leap-year snippet omits the
  400-year rule), so research has real, measurable limits.
* ``CANDIDATES`` — the offline coder's program library: for each problem a few
  plausible implementations ordered simplest-first (often subtly buggy), with
  keyword tags. The coder searches this space plus AST mutations of it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .messages import TaskSpec, TestCase


@dataclass(frozen=True)
class Snippet:
    """A knowledge-base document."""

    id: str
    title: str
    text: str


@dataclass(frozen=True)
class CandidateTemplate:
    """A library implementation with keyword tags used for requirement matching."""

    source: str
    tags: tuple[str, ...] = ()


@dataclass
class TaskDef:
    """A benchmark task: public spec + hidden scoring tests."""

    spec: TaskSpec
    hidden: list[TestCase] = field(default_factory=list)


def _t(prefix: str, rows: list[tuple], source: str = "spec") -> list[TestCase]:
    out = []
    for i, row in enumerate(rows):
        args, outcome = row
        if isinstance(outcome, dict) and "raises" in outcome:
            out.append(TestCase(id=f"{prefix}{i}", args=list(args), raises=outcome["raises"], source=source))
        else:
            out.append(TestCase(id=f"{prefix}{i}", args=list(args), expected=outcome, source=source))
    return out


def _task(name: str, func: str, params: list[str], desc: str, examples: list[tuple], hidden: list[tuple]) -> TaskDef:
    ex = _t("ex", examples)
    return TaskDef(
        spec=TaskSpec(name=name, func_name=func, params=params, description=desc, examples=ex),
        hidden=ex + _t("h", hidden, source="hidden"),
    )


RAISES_VALUE = {"raises": "ValueError"}

TASKS: list[TaskDef] = [
    _task("palindrome", "is_palindrome", ["s"],
          "Return True if the string s is a palindrome when read as a sentence.",
          [(("racecar",), True), (("hello",), False)],
          [(("A man, a plan, a canal: Panama",), True), (("",), True), (("No 'x' in Nixon",), True),
           (("ab",), False), (("Abba",), True)]),
    _task("fizzbuzz", "fizzbuzz", ["n"],
          "Return the FizzBuzz sequence from 1 to n as a list of strings.",
          [((5,), ["1", "2", "Fizz", "4", "Buzz"]), ((3,), ["1", "2", "Fizz"])],
          [((0,), []), ((15,), ["1", "2", "Fizz", "4", "Buzz", "Fizz", "7", "8", "Fizz", "Buzz", "11",
                                "Fizz", "13", "14", "FizzBuzz"]), ((1,), ["1"])]),
    _task("reverse_words", "reverse_words", ["s"],
          "Reverse the order of words in a sentence string s.",
          [(("hello world",), "world hello"), (("a b c",), "c b a")],
          [(("  hello   world ",), "world hello"), (("",), ""), (("single",), "single")]),
    _task("flatten", "flatten", ["xs"],
          "Flatten a nested list xs into a single flat list.",
          [(([[1, 2], [3]],), [1, 2, 3]), (([[4], [5, 6]],), [4, 5, 6])],
          [(([1, [2, [3, [4]]]],), [1, 2, 3, 4]), (([],), []), (([[], [[]]],), []), (([1, 2],), [1, 2])]),
    _task("merge_intervals", "merge_intervals", ["intervals"],
          "Merge all overlapping intervals in a list of [start, end] pairs.",
          [(([[1, 3], [2, 6], [8, 10]],), [[1, 6], [8, 10]]), (([[1, 2]],), [[1, 2]])],
          [(([[1, 4], [4, 5]],), [[1, 5]]), (([[8, 10], [1, 3], [2, 6]],), [[1, 6], [8, 10]]),
           (([],), []), (([[1, 10], [2, 3]],), [[1, 10]]), (([[5, 6], [1, 2]],), [[1, 2], [5, 6]])]),
    _task("roman", "roman_to_int", ["s"],
          "Convert a Roman numeral string s to an integer.",
          [(("III",), 3), (("VIII",), 8)],
          [(("IV",), 4), (("IX",), 9), (("MCMXCIV",), 1994), (("LVIII",), 58), (("XL",), 40)]),
    _task("binary_search", "binary_search", ["xs", "target"],
          "Return the index of target in the sorted list xs using binary search.",
          [(([1, 3, 5, 7], 5), 2), (([1, 3, 5, 7], 1), 0)],
          [(([1, 3, 5, 7], 7), 3), (([], 1), -1), (([2], 2), 0), (([1, 3], 4), -1), (([1, 3, 5], 0), -1)]),
    _task("rle", "run_length_encode", ["s"],
          "Run-length encode a string s as character followed by run count.",
          [(("aaabbb",), "a3b3"), (("zzzz",), "z4")],
          [(("abc",), "a1b1c1"), (("",), ""), (("aabcc",), "a2b1c2"), (("a",), "a1")]),
    _task("two_sum", "two_sum", ["nums", "target"],
          "Find the indices of two numbers in nums that add up to target.",
          [(([2, 7, 11, 15], 9), [0, 1]), (([1, 5, 9], 14), [1, 2])],
          [(([3, 2, 4], 6), [1, 2]), (([1, 2], 7), None), (([3, 3], 6), [0, 1]), (([], 0), None)]),
    _task("parens", "valid_parentheses", ["s"],
          "Check whether the brackets in string s are balanced and properly nested.",
          [(("()",), True), (("(()",), False)],
          [(("([)]",), False), (("{[]}",), True), (("",), True), ((")(",), False), (("(]",), False)]),
    _task("chunk", "chunk", ["xs", "size"],
          "Split list xs into consecutive chunks of length size.",
          [(([1, 2, 3, 4], 2), [[1, 2], [3, 4]]), (([1, 2, 3], 3), [[1, 2, 3]])],
          [(([1, 2, 3, 4, 5], 2), [[1, 2], [3, 4], [5]]), (([], 2), []), (([1, 2], -1), RAISES_VALUE),
           (([1, 2], 0), RAISES_VALUE)]),
    _task("second_largest", "second_largest", ["xs"],
          "Return the second largest value in the list xs.",
          [(([1, 5, 3],), 3), (([10, 20, 30],), 20)],
          [(([5, 5, 4],), 4), (([1],), None), (([3, 3],), None), (([],), None), (([-1, -2],), -2)]),
    _task("word_freq", "word_frequencies", ["text"],
          "Count how often each word occurs in the text and return a dict.",
          [(("a b a",), {"a": 2, "b": 1}), (("x y",), {"x": 1, "y": 1})],
          [(("Hello, hello!",), {"hello": 2}), (("",), {}), (("The cat. the CAT?",), {"the": 2, "cat": 2})]),
    _task("median", "median", ["xs"],
          "Return the median value of a list of numbers xs.",
          [(([3, 1, 2],), 2), (([5],), 5)],
          [(([1, 2, 3, 4],), 2.5), (([],), RAISES_VALUE), (([7, 1],), 4.0), (([2, 2, 9],), 2)]),
    _task("leap_year", "is_leap_year", ["year"],
          "Return True if year is a leap year in the Gregorian calendar.",
          [((2024,), True), ((2023,), False)],
          [((1900,), False), ((2000,), True), ((2100,), False), ((1996,), True)]),
]

TASKS_BY_NAME: dict[str, TaskDef] = {t.spec.name: t for t in TASKS}


KNOWLEDGE: list[Snippet] = [
    Snippet("kb-palindrome", "Sentence palindromes",
            "A sentence palindrome reads the same forwards and backwards once you ignore letter case, "
            "punctuation and spaces.\n"
            "REQ: comparison is case insensitive and only alphanumeric characters count, punctuation and spaces are ignored\n"
            "EDGE: is_palindrome('A man, a plan, a canal: Panama') -> True\n"
            "EDGE: is_palindrome('') -> True\n"
            "EDGE: is_palindrome('No \\'x\\' in Nixon') -> True\n"
            "EDGE: is_palindrome('ab') -> False"),
    Snippet("kb-palnum", "Palindrome numbers",
            "Integer palindromes such as 121; negative numbers are never palindromes.\n"
            "EDGE: is_palindrome_number(-121) -> False\n"
            "EDGE: is_palindrome_number(121) -> True"),
    Snippet("kb-fizzbuzz", "FizzBuzz rules",
            "FizzBuzz: for multiples of both three and five output FizzBuzz, which must be checked before "
            "the single Fizz or Buzz rules. The sequence starts at 1.\n"
            "REQ: multiples of 15 produce FizzBuzz, check the combined rule first\n"
            "REQ: n of zero produces an empty list\n"
            "EDGE: fizzbuzz(0) -> []\n"
            "EDGE: fizzbuzz(15) -> ['1', '2', 'Fizz', '4', 'Buzz', 'Fizz', '7', '8', 'Fizz', 'Buzz', '11', 'Fizz', '13', '14', 'FizzBuzz']"),
    Snippet("kb-words", "Word order reversal",
            "When reversing words in a sentence, collapse runs of whitespace and strip leading and trailing "
            "spaces; split on any whitespace rather than a single space.\n"
            "REQ: split on arbitrary whitespace and collapse repeated spaces\n"
            "EDGE: reverse_words('  hello   world ') -> 'world hello'\n"
            "EDGE: reverse_words('') -> ''"),
    Snippet("kb-flatten", "Flattening nested lists",
            "Nested lists can be arbitrarily deep, so flattening must be recursive, not a single level.\n"
            "REQ: recursive flattening of arbitrary depth nesting\n"
            "EDGE: flatten([1, [2, [3, [4]]]]) -> [1, 2, 3, 4]\n"
            "EDGE: flatten([]) -> []\n"
            "EDGE: flatten([[], [[]]]) -> []"),
    Snippet("kb-intervals", "Interval merging",
            "Merging intervals: sort by start first, treat touching intervals such as [1,4] and [4,5] as "
            "overlapping, and keep the maximum end so contained intervals are absorbed.\n"
            "REQ: sort intervals by start, touching intervals overlap, keep the max end for containment\n"
            "EDGE: merge_intervals([[1, 4], [4, 5]]) -> [[1, 5]]\n"
            "EDGE: merge_intervals([[8, 10], [1, 3], [2, 6]]) -> [[1, 6], [8, 10]]\n"
            "EDGE: merge_intervals([]) -> []\n"
            "EDGE: merge_intervals([[1, 10], [2, 3]]) -> [[1, 10]]"),
    Snippet("kb-roman", "Roman numerals",
            "Roman numerals use subtractive notation: a smaller symbol before a larger one is subtracted, "
            "as in IV (4), IX (9), XL (40) and CM (900).\n"
            "REQ: subtractive notation, subtract a symbol smaller than its successor\n"
            "EDGE: roman_to_int('IV') -> 4\n"
            "EDGE: roman_to_int('IX') -> 9\n"
            "EDGE: roman_to_int('MCMXCIV') -> 1994"),
    Snippet("kb-bsearch", "Binary search pitfalls",
            "Classic binary search bugs are off-by-one errors in the loop condition: with an inclusive upper "
            "bound the loop must run while lo <= hi, otherwise the last element is missed. Return -1 when "
            "the target is absent, including for an empty list.\n"
            "REQ: return -1 when the target is missing, inclusive bounds\n"
            "EDGE: binary_search([1, 3, 5, 7], 7) -> 3\n"
            "EDGE: binary_search([], 1) -> -1\n"
            "EDGE: binary_search([2], 2) -> 0\n"
            "EDGE: binary_search([1, 3], 4) -> -1"),
    Snippet("kb-rle", "Run-length encoding format",
            "In this run-length encoding format every run is written as the character followed by its count, "
            "even when the count is one.\n"
            "REQ: always emit the count, even for runs of length one\n"
            "EDGE: run_length_encode('abc') -> 'a1b1c1'\n"
            "EDGE: run_length_encode('') -> ''"),
    Snippet("kb-twosum", "Two-sum lookup",
            "Two sum with a hash map: look up the complement before inserting the current number so an "
            "element is never paired with itself; return None when no pair exists.\n"
            "REQ: never reuse the same element, return None if no pair\n"
            "EDGE: two_sum([3, 2, 4], 6) -> [1, 2]\n"
            "EDGE: two_sum([1, 2], 7) -> None\n"
            "EDGE: two_sum([3, 3], 6) -> [0, 1]"),
    Snippet("kb-parens", "Bracket matching",
            "Balanced brackets require a stack: each closing bracket must match the most recent unmatched "
            "opening bracket of the same type. Counting alone fails for interleaved types.\n"
            "REQ: use a stack and match bracket types\n"
            "EDGE: valid_parentheses('([)]') -> False\n"
            "EDGE: valid_parentheses('{[]}') -> True\n"
            "EDGE: valid_parentheses(')(') -> False\n"
            "EDGE: valid_parentheses('') -> True"),
    Snippet("kb-chunk", "Chunking sequences",
            "Chunking splits a list into consecutive pieces; the final chunk may be shorter. A size of zero "
            "or a negative size is invalid and must raise ValueError.\n"
            "REQ: validate size, raise ValueError for non positive size\n"
            "EDGE: chunk([1, 2, 3, 4, 5], 2) -> [[1, 2], [3, 4], [5]]\n"
            "EDGE: chunk([1, 2], -1) raises ValueError\n"
            "EDGE: chunk([], 2) -> []"),
    Snippet("kb-second", "Second largest element",
            "The second largest value means the second largest distinct value; duplicates of the maximum do "
            "not count. Return None when fewer than two distinct values exist.\n"
            "REQ: distinct values only, return None when there is no second value\n"
            "EDGE: second_largest([5, 5, 4]) -> 4\n"
            "EDGE: second_largest([1]) -> None\n"
            "EDGE: second_largest([3, 3]) -> None"),
    Snippet("kb-wordfreq", "Word frequency counting",
            "Word counting should be case insensitive and strip punctuation, so Hello and hello! are the "
            "same word.\n"
            "REQ: lowercase words and strip punctuation using a regex\n"
            "EDGE: word_frequencies('Hello, hello!') -> {'hello': 2}\n"
            "EDGE: word_frequencies('') -> {}"),
    Snippet("kb-median", "Median of a list",
            "The median of an even-length list is the mean of the two middle values; the median of an empty "
            "list is undefined and should raise ValueError.\n"
            "REQ: average the two middle values for even length, raise ValueError when empty\n"
            "EDGE: median([1, 2, 3, 4]) -> 2.5\n"
            "EDGE: median([]) raises ValueError"),
    Snippet("kb-leap", "Leap years (partial note)",
            "Leap years occur every four years, except that century years such as 1900 are not leap years.\n"
            "REQ: divisible by four, but century years are excluded\n"
            "EDGE: is_leap_year(1900) -> False"),
    Snippet("kb-sorting", "Sorting stability",
            "Python's sort is stable and runs in O(n log n). Use key functions instead of cmp.\n"
            "EDGE: stable_sort([3, 1, 2]) -> [1, 2, 3]"),
]


CANDIDATES: dict[str, list[CandidateTemplate]] = {
    "is_palindrome": [
        CandidateTemplate("def is_palindrome(s):\n    return s == s[::-1]\n"),
        CandidateTemplate("def is_palindrome(s):\n    t = s.lower()\n    return t == t[::-1]\n", ("case", "insensitive")),
        CandidateTemplate(
            "def is_palindrome(s):\n    t = [c.lower() for c in s if c.isalnum()]\n    return t == t[::-1]\n",
            ("case", "insensitive", "alphanumeric", "punctuation")),
    ],
    "fizzbuzz": [
        CandidateTemplate(
            "def fizzbuzz(n):\n    out = []\n    for i in range(1, n + 1):\n        if i % 3 == 0:\n"
            "            out.append('Fizz')\n        elif i % 5 == 0:\n            out.append('Buzz')\n"
            "        elif i % 15 == 0:\n            out.append('FizzBuzz')\n        else:\n"
            "            out.append(str(i))\n    return out\n"),
        CandidateTemplate(
            "def fizzbuzz(n):\n    out = []\n    for i in range(1, n + 1):\n        if i % 15 == 0:\n"
            "            out.append('FizzBuzz')\n        elif i % 3 == 0:\n            out.append('Fizz')\n"
            "        elif i % 5 == 0:\n            out.append('Buzz')\n        else:\n"
            "            out.append(str(i))\n    return out\n", ("combined", "first", "fizzbuzz")),
    ],
    "reverse_words": [
        CandidateTemplate("def reverse_words(s):\n    return ' '.join(s.split(' ')[::-1])\n"),
        CandidateTemplate("def reverse_words(s):\n    return ' '.join(reversed(s.split()))\n",
                          ("whitespace", "collapse", "arbitrary")),
    ],
    "flatten": [
        CandidateTemplate("def flatten(xs):\n    return [y for x in xs for y in x]\n"),
        CandidateTemplate(
            "def flatten(xs):\n    out = []\n    for x in xs:\n        if isinstance(x, list):\n"
            "            out.extend(flatten(x))\n        else:\n            out.append(x)\n    return out\n",
            ("recursive", "depth", "arbitrary")),
    ],
    "merge_intervals": [
        CandidateTemplate(
            "def merge_intervals(intervals):\n    out = []\n    for s, e in intervals:\n"
            "        if out and s < out[-1][1]:\n            out[-1][1] = e\n        else:\n"
            "            out.append([s, e])\n    return out\n"),
        CandidateTemplate(
            "def merge_intervals(intervals):\n    out = []\n    for s, e in sorted(intervals):\n"
            "        if out and s < out[-1][1]:\n            out[-1][1] = e\n        else:\n"
            "            out.append([s, e])\n    return out\n", ("sort",)),
        CandidateTemplate(
            "def merge_intervals(intervals):\n    out = []\n    for s, e in sorted(intervals):\n"
            "        if out and s <= out[-1][1]:\n            out[-1][1] = e\n        else:\n"
            "            out.append([s, e])\n    return out\n", ("sort", "touching")),
        CandidateTemplate(
            "def merge_intervals(intervals):\n    out = []\n    for s, e in sorted(intervals):\n"
            "        if out and s <= out[-1][1]:\n            out[-1][1] = max(out[-1][1], e)\n        else:\n"
            "            out.append([s, e])\n    return out\n", ("sort", "touching", "max", "containment")),
    ],
    "roman_to_int": [
        CandidateTemplate(
            "def roman_to_int(s):\n    v = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}\n"
            "    return sum(v[c] for c in s)\n"),
        CandidateTemplate(
            "def roman_to_int(s):\n    v = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}\n"
            "    total = 0\n    for i, c in enumerate(s):\n        if i + 1 < len(s) and v[c] < v[s[i + 1]]:\n"
            "            total -= v[c]\n        else:\n            total += v[c]\n    return total\n",
            ("subtractive", "subtract", "smaller")),
    ],
    "binary_search": [
        CandidateTemplate("def binary_search(xs, target):\n    return xs.index(target)\n"),
        CandidateTemplate(
            "def binary_search(xs, target):\n    lo, hi = 0, len(xs) - 1\n    while lo < hi:\n"
            "        mid = (lo + hi) // 2\n        if xs[mid] == target:\n            return mid\n"
            "        if xs[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n"
            "    return -1\n", ("missing", "bounds", "binary")),
    ],
    "run_length_encode": [
        CandidateTemplate(
            "def run_length_encode(s):\n    out = []\n    i = 0\n    while i < len(s):\n        j = i\n"
            "        while j < len(s) and s[j] == s[i]:\n            j += 1\n        n = j - i\n"
            "        out.append(s[i] + (str(n) if n > 1 else ''))\n        i = j\n    return ''.join(out)\n"),
        CandidateTemplate(
            "def run_length_encode(s):\n    out = []\n    i = 0\n    while i < len(s):\n        j = i\n"
            "        while j < len(s) and s[j] == s[i]:\n            j += 1\n        out.append(s[i] + str(j - i))\n"
            "        i = j\n    return ''.join(out)\n", ("always", "count")),
    ],
    "two_sum": [
        CandidateTemplate(
            "def two_sum(nums, target):\n    pos = {x: i for i, x in enumerate(nums)}\n    for i, x in enumerate(nums):\n"
            "        if target - x in pos:\n            return sorted([i, pos[target - x]])\n    return None\n"),
        CandidateTemplate(
            "def two_sum(nums, target):\n    seen = {}\n    for i, x in enumerate(nums):\n        if target - x in seen:\n"
            "            return [seen[target - x], i]\n        seen[x] = i\n    return None\n",
            ("reuse", "complement", "none")),
    ],
    "valid_parentheses": [
        CandidateTemplate(
            "def valid_parentheses(s):\n    depth = 0\n    for c in s:\n        if c in '([{':\n            depth += 1\n"
            "        elif c in ')]}':\n            depth -= 1\n    return depth == 0\n"),
        CandidateTemplate(
            "def valid_parentheses(s):\n    pairs = {')': '(', ']': '[', '}': '{'}\n    stack = []\n    for c in s:\n"
            "        if c in '([{':\n            stack.append(c)\n        elif c in pairs:\n"
            "            if not stack or stack.pop() != pairs[c]:\n                return False\n"
            "    return not stack\n", ("stack", "match", "types")),
    ],
    "chunk": [
        CandidateTemplate("def chunk(xs, size):\n    return [xs[i:i + size] for i in range(0, len(xs), size)]\n"),
        CandidateTemplate(
            "def chunk(xs, size):\n    if size <= 0:\n        raise ValueError('size must be positive')\n"
            "    return [xs[i:i + size] for i in range(0, len(xs), size)]\n", ("validate", "valueerror", "positive")),
    ],
    "second_largest": [
        CandidateTemplate("def second_largest(xs):\n    return sorted(xs)[-2]\n"),
        CandidateTemplate(
            "def second_largest(xs):\n    if len(xs) < 2:\n        return None\n    return sorted(xs)[-2]\n", ("none",)),
        CandidateTemplate(
            "def second_largest(xs):\n    u = sorted(set(xs))\n    return u[-2] if len(u) >= 2 else None\n",
            ("distinct", "none", "duplicates")),
    ],
    "word_frequencies": [
        CandidateTemplate(
            "def word_frequencies(text):\n    out = {}\n    for w in text.split():\n        out[w] = out.get(w, 0) + 1\n"
            "    return out\n"),
        CandidateTemplate(
            "import re\n\ndef word_frequencies(text):\n    out = {}\n    for w in re.findall(r'[a-z0-9]+', text.lower()):\n"
            "        out[w] = out.get(w, 0) + 1\n    return out\n", ("lowercase", "punctuation", "regex", "insensitive")),
    ],
    "median": [
        CandidateTemplate("def median(xs):\n    return sorted(xs)[len(xs) // 2]\n"),
        CandidateTemplate(
            "def median(xs):\n    s = sorted(xs)\n    n = len(s)\n    if n % 2:\n        return s[n // 2]\n"
            "    return (s[n // 2 - 1] + s[n // 2]) / 2\n", ("average", "even", "middle")),
        CandidateTemplate(
            "def median(xs):\n    if not xs:\n        raise ValueError('median of empty list')\n    s = sorted(xs)\n"
            "    n = len(s)\n    if n % 2:\n        return s[n // 2]\n    return (s[n // 2 - 1] + s[n // 2]) / 2\n",
            ("average", "even", "middle", "valueerror", "empty")),
    ],
    "is_leap_year": [
        CandidateTemplate("def is_leap_year(year):\n    return year % 4 == 0\n"),
        CandidateTemplate("def is_leap_year(year):\n    return year % 4 == 0 and year % 100 != 0\n", ("century",)),
        CandidateTemplate(
            "def is_leap_year(year):\n    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)\n",
            ("century", "400")),
    ],
}

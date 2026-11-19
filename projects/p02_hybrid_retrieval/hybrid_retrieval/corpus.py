"""Bundled synthetic-but-realistic labeled corpus (product support knowledge base).

6 product domains x 5 products x 8 support aspects = 240 documents, each
written from aspect-specific templates with randomized specifics (prices,
day counts, error codes, part numbers), domain filler, and occasional
mentions of sibling products (hard negatives for lexical matching).

Four query families stress different retrievers:

* ``keyword``    - pasted identifiers (error codes, part numbers) wrapped in
                   generic symptom words. BM25's IDF nails the rare token;
                   dense similarity is diluted by the generic words. Sparse wins.
* ``paraphrase`` - misspelled product name + synonyms for the aspect
                   ("how expensive is Veltrra" vs. a doc about "pricing").
                   No exact overlap for BM25; char n-grams + concept features
                   of the dense embedder still match. Dense wins.
* ``mixed``      - exact product name + synonym aspect ("Veltra X2
                   reimbursement"): BM25 finds the product, dense finds the
                   aspect - only fusion gets both.
* ``lexical``    - the doc's own vocabulary; everyone should do well.

Relevance: the target doc is grade 2; other docs about the same product are
grade 1 (right product, wrong question).
"""
from __future__ import annotations

import random
from collections import defaultdict

from .types import Document, Query

DOMAINS: dict[str, dict] = {
    "automotive": {
        "noun": "car",
        "products": ["Veltra X2", "Kestrel S4", "Orvane GT", "Brisa E7", "Tamsin R1"],
        "filler": [
            "Tire pressure should be checked monthly at a certified dealer.",
            "The infotainment screen supports wireless phone mirroring.",
            "Highway mileage improves noticeably with eco driving mode enabled.",
            "Dealers can schedule service visits through the owner portal.",
            "Winter packages include heated seats and a battery blanket.",
            "Brake pads are inspected at every scheduled service interval.",
            "The towing capacity varies with the trim level and axle ratio.",
            "Owners receive over-the-air updates for the navigation maps.",
        ],
    },
    "smart_home": {
        "noun": "device",
        "products": ["Lumora Hub", "Nestiva Cam", "Quillon Lock", "Arbo Thermostat", "Sylph Speaker"],
        "filler": [
            "Pairing with the mobile app requires Bluetooth and location permissions.",
            "Firmware updates install overnight while the home is quiet.",
            "Rooms and scenes can be grouped for voice assistant control.",
            "The unit works on 2.4 GHz wifi networks with WPA2 encryption.",
            "Battery level is shown in the household dashboard.",
            "Guests can be granted temporary access from the family settings.",
            "Motion zones reduce notifications from passing traffic.",
            "Energy reports summarize usage for every room each week.",
        ],
    },
    "cloud_database": {
        "noun": "database cluster",
        "products": ["Strata DB", "Cobalt Store", "Ferrix SQL", "Helion Cache", "Zephyr Graph"],
        "filler": [
            "Read replicas can be placed in up to five regions.",
            "Automated backups are retained according to the snapshot schedule.",
            "Sharding distributes tables across nodes by partition key.",
            "Query plans are visible in the console under the insights tab.",
            "Connections are encrypted with TLS by default.",
            "Schema migrations run online without locking tables.",
            "Storage autoscaling adds capacity in ten gigabyte increments.",
            "Audit logs record every administrative action on the cluster.",
        ],
    },
    "banking_app": {
        "noun": "account",
        "products": ["Pennant Pay", "Corvid Wallet", "Mirelle Bank", "Ostrum Card", "Quayle Savings"],
        "filler": [
            "Transfers between linked accounts settle within one business day.",
            "Monthly statements are available as downloadable files.",
            "Spending insights categorize transactions automatically.",
            "Virtual cards can be frozen instantly from the wallet screen.",
            "Direct deposit details are listed under account information.",
            "Savings goals round up purchases to the nearest dollar.",
            "Foreign transactions show the exchange rate applied.",
            "Statements include a summary of interest earned this year.",
        ],
    },
    "telehealth": {
        "noun": "appointment",
        "products": ["Medora Connect", "Curalis Visit", "Theron Care", "Vesta Clinic", "Ilex Health"],
        "filler": [
            "Video consultations are available every day of the week.",
            "Prescriptions can be sent to any pharmacy in the network.",
            "Insurance details are verified before each consultation.",
            "Lab results appear in the patient portal within two days.",
            "Patients can message their care team between visits.",
            "Visit summaries include next steps and follow-up dates.",
            "Interpreters can join consultations on request.",
            "Records can be shared securely with outside specialists.",
        ],
    },
    "project_mgmt": {
        "noun": "workspace",
        "products": ["Taskwell Teams", "Planara Suite", "Kanbrio Boards", "Roadmapr Pro", "Sprintly Cloud"],
        "filler": [
            "Boards can be organized by sprint, owner, or due date.",
            "Integrations sync issues with popular code hosting services.",
            "Gantt views show dependencies across milestones.",
            "Guests can comment on tasks without a paid seat.",
            "Automations move cards when a pull request is merged.",
            "Time tracking reports roll up hours by project.",
            "Templates help teams standardize recurring workflows.",
            "Workload views highlight overallocated team members.",
        ],
    },
}

ASPECTS: dict[str, dict] = {
    "pricing": {
        "title": "pricing and plans",
        "doc": [
            "Pricing for {p} is ${price} per month on the standard tier.",
            "Annual billing lowers the monthly rate by {pct} percent, and the premium tier adds priority support.",
        ],
        "lexical": ["{p} pricing per month", "{p} pricing tier"],
        "synonym": ["how expensive is {p}", "is {p} cheap", "what would I spend on {p}", "{p} price"],
    },
    "refund": {
        "title": "refund policy",
        "doc": [
            "{p} offers a full refund within {days} days of the original order.",
            "To request a refund, contact support with your order number and receipt.",
        ],
        "lexical": ["{p} refund policy", "{p} refund order receipt"],
        "synonym": ["reimbursement for {p}", "can {p} reimburse me", "{p} reimbursement"],
    },
    "login": {
        "title": "login and password help",
        "doc": [
            "If login fails on {p}, reset your password from the account settings page.",
            "Two-factor codes are required after {n} failed login attempts.",
        ],
        "lexical": ["{p} login password reset", "{p} login fails"],
        "synonym": ["cannot signin to {p}", "unable to authenticate with {p}", "{p} credentials rejected"],
    },
    "errors": {
        "title": "error and crash troubleshooting",
        "doc": [
            "{p} may crash with error code {code} when the local cache is corrupted.",
            "Clearing cached data resolves the crash in most cases; persistent errors should be reported.",
        ],
        "lexical": ["{p} error crash troubleshooting", "{p} crash cache"],
        "synonym": ["{p} keeps hitting a bug", "{p} fault", "exception thrown by {p}"],
        "keyword": [
            "{code} keeps appearing after the update, please help",
            "what does {code} mean on my {noun}",
            "seeing {code} again since yesterday",
        ],
    },
    "performance": {
        "title": "latency and performance",
        "doc": [
            "{p} keeps latency under {ms} milliseconds for typical workloads.",
            "Performance tuning guides recommend enabling the local cache and reviewing usage peaks.",
        ],
        "lexical": ["{p} latency performance", "{p} performance tuning"],
        "synonym": ["is {p} fast", "{p} speed", "how quick is {p}"],
    },
    "startup": {
        "title": "boot and start-up issues",
        "doc": [
            "The {p} boot sequence takes about {sec} seconds after a power cycle.",
            "If {p} does not start, hold the reset button for ten seconds and try again.",
        ],
        "lexical": ["{p} boot sequence", "{p} does not start"],
        "synonym": ["{p} will not launch", "how to initialize {p}", "{p} won't begin"],
    },
    "deletion": {
        "title": "account and data deletion",
        "doc": [
            "You can delete your {p} profile and all stored data from the privacy menu.",
            "Deleted records are purged permanently after {days} days.",
        ],
        "lexical": ["{p} delete profile data", "{p} deleted records purged"],
        "synonym": ["erase everything from {p}", "remove my data on {p}", "how to erase {p} history"],
    },
    "purchase": {
        "title": "where to purchase",
        "doc": [
            "{p} can be purchased online or through authorized partners using part number {part}.",
            "Volume purchase discounts apply to orders of {n} units or more.",
        ],
        "lexical": ["where to purchase {p}", "{p} purchase partners"],
        "synonym": ["where can I buy {p}", "acquire {p} for my team", "{p} buy"],
        "keyword": [
            "is {part} in stock",
            "order {part} for my {noun}",
            "{part} availability and shipping",
        ],
    },
}

SOURCES = ("docs", "support", "forum", "blog")


def misspell(word: str, rng: random.Random) -> str:
    """Deterministic, realistic typo: doubled letter, swapped letters, or dropped vowel."""
    if len(word) < 4:
        return word + word[-1]
    for _ in range(10):
        i = rng.randrange(1, len(word) - 1)
        op = rng.choice(("double", "swap", "drop"))
        if op == "double":
            out = word[: i + 1] + word[i] + word[i + 1 :]
        elif op == "swap" and i + 1 < len(word):
            out = word[:i] + word[i + 1] + word[i] + word[i + 2 :]
        else:
            vowels = [j for j in range(1, len(word) - 1) if word[j].lower() in "aeiou"]
            if not vowels:
                continue
            j = rng.choice(vowels)
            out = word[:j] + word[j + 1 :]
        if out.lower() != word.lower():
            return out
    return word[:-1]


def build_corpus(seed: int = 7) -> list[Document]:
    """Generate the 240-document knowledge base deterministically."""
    rng = random.Random(seed)
    used_codes: set[str] = set()
    docs: list[Document] = []

    def unique(fn) -> str:
        while True:
            v = fn()
            if v not in used_codes:
                used_codes.add(v)
                return v

    for domain, dspec in DOMAINS.items():
        for p in dspec["products"]:
            siblings = [q for q in dspec["products"] if q != p]
            for aspect, aspec in ASPECTS.items():
                fields = {
                    "p": p,
                    "price": rng.choice([9, 12, 19, 24, 29, 39, 49, 79, 99]),
                    "pct": rng.choice([10, 15, 20, 25]),
                    "days": rng.choice([14, 30, 45, 60, 90]),
                    "n": rng.choice([3, 5, 10, 25, 50]),
                    "ms": rng.choice([5, 10, 20, 50, 120]),
                    "sec": rng.choice([8, 12, 20, 30, 45]),
                    "code": unique(lambda: f"0x{rng.randrange(16**5, 16**6):06x}"),
                    "part": unique(lambda: f"PN{rng.randrange(10000, 99999)}"),
                }
                sentences = [s.format(**fields) for s in aspec["doc"]]
                sentences += rng.sample(dspec["filler"], 2)
                if rng.random() < 0.35:
                    sentences.append(f"Owners migrating from {rng.choice(siblings)} can import their settings.")
                rest = sentences[1:]
                rng.shuffle(rest)
                sentences = [sentences[0], *rest]  # keep the topical lead sentence first
                doc_id = f"{domain[:4]}-{p.split()[0].lower()}-{aspect}"
                docs.append(
                    Document(
                        id=doc_id,
                        title=f"{p}: {aspec['title']}",
                        text=" ".join(sentences),
                        metadata={
                            "domain": domain,
                            "product": p,
                            "aspect": aspect,
                            "year": rng.randint(2018, 2025),
                            "source": rng.choice(SOURCES),
                            "code": fields["code"] if aspect == "errors" else None,
                            "part": fields["part"] if aspect == "purchase" else None,
                        },
                    )
                )
    return docs


def build_queries(docs: list[Document], seed: int = 11, per_type: int = 60) -> list[Query]:
    """Generate `per_type` queries for each family with graded qrels."""
    rng = random.Random(seed)
    by_product: dict[str, list[str]] = defaultdict(list)
    for d in docs:
        by_product[d.metadata["product"]].append(d.id)

    def qrels_for(d: Document) -> dict[str, int]:
        rel = {i: 1 for i in by_product[d.metadata["product"]]}
        rel[d.id] = 2
        return rel

    queries: list[Query] = []
    families = {
        "lexical": docs,
        "paraphrase": docs,
        "mixed": docs,
        "keyword": [d for d in docs if "keyword" in ASPECTS[d.metadata["aspect"]]],
    }
    for qtype, pool in families.items():
        picks = rng.sample(pool, min(per_type, len(pool)))
        for n, d in enumerate(picks):
            meta = d.metadata
            aspec = ASPECTS[meta["aspect"]]
            noun = DOMAINS[meta["domain"]]["noun"]
            product = meta["product"]
            if qtype == "lexical":
                text = rng.choice(aspec["lexical"]).format(p=product)
            elif qtype == "mixed":
                text = rng.choice(aspec["synonym"]).format(p=product)
            elif qtype == "paraphrase":
                typo = misspell(product.split()[0], rng)
                text = rng.choice(aspec["synonym"]).format(p=typo)
            else:
                ident = meta["code"] if meta["aspect"] == "errors" else meta["part"]
                text = rng.choice(aspec["keyword"]).format(code=ident, part=ident, noun=noun)
                # an identifier query is about that exact doc only
                queries.append(Query(f"{qtype}-{n:03d}", text, {d.id: 2}, qtype))
                continue
            queries.append(Query(f"{qtype}-{n:03d}", text, qrels_for(d), qtype))
    return queries


def split_queries(queries: list[Query], train_frac: float = 0.5, seed: int = 3) -> tuple[list[Query], list[Query]]:
    """Stratified (per query family) deterministic train/test split."""
    rng = random.Random(seed)
    by_type: dict[str, list[Query]] = defaultdict(list)
    for q in queries:
        by_type[q.qtype].append(q)
    train: list[Query] = []
    test: list[Query] = []
    for qtype in sorted(by_type):
        qs = list(by_type[qtype])
        rng.shuffle(qs)
        cut = int(round(len(qs) * train_frac))
        train += qs[:cut]
        test += qs[cut:]
    return train, test


def load_benchmark(seed: int = 7) -> tuple[list[Document], list[Query]]:
    """Convenience: the canonical corpus and its 240 labeled queries."""
    docs = build_corpus(seed)
    return docs, build_queries(docs)

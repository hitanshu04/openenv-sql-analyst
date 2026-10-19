"""
Generate data/mock_data.sql, the benchmark database.

    python data/generate.py           # rewrite data/mock_data.sql
    python data/generate.py --check   # verify only, write nothing

Rows are random but seeded, so every run produces the same file. A few rows
are then adjusted on purpose so each task tests what it claims, and check()
refuses data where an answer would be ambiguous or a trap missing. ORACLES
compute every task's answer in plain Python, independently of its SQL.
"""

import argparse
import datetime
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from environment.tasks import MIN_CATEGORIES, RANK_TARGET, REPORT_YEAR  # noqa: E402

SEED = 2026
OUT = Path(__file__).resolve().parent / "mock_data.sql"

N_PURCHASES = 400
N_RANDOM_REFUNDS = 40
N_UNSOLD = 4
COUNTRIES = ["USA", "India", "UK", "Germany", "Canada", "France", "Brazil", "Spain"]

# The original v1 users and products are kept, so familiar names still exist.
V1_USERS = [
    (1, "alice", "USA", "2023-01-15"), (2, "bob", "Canada", "2023-02-20"),
    (3, "charlie", "UK", "2023-03-10"), (4, "diana", "USA", "2023-04-05"),
    (5, "eve", "Germany", "2023-05-12"), (6, "frank", "France", "2023-06-18"),
    (7, "grace", "USA", "2023-07-22"), (8, "henry", "Canada", "2023-08-30"),
    (9, "iris", "UK", "2023-09-14"), (10, "jack", "USA", "2023-10-01"),
    (11, "karen", "Germany", "2023-10-15"), (12, "leo", "France", "2023-11-02"),
    (13, "maria", "Spain", "2023-11-20"), (14, "nathan", "USA", "2023-12-05"),
    (15, "olivia", "Canada", "2023-12-18"),
]
NEW_USERNAMES = [
    "aaron", "bianca", "carlos", "daisy", "ethan", "fiona", "george", "hannah",
    "isaac", "julia", "kevin", "lily", "mason", "nora", "oscar", "priya", "quinn",
    "rachel", "samir", "tina", "umar", "vera", "wyatt", "ximena", "yara", "zane",
    "arjun", "bella", "chen", "devika", "elena", "farah", "gabriel", "hiro",
    "isla", "jonas", "kiara", "lucas", "meera", "nikhil", "omar", "paula", "ravi",
    "sofia", "tara",
]
V1_PRODUCTS = [
    (1, "Laptop Pro", "Electronics", 1299.99, 50), (2, "Wireless Mouse", "Electronics", 29.99, 200),
    (3, "USB-C Hub", "Electronics", 49.99, 150), (4, "Mechanical Keyboard", "Electronics", 89.99, 100),
    (5, 'Monitor 27"', "Electronics", 349.99, 75), (6, "Desk Chair", "Furniture", 199.99, 40),
    (7, "Standing Desk", "Furniture", 449.99, 25), (8, "Desk Lamp", "Furniture", 34.99, 120),
    (9, "Notebook Pack", "Office", 12.99, 300), (10, "Pen Set", "Office", 8.99, 500),
    (11, "Headphones", "Electronics", 149.99, 80), (12, "Webcam HD", "Electronics", 79.99, 90),
    (13, "Mousepad XL", "Electronics", 19.99, 250), (14, "Cable Organizer", "Office", 14.99, 180),
    (15, "Monitor Stand", "Furniture", 59.99, 60),
]
NEW_PRODUCTS = [
    ("SQL Pocket Guide", "Books", 29.99), ("Data Science Handbook", "Books", 49.99),
    ("Mystery Novel", "Books", 14.99), ("Cookbook", "Books", 24.99),
    ("Coffee Maker", "Kitchen", 89.99), ("Chef Knife", "Kitchen", 59.99),
    ("Blender", "Kitchen", 69.99), ("Water Bottle", "Kitchen", 19.99),
    ("Yoga Mat", "Sports", 29.99), ("Running Shoes", "Sports", 119.99),
    ("Dumbbell Set", "Sports", 149.99), ("Fitness Tracker", "Sports", 99.99),
    ("Hoodie", "Clothing", 44.99), ("Rain Jacket", "Clothing", 89.99),
    ("Wool Socks", "Clothing", 12.99),
]

# hard_top_buyer_tiebreak: these three tie for the most purchases. The
# alphabetical winner (bianca, id 17) sits between the lowest id (olivia, 15)
# and the highest (zane, 41), so a query without the tie-break is wrong whether
# SQLite keeps the first or the last of the tied rows (3.46 keeps the last).
TIED_BUYERS = ("olivia", "bianca", "zane")


@dataclass
class Dataset:
    users: list
    products: list
    purchases: list
    refunds: list
    signups: list = field(default_factory=list)


def _date(rng, year):
    return f"{year}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"


def _add_purchase(purchases, rng, user, product):
    quantity = rng.choice([1, 1, 1, 2, 2, 3])
    purchases.append({
        "purchase_id": len(purchases) + 1,
        "user_id": user["user_id"],
        "product_id": product["product_id"],
        "quantity": quantity,
        "purchase_date": _date(rng, REPORT_YEAR),
        "total_amount": round(product["price"] * quantity, 2),
    })


def _add_refund(refunds, rng, purchase, full):
    amount = purchase["total_amount"] if full else round(
        purchase["total_amount"] * rng.choice([0.25, 0.5]), 2)
    day = datetime.date.fromisoformat(purchase["purchase_date"])
    refunds.append({
        "refund_id": len(refunds) + 1,
        "purchase_id": purchase["purchase_id"],
        "refund_amount": amount,
        "refund_date": (day + datetime.timedelta(days=rng.randint(1, 20))).isoformat(),
    })


def build(seed=SEED):
    rng = random.Random(seed)

    users = [{"user_id": i, "username": n, "email": f"{n}@example.com", "country": c,
              "created_at": d} for i, n, c, d in V1_USERS]
    for name in NEW_USERNAMES:
        users.append({"user_id": len(users) + 1, "username": name,
                      "email": f"{name}@example.com", "country": rng.choice(COUNTRIES),
                      "created_at": _date(rng, REPORT_YEAR - 1)})

    products = [{"product_id": i, "product_name": n, "category": c, "price": p,
                 "stock": s} for i, n, c, p, s in V1_PRODUCTS]
    for name, category, price in NEW_PRODUCTS:
        products.append({"product_id": len(products) + 1, "product_name": name,
                         "category": category, "price": price,
                         "stock": rng.randint(20, 300)})

    # medium_unsold_products: some products are never offered for sale.
    unsold = set(rng.sample([p["product_id"] for p in products], N_UNSOLD))
    for_sale = [p for p in products if p["product_id"] not in unsold]

    weights = [rng.uniform(0.3, 2.0) for _ in users]
    purchases = []
    for _ in range(N_PURCHASES):
        _add_purchase(purchases, rng, rng.choices(users, weights)[0], rng.choice(for_sale))
    _force_top_buyer_tie(purchases, rng, users, for_sale)

    refunds = []
    for purchase in sorted(rng.sample(purchases, N_RANDOM_REFUNDS), key=lambda p: p["purchase_id"]):
        _add_refund(refunds, rng, purchase, full=rng.random() < 0.5)
    _force_net_revenue_upset(refunds, rng, users, purchases)

    # Generated last, so every table above is unchanged by its existence.
    signups = _signups(rng)
    return Dataset(users, products, purchases, refunds, signups)


# hard_us_signups_dirty_country: the same country, typed by users. Exact
# "USA" is under half of them, so filtering on it misses most US signups.
US_SPELLINGS = ["USA", "usa", "US", "United States", "united states ", " U.S.", "U.S.A"]
US_WEIGHTS = [4, 2, 2, 2, 1, 1, 1]
OTHER_COUNTRIES = ["India", "india ", "UK", "United Kingdom", "Germany", "germany",
                   "Canada", "Brazil", "France"]
N_SIGNUPS = 150
N_REPEAT_SIGNUPS = 25  # hard_distinct_real_signups: same person, email cased or spaced differently
N_TEST_SIGNUPS = 10    # internal test accounts that must be excluded


def _signups(rng):
    names = [n for _, n, _, _ in V1_USERS] + NEW_USERNAMES
    rows = []
    for k in range(N_SIGNUPS):
        us = rng.random() < 0.4
        rows.append({
            "email": f"{rng.choice(names)}.{k}@example.com",
            "country": rng.choices(US_SPELLINGS, US_WEIGHTS)[0] if us else rng.choice(OTHER_COUNTRIES),
            "channel": rng.choice(["organic", "ads", "referral", "affiliate"]),
            "signup_date": _date(rng, REPORT_YEAR),
        })
    for original in rng.sample(rows, N_REPEAT_SIGNUPS):
        email = original["email"]
        variant = rng.choice([email.upper(), email.capitalize(), f" {email}", f"{email} "])
        rows.append({**original, "email": variant, "signup_date": _date(rng, REPORT_YEAR)})
    for k in range(N_TEST_SIGNUPS):
        email = rng.choice([f"qa{k}@sqlgym.test", f"QA{k}@SQLGYM.TEST", f"qa{k}@sqlgym.test "])
        rows.append({"email": email, "country": "N/A", "channel": "internal",
                     "signup_date": _date(rng, REPORT_YEAR)})
    rng.shuffle(rows)
    return [{"signup_id": i, **row} for i, row in enumerate(rows, start=1)]


def _force_top_buyer_tie(purchases, rng, users, for_sale):
    by_name = {u["username"]: u for u in users}
    counts = Counter(p["user_id"] for p in purchases)
    target = max(counts.values()) + 1
    for name in TIED_BUYERS:
        user = by_name[name]
        while counts[user["user_id"]] < target:
            _add_purchase(purchases, rng, user, rng.choice(for_sale))
            counts[user["user_id"]] += 1


def _force_net_revenue_upset(refunds, rng, users, purchases):
    """hard_net_revenue_country: refund the top gross country's biggest
    purchases until a different country leads on net revenue."""
    gross, net = country_revenue(Dataset(users, [], purchases, refunds))
    leader = max(gross, key=gross.get)
    country = {u["user_id"]: u["country"] for u in users}
    refunded = {r["purchase_id"] for r in refunds}
    candidates = sorted(
        (p for p in purchases if country[p["user_id"]] == leader and p["purchase_id"] not in refunded),
        key=lambda p: -p["total_amount"],
    )
    for purchase in candidates:
        if net[leader] < 0.97 * max(v for c, v in net.items() if c != leader):
            return
        _add_refund(refunds, rng, purchase, full=True)
        net[leader] -= purchase["total_amount"]


# --------------------------------------------------------------------------
# Plain-Python views of the data, shared by ORACLES and check()
# --------------------------------------------------------------------------

def user_spend(ds):
    names = {u["user_id"]: u["username"] for u in ds.users}
    spend = defaultdict(float)
    for p in ds.purchases:
        spend[names[p["user_id"]]] += p["total_amount"]
    return spend


def country_revenue(ds):
    country = {u["user_id"]: u["country"] for u in ds.users}
    refund = {r["purchase_id"]: r["refund_amount"] for r in ds.refunds}
    gross, net = defaultdict(float), defaultdict(float)
    for p in ds.purchases:
        c = country[p["user_id"]]
        gross[c] += p["total_amount"]
        net[c] += p["total_amount"] - refund.get(p["purchase_id"], 0.0)
    return gross, net


def month_revenue(ds):
    revenue = defaultdict(float)
    for p in ds.purchases:
        if p["purchase_date"].startswith(f"{REPORT_YEAR}-"):
            revenue[p["purchase_date"][:7]] += p["total_amount"]
    return revenue


def purchase_counts(ds):
    names = {u["user_id"]: u["username"] for u in ds.users}
    return Counter(names[p["user_id"]] for p in ds.purchases)


def categories_bought(ds):
    category = {p["product_id"]: p["category"] for p in ds.products}
    bought = defaultdict(set)
    for p in ds.purchases:
        bought[p["user_id"]].add(category[p["product_id"]])
    return bought


def _top(scores):
    return max(scores, key=scores.get)


def is_us(country):
    return country.strip().lower().replace(".", "") in ("usa", "us", "united states")


def real_people(ds):
    emails = {s["email"].strip().lower() for s in ds.signups}
    return {e for e in emails if not e.endswith("@sqlgym.test")}


def quarter_change(ds):
    """Relative and absolute Q2 -> Q3 revenue change per category."""
    category = {p["product_id"]: p["category"] for p in ds.products}
    q2, q3 = defaultdict(float), defaultdict(float)
    for p in ds.purchases:
        month = int(p["purchase_date"][5:7])
        c = category[p["product_id"]]
        q2[c] += p["total_amount"] if 4 <= month <= 6 else 0.0
        q3[c] += p["total_amount"] if 7 <= month <= 9 else 0.0
    return ({c: (q3[c] - q2[c]) / q2[c] for c in q2 if q2[c]},
            {c: q3[c] - q2[c] for c in q2})


def repeat_buyers(ds, from_first=True):
    dates = defaultdict(list)
    for p in ds.purchases:
        dates[p["user_id"]].append(datetime.date.fromisoformat(p["purchase_date"]))
    if from_first:
        return sum(1 for ds_ in dates.values()
                   if any(1 <= (d - min(ds_)).days <= 30 for d in ds_))
    return sum(1 for ds_ in dates.values()
               if any(1 <= (b - a).days <= 30 for a in ds_ for b in ds_))


def median(values):
    values = sorted(values)
    mid = len(values) // 2
    return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2


ORACLES = {
    "easy_user_count": lambda ds: len(ds.users),
    "medium_usa_revenue": lambda ds: round(country_revenue(ds)[0]["USA"], 2),
    "hard_top_spender": lambda ds: _top(user_spend(ds)),
    "medium_unsold_products": lambda ds: len(ds.products) - len({p["product_id"] for p in ds.purchases}),
    "medium_multi_category_buyers": lambda ds: sum(
        1 for cats in categories_bought(ds).values() if len(cats) >= MIN_CATEGORIES),
    "medium_best_month": lambda ds: _top(month_revenue(ds)),
    "hard_spend_rank": lambda ds: 1 + sum(
        1 for v in user_spend(ds).values() if v > user_spend(ds)[RANK_TARGET]),
    "hard_net_revenue_country": lambda ds: _top(country_revenue(ds)[1]),
    "hard_top_buyer_tiebreak": lambda ds: min(
        purchase_counts(ds).items(), key=lambda item: (-item[1], item[0]))[0],
    "hard_us_signups_dirty_country": lambda ds: sum(1 for s in ds.signups if is_us(s["country"])),
    "hard_distinct_real_signups": lambda ds: len(real_people(ds)),
    "hard_category_q2_q3_drop": lambda ds: min(quarter_change(ds)[0].items(), key=lambda kv: kv[1])[0],
    "hard_repeat_within_30_days": lambda ds: repeat_buyers(ds),
    "medium_median_order_value": lambda ds: round(median(p["total_amount"] for p in ds.purchases), 2),
}


def check(ds):
    """Every way the data could make a task ambiguous or toothless."""
    problems = []

    def clear_winner(scores, label):
        first, second = sorted(scores.values(), reverse=True)[:2]
        if first - second < 0.01:
            problems.append(f"{label}: tie at the top")

    names = [u["username"] for u in ds.users]
    if len(set(names)) != len(names):
        problems.append("usernames are not unique")
    if len({r["purchase_id"] for r in ds.refunds}) != len(ds.refunds):
        problems.append("a purchase is refunded twice (would double-count in joins)")

    spend = user_spend(ds)
    clear_winner(spend, "hard_top_spender")

    gross, net = country_revenue(ds)
    clear_winner(net, "hard_net_revenue_country")
    if _top(gross) == _top(net):
        problems.append("hard_net_revenue_country: gross and net leaders are the same, no trap")

    revenue_by_month = month_revenue(ds)
    clear_winner(revenue_by_month, "medium_best_month")
    rows_by_month = Counter(p["purchase_date"][:7] for p in ds.purchases)
    if _top(revenue_by_month) == rows_by_month.most_common(1)[0][0]:
        problems.append("medium_best_month: counting rows finds the same month as summing revenue, no trap")

    counts = purchase_counts(ds)
    most = max(counts.values())
    tied = sorted(name for name, c in counts.items() if c == most)
    user_id = {u["username"]: u["user_id"] for u in ds.users}
    if len(tied) < 3:
        problems.append("hard_top_buyer_tiebreak: fewer than three users tied at the top")
    elif tied[0] in (min(tied, key=user_id.get), max(tied, key=user_id.get)):
        problems.append("hard_top_buyer_tiebreak: tie-break winner has the lowest or highest id, no trap")

    unsold = ORACLES["medium_unsold_products"](ds)
    if not 0 < unsold < len(ds.products):
        problems.append(f"medium_unsold_products: answer {unsold} is trivial")

    buyers = len(spend)
    multi = ORACLES["medium_multi_category_buyers"](ds)
    if not 0.2 * buyers <= multi <= 0.8 * buyers:
        problems.append(f"medium_multi_category_buyers: {multi} of {buyers} buyers is trivial")

    rank = ORACLES["hard_spend_rank"](ds)
    if not 5 <= rank <= buyers - 5:
        problems.append(f"hard_spend_rank: rank {rank} of {buyers} is trivial")
    if sum(1 for v in spend.values() if abs(v - spend[RANK_TARGET]) < 0.01) > 1:
        problems.append("hard_spend_rank: the target's spend is tied")

    # Harder tasks: each trap must miss the truth by more than the 10% partial-credit band.
    def far(naive, truth):
        return abs(naive - truth) > 0.1 * abs(truth)

    us = ORACLES["hard_us_signups_dirty_country"](ds)
    if not far(sum(1 for s in ds.signups if s["country"] == "USA"), us):
        problems.append("hard_us_signups_dirty_country: exact 'USA' is close to the answer, no trap")

    people = len(real_people(ds))
    raw = len({s["email"] for s in ds.signups if not s["email"].lower().endswith("@sqlgym.test")})
    if not far(raw, people):
        problems.append("hard_distinct_real_signups: raw-email count is close to the answer, no trap")

    pct, absolute = quarter_change(ds)
    ranked = sorted(pct.values())
    if ranked[0] >= 0 or ranked[1] - ranked[0] < 0.01:
        problems.append("hard_category_q2_q3_drop: no clear percentage drop")
    if min(pct, key=pct.get) == min(absolute, key=absolute.get):
        problems.append("hard_category_q2_q3_drop: absolute and percentage drops agree, no trap")

    repeat = repeat_buyers(ds)
    if not 0.2 * buyers <= repeat <= 0.8 * buyers:
        problems.append(f"hard_repeat_within_30_days: {repeat} of {buyers} buyers is trivial")
    if not far(repeat_buyers(ds, from_first=False), repeat):
        problems.append("hard_repeat_within_30_days: any-two-purchases count is close, no trap")

    totals = [p["total_amount"] for p in ds.purchases]
    if not far(sum(totals) / len(totals), median(totals)):
        problems.append("medium_median_order_value: mean is close to the median, no trap")

    return problems


# --------------------------------------------------------------------------
# SQL output
# --------------------------------------------------------------------------

SCHEMA = """\
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY,
    username TEXT NOT NULL UNIQUE,
    email TEXT NOT NULL,
    country TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS products (
    product_id INTEGER PRIMARY KEY,
    product_name TEXT NOT NULL,
    category TEXT NOT NULL,
    price REAL NOT NULL,
    stock INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS purchases (
    purchase_id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    quantity INTEGER NOT NULL,
    purchase_date TEXT NOT NULL,
    total_amount REAL NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(user_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

CREATE TABLE IF NOT EXISTS refunds (
    refund_id INTEGER PRIMARY KEY,
    purchase_id INTEGER NOT NULL UNIQUE,
    refund_amount REAL NOT NULL,
    refund_date TEXT NOT NULL,
    FOREIGN KEY (purchase_id) REFERENCES purchases(purchase_id)
);

CREATE TABLE IF NOT EXISTS signups (
    signup_id INTEGER PRIMARY KEY,
    email TEXT NOT NULL,
    country TEXT NOT NULL,
    channel TEXT NOT NULL,
    signup_date TEXT NOT NULL
);
"""


def _sql_value(value):
    if isinstance(value, float):
        return f"{value:.2f}"
    if isinstance(value, int):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


def _inserts(table, rows, chunk=100):
    columns = list(rows[0])
    out = []
    for start in range(0, len(rows), chunk):
        values = ",\n".join(
            "(" + ", ".join(_sql_value(row[c]) for c in columns) + ")"
            for row in rows[start:start + chunk]
        )
        out.append(f"INSERT INTO {table} ({', '.join(columns)}) VALUES\n{values};\n")
    return "\n".join(out)


def to_sql(ds):
    return "\n".join([
        f"-- SQLGym benchmark data. GENERATED by data/generate.py (seed {SEED}).",
        "-- Do not edit by hand: change the generator and rerun it.",
        "",
        SCHEMA,
        _inserts("users", ds.users),
        _inserts("products", ds.products),
        _inserts("purchases", ds.purchases),
        _inserts("refunds", ds.refunds),
        _inserts("signups", ds.signups),
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--check", action="store_true", help="verify only, write nothing")
    args = parser.parse_args()

    ds = build()
    problems = check(ds)
    for problem in problems:
        print(f"PROBLEM: {problem}")
    print(f"{len(ds.users)} users, {len(ds.products)} products, "
          f"{len(ds.purchases)} purchases, {len(ds.refunds)} refunds")
    for task_id, oracle in ORACLES.items():
        print(f"  {task_id:<30} {oracle(ds)}")
    if problems:
        return 1
    if not args.check:
        OUT.write_text(to_sql(ds))
        print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

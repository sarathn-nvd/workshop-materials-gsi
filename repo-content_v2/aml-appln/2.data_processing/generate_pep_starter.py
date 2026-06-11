"""Generate a synthetic PEP (Politically Exposed Person) starter list.

This is a placeholder seed pool used by Stage 4 of the SFT synthetic data
generation pipeline (see sgd_understanding.md). It will be replaced once the
OpenSanctions consolidated PEP dataset is ingested.

Output: 2.data_processing/data/transactional/pep_starter/names.csv
Schema mirrors OFAC's targets.simple.csv so Stage 4 can union the two pools.

The names are deterministically generated from a fixed RNG seed and combine
common regional first names with common surnames in unlikely permutations.
They are NOT real people. The dataset column is tagged accordingly.
"""
from __future__ import annotations

import csv
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

SEED = 42
N_ENTRIES = 200
OUT_DIR = Path(__file__).parent / "data" / "transactional" / "pep_starter"
OUT_CSV = OUT_DIR / "names.csv"
OUT_README = OUT_DIR / "README.md"

# Country weights — high-risk jurisdictions get larger share to mirror the
# distribution Stage 4's sampler is expected to draw from.
COUNTRIES: list[tuple[str, str, float]] = [
    ("ru", "Russia",            0.10),
    ("cn", "China",             0.10),
    ("ve", "Venezuela",         0.05),
    ("ir", "Iran",              0.05),
    ("kp", "North Korea",       0.02),
    ("ng", "Nigeria",           0.06),
    ("ao", "Angola",            0.04),
    ("cd", "DR Congo",          0.04),
    ("gq", "Equatorial Guinea", 0.02),
    ("kz", "Kazakhstan",        0.04),
    ("az", "Azerbaijan",        0.03),
    ("tm", "Turkmenistan",      0.02),
    ("pk", "Pakistan",          0.03),
    ("mm", "Myanmar",           0.03),
    ("kh", "Cambodia",          0.02),
    ("sy", "Syria",             0.02),
    ("by", "Belarus",           0.03),
    # mid-tier risk / EU
    ("hu", "Hungary",           0.03),
    ("cy", "Cyprus",            0.02),
    ("mt", "Malta",             0.02),
    ("lu", "Luxembourg",        0.02),
    # Latin America
    ("br", "Brazil",            0.03),
    ("mx", "Mexico",            0.03),
    ("ar", "Argentina",         0.02),
    ("pa", "Panama",            0.02),
    # baseline US/EU spread
    ("us", "United States",     0.05),
    ("gb", "United Kingdom",    0.03),
    ("de", "Germany",           0.02),
    ("fr", "France",            0.02),
]

# Roles — first element is program_ids tag, second is human-readable
ROLES: list[tuple[str, str, float]] = [
    ("HoS-current",     "Head of State / Government (current)",   0.04),
    ("HoS-former",      "Head of State / Government (former)",    0.04),
    ("Minister",        "Senior Cabinet Minister",                0.16),
    ("Legislator",      "Senior Legislator",                      0.10),
    ("Family-PEP",      "Family Member of PEP",                   0.15),
    ("Close-Assoc",     "Close Associate of PEP",                 0.10),
    ("Judge-Senior",    "Senior Judiciary",                       0.10),
    ("Military-Senior", "Senior Military / Defense",              0.08),
    ("CB-Regulator",    "Central Bank / Financial Regulator",     0.08),
    ("SOE-Exec",        "State-Owned Enterprise Executive",       0.10),
    ("IO-Official",     "International Org Official",             0.05),
]

# Per-region first-name banks. Compact but distinctive; the generator
# permutes them with surnames so individual names are unlikely permutations.
FIRST_M: dict[str, list[str]] = {
    "ru": ["Aleksei", "Igor", "Dmitri", "Nikolai", "Sergei", "Vladimir", "Mikhail", "Pavel", "Yuri", "Konstantin"],
    "cn": ["Wei", "Jian", "Ming", "Hao", "Lei", "Xin", "Bo", "Cheng", "Yang", "Long"],
    "ve": ["Carlos", "Javier", "Eduardo", "Rafael", "Hector", "Manuel", "Ramon", "Diego"],
    "ir": ["Reza", "Ali", "Hassan", "Mehdi", "Kazem", "Farhad", "Behzad"],
    "kp": ["Jong", "Sung", "Min", "Chol", "Hyun"],
    "ng": ["Adewale", "Chukwuma", "Olumide", "Babatunde", "Emeka", "Tunde"],
    "ao": ["Joao", "Paulo", "Antonio", "Manuel"],
    "cd": ["Joseph", "Patrice", "Etienne", "Laurent"],
    "gq": ["Teodoro", "Gabriel", "Salvador"],
    "kz": ["Nurlan", "Bauyrzhan", "Daniyar", "Askar"],
    "az": ["Ilham", "Rashad", "Elchin", "Vugar"],
    "tm": ["Gurbanguly", "Berdymukhamed", "Murat"],
    "pk": ["Asif", "Imran", "Tariq", "Naveed", "Shahid"],
    "mm": ["Min", "Aung", "Thein", "Soe"],
    "kh": ["Sok", "Hun", "Vann", "Kheng"],
    "sy": ["Bashar", "Maher", "Rami", "Ayman"],
    "by": ["Aliaksandr", "Viktar", "Mikalai"],
    "hu": ["Viktor", "Laszlo", "Sandor", "Zoltan"],
    "cy": ["Andreas", "Christos", "Nikos"],
    "mt": ["Joseph", "Lawrence", "Edward"],
    "lu": ["Jean-Claude", "Henri", "Pierre"],
    "br": ["Joao", "Paulo", "Antonio", "Carlos", "Ricardo"],
    "mx": ["Juan", "Jose", "Miguel", "Francisco", "Roberto"],
    "ar": ["Mauricio", "Alberto", "Ricardo", "Diego"],
    "pa": ["Juan", "Carlos", "Ricardo"],
    "us": ["James", "Robert", "William", "David", "Michael"],
    "gb": ["James", "William", "George", "Henry", "Edward"],
    "de": ["Hans", "Klaus", "Wolfgang", "Dieter"],
    "fr": ["Jean", "Pierre", "Philippe", "Francois"],
}

FIRST_F: dict[str, list[str]] = {
    "ru": ["Olga", "Natalya", "Yelena", "Irina", "Svetlana"],
    "cn": ["Mei", "Ying", "Lan", "Xia", "Hua"],
    "ve": ["Maria", "Carmen", "Gabriela", "Yolanda"],
    "ir": ["Fatemeh", "Zahra", "Maryam"],
    "kp": ["Sun", "Yong", "Hye"],
    "ng": ["Adaeze", "Chinonso", "Folake", "Ngozi"],
    "ao": ["Maria", "Ana", "Joana"],
    "cd": ["Therese", "Marie", "Solange"],
    "gq": ["Constancia"],
    "kz": ["Aigul", "Bibigul", "Saule"],
    "az": ["Aysel", "Leyla", "Konul"],
    "tm": ["Maral", "Bibigul"],
    "pk": ["Nadia", "Fatima", "Sana"],
    "mm": ["Hla", "Su", "Khin"],
    "kh": ["Sokha", "Channary"],
    "sy": ["Asma", "Rania"],
    "by": ["Sviatlana", "Volha"],
    "hu": ["Eszter", "Anna", "Katalin"],
    "cy": ["Maria", "Eleni"],
    "mt": ["Maria", "Anna"],
    "lu": ["Marie", "Anne"],
    "br": ["Maria", "Ana", "Beatriz", "Camila"],
    "mx": ["Maria", "Guadalupe", "Patricia"],
    "ar": ["Maria", "Sofia", "Valentina"],
    "pa": ["Maria", "Ana"],
    "us": ["Mary", "Patricia", "Jennifer", "Linda"],
    "gb": ["Elizabeth", "Margaret", "Catherine"],
    "de": ["Helga", "Ingrid", "Ursula"],
    "fr": ["Marie", "Claire", "Sophie"],
}

LAST_NAMES: dict[str, list[str]] = {
    "ru": ["Volkov", "Petrov", "Smirnov", "Kuznetsov", "Sokolov", "Popov", "Lebedev", "Kozlov", "Novikov", "Morozov"],
    "cn": ["Wang", "Li", "Zhang", "Liu", "Chen", "Yang", "Huang", "Zhou", "Wu", "Zhu"],
    "ve": ["Rodriguez", "Gonzalez", "Hernandez", "Martinez", "Perez", "Garcia", "Sanchez", "Romero"],
    "ir": ["Ahmadi", "Hosseini", "Rahimi", "Karimi", "Mousavi", "Rezaei"],
    "kp": ["Kim", "Pak", "Ri", "Choe"],
    "ng": ["Adebayo", "Okonkwo", "Eze", "Ogundipe", "Okafor", "Bello"],
    "ao": ["dos Santos", "Lopes", "Pereira", "Cardoso"],
    "cd": ["Mukendi", "Kabamba", "Tshisekedi", "Mbiya"],
    "gq": ["Obiang", "Nguema", "Mbasogo"],
    "kz": ["Tokayev", "Nazarbayev", "Bekmurzayev", "Asanov"],
    "az": ["Aliyev", "Hasanov", "Mammadov", "Quliyev"],
    "tm": ["Berdymukhamedov", "Atayev", "Annayev"],
    "pk": ["Khan", "Ahmed", "Malik", "Shah", "Hussain", "Sheikh"],
    "mm": ["Aung", "Hlaing", "Maung", "Nyi"],
    "kh": ["Sen", "Heng", "Sok", "Chea"],
    "sy": ["Assad", "Makhlouf", "Shaaban", "Mualem"],
    "by": ["Lukashenka", "Karpenka", "Sianko"],
    "hu": ["Szabo", "Nagy", "Kovacs", "Toth"],
    "cy": ["Constantinou", "Georgiou", "Demetriou"],
    "mt": ["Borg", "Camilleri", "Vella"],
    "lu": ["Schneider", "Weber", "Muller"],
    "br": ["Silva", "Souza", "Oliveira", "Santos", "Ferreira"],
    "mx": ["Garcia", "Hernandez", "Lopez", "Martinez", "Rodriguez"],
    "ar": ["Garcia", "Rodriguez", "Gonzalez", "Fernandez", "Lopez"],
    "pa": ["Martinez", "Gonzalez", "Rodriguez"],
    "us": ["Smith", "Johnson", "Williams", "Brown", "Jones"],
    "gb": ["Smith", "Jones", "Taylor", "Brown", "Davies"],
    "de": ["Mueller", "Schmidt", "Schneider", "Fischer"],
    "fr": ["Martin", "Bernard", "Dubois", "Petit"],
}

# State-owned enterprise name templates by country
SOE_TEMPLATES: list[str] = [
    "Banco Central de {country}",
    "{country} National Petroleum Corporation",
    "{country} State Investment Authority",
    "Sovereign Wealth Fund of {country}",
    "{country} Development Bank",
    "{country} Telecommunications State Holding",
    "Industrial Bank of {country}",
    "{country} National Mining Company",
]


def weighted_choice(items: list[tuple], rng: random.Random):
    """Pick from list of (key, value, weight) by weight."""
    total = sum(w for *_, w in items)
    r = rng.random() * total
    cum = 0.0
    for entry in items:
        cum += entry[-1]
        if r <= cum:
            return entry
    return items[-1]


def make_id(rng: random.Random, idx: int) -> str:
    suffix = "".join(rng.choice("23456789ABCDEFGHJKMNPQRSTUVWXYZ") for _ in range(8))
    return f"PEP-SYN-{idx:03d}-{suffix}"


def make_person_name(country_code: str, rng: random.Random) -> tuple[str, list[str]]:
    use_female = rng.random() < 0.20
    pool_first = FIRST_F if use_female else FIRST_M
    first = rng.choice(pool_first.get(country_code, FIRST_M["us"]))
    last = rng.choice(LAST_NAMES.get(country_code, LAST_NAMES["us"]))
    name = f"{first} {last}"
    aliases: list[str] = []
    if rng.random() < 0.20:
        initial = first[0]
        aliases.append(f"{initial}. {last}")
    return name, aliases


def make_company_name(country_name: str, rng: random.Random) -> str:
    template = rng.choice(SOE_TEMPLATES)
    return template.format(country=country_name)


def make_birth_date(rng: random.Random) -> str:
    if rng.random() < 0.40:
        return ""
    year = rng.randint(1945, 1985)
    return str(year)


def make_address(country_name: str, rng: random.Random) -> str:
    if rng.random() < 0.50:
        return ""
    cities = {
        "Russia": ["Moscow", "St. Petersburg"],
        "China": ["Beijing", "Shanghai", "Shenzhen"],
        "Venezuela": ["Caracas", "Maracaibo"],
        "Iran": ["Tehran", "Isfahan"],
        "Nigeria": ["Abuja", "Lagos"],
        "Kazakhstan": ["Astana", "Almaty"],
        "Pakistan": ["Islamabad", "Karachi"],
        "Brazil": ["Brasilia", "Sao Paulo"],
        "Mexico": ["Mexico City"],
        "United States": ["Washington DC", "New York"],
        "United Kingdom": ["London"],
        "Germany": ["Berlin"],
        "France": ["Paris"],
    }
    city = rng.choice(cities.get(country_name, [country_name]))
    return f"{city}, {country_name}"


def main() -> None:
    rng = random.Random(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    today = datetime.now(timezone.utc)
    snapshot_iso = today.strftime("%Y-%m-%dT%H:%M:%S")
    first_seen_iso = (today - timedelta(days=180)).strftime("%Y-%m-%dT%H:%M:%S")

    columns = [
        "id", "schema", "name", "aliases", "birth_date", "countries",
        "addresses", "identifiers", "sanctions", "phones", "emails",
        "program_ids", "dataset", "first_seen", "last_seen", "last_change",
    ]

    rows: list[dict[str, str]] = []
    for i in range(N_ENTRIES):
        country_code, country_name, _ = weighted_choice(COUNTRIES, rng)
        role_code, role_name, _ = weighted_choice(ROLES, rng)

        is_company = role_code == "SOE-Exec" and rng.random() < 0.40
        if is_company:
            schema = "Company"
            name = make_company_name(country_name, rng)
            aliases: list[str] = []
            birth_date = ""
        else:
            schema = "Person"
            name, aliases = make_person_name(country_code, rng)
            birth_date = make_birth_date(rng)

        rows.append({
            "id": make_id(rng, i),
            "schema": schema,
            "name": name,
            "aliases": "; ".join(aliases),
            "birth_date": birth_date,
            "countries": country_code,
            "addresses": make_address(country_name, rng),
            "identifiers": "",
            "sanctions": "",
            "phones": "",
            "emails": "",
            "program_ids": role_code,
            "dataset": "Synthetic PEP Starter v1",
            "first_seen": first_seen_iso,
            "last_seen": snapshot_iso,
            "last_change": first_seen_iso,
        })

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    n_person = sum(1 for r in rows if r["schema"] == "Person")
    n_company = len(rows) - n_person
    print(f"Wrote {len(rows)} entries to {OUT_CSV}")
    print(f"  Person:  {n_person}")
    print(f"  Company: {n_company}")
    role_counts: dict[str, int] = {}
    for r in rows:
        role_counts[r["program_ids"]] = role_counts.get(r["program_ids"], 0) + 1
    print("  Roles:")
    for role, n in sorted(role_counts.items(), key=lambda x: -x[1]):
        print(f"    {role:18s} {n}")

    readme = f"""# Synthetic PEP Starter Pool

Synthetic placeholder for the PEP (Politically Exposed Person) name pool used
by Stage 4 of the SFT synthetic data generation pipeline (see
`sgd_understanding.md`). To be replaced once OpenSanctions consolidated PEP
data is downloaded into a separate pool.

## Provenance

- **Generator**: `2.data_processing/generate_pep_starter.py`
- **Seed**: `{SEED}` (deterministic)
- **Entries**: `{N_ENTRIES}`
- **Schema**: matches `2.data_processing/data/transactional/ofac_enforcement/targets.simple.csv`
  so Stage 4 can union the two pools transparently.
- **Names**: synthetic permutations of common regional first names and
  surnames. Not real people.
- **Tagging**: `dataset` column = `"Synthetic PEP Starter v1"`; `id` prefix =
  `"PEP-SYN-"`.

## Distribution

- ~95% `Person`, ~5% `Company` (state-owned enterprises).
- Country mix weighted toward high-risk jurisdictions
  (Russia, China, Venezuela, Iran, Nigeria, Pakistan, Kazakhstan, etc.)
  with EU / US / Latin America baseline.
- Role mix (`program_ids`):
  HoS, Minister, Legislator, Family-PEP, Close-Assoc,
  Judge-Senior, Military-Senior, CB-Regulator, SOE-Exec, IO-Official.

## Regenerating

```bash
cd 2.data_processing
python generate_pep_starter.py
```
"""
    OUT_README.write_text(readme, encoding="utf-8")
    print(f"Wrote README to {OUT_README}")


if __name__ == "__main__":
    main()

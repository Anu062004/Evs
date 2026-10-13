"""Download photos of Indian bird species from iNaturalist.

Only research-grade observations made in India with a Creative Commons photo licence are used.

Usage:
    python fetch_inat.py --out-dir data/download/india --per-species 60

Writes one folder per species under --out-dir, plus data/manifest/inat_photos.csv with the
observer, licence and attribution of every photo. That CSV doubles as the --groups-csv input of
prepare_data.py, so all photos by one observer stay in the same split.
"""

import argparse
import csv
import json
import time
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

API = "https://api.inaturalist.org/v1/"
HEADERS = {"User-Agent": "EvsBirdProject/1.0 (student project; github.com/Anu062004/Evs)"}
LICENCES = "cc0,cc-by,cc-by-nc"
MAX_PER_OBSERVER = 3  # keeps one prolific photographer from dominating a species

SPECIES = [
    "Indian Peafowl", "House Crow", "Common Myna", "Rose-ringed Parakeet", "Indian Roller",
    "White-throated Kingfisher", "Red-vented Bulbul", "Asian Koel", "Oriental Magpie-Robin",
    "Black Drongo", "Indian Pond Heron", "Purple Sunbird", "Asian Green Bee-eater",
    "Coppersmith Barbet", "Indian Robin", "Jungle Babbler", "Red-wattled Lapwing", "Brahminy Kite",
    "Eurasian Hoopoe", "House Sparrow", "Greater Coucal", "Rufous Treepie", "Shikra", "Black Kite",
    "Laughing Dove",
]


def get(url, binary=False):
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60) as response:
                data = response.read()
            return data if binary else json.loads(data)
        except Exception as error:  # rate limits and network hiccups: back off and retry
            last = error
            time.sleep(5 * (attempt + 1))
    raise last


def api(endpoint, **params):
    time.sleep(1.1)  # iNaturalist asks for at most about one API request per second
    return get(API + endpoint + "?" + urllib.parse.urlencode(params))


def plain(name):
    """Common names differ in hyphenation between sources ('Pond Heron' / 'Pond-Heron')."""
    return name.lower().replace("-", " ")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default="data/download/india")
    parser.add_argument("--credits", default="data/manifest/inat_photos.csv")
    parser.add_argument("--per-species", type=int, default=60)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    india = next(p for p in api("places/autocomplete", q="India")["results"] if p["name"] == "India" and p["admin_level"] == 0)
    rows, jobs = [], []
    for name in SPECIES:
        taxon = next(t for t in api("taxa", q=name, rank="species", iconic_taxa="Aves")["results"]
                     if plain(t.get("preferred_common_name") or "") == plain(name))
        folder = name.replace(" ", "_")
        (out_dir / folder).mkdir(parents=True, exist_ok=True)
        per_observer, picked = Counter(), 0
        for page in (1, 2, 3):
            found = api("observations", taxon_id=taxon["id"], place_id=india["id"], quality_grade="research",
                        photos="true", photo_license=LICENCES, per_page=200, page=page, order_by="id")["results"]
            for obs in found:
                photo, observer = obs["photos"][0], obs["user"]["login"]
                if picked >= args.per_species or per_observer[observer] >= MAX_PER_OBSERVER:
                    continue
                per_observer[observer] += 1
                picked += 1
                rel = f"{folder}/{photo['id']}.jpg"
                jobs.append((photo["url"].replace("/square.", "/medium."), out_dir / rel))
                rows.append({"path": rel, "group": f"inat:{observer}", "species": name,
                             "scientific_name": taxon["name"], "licence": photo["license_code"],
                             "attribution": photo["attribution"], "observation": obs["uri"]})
            if picked >= args.per_species or len(found) < 200:
                break
        print(f"{name} ({taxon['name']}): {picked} photos from {len(per_observer)} observers", flush=True)

    def download(job):
        url, dest = job
        if not dest.exists():
            dest.write_bytes(get(url, binary=True))

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(download, jobs))

    Path(args.credits).parent.mkdir(parents=True, exist_ok=True)
    with open(args.credits, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Downloaded {len(jobs)} photos to {out_dir}; credits in {args.credits}")


if __name__ == "__main__":
    main()

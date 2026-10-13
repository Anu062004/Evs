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

# Common name -> scientific name. The scientific name is the fallback when iNaturalist
# uses a different common name (for example "Common Hoopoe").
SPECIES = {
    "Indian Peafowl": "Pavo cristatus", "House Crow": "Corvus splendens", "Common Myna": "Acridotheres tristis",
    "Rose-ringed Parakeet": "Psittacula krameri", "Indian Roller": "Coracias benghalensis",
    "White-throated Kingfisher": "Halcyon smyrnensis", "Red-vented Bulbul": "Pycnonotus cafer",
    "Asian Koel": "Eudynamys scolopaceus", "Oriental Magpie-Robin": "Copsychus saularis",
    "Black Drongo": "Dicrurus macrocercus", "Indian Pond Heron": "Ardeola grayii",
    "Purple Sunbird": "Cinnyris asiaticus", "Asian Green Bee-eater": "Merops orientalis",
    "Coppersmith Barbet": "Psilopogon haemacephalus", "Indian Robin": "Copsychus fulicatus",
    "Jungle Babbler": "Argya striata", "Red-wattled Lapwing": "Vanellus indicus",
    "Brahminy Kite": "Haliastur indus", "Eurasian Hoopoe": "Upupa epops", "House Sparrow": "Passer domesticus",
    "Greater Coucal": "Centropus sinensis", "Rufous Treepie": "Dendrocitta vagabunda",
    "Shikra": "Accipiter badius", "Black Kite": "Milvus migrans", "Laughing Dove": "Spilopelia senegalensis",
}


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


def find_taxon(name, scientific):
    for query in (name, scientific):
        for taxon in api("taxa", q=query, rank="species", iconic_taxa="Aves")["results"]:
            if plain(taxon.get("preferred_common_name") or "") == plain(name) or taxon["name"] == scientific:
                return taxon
    raise SystemExit(f"Could not find {name} ({scientific}) on iNaturalist")


def download(job):
    url, dest = job
    if not dest.exists():
        dest.write_bytes(get(url, binary=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default="data/download/india")
    parser.add_argument("--credits", default="data/manifest/inat_photos.csv")
    parser.add_argument("--per-species", type=int, default=60)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    india = next(p for p in api("places/autocomplete", q="India")["results"] if p["name"] == "India" and p["admin_level"] == 0)
    rows, total = [], 0
    for name, scientific in SPECIES.items():
        taxon = find_taxon(name, scientific)
        jobs = []
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
        with ThreadPoolExecutor(8) as pool:
            list(pool.map(download, jobs))
        total += len(jobs)
        print(f"{name} ({taxon['name']}): {picked} photos from {len(per_observer)} observers", flush=True)

    Path(args.credits).parent.mkdir(parents=True, exist_ok=True)
    with open(args.credits, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Downloaded {total} photos to {out_dir}; credits in {args.credits}")


if __name__ == "__main__":
    main()

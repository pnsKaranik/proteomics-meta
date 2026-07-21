"""External service clients: UniProt gene-name resolution and STRING interactions."""
from __future__ import annotations

import json
import os
import tempfile

import pandas as pd
import requests

from .capabilities import get_logger

logger = get_logger(__name__)


def get_gene_names_optimized(ids: list, cache_file: str = "gene_cache.json") -> dict:
    mapping: dict = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file) as f:
                mapping = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Cache load failed (%s).", exc)

    missing = [x for x in ids if x not in mapping]
    if not missing:
        return mapping

    logger.info("Fetching %d gene names from mygene.info …", len(missing))
    for i in range(0, len(missing), 1000):
        chunk = missing[i: i + 1000]
        try:
            r = requests.post(
                "https://mygene.info/v3/query",
                data={"q": ",".join(chunk), "scopes": "uniprot",
                      "fields": "symbol", "species": "human"},
                timeout=15,
            )
            r.raise_for_status()
            for item in r.json():
                mapping[item["query"]] = item.get("symbol", item["query"])
        except requests.RequestException as exc:
            logger.warning("mygene.info error: %s", exc)
            break

    dir_ = os.path.dirname(cache_file) or "."
    try:
        fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(mapping, f)
        os.replace(tmp, cache_file)
    except OSError as exc:
        logger.warning("Cache save failed: %s", exc)
    return mapping


def fetch_string_interactions(
    gene_list: list,
    score_threshold: int = 400,
    species: int = 9606,
    max_genes: int = 200,
    timeout: int = 30,
) -> pd.DataFrame:
    """
    Fetch protein-protein interactions from STRING API v11.5.
    Returns DataFrame with columns: gene_a, gene_b, score.
    Falls back to empty DataFrame on network error.

    score_threshold: 0-1000 (400=medium, 700=high, 900=very high)
    """
    genes = [g for g in gene_list if isinstance(g, str) and len(g) > 1][:max_genes]
    if len(genes) < 2:
        return pd.DataFrame()

    logger.info("Fetching STRING interactions for %d genes …", len(genes))
    try:
        # Step 1: map gene symbols to STRING IDs
        map_url = "https://string-db.org/api/json/get_string_ids"
        map_r = requests.post(map_url, data={
            "identifiers": "\r".join(genes),
            "species":     species,
            "limit":       1,
            "echo_query":  1,
        }, timeout=timeout)
        map_r.raise_for_status()
        id_map = {item["queryItem"]: item["stringId"]
                  for item in map_r.json() if "stringId" in item}

        if not id_map:
            logger.warning("STRING: no IDs mapped.")
            return pd.DataFrame()

        # Step 2: fetch interactions
        int_url = "https://string-db.org/api/json/network"
        int_r = requests.post(int_url, data={
            "identifiers":    "\r".join(id_map.values()),
            "species":        species,
            "required_score": score_threshold,
        }, timeout=timeout)
        int_r.raise_for_status()
        interactions = int_r.json()

        if not interactions:
            logger.info("STRING: no interactions above threshold.")
            return pd.DataFrame()

        # Reverse map STRING ID → gene symbol
        rev_map = {v: k for k, v in id_map.items()}
        records = []
        for item in interactions:
            ga = rev_map.get(item.get("stringId_A", ""), item.get("preferredName_A", ""))
            gb = rev_map.get(item.get("stringId_B", ""), item.get("preferredName_B", ""))
            sc = item.get("score", 0)
            if ga and gb and ga != gb:
                records.append({"gene_a": ga, "gene_b": gb, "score": float(sc)})

        df_ppi = pd.DataFrame(records).drop_duplicates()
        logger.info("STRING: %d interactions fetched.", len(df_ppi))
        return df_ppi

    except requests.RequestException as exc:
        logger.warning("STRING API failed: %s", exc)
        return pd.DataFrame()
    except Exception as exc:
        logger.warning("STRING processing failed: %s", exc)
        return pd.DataFrame()

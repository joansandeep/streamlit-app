# src/recommender_engine.py
import re
import math
import pandas as pd
from rapidfuzz import process, fuzz

# -----------------------
# Helpers
# -----------------------
_dosage_regex = re.compile(r"(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>mcg|mg|g|iu|ml|%)", flags=re.I)

def parse_price(val):
    """Return numeric price (float) or None from strings like '₹27.83' or '27.83'."""
    if pd.isna(val):
        return None
    s = str(val)
    s = s.replace("₹", "").replace(",", "").strip()
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)", s)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None

def split_salt_strength(s: str):
    """
    Input: "Paracetamol (650mg)" or "Paracetamol 650 mg" -> returns ("paracetamol", "650mg")
    If strength not found, returns (salt_lower, "").
    """
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return "", ""
    t = str(s).strip().lower()
    # find dosage like 650mg, 500 mg, 1000 iu, 0.5 ml etc.
    m = _dosage_regex.search(t)
    if m:
        num = m.group("num")
        unit = m.group("unit").lower()
        strength = f"{num}{unit}"
        salt_name = _dosage_regex.sub("", t).replace("(", "").replace(")", "").strip()
        # remove leftover punctuation
        salt_name = re.sub(r"[^a-z0-9\s\+\-]", " ", salt_name)
        salt_name = re.sub(r"\s+", " ", salt_name).strip()
        return salt_name, strength
    else:
        # no dosage found
        salt_name = re.sub(r"[^a-z0-9\s\+\-]", " ", t)
        salt_name = re.sub(r"\s+", " ", salt_name).strip()
        return salt_name, ""

def canonical_salt_key(salt, strength):
    """Return canonical combined key for comparisons: 'saltoname|||strength' (strength may be empty)."""
    salt_norm = re.sub(r"\s+", " ", (salt or "").lower().strip())
    strength_norm = re.sub(r"\s+", "", (strength or "").lower().strip())
    # keep empty strength as empty string
    return f"{salt_norm}|||{strength_norm}"

def normalize_for_matching(text: str):
    """Lowercase, collapse spaces, remove parentheses & common dosage words for fuzzy matching."""
    if text is None:
        return ""
    t = str(text).lower()
    t = t.replace("(", " ").replace(")", " ")
    # remove explicit dosage tokens but keep numbers if present (we will parse dosage separately)
    t = re.sub(r"\b(mg|ml|tablet|capsule|injection|mcg|iu)\b", " ", t)
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

# -----------------------
# RecommenderEngine
# -----------------------
class RecommenderEngine:
    """
    RecommenderEngine built for the 1mg-like dataset with columns:
      - Generic Name, Brand Name, Price, Manufacturer, Salt, Uses, Side Effects, URL, text
    Key behaviors:
      - Precomputes canonical salt keys (salt name + strength)
      - Fuzzy matches brand names for queries
      - Finds alternatives with same salt (and optionally same dosage)
      - Returns cheapest alternative and price when Price column exists
    """

    def __init__(self, data_df: pd.DataFrame):
        # copy to avoid mutating original
        self.df = data_df.copy().reset_index(drop=True)

        # Ensure expected columns exist
        if "Brand Name" not in self.df.columns:
            raise KeyError("'Brand Name' column required in dataset")
        if "Salt" not in self.df.columns:
            # fallback: use Generic Name as salt (no dosage)
            self.df["Salt"] = self.df.get("Generic Name", "")

        # Normalize textual columns (keep dosage text in Salt intact for parsing)
        for col in ["Brand Name", "Generic Name", "Salt"]:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).fillna("").str.strip()

        # Parse price into numeric column _price_val (None if cannot parse)
        self.df["_price_val"] = self.df["Price"].apply(parse_price) if "Price" in self.df.columns else None

        # Build canonical keys from Salt column
        canonical_keys = []
        parsed_salt = []  # list of (salt_name, strength)
        for s in self.df["Salt"].tolist():
            salt_name, strength = split_salt_strength(s)
            parsed_salt.append((salt_name, strength))
            canonical_keys.append(canonical_salt_key(salt_name, strength))
        self.df["_salt_name"] = [p[0] for p in parsed_salt]
        self.df["_strength"] = [p[1] for p in parsed_salt]
        self.df["_canonical_salt"] = canonical_keys

        # Precompute brand list and lookup mapping for exact lookup
        self.brand_list = self.df["Brand Name"].dropna().unique().tolist()
        # also store normalized brand list (for fuzzy)
        self._brand_choices_for_fuzzy = [normalize_for_matching(b) for b in self.brand_list]
        self._brand_map_norm_to_original = dict(zip(self._brand_choices_for_fuzzy, self.brand_list))

        # Precompute generic list if needed
        if "Generic Name" in self.df.columns:
            self.generic_list = self.df["Generic Name"].dropna().unique().tolist()
            self._generic_choices_for_fuzzy = [normalize_for_matching(g) for g in self.generic_list]
        else:
            self.generic_list = []
            self._generic_choices_for_fuzzy = []

    # -----------------------
    # Internal fuzzy brand lookup
    # -----------------------
    def _fuzzy_match_brand(self, query: str, score_cutoff: int = 75):
        """
        Return matched_brand (original exact-case from dataset) or None.
        Uses token_sort_ratio via rapidfuzz.
        """
        if not query:
            return None, 0
        q_norm = normalize_for_matching(query)
        # use rapidfuzz process.extractOne against normalized choices
        match = process.extractOne(q_norm, self._brand_choices_for_fuzzy, scorer=fuzz.token_sort_ratio)
        if not match:
            return None, 0
        matched_norm, score, idx = match  # matched_norm is normalized brand text
        if score >= score_cutoff:
            original_brand = self._brand_map_norm_to_original.get(matched_norm)
            return original_brand, score
        return None, score

    # -----------------------
    # Public helpers
    # -----------------------
    def get_salt_and_dosage(self, brand_name: str):
        """
        For a given brand name, return (salt_name, strength, canonical_key).
        If brand not found, tries fuzzy match. Returns ("", "", "") if nothing found.
        """
        if brand_name is None:
            return "", "", ""
        # try exact match case-insensitive
        mask = self.df["Brand Name"].astype(str).str.lower() == str(brand_name).lower()
        if mask.any():
            row = self.df[mask].iloc[0]
            return row["_salt_name"], row["_strength"], row["_canonical_salt"]

        # fuzzy match
        matched_brand, score = self._fuzzy_match_brand(brand_name)
        if matched_brand:
            row = self.df[self.df["Brand Name"] == matched_brand].iloc[0]
            return row["_salt_name"], row["_strength"], row["_canonical_salt"]

        return "", "", ""

    def find_alternatives(self, brand_name: str, require_dosage: bool = True, top_k: int = 5):
        """
        Find alternatives for brand_name.
        - require_dosage: if True, only alternatives with same salt+strength are considered.
                          if False, alternatives with same salt (any strength) are considered.
        Returns: (alternatives_df, status_message, cheapest_brand_or_None, cheapest_price_or_None)
        alternatives_df will include '_price_val' numeric column if Price exists.
        """
        if brand_name is None:
            return None, "No brand name provided.", None, None

        # find canonical key for queried brand
        salt_name, strength, canonical_key = self.get_salt_and_dosage(brand_name)
        if not canonical_key:
            return None, f"Drug '{brand_name}' not found in database (no match).", None, None

        # find alternatives by canonical comparison (avoid regex)
        if require_dosage:
            mask = self.df["_canonical_salt"] == canonical_key
        else:
            # match only salt name (ignore strength)
            salt_only = canonical_key.split("|||")[0]
            mask = self.df["_canonical_salt"].str.startswith(salt_only + "|||")

        # exclude the same brand(s)
        df_alts = self.df[mask].copy()
        df_alts = df_alts[df_alts["Brand Name"].astype(str).str.lower() != str(brand_name).lower()]

        if df_alts.empty:
            msg = f"No generic alternatives found for {brand_name} (require_dosage={require_dosage})."
            # if require_dosage True, try fallback to ignore dosage (increase recall)
            if require_dosage:
                # attempt fallback
                salt_only = canonical_key.split("|||")[0]
                fallback_mask = self.df["_canonical_salt"].str.startswith(salt_only + "|||")
                fallback_alts = self.df[fallback_mask].copy()
                fallback_alts = fallback_alts[fallback_alts["Brand Name"].astype(str).str.lower() != str(brand_name).lower()]
                if not fallback_alts.empty:
                    # sort fallback by price if available and return with note
                    if "_price_val" in fallback_alts.columns:
                        fallback_alts = fallback_alts.sort_values(by="_price_val", ascending=True)
                    cheapest = None
                    cheapest_price = None
                    if "_price_val" in fallback_alts.columns and not fallback_alts["_price_val"].isnull().all():
                        cheapest_row = fallback_alts.loc[fallback_alts["_price_val"].idxmin()]
                        cheapest = cheapest_row["Brand Name"]
                        cheapest_price = cheapest_row["_price_val"]
                    return fallback_alts.head(top_k), f"Fallback: matched same salt but different strength for {brand_name}.", cheapest, cheapest_price
            return None, msg, None, None

        # if price exists, sort by numeric price
        if "_price_val" in df_alts.columns and df_alts["_price_val"].notna().any():
            df_alts = df_alts.sort_values(by="_price_val", ascending=True)
        # compute cheapest
        cheapest = None
        cheapest_price = None
        if "_price_val" in df_alts.columns and df_alts["_price_val"].notna().any():
            cheapest_row = df_alts.loc[df_alts["_price_val"].idxmin()]
            cheapest = cheapest_row["Brand Name"]
            cheapest_price = cheapest_row["_price_val"]

        return df_alts.head(top_k), f"Top {top_k} alternatives found for {brand_name}.", cheapest, cheapest_price

    def find_brands_by_generic(self, generic_name: str, top_k: int = 5):
        """
        Find brands that implement the provided generic (generic_name).
        Uses fuzzy matching on Generic Name.
        """
        if "Generic Name" not in self.df.columns:
            return None, f"No 'Generic Name' column in dataset."

        # fuzzy match on generic list
        g_norm = normalize_for_matching(generic_name)
        best = process.extractOne(g_norm, self._generic_choices_for_fuzzy, scorer=fuzz.token_sort_ratio)
        if not best:
            return None, f"No brands found for generic '{generic_name}'."
        matched_norm, score, idx = best
        matched_generic = self.generic_list[idx] if idx < len(self.generic_list) else None
        if not matched_generic:
            return None, f"No brands found for generic '{generic_name}'."

        brands = self.df[self.df["Generic Name"] == matched_generic].copy()
        if brands.empty:
            return None, f"No brands found for generic '{generic_name}'."

        if "_price_val" in brands.columns and brands["_price_val"].notna().any():
            brands = brands.sort_values(by="_price_val", ascending=True)

        return brands.head(top_k), f"Top {top_k} brands for generic {generic_name}."

# If run directly, basic smoke test (not executed during import)
if __name__ == "__main__":
    print("RecommenderEngine module loaded.")

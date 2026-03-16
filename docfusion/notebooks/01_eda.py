"""
DocFusion EDA — Level 1: Document Understanding & Exploratory Data Analysis
=============================================================================
Run this as a script:  python notebooks/01_eda.py
Or convert to a Jupyter notebook:  jupytext --to notebook notebooks/01_eda.py

This script explores all three data sources (SROIE, CORD, Find-It-Again),
analyses distributions, visualises statistics, and identifies anomalies.
"""

# %% [markdown]
# # DocFusion Level 1 — EDA
# ## 1. Setup & Imports

# %%
import os, json, csv, re, glob, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter

# Paths (adjust if needed)
BASE      = os.path.join(os.path.dirname(__file__), "..")
SROIE_DIR = os.path.join(BASE, "SROIE2019", "train")
CORD_DIR  = os.path.join(BASE, "cord-v2-data")
FINDIT_DIR= os.path.join(BASE, "findit2")
OUT_DIR   = os.path.join(BASE, "notebooks", "eda_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# %% [markdown]
# ## 2. SROIE Dataset Exploration

# %%
def load_sroie(split_dir):
    """Load SROIE entity JSON files and return a DataFrame."""
    entity_dir = os.path.join(split_dir, "entities")
    records = []
    if not os.path.isdir(entity_dir):
        print(f"[SROIE] Entity dir not found: {entity_dir}")
        return pd.DataFrame()
    for fname in sorted(os.listdir(entity_dir)):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(entity_dir, fname)
        try:
            with open(path) as f:
                data = json.load(f)
            data["file_id"] = fname.replace(".txt", "")
            data["source"] = "SROIE"
            records.append(data)
        except Exception as e:
            print(f"  skipping {fname}: {e}")
    return pd.DataFrame(records)

sroie_df = load_sroie(SROIE_DIR)
print(f"[SROIE] Loaded {len(sroie_df)} records")
if len(sroie_df):
    print(sroie_df.head())

# %%
# Parse total to float
if len(sroie_df):
    sroie_df["total_num"] = pd.to_numeric(
        sroie_df["total"].str.replace(",", ""), errors="coerce"
    )

    # --- Total distribution ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sroie_df["total_num"].dropna().hist(bins=50, ax=axes[0], color="steelblue")
    axes[0].set_title("SROIE — Total Amount Distribution")
    axes[0].set_xlabel("Total ($)")

    sroie_df["total_num"].dropna().apply(np.log1p).hist(bins=50, ax=axes[1], color="teal")
    axes[1].set_title("SROIE — Log-Scaled Total Distribution")
    axes[1].set_xlabel("log(1 + Total)")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "sroie_total_dist.png"), dpi=120)
    plt.close()
    print("[SROIE] Saved total distribution plot.")

    # --- Vendor frequency ---
    top_vendors = sroie_df["company"].value_counts().head(20)
    fig, ax = plt.subplots(figsize=(12, 6))
    top_vendors.plot.barh(ax=ax, color="coral")
    ax.set_title("SROIE — Top 20 Vendors")
    ax.set_xlabel("Count")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "sroie_vendor_freq.png"), dpi=120)
    plt.close()
    print("[SROIE] Saved vendor frequency plot.")

    # --- Date distribution ---
    def parse_date(d):
        for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d.%m.%Y"):
            try:
                return pd.to_datetime(d, format=fmt)
            except Exception:
                continue
        return pd.NaT

    sroie_df["date_parsed"] = sroie_df["date"].apply(parse_date)
    valid_dates = sroie_df["date_parsed"].dropna()
    if len(valid_dates):
        fig, ax = plt.subplots(figsize=(12, 4))
        valid_dates.dt.month.value_counts().sort_index().plot.bar(ax=ax, color="mediumpurple")
        ax.set_title("SROIE — Receipts by Month")
        ax.set_xlabel("Month")
        ax.set_ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "sroie_month_dist.png"), dpi=120)
        plt.close()
        print("[SROIE] Saved date distribution plot.")

    # --- Summary stats ---
    print("\n[SROIE] Summary Statistics:")
    print(f"  Records:       {len(sroie_df)}")
    print(f"  Unique vendors:{sroie_df['company'].nunique()}")
    print(f"  Total range:   ${sroie_df['total_num'].min():.2f} – ${sroie_df['total_num'].max():.2f}")
    print(f"  Total mean:    ${sroie_df['total_num'].mean():.2f}")
    print(f"  Total median:  ${sroie_df['total_num'].median():.2f}")
    print(f"  Missing vendor:{sroie_df['company'].isna().sum()}")
    print(f"  Missing date:  {sroie_df['date'].isna().sum()}")
    print(f"  Missing total: {sroie_df['total'].isna().sum()}")

# %% [markdown]
# ## 3. CORD Dataset Exploration

# %%
def load_cord(base_dir, split="train"):
    """Load CORD dataset from Arrow files via HuggingFace datasets."""
    try:
        from datasets import load_from_disk
        ds = load_from_disk(base_dir)
        df = ds[split].to_pandas()
        df["source"] = "CORD"
        return df
    except Exception as e:
        print(f"[CORD] Could not load via datasets: {e}")
        return pd.DataFrame()

cord_df = load_cord(CORD_DIR, "train")
print(f"[CORD] Loaded {len(cord_df)} records")
if len(cord_df):
    print(f"[CORD] Columns: {list(cord_df.columns)}")
    print(cord_df.head(2))

# %%
if len(cord_df) and "ground_truth" in cord_df.columns:
    # Parse ground_truth JSON strings
    def parse_cord_gt(gt_str):
        try:
            gt = json.loads(gt_str)
            # CORD ground_truth is {"gt_parse": {...}}
            return gt.get("gt_parse", gt)
        except Exception:
            return {}

    cord_parsed = cord_df["ground_truth"].apply(parse_cord_gt)

    # Extract totals where available
    totals = []
    for parsed in cord_parsed:
        total_section = parsed.get("total", {})
        if isinstance(total_section, dict):
            tp = total_section.get("total_price", [])
            if isinstance(tp, list):
                for item in tp:
                    if isinstance(item, dict) and "price" in item:
                        cleaned = re.sub(r"[^\d.]", "", str(item["price"]))
                        if cleaned:
                            try:
                                totals.append(float(cleaned))
                            except ValueError:
                                pass
            elif isinstance(tp, str):
                cleaned = re.sub(r"[^\d.]", "", tp)
                if cleaned:
                    try:
                        totals.append(float(cleaned))
                    except ValueError:
                        pass

    if totals:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(totals, bins=50, color="darkorange", edgecolor="white")
        ax.set_title(f"CORD — Total Price Distribution (n={len(totals)})")
        ax.set_xlabel("Total Price")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "cord_total_dist.png"), dpi=120)
        plt.close()
        print(f"[CORD] Saved total distribution plot ({len(totals)} values).")

    print(f"\n[CORD] Summary Statistics:")
    print(f"  Records:  {len(cord_df)}")
    if totals:
        print(f"  Totals found:  {len(totals)}")
        print(f"  Total range:   {min(totals):.2f} – {max(totals):.2f}")
        print(f"  Total mean:    {np.mean(totals):.2f}")

# %% [markdown]
# ## 4. Find-It-Again Dataset Exploration (Anomaly Ground Truth)

# %%
def load_findit(base_dir, split="train"):
    """Load Find-It-Again CSV annotation file."""
    csv_path = os.path.join(base_dir, f"{split}.txt")
    if not os.path.exists(csv_path):
        print(f"[FindIt] File not found: {csv_path}")
        return pd.DataFrame()

    records = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["source"] = "FindIt"
            records.append(row)
    return pd.DataFrame(records)

findit_df = load_findit(FINDIT_DIR, "train")
print(f"[FindIt] Loaded {len(findit_df)} records")
if len(findit_df):
    print(f"[FindIt] Columns: {list(findit_df.columns)}")
    print(findit_df.head())

# %%
if len(findit_df) and "forged" in findit_df.columns:
    findit_df["forged"] = findit_df["forged"].astype(int)

    # --- Forged vs Genuine ---
    counts = findit_df["forged"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 5))
    counts.plot.bar(ax=ax, color=["#2ecc71", "#e74c3c"])
    ax.set_title("Find-It-Again — Genuine vs Forged Documents")
    ax.set_xticklabels(["Genuine (0)", "Forged (1)"], rotation=0)
    ax.set_ylabel("Count")
    for i, v in enumerate(counts):
        ax.text(i, v + 5, str(v), ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "findit_forged_dist.png"), dpi=120)
    plt.close()
    print("[FindIt] Saved forged distribution plot.")

    # --- Annotations per forged doc ---
    if "forgery annotations" in findit_df.columns:
        forged_docs = findit_df[findit_df["forged"] == 1]
        annotation_counts = []
        for _, row in forged_docs.iterrows():
            ann = row["forgery annotations"]
            if isinstance(ann, str) and ann != "0":
                try:
                    parsed = eval(ann)  # safe here, it's our data
                    regions = parsed.get("regions", [])
                    annotation_counts.append(len(regions))
                except Exception:
                    annotation_counts.append(0)
            else:
                annotation_counts.append(0)

        if annotation_counts:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(annotation_counts, bins=range(0, max(annotation_counts)+2),
                    color="#e74c3c", edgecolor="white", align="left")
            ax.set_title("Find-It-Again — Forgery Regions per Forged Document")
            ax.set_xlabel("Number of Forged Regions")
            ax.set_ylabel("Count")
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, "findit_regions_per_doc.png"), dpi=120)
            plt.close()
            print("[FindIt] Saved forgery regions per document plot.")

    print(f"\n[FindIt] Summary Statistics:")
    print(f"  Total documents: {len(findit_df)}")
    print(f"  Genuine:         {(findit_df['forged'] == 0).sum()}")
    print(f"  Forged:          {(findit_df['forged'] == 1).sum()}")
    print(f"  Forgery rate:    {findit_df['forged'].mean()*100:.1f}%")

# %% [markdown]
# ## 5. Sample Document Visualisation

# %%
def show_sample_images(img_dir, title, n=4):
    """Save a grid of sample images from the dataset."""
    if not os.path.isdir(img_dir):
        print(f"  Directory not found: {img_dir}")
        return
    images = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])[:n]
    if not images:
        print(f"  No images found in {img_dir}")
        return
    fig, axes = plt.subplots(1, min(n, len(images)), figsize=(5*min(n, len(images)), 6))
    if len(images) == 1:
        axes = [axes]
    for ax, fname in zip(axes, images):
        img = Image.open(os.path.join(img_dir, fname))
        ax.imshow(img)
        ax.set_title(fname[:20], fontsize=9)
        ax.axis("off")
    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    safe_title = title.replace(" ", "_").replace("—", "").lower()
    plt.savefig(os.path.join(OUT_DIR, f"samples_{safe_title}.png"), dpi=100)
    plt.close()
    print(f"  Saved sample grid for {title}")

print("\n[Samples] Generating sample image grids...")
show_sample_images(os.path.join(SROIE_DIR, "img"), "SROIE — Sample Receipts")
show_sample_images(os.path.join(FINDIT_DIR, "train"), "Find-It-Again — Sample Documents")

# %% [markdown]
# ## 6. Cross-Dataset Comparison

# %%
print("\n" + "="*60)
print("CROSS-DATASET SUMMARY")
print("="*60)
summary = {
    "Dataset": ["SROIE", "CORD", "Find-It-Again"],
    "Records": [
        len(sroie_df) if len(sroie_df) else "N/A",
        len(cord_df) if len(cord_df) else "N/A",
        len(findit_df) if len(findit_df) else "N/A",
    ],
    "Purpose": [
        "Baseline OCR extraction",
        "Volume / layout diversity",
        "Anomaly detection ground truth",
    ],
}
summary_df = pd.DataFrame(summary)
print(summary_df.to_string(index=False))

# %% [markdown]
# ## 7. Potential Anomalies & Observations

# %%
if len(sroie_df):
    print("\n[Anomaly Scan] SROIE outliers (total > 3σ from mean):")
    mean = sroie_df["total_num"].mean()
    std = sroie_df["total_num"].std()
    outliers = sroie_df[sroie_df["total_num"] > mean + 3 * std]
    if len(outliers):
        for _, row in outliers.iterrows():
            print(f"  {row['file_id']}: ${row['total_num']:.2f} — {row['company']}")
    else:
        print("  None found (within 3σ).")

print("\n✅ EDA complete. All plots saved to:", OUT_DIR)

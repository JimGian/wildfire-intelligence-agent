# visualize_sentinel2.py
"""
Visualizes the downloaded Sentinel-2 GeoTIFF files.
Shows true-color RGB, burn scar (NBR), and vegetation (NDVI)
for each region — before, during, and after the 2023 fires.
"""

import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# Band indices in our GeoTIFF (0-indexed):
# 0=B2 (Blue), 1=B3 (Green), 2=B4 (Red)
# 3=B8 (NIR),  4=B12 (SWIR)
# 5=NDVI, 6=NBR, 7=NDWI
BAND_IDX = {
    "blue": 0, "green": 1, "red": 2,
    "nir": 3,  "swir": 4,
    "ndvi": 5, "nbr": 6, "ndwi": 7,
}

PERIODS   = ["pre_fire", "during", "post_fire"]
PERIOD_LABELS = ["Pre-fire", "During fire", "Post-fire"]

REGIONS = ["evros", "rhodes", "attica", "evia", "peloponnese"]


def load_band(tif_path, band_idx):
    """Load a single band from a GeoTIFF as a numpy array."""
    with rasterio.open(tif_path) as src:
        band = src.read(band_idx + 1).astype(np.float32)  # rasterio is 1-indexed
    return band


def normalize_for_display(arr, percentile_low=2, percentile_high=98):
    """
    Stretch values to 0-1 range for display using percentile clipping.
    This handles the fact that satellite images often have outlier values
    from clouds or sensor artifacts — percentile clipping ignores those.
    """
    valid = arr[arr > 0]  # ignore nodata pixels (value = 0)
    if len(valid) == 0:
        return np.zeros_like(arr)
    lo = np.percentile(valid, percentile_low)
    hi = np.percentile(valid, percentile_high)
    clipped = np.clip(arr, lo, hi)
    return (clipped - lo) / (hi - lo + 1e-9)


def make_rgb(tif_path):
    """
    Creates a true-color RGB image (bands R, G, B).
    This is what the human eye would see from space.
    """
    r = load_band(tif_path, BAND_IDX["red"])
    g = load_band(tif_path, BAND_IDX["green"])
    b = load_band(tif_path, BAND_IDX["blue"])

    rgb = np.stack([
        normalize_for_display(r),
        normalize_for_display(g),
        normalize_for_display(b),
    ], axis=-1)   # shape: (H, W, 3)
    return rgb


def make_nbr_colormap(tif_path):
    """
    Returns the NBR (Normalized Burn Ratio) band.
    NBR range: -1 to +1
      High NBR (+0.4 to +1.0) = healthy vegetation (green)
      Low NBR  (-1.0 to +0.1) = burned area (red/orange)

    This is the most important band for fire detection —
    burned areas show up dramatically as dark red patches.
    """
    nbr = load_band(tif_path, BAND_IDX["nbr"])
    # Mask nodata
    nbr[nbr == 0] = np.nan
    return nbr


def make_ndvi_colormap(tif_path):
    """
    Returns NDVI (Normalized Difference Vegetation Index).
    High NDVI = dense healthy vegetation (bright green)
    Low NDVI  = bare soil, burned area, or water (brown/blue)
    """
    ndvi = load_band(tif_path, BAND_IDX["ndvi"])
    ndvi[ndvi == 0] = np.nan
    return ndvi


def plot_region(region_name, output_dir="data/sentinel2", save=True):
    """
    Creates a 3x3 grid for one region:
      Rows: pre_fire | during | post_fire
      Cols: True color RGB | NBR (burn ratio) | NDVI (vegetation)
    """
    fig = plt.figure(figsize=(15, 12))
    fig.suptitle(
        f"Sentinel-2 — {region_name.upper()} — 2023 Fire Season",
        fontsize=16, fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    for row, (period, period_label) in enumerate(zip(PERIODS, PERIOD_LABELS)):
        tif_path = os.path.join(output_dir, region_name, f"{period}.tif")

        if not os.path.exists(tif_path):
            print(f"  Missing: {tif_path}")
            continue

        # ── Column 0: True color RGB ──
        ax = fig.add_subplot(gs[row, 0])
        rgb = make_rgb(tif_path)
        ax.imshow(rgb)
        ax.set_title(f"{period_label}\nTrue color (RGB)", fontsize=9)
        ax.axis("off")

        # ── Column 1: NBR — burn scar detection ──
        ax = fig.add_subplot(gs[row, 1])
        nbr = make_nbr_colormap(tif_path)
        im = ax.imshow(nbr, cmap="RdYlGn", vmin=-0.5, vmax=0.8)
        ax.set_title(f"{period_label}\nNBR (burn ratio)", fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # ── Column 2: NDVI — vegetation health ──
        ax = fig.add_subplot(gs[row, 2])
        ndvi = make_ndvi_colormap(tif_path)
        im = ax.imshow(ndvi, cmap="YlGn", vmin=-0.2, vmax=0.8)
        ax.set_title(f"{period_label}\nNDVI (vegetation)", fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if save:
        os.makedirs("outputs", exist_ok=True)
        out_path = f"outputs/{region_name}_visualization.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out_path}")

    plt.show()
    plt.close()


def plot_burn_scar_comparison(output_dir="data/sentinel2"):
    """
    Side-by-side NBR comparison: pre vs post fire for all regions.
    This clearly shows the burn scar that appeared during summer 2023.
    The difference (dNBR = pre_NBR - post_NBR) highlights burned pixels.
    """
    fig, axes = plt.subplots(
        len(REGIONS), 3,
        figsize=(14, len(REGIONS) * 3.5)
    )
    fig.suptitle(
        "Burn Scar Analysis — Greece 2023\n"
        "dNBR = Pre-fire NBR minus Post-fire NBR  (red = burned area)",
        fontsize=14, fontweight="bold"
    )

    for row, region_name in enumerate(REGIONS):
        base = os.path.join(output_dir, region_name)
        pre_path  = os.path.join(base, "pre_fire.tif")
        post_path = os.path.join(base, "post_fire.tif")

        if not (os.path.exists(pre_path) and os.path.exists(post_path)):
            continue

        pre_nbr  = make_nbr_colormap(pre_path)
        post_nbr = make_nbr_colormap(post_path)

        # dNBR: positive values = vegetation loss = burned area
        # This is the standard metric used by fire agencies worldwide
        dnbr = pre_nbr - post_nbr

        # Pre-fire NBR
        im0 = axes[row, 0].imshow(pre_nbr, cmap="RdYlGn", vmin=-0.5, vmax=0.8)
        axes[row, 0].set_title(f"{region_name.upper()}\nPre-fire NBR", fontsize=9)
        axes[row, 0].axis("off")
        plt.colorbar(im0, ax=axes[row, 0], fraction=0.046)

        # Post-fire NBR
        im1 = axes[row, 1].imshow(post_nbr, cmap="RdYlGn", vmin=-0.5, vmax=0.8)
        axes[row, 1].set_title(f"Post-fire NBR", fontsize=9)
        axes[row, 1].axis("off")
        plt.colorbar(im1, ax=axes[row, 1], fraction=0.046)

        # dNBR — the burn scar
        # Values > 0.1 are considered burned by fire agencies
        im2 = axes[row, 2].imshow(dnbr, cmap="hot_r", vmin=0, vmax=0.8)
        axes[row, 2].set_title(f"dNBR (burn scar)\nred = burned", fontsize=9)
        axes[row, 2].axis("off")
        plt.colorbar(im2, ax=axes[row, 2], fraction=0.046)

        # Count burned pixels (dNBR > 0.1 threshold used by fire agencies)
        burned_pixels = np.sum(dnbr > 0.1)
        total_pixels  = np.sum(~np.isnan(dnbr))
        burned_pct    = burned_pixels / total_pixels * 100 if total_pixels > 0 else 0
        print(f"{region_name:12s} — burned pixels: {burned_pixels:6,d} "
              f"({burned_pct:.1f}% of area)")

    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/burn_scar_comparison.png", dpi=150, bbox_inches="tight")
    print("\nSaved: outputs/burn_scar_comparison.png")
    plt.show()
    plt.close()


if __name__ == "__main__":
    print("Generating per-region visualizations...")
    for region in REGIONS:
        print(f"\n  Processing {region}...")
        plot_region(region)

    print("\nGenerating burn scar comparison...")
    plot_burn_scar_comparison()

    print("\nAll visualizations saved to outputs/")
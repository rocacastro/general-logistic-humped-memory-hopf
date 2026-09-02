"""
00_memory_kernels_comparison_en.py

Generates the figure comparing two memory kernels:
    G_0(s) = a exp(-a s),
    G_1(s) = a^2 s exp(-a s).

This script is used for the introductory figure that explains the difference
between exponential memory and humped memory.

Output:
    figures/memory_kernels_comparison_english.pdf
    figures/memory_kernels_comparison_english.png
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Output paths are relative to the location of this script.
# ---------------------------------------------------------------------
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

FIG_DIR = BASE_DIR.parent / "figures" if BASE_DIR.name == "scripts" else BASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------
a = 1.0
s = np.linspace(0.0, 6.0, 800)
G0 = a * np.exp(-a * s)
G1 = a**2 * s * np.exp(-a * s)
s_max = 1.0 / a


# ---------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.0, 4.4))

ax.plot(s, G0, linewidth=2.0, label=r"$G_0(s)=a e^{-a s}$")
ax.plot(s, G1, linewidth=2.0, label=r"$G_1(s)=a^2 s e^{-a s}$")
ax.axvline(s_max, linestyle="--", linewidth=1.5)
ax.text(s_max + 0.08, 0.92 * max(G0.max(), G1.max()), r"$s=1/a$", fontsize=11)

ax.set_xlabel(r"$s$")
ax.set_ylabel("Weight of the past")
ax.set_xlim(0.0, 6.0)
ax.set_ylim(bottom=0.0)
ax.legend(frameon=True)

fig.tight_layout()

pdf_path = FIG_DIR / "memory_kernels_comparison_english.pdf"
png_path = FIG_DIR / "memory_kernels_comparison_english.png"
fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(png_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print("Generated files:")
print(f"  {pdf_path}")
print(f"  {png_path}")

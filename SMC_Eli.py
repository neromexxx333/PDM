import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm, norm
from matplotlib.lines import Line2D
import time
from io import BytesIO

st.set_page_config(layout="wide")
st.title("Aplikasi Penjadwalan Proyek Konstruksi Berbasis Resiko Produktivitas Tenaga Kerja")
st.title("Pengembang: Ir. Eliatun, ST., MT.")
st.title("Fakultas Teknik Universitas Lambung Mangkurat")
st.markdown(
    """
    <style>
    div[data-testid="stDataFrame"] [role="gridcell"],
    div[data-testid="stDataFrame"] [role="columnheader"] {
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =============================
# UPLOAD
# =============================
file = st.file_uploader("Upload Excel Dataset", type=["xlsx"])

if file is None:
    st.warning("Upload file Excel terlebih dahulu")
    st.stop()

excel_book = pd.ExcelFile(BytesIO(file.getvalue()))

# =============================
# LOAD DATA
# =============================
df_proj = pd.read_excel(excel_book, sheet_name="Data_Proyek")
df_prod = pd.read_excel(excel_book, sheet_name="Data_Produktivitas")

st.subheader("Data Proyek")
st.dataframe(df_proj)

st.subheader("Data Produktivitas per Jenis Pekerjaan")
st.dataframe(df_prod)

# =============================
# HITUNG PRODUKTIVITAS
# =============================
df_prod['p'] = df_prod['Output'] / (df_prod['Tenaga'] * df_prod['Waktu'])
df_prod['Koefisien Data'] = (df_prod['Tenaga'] * df_prod['Waktu']) / df_prod['Output']

# =============================
# KALIBRASI
# =============================
st.subheader("Kalibrasi Produktivitas per Jenis Pekerjaan")

kal = []
for act in df_prod['Aktivitas'].unique():
    data = df_prod[df_prod['Aktivitas'] == act]['p']
    kal.append([act, data.mean(), data.std()])

df_kal = pd.DataFrame(kal, columns=["Aktivitas","Mean p","Std Dev"])
st.dataframe(
    df_kal.style.format({"Mean p": "{:.3f}", "Std Dev": "{:.3f}"}),
    use_container_width=True,
    hide_index=True
)
mean_p_map = dict(zip(df_kal["Aktivitas"], df_kal["Mean p"]))
std_p_map = dict(zip(df_kal["Aktivitas"], df_kal["Std Dev"].fillna(0.0)))

# =============================
# FIT DISTRIBUSI
# =============================
# =============================
# HELPER
# =============================
def parse_predecessors(val):
    if pd.isna(val):
        return []
    return [
        p.strip()
        for p in str(val).split(';')
        if p.strip() and p.strip().lower() != "nan"
    ]


def fit_distribution_params(df, distribution_name):
    dist_param = {}

    for act in df['Aktivitas'].unique():
        data = df[df['Aktivitas'] == act]['p'].dropna().to_numpy()

        if len(data) == 0:
            dist_param[act] = None
            continue

        try:
            if distribution_name == "Normal":
                mu = float(np.mean(data))
                sigma = float(np.std(data, ddof=1)) if len(data) > 1 else 0.0
                sigma = sigma if np.isfinite(sigma) and sigma > 0 else 1e-6
                dist_param[act] = (mu, sigma)
            else:
                if np.any(data <= 0):
                    dist_param[act] = None
                    continue
                shape, loc, scale = lognorm.fit(data, floc=0)
                dist_param[act] = (shape, loc, scale)
        except Exception:
            dist_param[act] = None

    return dist_param


def sample_productivity(act, dist_param, distribution_name):
    params = dist_param.get(act)

    if params is not None:
        if distribution_name == "Normal":
            mu, sigma = params
            p = np.random.normal(mu, sigma)
        else:
            shape, loc, scale = params
            p = lognorm.rvs(shape, loc=loc, scale=scale)
    else:
        fallback_mean = mean_p_map.get(act, 2.0)
        fallback_std = std_p_map.get(act, 0.1)
        if pd.isna(fallback_std) or fallback_std <= 0:
            fallback_std = 0.1
        p = np.random.normal(fallback_mean, fallback_std)

    return max(p, 0.1)


def plot_productivity_histogram(act, distribution_name, dist_param):
    data = df_prod[df_prod['Aktivitas'] == act]['p'].dropna().to_numpy()
    fig, ax = plt.subplots()

    bins = min(max(len(data), 5), 15)
    ax.hist(
        data,
        bins=bins,
        density=True,
        alpha=0.70,
        color="#4C78A8",
        edgecolor="white",
        label="Data"
    )

    x_min = max(np.min(data) * 0.8, 1e-6)
    x_max = np.max(data) * 1.2
    x = np.linspace(x_min, x_max, 300)

    params = dist_param.get(act)
    if params is not None:
        if distribution_name == "Normal":
            mu, sigma = params
            y = norm.pdf(x, loc=mu, scale=max(sigma, 1e-6))
            label = f"Fit Normal (mu={mu:.3f}, sigma={sigma:.3f})"
        else:
            shape, loc, scale = params
            y = lognorm.pdf(x, shape, loc=loc, scale=scale)
            label = f"Fit Lognormal (shape={shape:.3f})"

        ax.plot(x, y, color="#E45756", linewidth=2, label=label)

    ax.set_title(f"Histogram Produktivitas per Jenis Pekerjaan - {act}")
    ax.set_xlabel("Produktivitas (p)")
    ax.set_ylabel("Density")
    ax.legend()
    return fig


def standardize_ahsp_reference(df):
    aliases = {
        "Aktivitas": ["Aktivitas", "Jenis Pekerjaan", "Uraian Pekerjaan"],
        "Kode AHSP": ["Kode AHSP", "Kode SNI", "Kode"],
        "Koef AHSP": ["Koef AHSP", "Koefisien AHSP", "Koef SNI", "Koefisien SNI"]
    }

    rename_map = {}
    for target, options in aliases.items():
        for col in options:
            if col in df.columns:
                rename_map[col] = target
                break

    df_ref = df.rename(columns=rename_map).copy()
    required_cols = ["Aktivitas", "Kode AHSP", "Koef AHSP"]

    if not all(col in df_ref.columns for col in required_cols):
        return None

    df_ref = df_ref[required_cols].dropna(subset=["Aktivitas", "Koef AHSP"]).copy()
    df_ref["Aktivitas"] = df_ref["Aktivitas"].astype(str).str.strip()
    df_ref["Kode AHSP"] = df_ref["Kode AHSP"].astype(str).str.strip()
    df_ref["Koef AHSP"] = pd.to_numeric(df_ref["Koef AHSP"], errors="coerce")
    df_ref = df_ref.dropna(subset=["Koef AHSP"])
    return df_ref


def build_productivity_coefficient_table(df):
    summary = (
        df.groupby("Aktivitas")
        .agg(
            Sampel=("Aktivitas", "size"),
            Mean_p=("p", "mean"),
            Std_p=("p", "std"),
            Mean_Koef_Data=("Koefisien Data", "mean"),
            Std_Koef_Data=("Koefisien Data", "std"),
            P50_Koef_Data=("Koefisien Data", "median"),
            Min_Koef_Data=("Koefisien Data", "min"),
            Max_Koef_Data=("Koefisien Data", "max")
        )
        .reset_index()
    )

    summary["Koef_Setara_1_per_Mean_p"] = 1 / summary["Mean_p"]
    return summary


def build_ahsp_comparison(df_coef, df_ref):
    df_cmp = df_coef.merge(df_ref, on="Aktivitas", how="left")

    if "Koef AHSP" not in df_cmp.columns:
        return df_cmp

    df_cmp["Selisih"] = df_cmp["Mean_Koef_Data"] - df_cmp["Koef AHSP"]
    df_cmp["Selisih_%"] = np.where(
        df_cmp["Koef AHSP"].notna() & (df_cmp["Koef AHSP"] != 0),
        (df_cmp["Selisih"] / df_cmp["Koef AHSP"]) * 100,
        np.nan
    )
    df_cmp["Rasio_Data_vs_AHSP"] = np.where(
        df_cmp["Koef AHSP"].notna() & (df_cmp["Koef AHSP"] != 0),
        df_cmp["Mean_Koef_Data"] / df_cmp["Koef AHSP"],
        np.nan
    )
    df_cmp["Interpretasi"] = np.where(
        df_cmp["Koef AHSP"].isna(),
        "Referensi AHSP belum tersedia",
        np.where(
            df_cmp["Mean_Koef_Data"] < df_cmp["Koef AHSP"],
            "Data lebih efisien dari AHSP",
            np.where(
                df_cmp["Mean_Koef_Data"] > df_cmp["Koef AHSP"],
                "Data lebih boros dari AHSP",
                "Data setara dengan AHSP"
            )
        )
    )
    return df_cmp


def plot_coefficient_comparison(df_cmp):
    df_plot = df_cmp.dropna(subset=["Koef AHSP"]).copy()

    fig, ax = plt.subplots(figsize=(10, 5))

    if df_plot.empty:
        ax.text(
            0.5,
            0.5,
            "Data pembanding AHSP/SNI belum tersedia",
            ha="center",
            va="center",
            fontsize=11
        )
        ax.axis("off")
        return fig

    x = np.arange(len(df_plot))
    width = 0.36

    ax.bar(
        x - width / 2,
        df_plot["Mean_Koef_Data"],
        width,
        label="Observasi",
        color="#4C78A8"
    )
    ax.bar(
        x + width / 2,
        df_plot["Koef AHSP"],
        width,
        label="AHSP/SNI",
        color="#E45756"
    )

    ax.set_title("Perbandingan Koefisien Produktivitas per Jenis Pekerjaan: Observasi vs AHSP/SNI")
    ax.set_xlabel("Jenis Pekerjaan")
    ax.set_ylabel("Koefisien Produktivitas")
    ax.set_xticks(x)
    ax.set_xticklabels(df_plot["Aktivitas"], rotation=0)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig


def build_risk_map_table(df_cp, df_s):
    df_risk = df_cp.merge(df_s, on="Aktivitas", how="inner").copy()
    prob_threshold = df_risk["Prob"].median()
    impact_threshold = df_risk["Pengaruh"].median()

    def classify_risk(row):
        high_prob = row["Prob"] >= prob_threshold
        high_impact = row["Pengaruh"] >= impact_threshold

        if high_prob and high_impact:
            return "Tinggi"
        if high_prob or high_impact:
            return "Sedang"
        return "Rendah"

    df_risk["Kategori Risiko"] = df_risk.apply(classify_risk, axis=1)
    df_risk["Risk Score"] = df_risk["Prob"] * df_risk["Pengaruh"]

    df_risk["Prob_Threshold"] = prob_threshold
    df_risk["Impact_Threshold"] = impact_threshold
    category_order = {"Tinggi": 0, "Sedang": 1, "Rendah": 2}
    df_risk["Category_Order"] = df_risk["Kategori Risiko"].map(category_order)
    return df_risk.sort_values(
        by=["Category_Order", "Risk Score", "Prob", "Pengaruh"],
        ascending=[True, False, False, False]
    ).drop(columns=["Category_Order"])


def plot_risk_map(df_risk):
    fig, ax = plt.subplots(figsize=(8, 6))

    color_map = {
        "Tinggi": "#E45756",
        "Sedang": "#F2CF5B",
        "Rendah": "#54A24B"
    }

    for category in ["Tinggi", "Sedang", "Rendah"]:
        data = df_risk[df_risk["Kategori Risiko"] == category]
        if data.empty:
            continue

        ax.scatter(
            data["Pengaruh"],
            data["Prob"],
            s=130,
            color=color_map[category],
            edgecolors="black",
            alpha=0.85,
            label=category
        )

        for _, row in data.iterrows():
            ax.annotate(
                row["Aktivitas"],
                (row["Pengaruh"], row["Prob"]),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=9
            )

    impact_threshold = df_risk["Impact_Threshold"].iloc[0]
    prob_threshold = df_risk["Prob_Threshold"].iloc[0]

    ax.axvline(impact_threshold, color="gray", linestyle="--", linewidth=1)
    ax.axhline(prob_threshold, color="gray", linestyle="--", linewidth=1)
    ax.text(
        impact_threshold,
        1.02,
        f"Impact threshold = {impact_threshold:.3f}",
        ha="left",
        va="bottom",
        fontsize=9
    )
    ax.text(
        0.01,
        prob_threshold,
        f"Prob threshold = {prob_threshold:.3f}",
        ha="left",
        va="bottom",
        fontsize=9
    )

    ax.set_title("Peta Resiko Pekerjaan")
    ax.set_xlabel("Impact (Pengaruh terhadap Durasi Proyek)")
    ax.set_ylabel("Probability (Criticality Index)")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(title="Kategori")
    fig.tight_layout()
    return fig


def build_network_positions(df):
    activities = df["Aktivitas"].astype(str).tolist()
    order_map = {act: idx for idx, act in enumerate(activities)}
    pred_map = {
        str(row["Aktivitas"]): parse_predecessors(row["Predecessor"])
        for _, row in df.iterrows()
    }
    level_cache = {}

    def get_level(act):
        if act in level_cache:
            return level_cache[act]

        preds = pred_map.get(act, [])
        if not preds:
            level_cache[act] = 0
        else:
            level_cache[act] = max(get_level(pred) + 1 for pred in preds)

        return level_cache[act]

    for act in activities:
        get_level(act)

    level_groups = {}
    for act, level in level_cache.items():
        level_groups.setdefault(level, []).append(act)

    positions = {}
    for level in sorted(level_groups):
        level_acts = sorted(level_groups[level], key=lambda x: order_map[x])
        n = len(level_acts)
        y_positions = np.linspace((n - 1) / 2, -(n - 1) / 2, n) if n > 1 else [0]

        for act, y in zip(level_acts, y_positions):
            positions[act] = (level * 2.8, y * 2.0)

    return positions, pred_map


def plot_network_diagram(df, df_path, max_paths=5):
    positions, pred_map = build_network_positions(df)
    fig, ax = plt.subplots(figsize=(12, 6))

    base_edge_color = "#B0B0B0"
    base_node_edge = "#666666"
    highlight_colors = ["#D62728", "#1F77B4", "#2CA02C", "#9467BD", "#17BECF"]
    highlight_fill_colors = ["#FDECEC", "#EAF2FB", "#EAF7EA", "#F2ECFA", "#E8FAFA"]

    # Base network in gray.
    for act, preds in pred_map.items():
        x2, y2 = positions[act]
        for pred in preds:
            x1, y1 = positions[pred]
            ax.annotate(
                "",
                xy=(x2, y2),
                xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle="->",
                    color=base_edge_color,
                    lw=1.4,
                    shrinkA=20,
                    shrinkB=20
                ),
                zorder=1
            )

    path_legend = []
    node_color_map = {}
    node_fill_map = {}
    top_paths = df_path.head(max_paths).reset_index(drop=True)

    for idx, row in top_paths.iterrows():
        color = highlight_colors[idx % len(highlight_colors)]
        fill_color = highlight_fill_colors[idx % len(highlight_fill_colors)]
        activities = [act.strip() for act in str(row["Path"]).split("->")]

        for start_act, end_act in zip(activities, activities[1:]):
            x1, y1 = positions[start_act]
            x2, y2 = positions[end_act]
            ax.annotate(
                "",
                xy=(x2, y2),
                xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle="->",
                    color=color,
                    lw=3.2,
                    shrinkA=20,
                    shrinkB=20
                ),
                zorder=2 + idx
            )

        for act in activities:
            node_color_map.setdefault(act, color)
            node_fill_map.setdefault(act, fill_color)

        path_legend.append(
            Line2D(
                [0], [0],
                color=color,
                lw=3,
                label=f"CP{idx + 1} ({row['Prob']:.3f})"
            )
        )

    node_x = [positions[act][0] for act in df["Aktivitas"]]
    node_y = [positions[act][1] for act in df["Aktivitas"]]
    node_edge_colors = [node_color_map.get(act, base_node_edge) for act in df["Aktivitas"]]
    node_face_colors = [
        node_fill_map.get(act, "#F8F8F8")
        for act in df["Aktivitas"]
    ]

    ax.scatter(
        node_x,
        node_y,
        s=1800,
        c=node_face_colors,
        edgecolors=node_edge_colors,
        linewidths=2.4,
        zorder=4,
        clip_on=False
    )

    for act in df["Aktivitas"]:
        x, y = positions[act]
        ax.text(
            x,
            y,
            act,
            ha="center",
            va="center",
            fontsize=10,
            weight="bold",
            zorder=5
        )

    ax.set_title("Sketsa Network Diagram Probabilistik")
    x_min, x_max = min(node_x), max(node_x)
    y_min, y_max = min(node_y), max(node_y)
    x_margin = max(1.0, (x_max - x_min) * 0.12)
    y_margin = max(1.0, (y_max - y_min) * 0.18)
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    ax.axis("off")

    if path_legend:
        ax.legend(
            handles=path_legend,
            title="Lintasan Kritis",
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0)
        )

    fig.tight_layout()
    return fig

# =============================
# VALIDASI DEPENDENCY
# =============================
def validate(df):
    acts = set(df['Aktivitas'])
    errors = []

    for _, row in df.iterrows():
        act = row['Aktivitas']
        preds = parse_predecessors(row['Predecessor'])

        for p in preds:
            if p not in acts:
                errors.append(f"{p} tidak ditemukan")

    return errors

err = validate(df_proj)

if err:
    st.error("Dependency error:")
    for e in err:
        st.write("-", e)

# =============================
# PDM FUNCTION (FULL CP)
# =============================
def pdm_cp(df, durasi, tol=1e-9):
    ES, EF, row_map = {}, {}, {}

    for _, row in df.iterrows():
        act = row['Aktivitas']
        ES[act] = 0
        EF[act] = durasi[act]
        row_map[act] = {
            "rel": row['Relasi'],
            "lag": row['Lag'],
            "preds": parse_predecessors(row['Predecessor'])
        }

    changed = True
    iter_count = 0
    max_iter = max(len(df) * 2, 1)

    while changed:
        changed = False
        iter_count += 1

        if iter_count > max_iter:
            raise RuntimeError("Dependency proyek mengandung siklus atau relasi tidak valid.")

        for act, info in row_map.items():
            rel = info["rel"]
            lag = info["lag"]
            preds = info["preds"]

            if not preds:
                continue

            if rel == 'FS':
                new_ES = max(EF[p] for p in preds) + lag
            elif rel == 'SS':
                new_ES = max(ES[p] for p in preds) + lag
            elif rel == 'FF':
                new_ES = max(EF[p] for p in preds) + lag - durasi[act]
            elif rel == 'SF':
                new_ES = max(ES[p] for p in preds) + lag - durasi[act]
            else:
                continue

            if new_ES > ES[act] + tol:
                ES[act] = new_ES
                EF[act] = ES[act] + durasi[act]
                changed = True

    total = max(EF.values())
    end_acts = [
        act for act, ef in EF.items()
        if np.isclose(ef, total, atol=tol, rtol=0)
    ]

    def critical_predecessors(act):
        info = row_map[act]
        rel = info["rel"]
        lag = info["lag"]
        crit_preds = []

        for pred in info["preds"]:
            if rel == 'FS':
                candidate_es = EF[pred] + lag
            elif rel == 'SS':
                candidate_es = ES[pred] + lag
            elif rel == 'FF':
                candidate_es = EF[pred] + lag - durasi[act]
            elif rel == 'SF':
                candidate_es = ES[pred] + lag - durasi[act]
            else:
                continue

            if np.isclose(candidate_es, ES[act], atol=tol, rtol=0):
                crit_preds.append(pred)

        return crit_preds

    critical_paths = set()

    def backtrack(act, current_path):
        preds = critical_predecessors(act)
        next_path = current_path + [act]

        if not preds:
            critical_paths.add(tuple(reversed(next_path)))
            return

        for pred in preds:
            backtrack(pred, next_path)

    for end_act in end_acts:
        backtrack(end_act, [])

    return total, [list(path) for path in sorted(critical_paths)]


def build_durations_from_productivity(df, productivity_map):
    durasi = {}

    for _, row in df.iterrows():
        act = row['Aktivitas']
        Q = row['Volume']
        n = row['Tenaga']
        mean_p = productivity_map.get(act)

        if pd.isna(mean_p) or mean_p is None or mean_p <= 0:
            raise ValueError(
                f"Produktivitas rata-rata untuk aktivitas '{act}' tidak valid."
            )

        durasi[act] = Q / (n * mean_p)

    return durasi

# =============================
# PARAMETER
# =============================
st.sidebar.header("Parameter")
n_sim = st.sidebar.slider("Jumlah Simulasi", 100, 3000, 500)
distribution_name = st.sidebar.selectbox(
    "Distribusi Produktivitas",
    ["Lognormal", "Normal"],
    index=0
)
dist_param = fit_distribution_params(df_prod, distribution_name)

st.subheader("Histogram Produktivitas per Jenis Pekerjaan")
selected_activity = st.selectbox(
    "Pilih aktivitas untuk histogram produktivitas",
    sorted(df_prod["Aktivitas"].unique())
)
st.pyplot(plot_productivity_histogram(selected_activity, distribution_name, dist_param))

# =============================
# KOEFISIEN PRODUKTIVITAS
# =============================
st.subheader("Koefisien Produktivitas per Jenis Pekerjaan")
st.caption(
    "Asumsi koefisien data = (Tenaga x Waktu) / Output, sehingga satuannya mengikuti input waktu per satuan output."
)

df_coef = build_productivity_coefficient_table(df_prod)
st.dataframe(
    df_coef.style.format({
        "Mean_p": "{:.3f}",
        "Std_p": "{:.3f}",
        "Mean_Koef_Data": "{:.3f}",
        "Std_Koef_Data": "{:.3f}",
        "P50_Koef_Data": "{:.3f}",
        "Min_Koef_Data": "{:.3f}",
        "Max_Koef_Data": "{:.3f}",
        "Koef_Setara_1_per_Mean_p": "{:.3f}"
    }),
    use_container_width=True,
    hide_index=True
)

df_ahsp_ref = None
if "Referensi_AHSP" in excel_book.sheet_names:
    raw_ahsp_ref = pd.read_excel(excel_book, sheet_name="Referensi_AHSP")
    df_ahsp_ref = standardize_ahsp_reference(raw_ahsp_ref)

st.subheader("Perbandingan Koefisien Produktivitas per Jenis Pekerjaan")
if df_ahsp_ref is not None and not df_ahsp_ref.empty:
    df_coef_cmp = build_ahsp_comparison(df_coef, df_ahsp_ref)
    st.dataframe(
        df_coef_cmp.style.format({
            "Mean_p": "{:.3f}",
            "Mean_Koef_Data": "{:.3f}",
            "Koef AHSP": "{:.3f}",
            "Selisih": "{:.3f}",
            "Selisih_%": "{:.3f}",
            "Rasio_Data_vs_AHSP": "{:.3f}"
        }),
        use_container_width=True,
        hide_index=True
    )
    st.pyplot(plot_coefficient_comparison(df_coef_cmp))
else:
    st.info(
        "Referensi AHSP belum dibaca. Tambahkan sheet 'Referensi_AHSP' dengan kolom: "
        "'Aktivitas', 'Kode AHSP', dan 'Koef AHSP' agar tabel pembanding muncul otomatis."
    )

# =============================
# RUN
# =============================
if st.button("Jalankan Simulasi"):

    results = []
    critical_count = {act: 0 for act in df_proj['Aktivitas']}
    path_count = {}

    progress = st.progress(0)

    start = time.time()

    deterministic_durations = build_durations_from_productivity(df_proj, mean_p_map)
    deterministic_total, deterministic_paths = pdm_cp(df_proj, deterministic_durations)
    deterministic_acts = {
        act for path in deterministic_paths for act in path
    }

    for i in range(n_sim):

        durasi = {}

        for _, row in df_proj.iterrows():
            act = row['Aktivitas']
            Q = row['Volume']
            n = row['Tenaga']
            p = sample_productivity(act, dist_param, distribution_name)
            durasi[act] = Q / (n * p)

        total, critical_paths = pdm_cp(df_proj, durasi)

        critical_acts = {act for path in critical_paths for act in path}
        for act in critical_acts:
            critical_count[act] += 1

        for path in critical_paths:
            path_key = " -> ".join(path)
            path_count[path_key] = path_count.get(path_key, 0) + 1

        results.append(total)

        progress.progress((i+1)/n_sim)

    results = np.array(results)

    st.success("Simulasi selesai")

    # =============================
    # SIMULASI MONTE CARLO
    # =============================
    st.subheader("Hasil Simulasi Durasi Pekerjaan -- Simulasi Monte Carlo")
    st.dataframe(pd.DataFrame({"Durasi": results}).head(50))

    # =============================
    # STATISTIK DURASI PROBABILISTIK
    # =============================
    st.subheader("Analisis Probabilistik Durasi per Jenis Pekerjaan")
    df_stats = pd.DataFrame({
        "Statistik": ["Mean", "Std", "P50", "P60", "P70", "P80", "P90", "P100"],
        "Nilai": [
            np.mean(results),
            np.std(results),
            np.percentile(results, 50),
            np.percentile(results, 60),
            np.percentile(results, 70),
            np.percentile(results, 80),
            np.percentile(results, 90),
            np.percentile(results, 100)
        ]
    })
    st.dataframe(
        df_stats.style.format({"Nilai": "{:.3f}"}),
        use_container_width=True,
        hide_index=True
    )

    # =============================
    # HISTOGRAM
    # =============================
    st.subheader("Histogram Probabilistik Durasi per Jenis Pekerjaan")
    fig, ax = plt.subplots()
    ax.hist(results, bins=30)
    st.pyplot(fig)

    # =============================
    # PROBABILISTIC CRITICAL PATH
    # =============================
    st.subheader("Analisis Probabilistic Lintasan Kritis")

    df_path = pd.DataFrame({
        "Path": list(path_count.keys()),
        "Prob": [v/n_sim for v in path_count.values()]
    }).sort_values(by="Prob", ascending=False)

    st.dataframe(
        df_path.style.format({"Prob": "{:.3f}"}),
        use_container_width=True,
        hide_index=True
    )

    df_cp = pd.DataFrame({
        "Aktivitas": list(critical_count.keys()),
        "Prob": [v/n_sim for v in critical_count.values()]
    }).sort_values(by="Prob", ascending=False)

    st.subheader("Criticality Index (CI)")
    st.dataframe(
        df_cp.style.format({"Prob": "{:.3f}"}),
        use_container_width=True,
        hide_index=True
    )

    # =============================
    # NETWORK DIAGRAM
    # =============================
    st.subheader("Network Diagram")
    st.caption(
        "Lintasan kritis dengan probabilitas tertinggi diberi warna merah. "
        "Lintasan kritis berikutnya diberi warna berbeda berdasarkan urutan probabilitas."
    )

    df_network_legend = df_path.head(5).reset_index(drop=True).copy()
    df_network_legend.insert(
        0,
        "Warna",
        ["Merah", "Biru", "Hijau", "Ungu", "Toska"][:len(df_network_legend)]
    )
    st.dataframe(
        df_network_legend.style.format({"Prob": "{:.3f}"}),
        use_container_width=True,
        hide_index=True
    )
    st.pyplot(plot_network_diagram(df_proj, df_path))

    # =============================
    # DETERMINISTIC COMPARISON
    # =============================
    st.subheader("Analisis Deterministic")

    df_det_path = pd.DataFrame({
        "Path": [" -> ".join(path) for path in deterministic_paths],
        "Durasi Proyek": [deterministic_total] * len(deterministic_paths),
        "Status": ["Critical"] * len(deterministic_paths)
    })

    st.write("Deterministic Critical Path (berdasarkan mean productivity):")
    st.dataframe(
        df_det_path.style.format({"Durasi Proyek": "{:.3f}"}),
        use_container_width=True,
        hide_index=True
    )

    df_det_ci = pd.DataFrame({
        "Aktivitas": list(df_proj["Aktivitas"]),
        "Deterministik": [
            1.0 if act in deterministic_acts else 0.0
            for act in df_proj["Aktivitas"]
        ]
    })

    df_ci_compare = df_det_ci.merge(df_cp, on="Aktivitas", how="left")
    df_ci_compare = df_ci_compare.rename(columns={"Prob": "Probabilistik"})
    df_ci_compare["Probabilistik"] = df_ci_compare["Probabilistik"].fillna(0.0)
    df_ci_compare = df_ci_compare.sort_values(
        by=["Deterministik", "Probabilistik", "Aktivitas"],
        ascending=[False, False, True]
    )

    st.write("Perbandingan Criticality Index deterministik vs probabilistik:")
    st.dataframe(
        df_ci_compare.style.format({
            "Deterministik": "{:.3f}",
            "Probabilistik": "{:.3f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    # =============================
    # TORNADO
    # =============================
    st.subheader("Analisis Sensitivitas")

    sens = {}

    for act in df_proj['Aktivitas']:
        vals = []

        for f in [0.8, 1.2]:
            durasi = {}

            for _, row in df_proj.iterrows():
                a = row['Aktivitas']
                Q = row['Volume']
                n = row['Tenaga']

                mean_p = df_prod[df_prod['Aktivitas']==a]['p'].mean()

                p = mean_p * f if a == act else mean_p

                durasi[a] = Q / (n * p)

            total, _ = pdm_cp(df_proj, durasi)
            vals.append(total)

        sens[act] = abs(vals[1] - vals[0])

    df_s = pd.DataFrame({
        "Aktivitas": list(sens.keys()),
        "Pengaruh": list(sens.values())
    }).sort_values(by="Pengaruh")

    st.write("Tabel Hasil Tornado:")
    df_s_table = df_s.sort_values(by="Pengaruh", ascending=False).reset_index(drop=True)
    df_s_table.insert(0, "Rank", range(1, len(df_s_table) + 1))
    st.dataframe(
        df_s_table.style.format({"Pengaruh": "{:.3f}"}),
        use_container_width=True,
        hide_index=True
    )

    fig2, ax2 = plt.subplots()
    ax2.barh(df_s["Aktivitas"], df_s["Pengaruh"])
    st.pyplot(fig2)

    # =============================
    # PETA RESIKO
    # =============================
    st.subheader("Peta Resiko Pekerjaan")
    st.caption(
        "Probability diambil dari Criticality Index, Impact diambil dari nilai Tornado, dan Risk Score = CI x Impact Tornado."
    )

    df_risk = build_risk_map_table(df_cp, df_s)
    df_risk_display = df_risk.rename(columns={
        "Prob": "CI",
        "Pengaruh": "Impact Tornado"
    })
    st.dataframe(
        df_risk_display[["Aktivitas", "CI", "Impact Tornado", "Risk Score", "Kategori Risiko"]]
        .style.format({
            "CI": "{:.3f}",
            "Impact Tornado": "{:.3f}",
            "Risk Score": "{:.3f}"
        }),
        use_container_width=True,
        hide_index=True
    )
    st.pyplot(plot_risk_map(df_risk))

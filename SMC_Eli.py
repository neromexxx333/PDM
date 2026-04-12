import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy.stats import lognorm, norm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import time
import hashlib
from io import BytesIO
from html import escape

try:
    from PIL import Image, ImageChops
    PIL_AVAILABLE = True
except ImportError:
    Image = None
    ImageChops = None
    PIL_AVAILABLE = False

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    go = None
    PLOTLY_AVAILABLE = False

st.set_page_config(layout="wide")
st.title("Aplikasi Penjadwalan Proyek Konstruksi")
st.title("Berbasis Risiko Produktivitas Tenaga Kerja")
st.markdown(
    """
    <div style="font-size: 2.0rem; margin-top: 0.15rem; margin-bottom: 0.75rem;">
        Pengembang: Ir. Eliatun, ST., MT.
    </div>
    <div style="font-size: 2.0rem; margin-top: 0.15rem; margin-bottom: 0.75rem;">
        Fakultas Teknik Universitas Lambung Mangkurat
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
    div[data-testid="stDataFrame"] [role="gridcell"],
    div[data-testid="stDataFrame"] [role="columnheader"] {
        font-size: 0.95rem;
    }
    div[data-testid="stDataFrame"] [role="columnheader"] {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        text-align: center !important;
        justify-content: center !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


def prepare_table_dataframe(df, formats=None, show_index=False, start_index_at_one=False):
    table_df = df.copy()

    if start_index_at_one:
        table_df.index = np.arange(1, len(table_df) + 1)
        show_index = True

    return table_df, show_index


def render_table(df, formats=None, show_index=False, start_index_at_one=False):
    table_df, show_index = prepare_table_dataframe(
        df,
        formats=formats,
        show_index=show_index,
        start_index_at_one=start_index_at_one
    )

    styler = table_df.style
    if formats:
        styler = styler.format(formats)

    st.dataframe(
        styler,
        use_container_width=False,
        hide_index=not show_index
    )


def render_wrapped_table(
    df,
    formats=None,
    show_index=False,
    start_index_at_one=False,
    wide_columns=None
):
    table_df, show_index = prepare_table_dataframe(
        df,
        formats=formats,
        show_index=show_index,
        start_index_at_one=start_index_at_one
    )
    display_df = table_df.copy()

    if formats:
        for col, fmt in formats.items():
            if col not in display_df.columns:
                continue

            def format_value(value):
                if pd.isna(value):
                    return ""
                if callable(fmt):
                    return fmt(value)
                try:
                    return fmt.format(value)
                except Exception:
                    return value

            display_df[col] = display_df[col].map(format_value)

    wide_columns = set(wide_columns or [])
    header_html = []
    row_html = []

    if show_index:
        header_html.append('<th class="wrapped-col-index">No.</th>')

    for col in display_df.columns:
        classes = ["wrapped-col"]
        if col in wide_columns:
            classes.append("wrapped-col-wide")
        header_html.append(
            f'<th class="{" ".join(classes)}">{escape(str(col))}</th>'
        )

    for idx, row in display_df.iterrows():
        cells = []
        if show_index:
            cells.append(f'<td class="wrapped-col-index">{escape(str(idx))}</td>')

        for col in display_df.columns:
            classes = ["wrapped-col"]
            if col in wide_columns:
                classes.append("wrapped-col-wide")
            value = "" if pd.isna(row[col]) else row[col]
            cells.append(
                f'<td class="{" ".join(classes)}">{escape(str(value))}</td>'
            )

        row_html.append(f"<tr>{''.join(cells)}</tr>")

    table_html = f"""
    <style>
    .wrapped-table-container {{
        width: 100%;
        overflow-x: auto;
        margin-bottom: 0.5rem;
    }}
    .wrapped-table {{
        width: max-content;
        min-width: 100%;
        border-collapse: collapse;
        table-layout: auto;
    }}
    .wrapped-table th,
    .wrapped-table td {{
        border: 1px solid rgba(127, 127, 127, 0.30);
        padding: 0.55rem 0.70rem;
        vertical-align: top;
        text-align: left;
        white-space: nowrap;
        word-break: normal;
        overflow-wrap: normal;
    }}
    .wrapped-table th {{
        background: rgba(127, 127, 127, 0.12);
        font-weight: 700;
    }}
    .wrapped-table .wrapped-col-index {{
        width: 3.8rem;
        text-align: center;
    }}
    .wrapped-table .wrapped-col-wide {{
        min-width: 24rem;
        max-width: 38rem;
        white-space: normal !important;
        word-break: break-word;
        overflow-wrap: anywhere;
    }}
    </style>
    <div class="wrapped-table-container">
        <table class="wrapped-table">
            <thead><tr>{''.join(header_html)}</tr></thead>
            <tbody>{''.join(row_html)}</tbody>
        </table>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)


EXPORT_DPI = 320


def make_safe_filename(value):
    safe_name = "".join(
        ch if ch.isalnum() or ch in {"-", "_"} else "_"
        for ch in str(value).strip().replace(" ", "_")
    )
    return safe_name.strip("_") or "gambar"


def export_matplotlib_figure_to_jpg(fig, dpi=EXPORT_DPI):
    try:
        png_buffer = BytesIO()
        fig.savefig(
            png_buffer,
            format="png",
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.05,
            facecolor="white"
        )
        png_buffer.seek(0)

        if PIL_AVAILABLE:
            image = Image.open(png_buffer).convert("RGB")
            jpg_buffer = BytesIO()
            image.save(
                jpg_buffer,
                format="JPEG",
                quality=100,
                subsampling=0,
                dpi=(dpi, dpi)
            )
            return jpg_buffer.getvalue()

        jpg_buffer = BytesIO()
        fig.savefig(
            jpg_buffer,
            format="jpeg",
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.05,
            facecolor="white"
        )
        return jpg_buffer.getvalue()
    except Exception:
        return None


def render_jpg_download_button(fig, filename_base, key, label="Simpan JPG HD (320 dpi)"):
    jpg_bytes = export_matplotlib_figure_to_jpg(fig)

    if jpg_bytes is None:
        st.info("Ekspor JPG belum tersedia pada lingkungan ini.")
        return

    st.download_button(
        label=label,
        data=jpg_bytes,
        file_name=f"{make_safe_filename(filename_base)}.jpg",
        mime="image/jpeg",
        key=key
    )

# =============================
# UPLOAD
# =============================
file = st.file_uploader("Upload Excel Dataset", type=["xlsx"])

if file is None:
    st.warning("Upload file Excel terlebih dahulu")
    st.stop()

excel_book = pd.ExcelFile(BytesIO(file.getvalue()))
current_file_signature = hashlib.md5(file.getvalue()).hexdigest()

# =============================
# LOAD DATA
# =============================
df_proj = pd.read_excel(excel_book, sheet_name="Data_Proyek")
df_prod = pd.read_excel(excel_book, sheet_name="Data_Produktivitas")
activity_work_type_map = {}
activity_work_type_source = "Deteksi keyword otomatis"

st.subheader("Data Proyek")
render_table(df_proj, show_index=True, start_index_at_one=True)

st.subheader("Data Produktivitas per Jenis Pekerjaan")
render_table(df_prod, show_index=True, start_index_at_one=True)

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
render_table(df_kal, formats={"Mean p": "{:.3f}", "Std Dev": "{:.3f}"})
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


def normalize_relation(val):
    if pd.isna(val):
        return "FS"

    relation = str(val).strip().upper()
    return relation if relation in {"FS", "SS", "FF", "SF"} else "FS"


def normalize_lag(val):
    if pd.isna(val):
        return 0.0

    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def build_activity_table(df):
    columns = ["Aktivitas", "Volume", "Tenaga"]
    activity_map = {}

    for _, row in df.iterrows():
        act = str(row.get("Aktivitas", "")).strip()
        if not act or act.lower() == "nan":
            continue

        if act not in activity_map:
            activity_map[act] = {
                "Aktivitas": act,
                "Volume": row.get("Volume"),
                "Tenaga": row.get("Tenaga")
            }
        else:
            if pd.isna(activity_map[act]["Volume"]) and pd.notna(row.get("Volume")):
                activity_map[act]["Volume"] = row.get("Volume")
            if pd.isna(activity_map[act]["Tenaga"]) and pd.notna(row.get("Tenaga")):
                activity_map[act]["Tenaga"] = row.get("Tenaga")

    if not activity_map:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(activity_map.values(), columns=columns)


def build_relation_table(df):
    relation_rows = []

    for _, row in df.iterrows():
        act = str(row.get("Aktivitas", "")).strip()
        if not act or act.lower() == "nan":
            continue

        relation = normalize_relation(row.get("Relasi"))
        lag = normalize_lag(row.get("Lag"))

        for pred in parse_predecessors(row.get("Predecessor")):
            relation_rows.append({
                "Aktivitas": act,
                "Predecessor": pred,
                "Relasi": relation,
                "Lag": lag
            })

    columns = ["Aktivitas", "Predecessor", "Relasi", "Lag"]
    if not relation_rows:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(relation_rows, columns=columns).drop_duplicates(ignore_index=True)


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


def sample_productivity(act, dist_param, distribution_name, rng):
    params = dist_param.get(act)

    if params is not None:
        if distribution_name == "Normal":
            mu, sigma = params
            p = rng.normal(mu, sigma)
        else:
            shape, loc, scale = params
            p = lognorm.rvs(shape, loc=loc, scale=scale, random_state=rng)
    else:
        fallback_mean = mean_p_map.get(act, 2.0)
        fallback_std = std_p_map.get(act, 0.1)
        if pd.isna(fallback_std) or fallback_std <= 0:
            fallback_std = 0.1
        p = rng.normal(fallback_mean, fallback_std)

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
        edgecolor="black",
        linewidth=1,
        label="Observed Data"
    )

    x_min = max(np.min(data) * 0.8, 1e-6)
    x_max = np.max(data) * 1.2
    x = np.linspace(x_min, x_max, 300)

    params = dist_param.get(act)
    if params is not None:
        if distribution_name == "Normal":
            mu, sigma = params
            y = norm.pdf(x, loc=mu, scale=max(sigma, 1e-6))
            label = f"Normal Fit (mu={mu:.3f}, sigma={sigma:.3f})"
        else:
            shape, loc, scale = params
            y = lognorm.pdf(x, shape, loc=loc, scale=scale)
            label = f"Lognormal Fit (shape={shape:.3f})"

        ax.plot(x, y, color="#E45756", linewidth=2, label=label)

    ax.set_title(f"Productivity Histogram by Work Item - {act}")
    ax.set_xlabel("Productivity (p)")
    ax.set_ylabel("Probability Density")
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


def standardize_activity_work_type_reference(df):
    aliases = {
        "Aktivitas": [
            "Aktivitas", "Kode Aktivitas", "Kode", "Activity", "ID Aktivitas"
        ],
        "Jenis Pekerjaan": [
            "Jenis Pekerjaan", "Uraian Pekerjaan", "Nama Pekerjaan",
            "Deskripsi Pekerjaan", "Work Item", "Item Pekerjaan"
        ]
    }

    rename_map = {}
    for target, options in aliases.items():
        for col in options:
            if col in df.columns:
                rename_map[col] = target
                break

    df_ref = df.rename(columns=rename_map).copy()
    required_cols = ["Aktivitas", "Jenis Pekerjaan"]

    if not all(col in df_ref.columns for col in required_cols):
        return None

    df_ref = df_ref[required_cols].dropna(subset=["Aktivitas", "Jenis Pekerjaan"]).copy()
    df_ref["Aktivitas"] = df_ref["Aktivitas"].astype(str).str.strip()
    df_ref["Jenis Pekerjaan"] = df_ref["Jenis Pekerjaan"].astype(str).str.strip()
    df_ref = df_ref[
        (df_ref["Aktivitas"] != "") &
        (df_ref["Jenis Pekerjaan"] != "") &
        (df_ref["Aktivitas"].str.lower() != "nan") &
        (df_ref["Jenis Pekerjaan"].str.lower() != "nan")
    ]
    return df_ref.drop_duplicates(subset=["Aktivitas"], keep="first")


def build_activity_work_type_map(df_project, excel_book=None):
    work_type_map = {}
    source_labels = []

    direct_ref = standardize_activity_work_type_reference(df_project)
    if direct_ref is not None and not direct_ref.empty:
        work_type_map.update(dict(zip(direct_ref["Aktivitas"], direct_ref["Jenis Pekerjaan"])))
        source_labels.append("Data_Proyek")

    if excel_book is not None:
        candidate_sheets = [
            "Mapping_Aktivitas",
            "Referensi_Aktivitas",
            "Master_Aktivitas",
            "Jenis_Pekerjaan",
            "Work_Type_Map"
        ]
        for sheet_name in candidate_sheets:
            if sheet_name not in excel_book.sheet_names:
                continue

            try:
                df_ref_raw = pd.read_excel(excel_book, sheet_name=sheet_name)
            except Exception:
                continue

            df_ref = standardize_activity_work_type_reference(df_ref_raw)
            if df_ref is None or df_ref.empty:
                continue

            for _, row in df_ref.iterrows():
                work_type_map.setdefault(row["Aktivitas"], row["Jenis Pekerjaan"])
            source_labels.append(sheet_name)

    source_label = ", ".join(source_labels) if source_labels else "Deteksi keyword otomatis"
    return work_type_map, source_label


activity_work_type_map, activity_work_type_source = build_activity_work_type_map(
    df_proj,
    excel_book
)


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
    df_cmp["Rasio_Data_vs_SNI(AHSP)"] = np.where(
        df_cmp["Koef AHSP"].notna() & (df_cmp["Koef AHSP"] != 0),
        df_cmp["Mean_Koef_Data"] / df_cmp["Koef AHSP"],
        np.nan
    )
    df_cmp["Interpretasi"] = np.where(
        df_cmp["Koef AHSP"].isna(),
        "Referensi SNI(AHSP) belum tersedia",
        np.where(
            df_cmp["Mean_Koef_Data"] < df_cmp["Koef AHSP"],
            "Data lebih efisien dari SNI(AHSP)",
            np.where(
                df_cmp["Mean_Koef_Data"] > df_cmp["Koef AHSP"],
                "Data lebih boros dari SNI(AHSP)",
                "Data setara dengan SNI(AHSP)"
            )
        )
    )
    df_cmp = df_cmp.rename(columns={
        "Kode AHSP": "Kode SNI(AHSP)",
        "Koef AHSP": "Koef SNI(AHSP)"
    })
    return df_cmp


def plot_coefficient_comparison(df_cmp):
    df_plot = df_cmp.dropna(subset=["Koef SNI(AHSP)"]).copy()

    fig, ax = plt.subplots(figsize=(10, 5))

    if df_plot.empty:
        ax.text(
            0.5,
            0.5,
            "SNI (AHSP) reference data are not available",
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
        label="Observed Data",
        color="#4C78A8",
        edgecolor="black",
        linewidth=1
    )
    ax.bar(
        x + width / 2,
        df_plot["Koef SNI(AHSP)"],
        width,
        label="SNI (AHSP) Reference",
        color="#E45756",
        edgecolor="black",
        linewidth=1
    )

    ax.set_title("Productivity Coefficient Comparison by Work Item: Observed Data vs. SNI (AHSP) Reference")
    ax.set_xlabel("Work Item")
    ax.set_ylabel("Productivity Coefficient")
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
    category_label_map = {
        "Tinggi": "High",
        "Sedang": "Medium",
        "Rendah": "Low"
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
            label=category_label_map[category]
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
        f"Impact Threshold = {impact_threshold:.3f}",
        ha="left",
        va="bottom",
        fontsize=9
    )
    ax.text(
        0.01,
        prob_threshold,
        f"Probability Threshold = {prob_threshold:.3f}",
        ha="left",
        va="bottom",
        fontsize=9
    )

    ax.set_title("Risk Map by Work Item")
    ax.set_xlabel("Impact on Project Duration")
    ax.set_ylabel("Probability (Criticality Index)")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(title="Risk Category")
    fig.tight_layout()
    return fig


def detect_work_type(activity_name, activity_work_type_map=None):
    mapped_name = None
    if activity_work_type_map:
        mapped_name = activity_work_type_map.get(str(activity_name).strip())

    name = str(mapped_name if mapped_name else activity_name).strip().lower()
    keyword_groups = [
        ("Pondasi", ["pondasi", "foundation", "bore pile", "pancang", "pile cap", "footplat", "sloof"]),
        ("Pekerjaan Tanah", ["galian", "urugan", "tanah", "cut and fill", "backfill", "excav", "subgrade"]),
        ("Bekisting", ["bekisting", "formwork", "shuttering", "form"]),
        ("Pembesian", ["pembesian", "rebar", "tulangan", "wiremesh", "besi"]),
        ("Beton", ["beton", "concrete", "cor", "pengecoran", "grouting"]),
        ("Struktur Baja", ["baja", "steel", "truss", "erection"]),
        ("Pasangan Dinding", ["pasangan", "bata", "hebel", "aac", "masonry", "dinding"]),
        ("Plester/Acian", ["plester", "aci", "plaster", "skim coat"]),
        ("Lantai/Keramik", ["keramik", "ubin", "tile", "granite", "lantai"]),
        ("Pengecatan/Finishing", ["cat", "pengecatan", "finishing", "coating", "sealant"]),
        ("Atap", ["atap", "roof", "genteng", "spandek", "rangka atap"]),
        ("Plumbing/Sanitary", ["plumbing", "sanitary", "pipa", "air bersih", "air kotor", "drainase"]),
        ("Elektrikal", ["listrik", "elektrikal", "kabel", "panel", "lampu", "conduit", "grounding"]),
        ("HVAC/MEP", ["hvac", "ducting", "ac", "mekanikal", "mep"]),
        ("Kusen/Pintu/Jendela", ["kusen", "pintu", "jendela", "aluminium", "kaca"]),
        ("Jalan/Perkerasan", ["jalan", "aspal", "perkerasan", "paving", "rigid pavement"])
    ]

    for work_type, keywords in keyword_groups:
        if any(keyword in name for keyword in keywords):
            return work_type

    if mapped_name:
        return str(mapped_name).strip()

    return "Umum"


def build_work_type_focus_action(work_type):
    action_map = {
        "Pondasi": "Pastikan titik/elevasi kerja, alat bor/pancang, tulangan, dan suplai beton siap sesuai urutan eksekusi.",
        "Pekerjaan Tanah": "Kontrol kondisi tanah, akses alat angkut, dewatering, dan hasil pemadatan agar produktivitas tidak turun.",
        "Bekisting": "Amankan shop drawing, siklus bongkar-pasang panel, ketersediaan material formwork, dan inspeksi alignment.",
        "Pembesian": "Siapkan bending schedule, cutting list, area fabrikasi, dan stok tulangan agar pemasangan tidak tersendat.",
        "Beton": "Kunci jadwal pengecoran, batching plant/pompa, vibrator cadangan, serta kontrol slump dan curing.",
        "Struktur Baja": "Sinkronkan fabrikasi, delivery member, crane/lifting plan, dan kesiapan baut maupun sambungan.",
        "Pasangan Dinding": "Jaga suplai material pasangan, ketelitian layout bukaan, dan ritme kerja antar zona.",
        "Plester/Acian": "Pastikan area siap kerja, kondisi dasar dinding sesuai, dan rotasi tenaga menjaga mutu serta output.",
        "Lantai/Keramik": "Amankan approval pola/potongan, leveling dasar, stok material, dan akses area bebas gangguan.",
        "Pengecatan/Finishing": "Kontrol kesiapan permukaan, waktu pengeringan, mockup mutu, dan konsistensi material finishing.",
        "Atap": "Pastikan area kerja aman, material atap lengkap, dan cuaca serta akses lifting terencana dengan baik.",
        "Plumbing/Sanitary": "Koordinasikan shop drawing, jalur instalasi, prefabrikasi pipa, dan pressure test sebelum penutupan area.",
        "Elektrikal": "Jaga kesiapan material kabel/panel, urutan instalasi tray-conduit, dan inspeksi pengujian bertahap.",
        "HVAC/MEP": "Sinkronkan jalur ducting/mep dengan ruang kerja lain dan pastikan item long lead tersedia tepat waktu.",
        "Kusen/Pintu/Jendela": "Pastikan ukuran lapangan final, material datang per zona, dan pemasangan terlindung dari rework finishing.",
        "Jalan/Perkerasan": "Kontrol urutan hamparan, kesiapan alat, kepadatan/lapisan dasar, dan pasokan material harian.",
        "Umum": "Pastikan keseimbangan tenaga, alat, material, dan akses kerja dijaga stabil sepanjang pelaksanaan."
    }
    return action_map.get(work_type, action_map["Umum"])


def build_risk_recommendation_table(
    df_risk,
    schedule_metrics,
    activity_work_type_map=None,
    near_critical_tf=2.0
):
    columns = [
        "Aktivitas",
        "Jenis Pekerjaan",
        "Kategori Risiko",
        "Status Jadwal",
        "Pemicu Dominan",
        "Prioritas",
        "Rekomendasi Teknis"
    ]
    if df_risk.empty:
        return pd.DataFrame(columns=columns)

    metrics_lookup = (
        schedule_metrics["metrics_df"]
        .set_index("Aktivitas")
        .to_dict("index")
    )
    prob_threshold = float(df_risk["Prob_Threshold"].iloc[0])
    impact_threshold = float(df_risk["Impact_Threshold"].iloc[0])
    recommendation_rows = []

    for _, row in df_risk.iterrows():
        act = row["Aktivitas"]
        prob = float(row["Prob"])
        impact = float(row["Pengaruh"])
        category = row["Kategori Risiko"]
        metrics = metrics_lookup.get(act, {})
        tf = float(metrics.get("TF", np.nan))
        work_type = detect_work_type(act, activity_work_type_map=activity_work_type_map)
        work_type_action = build_work_type_focus_action(work_type)

        high_prob = prob >= prob_threshold
        high_impact = impact >= impact_threshold

        if np.isfinite(tf) and np.isclose(tf, 0.0, atol=1e-9):
            schedule_status = "Kritis"
        elif np.isfinite(tf) and tf <= near_critical_tf:
            schedule_status = "Hampir Kritis"
        else:
            schedule_status = "Non-Kritis"

        if high_prob and high_impact:
            dominant_trigger = "Probabilitas & Impact Tinggi"
        elif high_prob:
            dominant_trigger = "Probabilitas Tinggi"
        elif high_impact:
            dominant_trigger = "Impact Tinggi"
        else:
            dominant_trigger = "Terkendali"

        if category == "Tinggi":
            if schedule_status == "Kritis":
                priority = "Segera"
                recommendation = (
                    "Tambah cadangan tenaga/alat, pastikan material siap sebelum ES, "
                    "kontrol produktivitas harian, dan siapkan skenario percepatan. "
                    f"Fokus {work_type.lower()}: {work_type_action}"
                )
            else:
                priority = "Tinggi"
                recommendation = (
                    "Kunci kesiapan predecessor, siapkan cadangan sumber daya, "
                    "audit metode kerja, dan aktifkan recovery plan saat deviasi muncul. "
                    f"Fokus {work_type.lower()}: {work_type_action}"
                )
        elif category == "Sedang":
            if dominant_trigger == "Probabilitas Tinggi":
                priority = "Menengah"
                recommendation = (
                    "Pantau output harian, stabilkan komposisi crew, "
                    "dan cek kesiapan alat/material sebelum pekerjaan dimulai. "
                    f"Fokus {work_type.lower()}: {work_type_action}"
                )
            elif dominant_trigger == "Impact Tinggi":
                priority = "Menengah"
                recommendation = (
                    "Siapkan buffer alat/material, review urutan kerja dan akses area, "
                    "serta percepat keputusan lapangan bila ada hambatan. "
                    f"Fokus {work_type.lower()}: {work_type_action}"
                )
            else:
                priority = "Menengah"
                recommendation = (
                    "Lakukan monitoring mingguan, validasi produktivitas aktual, "
                    "dan koreksi deviasi kecil lebih awal. "
                    f"Fokus {work_type.lower()}: {work_type_action}"
                )
        else:
            priority = "Rutin"
            recommendation = (
                "Pertahankan metode kerja yang berjalan, monitor periodik, "
                "dan jaga kontinuitas material, alat, serta tenaga inti. "
                f"Fokus {work_type.lower()}: {work_type_action}"
            )

        recommendation_rows.append({
            "Aktivitas": act,
            "Jenis Pekerjaan": work_type,
            "Kategori Risiko": category,
            "Status Jadwal": schedule_status,
            "Pemicu Dominan": dominant_trigger,
            "Prioritas": priority,
            "Rekomendasi Teknis": recommendation
        })

    return pd.DataFrame(recommendation_rows, columns=columns)


def build_network_positions(df):
    df_activities = build_activity_table(df)
    df_relations = build_relation_table(df)
    activities = df_activities["Aktivitas"].astype(str).tolist()
    order_map = {act: idx for idx, act in enumerate(activities)}
    pred_map = {act: [] for act in activities}

    for _, row in df_relations.iterrows():
        pred_map.setdefault(row["Aktivitas"], []).append(row["Predecessor"])

    pred_map = {act: list(dict.fromkeys(preds)) for act, preds in pred_map.items()}
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
        level_acts = level_groups[level]
        n = len(level_acts)

        if level == 0:
            level_acts = sorted(level_acts, key=lambda x: order_map[x])
        else:
            def barycenter(act):
                preds = pred_map.get(act, [])
                pred_y = [positions[pred][1] for pred in preds if pred in positions]
                if pred_y:
                    return -np.mean(pred_y)
                return order_map[act]

            level_acts = sorted(level_acts, key=lambda x: (barycenter(x), order_map[x]))

        y_positions = np.linspace((n - 1) / 2, -(n - 1) / 2, n) if n > 1 else [0]

        for act, y in zip(level_acts, y_positions):
            positions[act] = (level * 3.2, y * 2.25)

    return positions, pred_map


def calculate_schedule_metrics(df, durasi, tol=1e-9):
    df_activities = build_activity_table(df)
    df_relations = build_relation_table(df)
    activities = df_activities["Aktivitas"].astype(str).tolist()

    incoming_map = {act: [] for act in activities}
    for _, row in df_relations.iterrows():
        incoming_map.setdefault(row["Aktivitas"], []).append({
            "pred": row["Predecessor"],
            "rel": row["Relasi"],
            "lag": row["Lag"]
        })

    es = {act: 0.0 for act in activities}
    ef = {act: float(durasi[act]) for act in activities}
    max_iter = max(len(activities) * 5, 1)

    changed = True
    iter_count = 0
    while changed:
        changed = False
        iter_count += 1

        if iter_count > max_iter:
            raise RuntimeError("Perhitungan ES/EF tidak konvergen. Periksa relasi dan predecessor.")

        for act in activities:
            constraints = incoming_map.get(act, [])
            if not constraints:
                continue

            candidate_es = []
            for constraint in constraints:
                pred = constraint["pred"]
                rel = constraint["rel"]
                lag = constraint["lag"]

                if rel == "FS":
                    candidate_es.append(ef[pred] + lag)
                elif rel == "SS":
                    candidate_es.append(es[pred] + lag)
                elif rel == "FF":
                    candidate_es.append(ef[pred] + lag - durasi[act])
                else:
                    candidate_es.append(es[pred] + lag - durasi[act])

            new_es = max(candidate_es)

            if new_es > es[act] + tol:
                es[act] = new_es
                ef[act] = es[act] + durasi[act]
                changed = True

    project_duration = max(ef.values()) if ef else 0.0
    outgoing_map = {act: [] for act in activities}

    for _, row in df_relations.iterrows():
        outgoing_map.setdefault(row["Predecessor"], []).append(
            (row["Aktivitas"], row["Relasi"], row["Lag"])
        )

    lf = {act: float(project_duration) for act in activities}
    ls = {act: lf[act] - durasi[act] for act in activities}

    changed = True
    iter_count = 0
    while changed:
        changed = False
        iter_count += 1

        if iter_count > max_iter:
            raise RuntimeError("Perhitungan LS/LF tidak konvergen. Periksa relasi dan successor.")

        for act in reversed(activities):
            successors = outgoing_map.get(act, [])

            if not successors:
                new_lf = project_duration
            else:
                candidate_lf = []
                for succ, rel, lag in successors:
                    if rel == "FS":
                        candidate_lf.append(ls[succ] - lag)
                    elif rel == "SS":
                        candidate_lf.append(ls[succ] - lag + durasi[act])
                    elif rel == "FF":
                        candidate_lf.append(lf[succ] - lag)
                    else:
                        candidate_lf.append(lf[succ] - lag + durasi[act])

                new_lf = min(candidate_lf)

            if new_lf < lf[act] - tol:
                lf[act] = new_lf
                ls[act] = lf[act] - durasi[act]
                changed = True

    metrics_rows = []
    for act in activities:
        tf = lf[act] - ef[act]
        if abs(tf) < tol:
            tf = 0.0

        metrics_rows.append({
            "Aktivitas": act,
            "Durasi": float(durasi[act]),
            "ES": es[act],
            "EF": ef[act],
            "LS": ls[act],
            "LF": lf[act],
            "TF": tf
        })

    metrics_df = pd.DataFrame(metrics_rows)
    return {
        "project_duration": project_duration,
        "metrics_df": metrics_df
    }


def get_path_terminal_duration(path_label, schedule_metrics):
    metrics_lookup = (
        schedule_metrics["metrics_df"]
        .set_index("Aktivitas")
        .to_dict("index")
    )
    activities = [
        act.strip()
        for act in str(path_label).split("->")
        if act.strip()
    ]

    if not activities:
        return np.nan

    end_act = activities[-1]
    metrics = metrics_lookup.get(end_act)

    if metrics is None:
        return np.nan

    return float(metrics["EF"])


def build_path_conditioned_activity_durations(
    path_label,
    path_count,
    path_activity_duration_sum,
    fallback_durations
):
    if path_label not in path_activity_duration_sum or path_count.get(path_label, 0) <= 0:
        return dict(fallback_durations)

    count = path_count[path_label]
    conditioned_sum = path_activity_duration_sum[path_label]

    return {
        act: conditioned_sum.get(act, fallback_durations.get(act, 0.0)) / count
        for act in fallback_durations.keys()
    }


def scale_durations_to_target_path_duration(
    df,
    path_label,
    base_durations,
    target_duration,
    tol=1e-6,
    max_iter=60
):
    def build_scaled_durations(factor):
        return {
            act: max(float(duration) * factor, 0.0)
            for act, duration in base_durations.items()
        }

    def compute_path_duration(factor):
        schedule = calculate_schedule_metrics(df, build_scaled_durations(factor))
        return get_path_terminal_duration(path_label, schedule), schedule

    if not np.isfinite(target_duration) or target_duration <= 0:
        return calculate_schedule_metrics(df, base_durations)

    try:
        current_duration, current_schedule = compute_path_duration(1.0)
    except Exception:
        return calculate_schedule_metrics(df, base_durations)

    if not np.isfinite(current_duration) or current_duration <= 0:
        return current_schedule

    if abs(current_duration - target_duration) <= tol:
        return current_schedule

    low_factor = 0.0
    high_factor = 1.0

    try:
        low_duration, _ = compute_path_duration(low_factor)
    except Exception:
        low_duration = np.nan

    if current_duration < target_duration:
        high_duration = current_duration
        while high_duration < target_duration and high_factor < 1e6:
            low_factor = high_factor
            high_factor *= 2.0
            high_duration, _ = compute_path_duration(high_factor)
    else:
        if np.isfinite(low_duration) and low_duration > target_duration:
            return current_schedule

    best_schedule = current_schedule

    for _ in range(max_iter):
        mid_factor = (low_factor + high_factor) / 2.0
        mid_duration, mid_schedule = compute_path_duration(mid_factor)
        best_schedule = mid_schedule

        if not np.isfinite(mid_duration):
            break

        if abs(mid_duration - target_duration) <= tol:
            return mid_schedule

        if mid_duration < target_duration:
            low_factor = mid_factor
        else:
            high_factor = mid_factor

    return best_schedule


def scale_durations_to_target_project_duration(
    df,
    base_durations,
    target_duration,
    tol=1e-6,
    max_iter=60
):
    def build_scaled_durations(factor):
        return {
            act: max(float(duration) * factor, 0.0)
            for act, duration in base_durations.items()
        }

    def compute_project_duration(factor):
        schedule = calculate_schedule_metrics(df, build_scaled_durations(factor))
        return float(schedule["project_duration"]), schedule

    if not np.isfinite(target_duration) or target_duration <= 0:
        return calculate_schedule_metrics(df, base_durations)

    try:
        current_duration, current_schedule = compute_project_duration(1.0)
    except Exception:
        return calculate_schedule_metrics(df, base_durations)

    if not np.isfinite(current_duration) or current_duration <= 0:
        return current_schedule

    if abs(current_duration - target_duration) <= tol:
        return current_schedule

    low_factor = 0.0
    high_factor = 1.0

    try:
        low_duration, _ = compute_project_duration(low_factor)
    except Exception:
        low_duration = np.nan

    if current_duration < target_duration:
        high_duration = current_duration
        while high_duration < target_duration and high_factor < 1e6:
            low_factor = high_factor
            high_factor *= 2.0
            high_duration, _ = compute_project_duration(high_factor)
    else:
        if np.isfinite(low_duration) and low_duration > target_duration:
            return current_schedule

    best_schedule = current_schedule

    for _ in range(max_iter):
        mid_factor = (low_factor + high_factor) / 2.0
        mid_duration, mid_schedule = compute_project_duration(mid_factor)
        best_schedule = mid_schedule

        if not np.isfinite(mid_duration):
            break

        if abs(mid_duration - target_duration) <= tol:
            return mid_schedule

        if mid_duration < target_duration:
            low_factor = mid_factor
        else:
            high_factor = mid_factor

    return best_schedule


def plot_network_diagram(df, df_path, schedule_metrics, max_paths=5):
    df_activities = build_activity_table(df)
    df_relations = build_relation_table(df)
    positions, pred_map = build_network_positions(df)

    base_edge_color = "#B0B0B0"
    base_node_edge = "#666666"
    base_node_fill = "#F8F8F8"
    highlight_colors = ["#D62728", "#1F77B4", "#2CA02C", "#9467BD", "#17BECF"]
    highlight_fill_colors = ["#FDECEC", "#EAF2FB", "#EAF7EA", "#F2ECFA", "#E8FAFA"]
    node_width = 1.55
    node_height = 1.05
    endpoint_width = 1.55
    endpoint_height = 0.88

    metrics_lookup = (
        schedule_metrics["metrics_df"]
        .set_index("Aktivitas")
        .to_dict("index")
    )

    outgoing_map = {act: [] for act in df_activities["Aktivitas"]}
    for _, row in df_relations.iterrows():
        outgoing_map.setdefault(row["Predecessor"], []).append(row["Aktivitas"])

    incoming_lane_order = {}
    incoming_lane_count = {}
    for end_act, group in df_relations.groupby("Aktivitas"):
        ordered_preds = sorted(
            group["Predecessor"].dropna().astype(str).unique().tolist(),
            key=lambda pred: (
                -positions.get(pred, (0.0, 0.0))[1],
                positions.get(pred, (0.0, 0.0))[0],
                pred
            )
        )
        incoming_lane_order[end_act] = {
            pred: idx for idx, pred in enumerate(ordered_preds)
        }
        incoming_lane_count[end_act] = len(ordered_preds)

    start_acts = [act for act in df_activities["Aktivitas"] if not pred_map.get(act)]
    finish_acts = [act for act in df_activities["Aktivitas"] if not outgoing_map.get(act)]
    finish_lane_acts = sorted(
        finish_acts,
        key=lambda act: (
            -positions.get(act, (0.0, 0.0))[1],
            positions.get(act, (0.0, 0.0))[0],
            act
        )
    )
    incoming_lane_order["__FINISH__"] = {
        act: idx for idx, act in enumerate(finish_lane_acts)
    }
    incoming_lane_count["__FINISH__"] = len(finish_lane_acts)

    x_values = [pos[0] for pos in positions.values()]
    start_x = min(x_values) - 2.0
    finish_x = max(x_values) + 2.0
    positions["__START__"] = (start_x, 0.0)
    positions["__FINISH__"] = (finish_x, 0.0)
    endpoint_metrics = {
        "__START__": {"ES": 0.0, "EF": 0.0, "LS": 0.0, "LF": 0.0},
        "__FINISH__": {
            "ES": float(schedule_metrics["project_duration"]),
            "EF": float(schedule_metrics["project_duration"]),
            "LS": float(schedule_metrics["project_duration"]),
            "LF": float(schedule_metrics["project_duration"])
        }
    }
    all_pos_x = [pos[0] for pos in positions.values()]
    all_pos_y = [pos[1] for pos in positions.values()]
    span_x = max(all_pos_x) - min(all_pos_x)
    span_y = max(all_pos_y) - min(all_pos_y)
    fig_width = min(max(8.5, span_x * 0.95), 13.0)
    fig_height = min(max(4.8, span_y * 0.95), 7.2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    def draw_text(x, y, text, outline_width=1.1, **kwargs):
        text_artist = ax.text(x, y, text, **kwargs)
        text_artist.set_path_effects([
            pe.withStroke(linewidth=outline_width, foreground="white"),
            pe.Normal()
        ])
        return text_artist

    node_bounds = {
        act: (
            positions[act][0] - node_width / 2,
            positions[act][0] + node_width / 2,
            positions[act][1] - node_height / 2,
            positions[act][1] + node_height / 2,
        )
        for act in df_activities["Aktivitas"]
    }
    node_bounds["__START__"] = (
        start_x - endpoint_width / 2,
        start_x + endpoint_width / 2,
        -endpoint_height / 2,
        endpoint_height / 2
    )
    node_bounds["__FINISH__"] = (
        finish_x - endpoint_width / 2,
        finish_x + endpoint_width / 2,
        -endpoint_height / 2,
        endpoint_height / 2
    )
    finish_merge_y = positions["__FINISH__"][1]
    finish_port_x = node_bounds["__FINISH__"][0]

    if finish_lane_acts:
        rightmost_finish_node = max(node_bounds[act][1] for act in finish_lane_acts)
        left_bound = rightmost_finish_node + 0.20
        right_bound = finish_port_x - 0.24

        if right_bound > left_bound:
            finish_merge_x = left_bound + ((right_bound - left_bound) * 0.78)
        else:
            finish_merge_x = max(rightmost_finish_node + 0.12, finish_port_x - 0.16)

        finish_merge_x = min(
            max(finish_merge_x, rightmost_finish_node + 0.12),
            finish_port_x - 0.08
        )
    else:
        finish_merge_x = finish_port_x - 0.16

    def edge_points(start_act, end_act, relation):
        x1, y1 = positions[start_act]
        x2, y2 = positions[end_act]

        def incoming_port_y():
            lane_order = incoming_lane_order.get(end_act, {})
            lane_count = incoming_lane_count.get(end_act, 1)
            if start_act not in lane_order or lane_count <= 1:
                return y2

            if end_act == "__FINISH__":
                max_offset = min(endpoint_height * 0.28, 0.24)
            else:
                max_offset = min(node_height * 0.32, 0.30)

            offsets = np.linspace(max_offset, -max_offset, lane_count)
            return y2 + float(offsets[lane_order[start_act]])

        pred_start = (x1 - node_width / 2, y1)
        pred_finish = (x1 + node_width / 2, y1)
        succ_port_y = incoming_port_y()
        succ_start = (x2 - node_width / 2, succ_port_y)
        succ_finish = (x2 + node_width / 2, succ_port_y)

        if start_act == "__START__":
            pred_start = (x1 + endpoint_width / 2, y1)
            pred_finish = (x1 + endpoint_width / 2, y1)
        if end_act == "__FINISH__":
            succ_start = (x2 - endpoint_width / 2, succ_port_y)
            succ_finish = (x2 - endpoint_width / 2, succ_port_y)

        if relation == "FS":
            return pred_finish, succ_start
        if relation == "SS":
            return pred_start, succ_start
        if relation == "FF":
            return pred_finish, succ_finish
        if relation == "SF":
            return pred_start, succ_finish
        return pred_finish, succ_start

    def relation_label(rel, lag):
        if np.isclose(lag, 0.0):
            return rel
        sign = "+" if lag >= 0 else ""
        return f"{rel}{sign}{lag:g}"

    def compute_bend_x(start_act, end_act, start_point, end_point):
        x1, _ = start_point
        x2, _ = end_point
        bend_x = (x1 + x2) / 2

        lane_count = incoming_lane_count.get(end_act, 1)
        lane_order = incoming_lane_order.get(end_act, {})

        if lane_count > 1 and start_act in lane_order:
            if end_act == "__FINISH__":
                right_bound = x2 - 0.05
                left_bound = max(
                    max(node_bounds[act][1] for act in finish_lane_acts) + 0.18,
                    x1 + 0.22
                )
                if left_bound > right_bound:
                    left_bound = min(left_bound, x2 - 0.12)
                lane_t = lane_order[start_act] / max(lane_count - 1, 1)
                bend_x = left_bound + (right_bound - left_bound) * lane_t
                bend_x = min(max(bend_x, left_bound), right_bound)
                return bend_x

            node_entry_clearance = 0.34 + min(0.18 * max(lane_count - 1, 0), 0.24)

            if x2 >= x1:
                left_bound = x1 + max(0.28, min(0.60, abs(x2 - x1) * 0.18))
                right_bound = x2 - node_entry_clearance
                if right_bound <= left_bound:
                    right_bound = x2 - 0.22
                if lane_count == 2:
                    lane_t = float(lane_order[start_act])
                else:
                    lane_t = lane_order[start_act] / max(lane_count - 1, 1)
                bend_x = left_bound + (right_bound - left_bound) * lane_t
                bend_x = min(max(bend_x, left_bound), right_bound)
            else:
                left_bound = x2 + node_entry_clearance
                right_bound = x1 - max(0.28, min(0.60, abs(x2 - x1) * 0.18))
                if left_bound >= right_bound:
                    left_bound = x2 + 0.22
                if lane_count == 2:
                    lane_t = float(lane_order[start_act])
                else:
                    lane_t = lane_order[start_act] / max(lane_count - 1, 1)
                bend_x = left_bound + (right_bound - left_bound) * lane_t
                bend_x = min(max(bend_x, left_bound), right_bound)

        return bend_x

    def draw_orthogonal_arrow(start_point, end_point, color, width, zorder, bend_x=None):
        x1, y1 = start_point
        x2, y2 = end_point
        mid_x = bend_x if bend_x is not None else (x1 + x2) / 2

        ax.plot([x1, mid_x], [y1, y1], color=color, lw=width, zorder=zorder)
        ax.plot([mid_x, mid_x], [y1, y2], color=color, lw=width, zorder=zorder)
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(mid_x, y2),
            arrowprops=dict(
                arrowstyle="->",
                color=color,
                lw=width,
                shrinkA=0,
                shrinkB=0
            ),
            zorder=zorder
        )
        return mid_x, y1, y2

    def draw_finish_merge_edges(acts, color, width, zorder):
        if not acts:
            return

        branch_points = [
            (act, node_bounds[act][1], positions[act][1])
            for act in acts
        ]

        if len(branch_points) > 1:
            y_values = [y for _, _, y in branch_points]
            y_values.append(finish_merge_y)
            ax.plot(
                [finish_merge_x, finish_merge_x],
                [min(y_values), max(y_values)],
                color=color,
                lw=width,
                zorder=zorder
            )

        for act, start_x, start_y in branch_points:
            ax.plot(
                [start_x, finish_merge_x],
                [start_y, start_y],
                color=color,
                lw=width,
                zorder=zorder
            )

        ax.annotate(
            "",
            xy=(finish_port_x, finish_merge_y),
            xytext=(finish_merge_x, finish_merge_y),
            arrowprops=dict(
                arrowstyle="->",
                color=color,
                lw=width,
                shrinkA=0,
                shrinkB=0
            ),
            zorder=zorder
        )

    def estimate_relation_label_half_width(label_text):
        return 0.12 + (0.05 * len(str(label_text)))

    def place_relation_label(base_x, base_y, label_text):
        label_x = base_x
        label_y = base_y
        label_half_width = estimate_relation_label_half_width(label_text)
        x_pad = 0.18 + label_half_width
        y_pad = 0.22
        x_gap = 0.20 + (0.02 * len(str(label_text)))
        y_gap = 0.32

        for _ in range(12):
            moved = False
            for x_min, x_max, y_min, y_max in node_bounds.values():
                left_bound = x_min - x_pad
                right_bound = x_max + x_pad
                bottom_bound = y_min - y_pad
                top_bound = y_max + y_pad

                if left_bound <= label_x <= right_bound and bottom_bound <= label_y <= top_bound:
                    distances = {
                        "left": abs(label_x - left_bound),
                        "right": abs(right_bound - label_x),
                        "bottom": abs(label_y - bottom_bound),
                        "top": abs(top_bound - label_y)
                    }
                    nearest_side = min(distances, key=distances.get)

                    if nearest_side == "left":
                        label_x = left_bound - x_gap
                    elif nearest_side == "right":
                        label_x = right_bound + x_gap
                    elif nearest_side == "bottom":
                        label_y = bottom_bound - y_gap
                    else:
                        label_y = top_bound + y_gap

                    moved = True
                    break
            if not moved:
                break

        return label_x, label_y

    def fine_tune_relation_label(label_x, label_y, label_text, start_act, end_act):
        if (start_act, end_act, label_text) == ("I", "J", "FS+2"):
            label_half_width = estimate_relation_label_half_width(label_text)
            pred_right = node_bounds[start_act][1]
            succ_left = node_bounds[end_act][0]
            min_center_x = pred_right + label_half_width + 0.10
            max_center_x = succ_left - label_half_width - 0.08
            label_x = max(min_center_x, min(label_x - 0.30, max_center_x))

        if (start_act, end_act, label_text) == ("J", "__FINISH__", "FS"):
            label_half_width = estimate_relation_label_half_width(label_text)
            pred_right = node_bounds[start_act][1]
            succ_left = node_bounds[end_act][0]
            min_center_x = pred_right + label_half_width + 0.12
            max_center_x = succ_left - label_half_width - 0.08
            label_x = min(max_center_x, max(label_x + 0.18, min_center_x))

        return label_x, label_y

    path_legend = []
    node_color_map = {}
    node_fill_map = {}
    edge_color_map = {}
    top_paths = df_path.head(max_paths).reset_index(drop=True)

    for idx, row in top_paths.iterrows():
        color = highlight_colors[idx % len(highlight_colors)]
        fill_color = highlight_fill_colors[idx % len(highlight_fill_colors)]
        activities = [act.strip() for act in str(row["Path"]).split("->")]

        for start_act, end_act in zip(activities, activities[1:]):
            edge_color_map.setdefault((start_act, end_act), color)

        for act in activities:
            node_color_map.setdefault(act, color)
            node_fill_map.setdefault(act, fill_color)

        path_legend.append(
            Line2D(
                [0], [0],
                color=color,
                lw=3,
                label=f"Critical Path {idx + 1} (P={row['Prob']:.3f})"
            )
        )

    for act in start_acts:
        edge_color_map.setdefault(("__START__", act), base_edge_color)
    for act in finish_acts:
        edge_color_map.setdefault((act, "__FINISH__"), base_edge_color)

    for _, row in df_relations.iterrows():
        end_act = row["Aktivitas"]
        start_act = row["Predecessor"]
        relation = row["Relasi"]
        lag = row["Lag"]

        is_highlighted = (start_act, end_act) in {
            (a, b) for a, b in edge_color_map if a != "__START__" and b != "__FINISH__"
        }
        color = edge_color_map.get((start_act, end_act), base_edge_color)
        width = 3.0 if is_highlighted else 1.4
        zorder = 3 if is_highlighted else 1

        start_point, end_point = edge_points(start_act, end_act, relation)
        bend_x = compute_bend_x(start_act, end_act, start_point, end_point)
        mid_x, y1, y2 = draw_orthogonal_arrow(
            start_point,
            end_point,
            color,
            width,
            zorder,
            bend_x=bend_x
        )
        rel_text = relation_label(relation, lag)
        base_label_y = ((y1 + y2) / 2) + (0.18 if relation in {"SS", "FF"} else -0.18)
        label_x, label_y = place_relation_label(mid_x, base_label_y, rel_text)
        label_x, label_y = fine_tune_relation_label(label_x, label_y, rel_text, start_act, end_act)

        draw_text(
            label_x,
            label_y,
            rel_text,
            ha="center",
            va="center",
            fontsize=8.9,
            fontweight="semibold",
            color=color if is_highlighted else "#555555",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.92),
            zorder=6
        )

    for act in start_acts:
        start_point, end_point = edge_points("__START__", act, "FS")
        draw_orthogonal_arrow(start_point, end_point, "#BDBDBD", 1.2, 0.8)

    draw_finish_merge_edges(finish_lane_acts, "#BDBDBD", 1.2, 0.8)

    def draw_activity_node(act, x, y, edge_color, fill_color):
        patch = FancyBboxPatch(
            (x - node_width / 2, y - node_height / 2),
            node_width,
            node_height,
            boxstyle="round,pad=0.04,rounding_size=0.10",
            linewidth=2.1,
            edgecolor=edge_color,
            facecolor=fill_color,
            zorder=4,
            clip_on=False
        )
        ax.add_patch(patch)

        top_divider_y = y + node_height * 0.08
        bottom_divider_y = y - node_height * 0.14
        top_y = y + node_height * 0.27
        middle_y = y - node_height * 0.03
        duration_x = x - node_width * 0.46
        duration_y = middle_y
        bottom_y = y - node_height * 0.29
        tf_y = y - node_height * 0.43
        node_text_box = dict(
            boxstyle="round,pad=0.05",
            facecolor=fill_color,
            edgecolor="none",
            alpha=1.0
        )
        duration_text_box = dict(
            boxstyle="round,pad=0.02",
            facecolor=fill_color,
            edgecolor="none",
            alpha=1.0
        )

        ax.plot(
            [x - node_width / 2, x + node_width / 2],
            [top_divider_y, top_divider_y],
            color=edge_color,
            lw=1.0,
            zorder=5
        )
        ax.plot(
            [x - node_width / 2, x + node_width / 2],
            [bottom_divider_y, bottom_divider_y],
            color=edge_color,
            lw=1.0,
            zorder=5
        )
        ax.plot([x, x], [top_divider_y, y + node_height / 2], color=edge_color, lw=1.0, zorder=5)
        ax.plot([x, x], [y - node_height / 2, bottom_divider_y], color=edge_color, lw=1.0, zorder=5)

        metrics = metrics_lookup[act]
        draw_text(
            x - node_width * 0.24,
            top_y,
            f"{metrics['ES']:.1f}",
            ha="center",
            va="center",
            fontsize=9.0,
            fontweight="semibold",
            color="#111111",
            bbox=node_text_box,
            zorder=8
        )
        draw_text(
            x + node_width * 0.24,
            top_y,
            f"{metrics['EF']:.1f}",
            ha="center",
            va="center",
            fontsize=9.0,
            fontweight="semibold",
            color="#111111",
            bbox=node_text_box,
            zorder=8
        )
        draw_text(
            x,
            middle_y,
            act,
            ha="center",
            va="center",
            fontsize=11.6,
            weight="bold",
            color="#111111",
            bbox=node_text_box,
            zorder=8
        )
        draw_text(
            duration_x,
            duration_y,
            f"d={metrics['Durasi']:.1f}",
            ha="left",
            va="center",
            fontsize=8.4,
            fontweight="semibold",
            color="#111111",
            bbox=duration_text_box,
            zorder=8
        )
        draw_text(
            x - node_width * 0.24,
            bottom_y,
            f"{metrics['LS']:.1f}",
            ha="center",
            va="center",
            fontsize=9.0,
            fontweight="semibold",
            color="#111111",
            bbox=node_text_box,
            zorder=8
        )
        draw_text(
            x + node_width * 0.24,
            bottom_y,
            f"{metrics['LF']:.1f}",
            ha="center",
            va="center",
            fontsize=9.0,
            fontweight="semibold",
            color="#111111",
            bbox=node_text_box,
            zorder=8
        )
        draw_text(
            x,
            tf_y,
            f"TF={metrics['TF']:.1f}",
            ha="center",
            va="center",
            fontsize=8.2,
            fontweight="semibold",
            color="#444444",
            bbox=node_text_box,
            zorder=8
        )

    def draw_endpoint_node(label, x, y, metrics):
        patch = FancyBboxPatch(
            (x - endpoint_width / 2, y - endpoint_height / 2),
            endpoint_width,
            endpoint_height,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            linewidth=1.8,
            edgecolor="#444444",
            facecolor="#F2F2F2",
            zorder=4,
            clip_on=False
        )
        ax.add_patch(patch)

        top_divider_y = y + endpoint_height * 0.13
        bottom_divider_y = y - endpoint_height * 0.13
        top_y = y + endpoint_height * 0.30
        middle_y = y
        bottom_y = y - endpoint_height * 0.30
        half_w = endpoint_width / 2

        ax.plot(
            [x - half_w, x + half_w],
            [top_divider_y, top_divider_y],
            color="#444444",
            lw=1.0,
            zorder=5
        )
        ax.plot(
            [x - half_w, x + half_w],
            [bottom_divider_y, bottom_divider_y],
            color="#444444",
            lw=1.0,
            zorder=5
        )
        ax.plot([x, x], [top_divider_y, y + endpoint_height / 2], color="#444444", lw=1.0, zorder=5)
        ax.plot([x, x], [y - endpoint_height / 2, bottom_divider_y], color="#444444", lw=1.0, zorder=5)

        endpoint_text_box = dict(
            boxstyle="round,pad=0.04",
            facecolor="#F2F2F2",
            edgecolor="none",
            alpha=1.0
        )
        draw_text(
            x - endpoint_width * 0.24,
            top_y,
            f"{metrics['ES']:.1f}",
            ha="center",
            va="center",
            fontsize=8.6,
            fontweight="semibold",
            color="#111111",
            bbox=endpoint_text_box,
            zorder=7
        )
        draw_text(
            x + endpoint_width * 0.24,
            top_y,
            f"{metrics['EF']:.1f}",
            ha="center",
            va="center",
            fontsize=8.6,
            fontweight="semibold",
            color="#111111",
            bbox=endpoint_text_box,
            zorder=7
        )
        draw_text(
            x,
            middle_y,
            label,
            ha="center",
            va="center",
            fontsize=11.0,
            weight="bold",
            color="#111111",
            bbox=endpoint_text_box,
            zorder=7
        )
        draw_text(
            x - endpoint_width * 0.24,
            bottom_y,
            f"{metrics['LS']:.1f}",
            ha="center",
            va="center",
            fontsize=8.6,
            fontweight="semibold",
            color="#111111",
            bbox=endpoint_text_box,
            zorder=7
        )
        draw_text(
            x + endpoint_width * 0.24,
            bottom_y,
            f"{metrics['LF']:.1f}",
            ha="center",
            va="center",
            fontsize=8.6,
            fontweight="semibold",
            color="#111111",
            bbox=endpoint_text_box,
            zorder=7
        )

    for act in df_activities["Aktivitas"]:
        x, y = positions[act]
        draw_activity_node(
            act,
            x,
            y,
            node_color_map.get(act, base_node_edge),
            node_fill_map.get(act, base_node_fill)
        )

    draw_endpoint_node("Start", *positions["__START__"], endpoint_metrics["__START__"])
    draw_endpoint_node("Finish", *positions["__FINISH__"], endpoint_metrics["__FINISH__"])

    all_x = [pos[0] for pos in positions.values()]
    all_y = [pos[1] for pos in positions.values()]
    title = ax.set_title("Probabilistic PDM Network Diagram", fontsize=15, fontweight="bold", pad=10)
    title.set_path_effects([
        pe.withStroke(linewidth=1.2, foreground="white"),
        pe.Normal()
    ])
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    x_margin = max(0.65, (x_max - x_min) * 0.04)
    y_margin = max(0.55, (y_max - y_min) * 0.08)
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    ax.axis("off")

    if path_legend:
        legend = ax.legend(
            handles=path_legend,
            title="Critical Paths",
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            prop={"size": 10.5, "weight": "semibold"},
            title_fontsize=11.5
        )
        legend.get_title().set_fontweight("bold")

    fig.tight_layout(pad=0.15)
    return fig


def build_zoomable_network_figure(matplotlib_fig):
    if not PLOTLY_AVAILABLE:
        return None

    buffer = BytesIO()
    matplotlib_fig.savefig(
        buffer,
        format="png",
        dpi=320,
        bbox_inches="tight",
        pad_inches=0.03,
        facecolor="white"
    )
    buffer.seek(0)

    if PIL_AVAILABLE:
        image = Image.open(buffer).convert("RGBA")
        white_bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(white_bg, image).convert("RGB")
        diff = ImageChops.difference(image, Image.new("RGB", image.size, "white"))
        bbox = diff.getbbox()

        if bbox is not None:
            x_pad = 10
            bbox_height = max(bbox[3] - bbox[1], 1)
            y_pad = max(18, int(round(bbox_height * 0.08)))
            left = max(bbox[0] - x_pad, 0)
            top = max(bbox[1] - y_pad, 0)
            right = min(bbox[2] + x_pad, image.size[0])
            bottom = min(bbox[3] + y_pad, image.size[1])
            image = image.crop((left, top, right, bottom))

        image_array = np.array(image)
    else:
        image_array = plt.imread(buffer)

    if image_array.dtype != np.uint8:
        image_array = (np.clip(image_array, 0, 1) * 255).astype(np.uint8)

    height, width = image_array.shape[:2]
    vertical_view_padding = max(int(round(height * 0.08)), 60)
    effective_height = height + (2 * vertical_view_padding)
    max_display_width = 1100

    if width > max_display_width:
        display_width = max_display_width
        display_height = max(int(round(effective_height * (display_width / width))), 1)
    else:
        display_width = width
        display_height = effective_height

    fig = go.Figure(go.Image(z=image_array))
    fig.update_traces(hoverinfo="skip")
    fig.update_xaxes(
        visible=False,
        showgrid=False,
        zeroline=False,
        range=[-0.5, width - 0.5]
    )
    fig.update_yaxes(
        visible=False,
        showgrid=False,
        zeroline=False,
        scaleanchor="x",
        scaleratio=1,
        range=[height - 0.5 + vertical_view_padding, -0.5 - vertical_view_padding]
    )
    fig.update_layout(
        width=display_width,
        height=display_height,
        margin=dict(l=0, r=0, t=0, b=0),
        dragmode="pan",
        plot_bgcolor="white",
        paper_bgcolor="white",
        uirevision="network-diagram"
    )
    return fig

# =============================
# VALIDASI DEPENDENCY
# =============================
def validate(df):
    df_activities = build_activity_table(df)
    df_relations = build_relation_table(df)
    acts = set(df_activities['Aktivitas'])
    errors = []

    for _, row in df_relations.iterrows():
        if row["Predecessor"] not in acts:
            errors.append(f"{row['Predecessor']} tidak ditemukan")

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
    df_activities = build_activity_table(df)
    df_relations = build_relation_table(df)
    activities = df_activities["Aktivitas"].tolist()
    ES = {act: 0.0 for act in activities}
    EF = {act: float(durasi[act]) for act in activities}
    incoming_map = {act: [] for act in activities}

    for _, row in df_relations.iterrows():
        incoming_map.setdefault(row["Aktivitas"], []).append({
            "pred": row["Predecessor"],
            "rel": row["Relasi"],
            "lag": row["Lag"]
        })

    changed = True
    iter_count = 0
    max_iter = max(len(activities) * 5, 1)

    while changed:
        changed = False
        iter_count += 1

        if iter_count > max_iter:
            raise RuntimeError("Dependency proyek mengandung siklus atau relasi tidak valid.")

        for act in activities:
            constraints = incoming_map.get(act, [])
            if not constraints:
                continue

            candidate_es = []
            for constraint in constraints:
                pred = constraint["pred"]
                rel = constraint["rel"]
                lag = constraint["lag"]

                if rel == 'FS':
                    candidate_es.append(EF[pred] + lag)
                elif rel == 'SS':
                    candidate_es.append(ES[pred] + lag)
                elif rel == 'FF':
                    candidate_es.append(EF[pred] + lag - durasi[act])
                elif rel == 'SF':
                    candidate_es.append(ES[pred] + lag - durasi[act])

            if not candidate_es:
                continue

            new_ES = max(candidate_es)

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
        crit_preds = []

        for constraint in incoming_map.get(act, []):
            pred = constraint["pred"]
            rel = constraint["rel"]
            lag = constraint["lag"]

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

        return list(dict.fromkeys(crit_preds))

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

    for _, row in build_activity_table(df).iterrows():
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


def build_ci_percentile_comparison_table(
    df_ci_compare,
    results,
    critical_flags,
    percentiles=None,
    tol=1e-9
):
    percentiles = percentiles or list(range(0, 101, 10))
    base_columns = ["Aktivitas", "Deterministik"]
    if "Probabilistik (Global)" in df_ci_compare.columns:
        base_columns.append("Probabilistik (Global)")

    df_percentile = df_ci_compare[base_columns].copy()
    results_array = np.asarray(results, dtype=float)

    if results_array.size == 0 or not critical_flags:
        for percentile in percentiles:
            df_percentile[f"CI(<=P{percentile})"] = np.nan
        return df_percentile

    for percentile in percentiles:
        threshold = float(np.percentile(results_array, percentile))
        mask = results_array <= (threshold + tol)
        column_name = f"CI(<=P{percentile})"

        if not np.any(mask):
            df_percentile[column_name] = 0.0
            continue

        df_percentile[column_name] = [
            float(
                np.mean(
                    np.asarray(
                        critical_flags.get(act, np.zeros_like(results_array)),
                        dtype=float
                    )[mask]
                )
            )
            for act in df_percentile["Aktivitas"]
        ]

    return df_percentile


def build_critical_count_percentile_table(
    critical_count,
    results,
    critical_flags,
    percentiles=None,
    tol=1e-9
):
    percentiles = percentiles or list(range(0, 101, 10))
    results_array = np.asarray(results, dtype=float)
    activities = list(critical_count.keys())
    total_simulations = results_array.size

    table_data = {
        "Aktivitas": activities,
        "Critical Count (Global)": [
            int(critical_count.get(act, 0))
            for act in activities
        ],
        "CI (Global)": [
            float(critical_count.get(act, 0)) / total_simulations
            if total_simulations > 0 else np.nan
            for act in activities
        ]
    }

    if total_simulations == 0 or not critical_flags:
        for percentile in percentiles:
            table_data[f"Critical Count (<=P{percentile})"] = np.nan
            table_data[f"CI (<=P{percentile})"] = np.nan
        return pd.DataFrame(table_data)

    for percentile in percentiles:
        threshold = float(np.percentile(results_array, percentile))
        mask = results_array <= (threshold + tol)
        subset_size = int(np.sum(mask))
        count_column_name = f"Critical Count (<=P{percentile})"
        ci_column_name = f"CI (<=P{percentile})"

        if subset_size <= 0:
            table_data[count_column_name] = [0] * len(activities)
            table_data[ci_column_name] = [0.0] * len(activities)
            continue

        counts = [
            int(
                np.sum(
                    np.asarray(
                        critical_flags.get(act, np.zeros_like(results_array)),
                        dtype=float
                    )[mask]
                )
            )
            for act in activities
        ]
        table_data[count_column_name] = counts
        table_data[ci_column_name] = [
            float(count_value) / subset_size
            for count_value in counts
        ]

    return pd.DataFrame(table_data)

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
use_auto_seed = st.sidebar.checkbox(
    "Acak seed setiap klik simulasi",
    value=True
)

configured_seed = None
if use_auto_seed:
    st.sidebar.caption(
        "Setiap klik tombol 'Jalankan Simulasi' akan memakai seed acak baru."
    )
else:
    configured_seed = int(
        st.sidebar.number_input(
            "Random Seed",
            min_value=0,
            value=42,
            step=1
        )
    )

dist_param = fit_distribution_params(df_prod, distribution_name)

st.subheader("Histogram Produktivitas per Jenis Pekerjaan")
selected_activity = st.selectbox(
    "Pilih aktivitas untuk histogram produktivitas",
    sorted(df_prod["Aktivitas"].unique())
)
productivity_hist_fig = plot_productivity_histogram(selected_activity, distribution_name, dist_param)
st.pyplot(productivity_hist_fig)
render_jpg_download_button(
    productivity_hist_fig,
    f"histogram_produktivitas_{selected_activity}",
    key=f"download_histogram_produktivitas_{make_safe_filename(selected_activity)}"
)
plt.close(productivity_hist_fig)

# =============================
# KOEFISIEN PRODUKTIVITAS
# =============================
st.subheader("Koefisien Produktivitas per Jenis Pekerjaan")
st.caption(
    "Asumsi koefisien data = (Tenaga x Waktu) / Output, sehingga satuannya mengikuti input waktu per satuan output."
)

df_coef = build_productivity_coefficient_table(df_prod)
render_table(
    df_coef,
    formats={
        "Mean_p": "{:.3f}",
        "Std_p": "{:.3f}",
        "Mean_Koef_Data": "{:.3f}",
        "Std_Koef_Data": "{:.3f}",
        "P50_Koef_Data": "{:.3f}",
        "Min_Koef_Data": "{:.3f}",
        "Max_Koef_Data": "{:.3f}",
        "Koef_Setara_1_per_Mean_p": "{:.3f}"
    }
)

df_ahsp_ref = None
if "Referensi_AHSP" in excel_book.sheet_names:
    raw_ahsp_ref = pd.read_excel(excel_book, sheet_name="Referensi_AHSP")
    df_ahsp_ref = standardize_ahsp_reference(raw_ahsp_ref)

st.subheader("Perbandingan Koefisien Produktivitas per Jenis Pekerjaan")
if df_ahsp_ref is not None and not df_ahsp_ref.empty:
    df_coef_cmp = build_ahsp_comparison(df_coef, df_ahsp_ref)
    render_table(
        df_coef_cmp,
        formats={
            "Mean_p": "{:.3f}",
            "Mean_Koef_Data": "{:.3f}",
            "Koef SNI(AHSP)": "{:.3f}",
            "Selisih": "{:.3f}",
            "Selisih_%": "{:.3f}",
            "Rasio_Data_vs_SNI(AHSP)": "{:.3f}"
        }
    )
    coefficient_comparison_fig = plot_coefficient_comparison(df_coef_cmp)
    st.pyplot(coefficient_comparison_fig)
    render_jpg_download_button(
        coefficient_comparison_fig,
        "perbandingan_koefisien_produktivitas",
        key="download_perbandingan_koefisien"
    )
    plt.close(coefficient_comparison_fig)
else:
    st.info(
        "Referensi SNI(AHSP) belum dibaca. Tambahkan sheet 'Referensi_AHSP' dengan kolom: "
        "'Aktivitas', 'Kode SNI(AHSP)', dan 'Koef SNI(AHSP)' agar tabel pembanding muncul otomatis."
    )

# =============================
# RUN
# =============================
SIMULATION_STATE_KEY = "smc_simulation_results"
run_simulation = st.button("Jalankan Simulasi")
stored_simulation = st.session_state.get(SIMULATION_STATE_KEY)
stored_simulation_matches_file = (
    stored_simulation is not None and
    stored_simulation.get("file_signature") == current_file_signature
)

if run_simulation or stored_simulation_matches_file:
    if run_simulation:
        df_proj_unique = build_activity_table(df_proj)
        unique_activities = df_proj_unique["Aktivitas"].tolist()

        results = []
        activity_duration_sum = {act: 0.0 for act in unique_activities}
        critical_count = {act: 0 for act in unique_activities}
        critical_flags = {act: [] for act in unique_activities}
        path_count = {}
        path_duration_sum = {}

        progress = st.progress(0)

        start = time.time()

        deterministic_durations = build_durations_from_productivity(df_proj, mean_p_map)
        deterministic_schedule = calculate_schedule_metrics(df_proj, deterministic_durations)
        deterministic_total, deterministic_paths = pdm_cp(df_proj, deterministic_durations)
        deterministic_acts = {
            act for path in deterministic_paths for act in path
        }
        simulation_seed = (
            int(np.random.SeedSequence().generate_state(1)[0])
            if use_auto_seed
            else configured_seed
        )
        rng = np.random.default_rng(simulation_seed)

        for i in range(n_sim):

            durasi = {}

            for _, row in df_proj_unique.iterrows():
                act = row['Aktivitas']
                Q = row['Volume']
                n = row['Tenaga']
                p = sample_productivity(act, dist_param, distribution_name, rng)
                durasi[act] = Q / (n * p)

            total, critical_paths = pdm_cp(df_proj, durasi)

            for act, duration_value in durasi.items():
                activity_duration_sum[act] += duration_value

            critical_acts = {act for path in critical_paths for act in path}
            for act in critical_acts:
                critical_count[act] += 1
            for act in unique_activities:
                critical_flags[act].append(1 if act in critical_acts else 0)

            for path in critical_paths:
                path_key = " -> ".join(path)
                path_count[path_key] = path_count.get(path_key, 0) + 1
                path_duration_sum[path_key] = path_duration_sum.get(path_key, 0.0) + total

            results.append(total)

            progress.progress((i+1)/n_sim)

        results = np.array(results)
        mean_duration = np.mean(results)
        std_duration = np.std(results)
        probabilistic_activity_durations = {
            act: activity_duration_sum[act] / n_sim
            for act in unique_activities
        }
        simulation_n_sim = n_sim
        simulation_seed_was_auto = use_auto_seed

        st.session_state[SIMULATION_STATE_KEY] = {
            "file_signature": current_file_signature,
            "results": results.copy(),
            "mean_duration": float(mean_duration),
            "std_duration": float(std_duration),
            "probabilistic_activity_durations": dict(probabilistic_activity_durations),
            "critical_count": dict(critical_count),
            "critical_flags": {act: list(flags) for act, flags in critical_flags.items()},
            "path_count": dict(path_count),
            "path_duration_sum": dict(path_duration_sum),
            "deterministic_total": float(deterministic_total),
            "deterministic_paths": [list(path) for path in deterministic_paths],
            "deterministic_acts": sorted(deterministic_acts),
            "deterministic_schedule": deterministic_schedule,
            "simulation_seed": int(simulation_seed),
            "simulation_seed_was_auto": bool(simulation_seed_was_auto),
            "simulation_n_sim": int(simulation_n_sim)
        }
    else:
        df_proj_unique = build_activity_table(df_proj)
        unique_activities = df_proj_unique["Aktivitas"].tolist()
        simulation_payload = stored_simulation
        results = np.array(simulation_payload["results"], dtype=float)
        mean_duration = float(simulation_payload["mean_duration"])
        std_duration = float(simulation_payload["std_duration"])
        probabilistic_activity_durations = dict(simulation_payload["probabilistic_activity_durations"])
        critical_count = dict(simulation_payload["critical_count"])
        critical_flags = simulation_payload.get("critical_flags")
        path_count = dict(simulation_payload["path_count"])
        path_duration_sum = dict(simulation_payload["path_duration_sum"])
        deterministic_total = float(simulation_payload["deterministic_total"])
        deterministic_paths = [list(path) for path in simulation_payload["deterministic_paths"]]
        deterministic_acts = set(simulation_payload["deterministic_acts"])
        deterministic_schedule = simulation_payload["deterministic_schedule"]
        simulation_seed = int(simulation_payload["simulation_seed"])
        simulation_seed_was_auto = bool(simulation_payload["simulation_seed_was_auto"])
        simulation_n_sim = int(simulation_payload["simulation_n_sim"])

    if run_simulation:
        st.success("Simulasi selesai")
    else:
        st.caption("Menampilkan hasil simulasi terakhir untuk dataset ini.")

    if simulation_seed_was_auto:
        st.caption(f"Seed simulasi yang digunakan: {simulation_seed} (acak otomatis)")
    else:
        st.caption(f"Seed simulasi yang digunakan: {simulation_seed}")

    # =============================
    # SIMULASI MONTE CARLO
    # =============================
    st.subheader("Hasil Simulasi Durasi Pekerjaan Metode Monte Carlo")
    df_results = pd.DataFrame({"Durasi": results})
    with st.expander("Lihat Data Hasil Simulasi Monte Carlo"):
        st.write("Tabel Durasi Hasil Simulasi:")
        render_table(
            df_results,
            formats={"Durasi": "{:.3f}"},
            show_index=True,
            start_index_at_one=True
        )

        df_critical_count_table = build_critical_count_percentile_table(
            critical_count,
            results,
            critical_flags
        ).sort_values(
            by=["Critical Count (Global)", "Aktivitas"],
            ascending=[False, True]
        ).reset_index(drop=True)

        st.write("Tabel Rekap Critical Count per Aktivitas:")
        st.caption(
            "Kolom 'Critical Count (<=Pk)' menunjukkan jumlah simulasi saat aktivitas "
            "menjadi kritis pada subset durasi proyek kurang dari atau sama dengan "
            "batas persentil ke-k. Kolom 'CI (<=Pk)' adalah proporsinya di dalam "
            "subset persentil tersebut."
        )
        critical_count_formats = {
            "Critical Count (Global)": "{:.0f}",
            "CI (Global)": "{:.3f}"
        }
        critical_count_formats.update({
            f"Critical Count (<=P{percentile})": "{:.0f}"
            for percentile in range(0, 101, 10)
        })
        critical_count_formats.update({
            f"CI (<=P{percentile})": "{:.3f}"
            for percentile in range(0, 101, 10)
        })
        render_table(
            df_critical_count_table,
            formats=critical_count_formats,
            show_index=True,
            start_index_at_one=True
        )

    st.write("Resume Hasil Simulasi Monte Carlo")
    df_resume_mc = pd.DataFrame({
        "Statistik": ["Xmin", "Xmax", "Xrata-rata"],
        "Nilai": [
            np.min(results),
            np.max(results),
            mean_duration
        ]
    })
    render_table(df_resume_mc, formats={"Nilai": "{:.3f}"})

    # =============================
    # HISTOGRAM
    # =============================
    st.subheader("Histogram Probabilistik Durasi per Jenis Pekerjaan")
    fig, ax = plt.subplots()
    hist_freq, hist_bins, _ = ax.hist(results, bins=30, edgecolor="black", linewidth=1)
    ax.axvline(
        mean_duration,
        color="red",
        linewidth=2,
        label=f"Mean Duration = {mean_duration:.3f}"
    )
    ax.axvline(
        mean_duration - std_duration,
        color="red",
        linestyle="--",
        linewidth=1.8,
        label=f"Mean - 1 SD = {mean_duration - std_duration:.3f}"
    )
    ax.axvline(
        mean_duration + std_duration,
        color="red",
        linestyle="--",
        linewidth=1.8,
        label=f"Mean + 1 SD = {mean_duration + std_duration:.3f}"
    )
    ax.set_title("Probabilistic Project Duration Histogram")
    ax.set_xlabel("Project Duration")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)
    render_jpg_download_button(
        fig,
        "histogram_probabilistik_durasi_proyek",
        key="download_histogram_probabilistik_durasi"
    )

    df_hist_freq = pd.DataFrame({
        "Bin Ke": np.arange(1, len(hist_freq) + 1),
        "Batas Bawah": hist_bins[:-1],
        "Batas Atas": hist_bins[1:],
        "Lebar Bin": hist_bins[1:] - hist_bins[:-1],
        "Frekuensi": np.rint(hist_freq).astype(int)
    })
    st.write(f"Tabel Frekuensi Histogram Durasi Proyek (Jumlah Bin: {len(hist_freq)})")
    render_table(
        df_hist_freq,
        formats={
            "Batas Bawah": "{:.3f}",
            "Batas Atas": "{:.3f}",
            "Lebar Bin": "{:.3f}"
        },
        show_index=False
    )
    plt.close(fig)

    # =============================
    # STATISTIK DURASI PROBABILISTIK
    # =============================
    st.subheader("Analisis Probabilistik Durasi per Jenis Pekerjaan")
    percentile_points = list(range(0, 101, 10))
    df_stats = pd.DataFrame({
        "Statistik": ["Mean", "Std"] + [f"P{p}" for p in percentile_points],
        "Nilai": [mean_duration, std_duration] + [
            np.percentile(results, p) for p in percentile_points
        ]
    })
    render_table(df_stats, formats={"Nilai": "{:.3f}"})

    # =============================
    # PROBABILISTIC CRITICAL PATH
    # =============================
    st.subheader("Analisis Probabilistic Lintasan Kritis")

    df_path = pd.DataFrame({
        "Path": list(path_count.keys()),
        "Rata-rata Durasi Proyek Saat Path Kritis": [
            path_duration_sum[path] / path_count[path]
            for path in path_count.keys()
        ],
        "Prob": [v/simulation_n_sim for v in path_count.values()]
    }).sort_values(by="Prob", ascending=False)

    st.caption(
        "Kolom rata-rata durasi proyek dihitung dari rerata total durasi proyek "
        "pada simulasi-simulasi ketika path tersebut menjadi lintasan kritis."
    )
    render_table(
        df_path,
        formats={
            "Rata-rata Durasi Proyek Saat Path Kritis": "{:.3f}",
            "Prob": "{:.3f}"
        }
    )

    df_cp = pd.DataFrame({
        "Aktivitas": list(critical_count.keys()),
        "Prob": [v/simulation_n_sim for v in critical_count.values()]
    }).sort_values(by="Prob", ascending=False)

    st.subheader("Criticality Index (CI)")
    render_table(df_cp, formats={"Prob": "{:.3f}"})

    # =============================
    # NETWORK DIAGRAM
    # =============================
    st.subheader("Network Diagram Analisis Probabilistik")
    percentile_options = {
        f"P{percentile}": percentile
        for percentile in range(0, 101, 10)
    }
    percentile_labels = list(percentile_options.keys())
    selected_percentile_label = st.selectbox(
        "Pilih basis durasi proyek untuk network diagram",
        percentile_labels,
        index=percentile_labels.index("P50"),
        key=f"network_diagram_percentile_{current_file_signature}"
    )
    selected_percentile = percentile_options[selected_percentile_label]
    selected_percentile_duration = float(np.percentile(results, selected_percentile))
    network_schedule = scale_durations_to_target_project_duration(
        df_proj,
        probabilistic_activity_durations,
        selected_percentile_duration
    )
    network_project_duration = float(network_schedule["project_duration"])

    st.caption(
        "Lintasan kritis dengan probabilitas tertinggi diberi warna merah. "
        "Lintasan kritis berikutnya diberi warna berbeda berdasarkan urutan probabilitas. "
        f"Node pada network diagram dibangun dari rerata durasi aktivitas hasil simulasi Monte Carlo, "
        f"lalu total durasi diagram diselaraskan ke target durasi proyek {selected_percentile_label} "
        f"= {selected_percentile_duration:.3f}. Durasi proyek pada diagram yang terbentuk = "
        f"{network_project_duration:.3f}."
    )

    df_network_legend = df_path.head(5).reset_index(drop=True).copy()
    diagram_duration_column = f"Durasi Path pada Diagram ({selected_percentile_label})"
    df_network_legend.insert(
        0,
        "Warna",
        ["Merah", "Biru", "Hijau", "Ungu", "Toska"][:len(df_network_legend)]
    )
    df_network_legend.insert(
        2,
        "Durasi Path Deterministik",
        df_network_legend["Path"].apply(
            lambda path: get_path_terminal_duration(path, deterministic_schedule)
        )
    )
    df_network_legend.insert(
        3,
        diagram_duration_column,
        df_network_legend["Path"].apply(
            lambda path: get_path_terminal_duration(path, network_schedule)
        )
    )
    df_network_legend = df_network_legend[
        [
            "Warna",
            "Path",
            "Durasi Path Deterministik",
            diagram_duration_column,
            "Prob"
        ]
    ]
    st.caption(
        "Kolom 'Durasi Path Deterministik' dihitung dari jadwal deterministik berbasis mean productivity. "
        f"Kolom '{diagram_duration_column}' mengikuti nilai path yang benar-benar tergambar pada network diagram "
        f"untuk basis durasi proyek {selected_percentile_label}. "
        "Jika membutuhkan rerata total durasi proyek saat suatu path menjadi lintasan kritis, gunakan tabel "
        "'Analisis Probabilistic Lintasan Kritis' di atas."
    )
    render_table(
        df_network_legend,
        formats={
            "Durasi Path Deterministik": "{:.3f}",
            diagram_duration_column: "{:.3f}",
            "Prob": "{:.3f}"
        }
    )
    network_fig = plot_network_diagram(df_proj, df_path, network_schedule)
    zoomable_network_fig = build_zoomable_network_figure(network_fig)

    if zoomable_network_fig is not None:
        st.caption("Gunakan scroll mouse untuk zoom, drag untuk pan, dan tombol modebar untuk reset view.")
        st.plotly_chart(
            zoomable_network_fig,
            use_container_width=False,
            config={
                "scrollZoom": True,
                "displaylogo": False,
                "modeBarButtonsToAdd": ["zoomIn2d", "zoomOut2d", "resetScale2d"]
            }
        )
    else:
        st.info("Plotly belum terpasang, sehingga network diagram ditampilkan dalam mode statis.")
        st.pyplot(network_fig)
    render_jpg_download_button(
        network_fig,
        f"network_diagram_analisis_probabilistik_{selected_percentile_label.lower()}",
        key=f"download_network_diagram_probabilistik_{selected_percentile_label.lower()}"
    )
    plt.close(network_fig)

    # =============================
    # DETERMINISTIC COMPARISON
    # =============================
    st.subheader("Analisis Deterministik Lintasan Kritis")

    df_det_path = pd.DataFrame({
        "Path": [" -> ".join(path) for path in deterministic_paths],
        "Durasi Proyek": [deterministic_total] * len(deterministic_paths),
        "Status": ["Critical"] * len(deterministic_paths)
    })

    st.write("Deterministic Critical Path (berdasarkan mean productivity):")
    render_table(df_det_path, formats={"Durasi Proyek": "{:.3f}"})

    df_det_ci = pd.DataFrame({
        "Aktivitas": unique_activities,
        "Deterministik": [
            1.0 if act in deterministic_acts else 0.0
            for act in unique_activities
        ]
    })

    df_ci_compare = df_det_ci.merge(df_cp, on="Aktivitas", how="left")
    df_ci_compare = df_ci_compare.rename(columns={"Prob": "Probabilistik (Global)"})
    df_ci_compare["Probabilistik (Global)"] = df_ci_compare["Probabilistik (Global)"].fillna(0.0)
    df_ci_compare = df_ci_compare.sort_values(
        by=["Deterministik", "Probabilistik (Global)", "Aktivitas"],
        ascending=[False, False, True]
    )

    st.subheader("Perbandingan Criticality Index deterministik vs probabilistik")
    render_table(
        df_ci_compare,
        formats={
            "Deterministik": "{:.3f}",
            "Probabilistik (Global)": "{:.3f}"
        }
    )
    with st.expander("Lihat tabel Perbandingan CI kumulatif sampai P0 s.d. P100"):
        if critical_flags:
            df_ci_percentile_compare = build_ci_percentile_comparison_table(
                df_ci_compare,
                results,
                critical_flags
            )
            st.caption(
                "'Probabilistik (Global)' dihitung dari seluruh simulasi. "
                "Kolom 'CI(<=Pk)' dihitung dari subset simulasi dengan durasi proyek "
                "kurang dari atau sama dengan batas persentil ke-k, sehingga nilainya "
                "memang bisa berbeda dari kolom global."
            )
            st.caption(
                "Contoh: jika 'Probabilistik (Global)' = 0.994 tetapi 'CI(<=P20)' = 0.990, "
                "artinya pada kelompok 20% simulasi tercepat aktivitas tersebut sedikit "
                "lebih jarang menjadi kritis dibandingkan seluruh simulasi."
            )
            percentile_formats = {
                "Deterministik": "{:.3f}",
                "Probabilistik (Global)": "{:.3f}"
            }
            percentile_formats.update({
                f"CI(<=P{percentile})": "{:.3f}"
                for percentile in range(0, 101, 10)
            })
            render_table(
                df_ci_percentile_compare,
                formats=percentile_formats
            )
        else:
            st.info(
                "Data CI per persentil belum tersedia pada hasil simulasi yang sedang ditampilkan. "
                "Klik 'Jalankan Simulasi' sekali lagi untuk membangkitkan tabel CI kumulatif per persentil."
            )

    # ==============================
    # ANALISIS SENSITIVITAS (TORNADO)
    # ==============================
    st.subheader("Analisis Sensitivitas per Jenis Pekerjaan")

    activity_input_lookup = (
        df_proj_unique
        .set_index("Aktivitas")[["Volume", "Tenaga"]]
        .to_dict("index")
    )
    sens = {}
    sensitivity_rows = []

    for act in unique_activities:
        scenario_totals = {}
        activity_inputs = activity_input_lookup.get(act, {})
        act_volume = activity_inputs.get("Volume", np.nan)
        act_tenaga = activity_inputs.get("Tenaga", np.nan)
        mean_p_act = mean_p_map.get(act, np.nan)
        p_low = mean_p_act * 0.8 if pd.notna(mean_p_act) else np.nan
        p_high = mean_p_act * 1.2 if pd.notna(mean_p_act) else np.nan

        if (
            pd.notna(act_volume) and pd.notna(act_tenaga) and
            pd.notna(p_low) and p_low > 0 and
            pd.notna(p_high) and p_high > 0
        ):
            durasi_act_low = act_volume / (act_tenaga * p_low)
            durasi_act_high = act_volume / (act_tenaga * p_high)
        else:
            durasi_act_low = np.nan
            durasi_act_high = np.nan

        for f in [0.8, 1.2]:
            durasi = {}

            for _, row in df_proj_unique.iterrows():
                a = row['Aktivitas']
                Q = row['Volume']
                n = row['Tenaga']
                mean_p = mean_p_map.get(a)

                if pd.isna(mean_p) or mean_p is None or mean_p <= 0:
                    raise ValueError(
                        f"Produktivitas rata-rata untuk aktivitas '{a}' tidak valid."
                    )

                p = mean_p * f if a == act else mean_p

                durasi[a] = Q / (n * p)

            total, _ = pdm_cp(df_proj, durasi)
            scenario_totals[f] = total

        pengaruh = abs(scenario_totals[1.2] - scenario_totals[0.8])
        sens[act] = pengaruh
        sensitivity_rows.append({
            "Aktivitas": act,
            "Volume": act_volume,
            "Tenaga": act_tenaga,
            "Mean p": mean_p_act,
            "p(-20%)": p_low,
            "p(+20%)": p_high,
            "Durasi Aktivitas saat p(-20%)": durasi_act_low,
            "Durasi Aktivitas saat p(+20%)": durasi_act_high,
            "Durasi Proyek saat p(-20%)": scenario_totals[0.8],
            "Durasi Proyek saat p(+20%)": scenario_totals[1.2],
            "Pengaruh": pengaruh
        })

    df_s = pd.DataFrame({
        "Aktivitas": list(sens.keys()),
        "Pengaruh": list(sens.values())
    }).sort_values(by="Pengaruh")

    st.write("Tabel Hasil Sensitivitas per Jenis Pekerjaan:")
    st.caption("Keterangan: 1 hari kerja = 4 jam efektif.")
    df_s_table = (
        pd.DataFrame(sensitivity_rows)
        .sort_values(by="Pengaruh", ascending=False)
        .reset_index(drop=True)
    )
    df_s_table.insert(0, "Rank", range(1, len(df_s_table) + 1))
    render_table(
        df_s_table,
        formats={
            "Volume": "{:.3f}",
            "Tenaga": "{:.3f}",
            "Mean p": "{:.3f}",
            "p(-20%)": "{:.3f}",
            "p(+20%)": "{:.3f}",
            "Durasi Aktivitas saat p(-20%)": "{:.3f}",
            "Durasi Aktivitas saat p(+20%)": "{:.3f}",
            "Durasi Proyek saat p(-20%)": "{:.3f}",
            "Durasi Proyek saat p(+20%)": "{:.3f}",
            "Pengaruh": "{:.3f}"
        }
    )

    fig2, ax2 = plt.subplots()
    ax2.barh(df_s["Aktivitas"], df_s["Pengaruh"], edgecolor="black", linewidth=1, color="#4C78A8")
    ax2.set_title("Sensitivity Analysis by Work Item")
    ax2.set_xlabel("Impact on Project Duration")
    ax2.set_ylabel("Work Item")
    st.pyplot(fig2)
    render_jpg_download_button(
        fig2,
        "analisis_sensitivitas_jenis_pekerjaan",
        key="download_analisis_sensitivitas"
    )
    plt.close(fig2)

    # =============================
    # PETA RISIKO
    # =============================
    st.subheader("Peta Risiko per Jenis Pekerjaan")
    st.caption(
        "Probability diambil dari Criticality Index, Impact diambil dari nilai analisis sensitivitas, dan Skor Risiko = CI x Impact Sensitivitas."
    )

    df_risk = build_risk_map_table(df_cp, df_s)
    df_risk_display = df_risk.rename(columns={
        "Prob": "CI",
        "Pengaruh": "Impact Sensitivitas",
        "Risk Score": "Skor Risiko"
    })
    render_table(
        df_risk_display[["Aktivitas", "CI", "Impact Sensitivitas", "Skor Risiko", "Kategori Risiko"]],
        formats={
            "CI": "{:.3f}",
            "Impact Sensitivitas": "{:.3f}",
            "Skor Risiko": "{:.3f}"
        }
    )
    risk_map_fig = plot_risk_map(df_risk)
    st.pyplot(risk_map_fig)
    render_jpg_download_button(
        risk_map_fig,
        "peta_risiko_jenis_pekerjaan",
        key="download_peta_risiko"
    )
    plt.close(risk_map_fig)

    st.subheader("Rekomendasi Teknis Mitigasi Risiko")
    st.caption(
        "Rekomendasi berikut dibangkitkan otomatis dari kategori risiko, "
        "pemicu dominan (CI/impact), dan status jadwal deterministik."
    )
    st.caption(f"Sumber jenis pekerjaan: {activity_work_type_source}.")
    df_risk_recommendation = build_risk_recommendation_table(
        df_risk,
        deterministic_schedule,
        activity_work_type_map=activity_work_type_map
    )
    render_wrapped_table(
        df_risk_recommendation,
        wide_columns=["Rekomendasi Teknis"]
    )

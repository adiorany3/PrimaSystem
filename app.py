"""Sistem prediktor lesi & performa unggas — Streamlit.

Menjalankan dua model sesuai rekomendasi laporan:
  A. grade C saat chick-in (logistic regression)
  B. peringatan dini grade C dari performa harian <=21 hari (HistGradientBoosting)

Model C (mortalitas dari skor lesi) sengaja tidak dipakai: AUC 0,645 dan tidak layak
untuk keputusan operasional.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

APP = Path(__file__).resolve().parent
ROOT = APP.parent
MODELS = APP / "models"
SCHEMA = json.loads((APP / "schema.json").read_text())

st.set_page_config(page_title="Prediktor Unggas — Prima System", page_icon="🐔", layout="wide")

CSS = """
<style>
.block-container { padding-top: 2.2rem; max-width: 1180px; }
h1, h2, h3 { color: #1f3864; }
.kpi { background: #f4f7fb; border: 1px solid #dbe4ee; border-radius: 10px; padding: 14px 16px; }
.kpi .label { color: #6b7a8c; font-size: 0.82rem; letter-spacing: .02em; }
.kpi .value { color: #1f3864; font-size: 1.9rem; font-weight: 700; line-height: 1.1; }
.tag { display: inline-block; padding: 4px 10px; border-radius: 999px; font-size: .82rem; font-weight: 700; }
.low { background: #e6f4ea; color: #1e6b3a; }
.mid { background: #fff4e0; color: #8a5a00; }
.high { background: #fbe9ec; color: #a3243b; }
.note { color: #6b7a8c; font-size: .86rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_models():
    return {
        "A": joblib.load(MODELS / "grade_placement.joblib"),
        "B": joblib.load(MODELS / "grade_early.joblib"),
    }


@st.cache_data(show_spinner=False)
def load_metrics():
    for p in (APP / "training_results.json", ROOT / "reports/training_results.json"):
        if p.exists():
            return json.loads(p.read_text())
    return {}


CAT_A = ["Area", "Location", "Customer Group", "Customer Name", "Sub Customer", "House",
         "Fdg", "Plant", "Company Name", "Pre Starter", "Starter", "Finisher", "Product Group", "DOC"]
NUM_A = ["log_chick_in"]
CAT_B = ["Area", "Hou", "DOC", "Cust. Group", "Product Group", "Plant"]
NUM_B = ["Age-d", "Body Weight", "Mort %", "Act FCR", "Feed Intake", "Indeks", "n_obs"]
def risk_tag(p: float, high: float, mid: float) -> tuple[str, str]:
    if p >= high:
        return "Tinggi", "high"
    if p >= mid:
        return "Sedang", "mid"
    return "Rendah", "low"


def kpi(label: str, value: str, tag: str | None = None) -> str:
    badge = f"<div style='margin-top:6px'><span class='tag {tag[1]}'>{tag[0]}</span></div>" if tag else ""
    return f"<div class='kpi'><div class='label'>{label}</div><div class='value'>{value}</div>{badge}</div>"


def contributions(model, X: pd.DataFrame, top: int = 8) -> pd.DataFrame:
    prep = model.named_steps["prep"]
    lr = model.named_steps[list(model.named_steps)[-1]]
    if not hasattr(lr, "coef_"):
        return pd.DataFrame()
    Xt = prep.transform(X)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    coef = lr.coef_.ravel()
    names = prep.get_feature_names_out()
    vals = Xt.ravel() * coef
    df = pd.DataFrame({"fitur": [str(n).replace("cat__", "").replace("num__", "") for n in names],
                       "kontribusi": vals})
    df = df[df["kontribusi"].abs() > 1e-6]
    return df.reindex(df["kontribusi"].abs().sort_values(ascending=False).index).head(top)


def about_block():
    m = load_metrics()
    st.subheader("Cara pakai")
    st.markdown(
        "1. Pilih tab prediksi yang sesuai dengan ketersediaan data.\n"
        "2. Isi form (nilai awal sudah diisi median data latih) atau unggah CSV untuk prediksi massal.\n"
        "3. Baca probabilitas, kategori risiko, dan faktor yang paling menentukan.\n"
        "4. Gunakan hasil sebagai **prioritas kunjungan/verifikasi**, bukan keputusan otomatis."
    )
    st.subheader("Metrik model (out-of-fold)")
    rows = []
    if m:
        a = m["grade_placement"]
        rows.append(["A — grade C saat chick-in", f"{a['cv']['auc']:.3f}", f"{a['cv']['pr_auc']:.3f}",
                     f"{a['cv']['prevalence_pct']:.2f}%", f"{a['cv']['brier']:.3f}"])
        rows.append(["A — uji temporal 2026", f"{a['temporal_2025_2026']['auc']:.3f}",
                     f"{a['temporal_2025_2026']['pr_auc']:.3f}",
                     f"{a['temporal_2025_2026']['prevalence_pct']:.2f}%",
                     f"{a['temporal_2025_2026']['brier']:.3f}"])
        b = m["grade_early"]["hgb"]
        rows.append(["B — peringatan dini ≤21 hari", f"{b['auc']:.3f}", f"{b['pr_auc']:.3f}",
                     f"{b['prevalence_pct']:.2f}%", f"{b['brier']:.3f}"])
    st.table(pd.DataFrame(rows, columns=["Model", "AUC", "PR-AUC", "Prevalensi", "Brier"]))
    st.warning(
        "Probabilitas model A belum terkalibrasi (class_weight balanced): angka 0,8 bukan berarti 80% peluang. "
        "Gunakan kategori risiko/peringkat, bukan angka mentah sebagai peluang."
    )
    st.markdown(
        "<div class='note'>Sumber: data Prima System 2021–2026 (10.320 panen, 2.888 flock harian). "
        "Validasi grouped by farm + uji temporal 2025→2026. Model mortalitas dari skor lesi tidak dipakai.</div>",
        unsafe_allow_html=True)


def page_a():
    st.subheader("Prediksi grade C — saat chick-in")
    st.caption("Dipakai sebelum/saat DOC masuk. Tidak memakai FCR, mortalitas, atau umur panen.")
    s = SCHEMA["model_a"]
    with st.form("form_a"):
        c1, c2, c3 = st.columns(3)
        area = c1.selectbox("Area", s["cat"]["Area"], index=s["cat"]["Area"].index(s["defaults"]["Area"]) if s["defaults"]["Area"] in s["cat"]["Area"] else 0)
        locs = s["localities"].get(area) or s["cat"]["Location"]
        location = c2.selectbox("Lokasi", locs)
        plant = c3.selectbox("Plant", s["cat"]["Plant"], index=s["cat"]["Plant"].index(s["defaults"]["Plant"]) if s["defaults"]["Plant"] in s["cat"]["Plant"] else 0)
        c4, c5, c6 = st.columns(3)
        cust_group = c4.selectbox("Customer Group", s["cat"]["Customer Group"])
        customer = c5.selectbox("Customer Name", s["cat"]["Customer Name"])
        subcust = c6.selectbox("Sub Customer", s["cat"]["Sub Customer"])
        c7, c8, c9 = st.columns(3)
        house = c7.selectbox("House", s["cat"]["House"])
        fdg = c8.selectbox("Feeding", s["cat"]["Fdg"])
        doc = c9.selectbox("DOC (breed)", s["cat"]["DOC"])
        c10, c11, c12 = st.columns(3)
        pre = c10.selectbox("Pre Starter", s["cat"]["Pre Starter"])
        starter = c11.selectbox("Starter", s["cat"]["Starter"])
        finisher = c12.selectbox("Finisher", s["cat"]["Finisher"])
        c13, c14, c15 = st.columns(3)
        pgroup = c13.selectbox("Product Group", s["cat"]["Product Group"])
        company = c14.selectbox("Company Name", s["cat"]["Company Name"])
        chick_in = c15.number_input("Chick In (ekor)", min_value=1000, max_value=200000,
                                    value=int(SCHEMA["model_a"]["numeric"]["Chick In"]["default"]), step=500)
        submitted = st.form_submit_button("Hitung risiko", type="primary", use_container_width=True)

    if submitted:
        row = {"Area": area, "Location": location, "Customer Group": cust_group, "Customer Name": customer,
               "Sub Customer": subcust, "House": house, "Fdg": fdg, "Plant": plant, "Company Name": company,
               "Pre Starter": pre, "Starter": starter, "Finisher": finisher, "Product Group": pgroup,
               "DOC": doc, "Chick In": chick_in}
        X = pd.DataFrame([row])
        X["log_chick_in"] = np.log1p(X["Chick In"])
        model = load_models()["A"]
        p = float(model.predict_proba(X[CAT_A + NUM_A])[0, 1])
        t = SCHEMA["model_a"]["tiers"]
        label, cls = risk_tag(p, t["p90"], t["p75"])
        c1, c2, c3 = st.columns([1.1, 1, 1.4])
        c1.markdown(kpi("Probabilitas grade C", f"{p*100:.1f}%", (label, cls)), unsafe_allow_html=True)
        c2.markdown(kpi("Kategori risiko", label), unsafe_allow_html=True)
        c3.markdown(kpi("Ambang kategori", f"sedang ≥ {t['p75']*100:.1f}% · tinggi ≥ {t['p90']*100:.1f}%"),
                    unsafe_allow_html=True)
        st.info("Probabilitas belum terkalibrasi: pakai kategori/peringkat, bukan angka mentah sebagai peluang. "
                "Kategori dihitung relatif terhadap distribusi data latih (rendah < persentil 75, tinggi > persentil 90).")
        contrib = contributions(model, X[CAT_A + NUM_A])
        if not contrib.empty:
            st.markdown("**Faktor paling menentukan untuk input ini**")
            st.bar_chart(contrib.set_index("fitur")["kontribusi"], horizontal=True, color="#2f6f8f")
            st.caption("Merah/hijau tidak dipakai di sini; nilai positif menaikkan risiko, negatif menurunkan.")
        st.caption(f"Model A · AUC 0,931 (CV) · 0,896 (uji 2026) · PR-AUC 0,291 vs baseline 0,025")

    with st.expander("Prediksi massal dari CSV"):
        st.markdown("Kolom wajib: " + ", ".join(CAT_A + ["Chick In"]))
        tpl = pd.DataFrame([{c: SCHEMA["model_a"]["defaults"].get(c, "") for c in CAT_A} | {"Chick In": 7000}])
        st.download_button("Unduh template CSV", tpl.to_csv(index=False).encode(), "template_grade_c.csv", "text/csv")
        up = st.file_uploader("Unggah CSV", type=["csv"], key="up_a")
        if up is not None:
            df = pd.read_csv(up)
            missing = [c for c in CAT_A + ["Chick In"] if c not in df.columns]
            if missing:
                st.error("Kolom belum ada: " + ", ".join(missing))
            else:
                df["log_chick_in"] = np.log1p(pd.to_numeric(df["Chick In"], errors="coerce"))
                pr = load_models()["A"].predict_proba(df[CAT_A + NUM_A])[:, 1]
                df["prob_grade_c"] = np.round(pr, 4)
                df["kategori"] = ["Tinggi" if x >= SCHEMA["model_a"]["tiers"]["p90"] else
                                  "Sedang" if x >= SCHEMA["model_a"]["tiers"]["p75"] else "Rendah" for x in pr]
                st.dataframe(df.sort_values("prob_grade_c", ascending=False).head(200), hide_index=True,
                             use_container_width=True)
                st.download_button("Unduh hasil", df.to_csv(index=False).encode(), "prediksi_grade_c.csv", "text/csv")


def page_b():
    st.subheader("Peringatan dini grade C — umur ≤21 hari")
    st.caption("Dipakai saat recording harian; menandai flock yang berisiko gagal sebelum panen.")
    s = SCHEMA["model_b"]
    with st.form("form_b"):
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Umur (hari)", min_value=1, max_value=21, value=16)
        bw = c2.number_input("Body Weight (kg)", min_value=0.03, max_value=3.0, value=0.49, step=0.01)
        mort = c3.number_input("Mortalitas berjalan (%)", min_value=0.0, max_value=60.0, value=2.1, step=0.1)
        c4, c5, c6 = st.columns(3)
        fcr = c4.number_input("FCR harian", min_value=0.3, max_value=3.0, value=1.16, step=0.01)
        fi = c5.number_input("Feed Intake", min_value=0.0, max_value=6.0, value=0.57, step=0.01)
        idx = c6.number_input("Indeks", min_value=50.0, max_value=450.0, value=331.0, step=1.0)
        c7, c8, c9 = st.columns(3)
        area = c7.selectbox("Area", s["cat"]["Area"])
        house = c8.selectbox("House", s["cat"]["Hou"])
        doc = c9.selectbox("DOC (breed)", s["cat"]["DOC"])
        c10, c11, c12 = st.columns(3)
        cg = c10.selectbox("Customer Group", s["cat"]["Cust. Group"])
        pg = c11.selectbox("Product Group", s["cat"]["Product Group"])
        plant = c12.selectbox("Plant", s["cat"]["Plant"])
        n_obs = st.slider("Jumlah observasi sampai hari ini", 1, 15, 2)
        submitted = st.form_submit_button("Hitung risiko", type="primary", use_container_width=True)

    if submitted:
        row = {"Age-d": age, "Body Weight": bw, "Mort %": mort, "Act FCR": fcr, "Feed Intake": fi,
               "Indeks": idx, "n_obs": n_obs, "Area": area, "Hou": house, "DOC": doc,
               "Cust. Group": cg, "Product Group": pg, "Plant": plant}
        X = pd.DataFrame([row])[CAT_B + NUM_B]
        p = float(load_models()["B"].predict_proba(X)[0, 1])
        label, cls = risk_tag(p, 0.10, 0.02)
        c1, c2, c3 = st.columns([1.1, 1, 1.4])
        c1.markdown(kpi("Probabilitas grade C", f"{p*100:.1f}%", (label, cls)), unsafe_allow_html=True)
        c2.markdown(kpi("Kategori risiko", label), unsafe_allow_html=True)
        c3.markdown(kpi("Ambang kategori", "sedang ≥ 2% · tinggi ≥ 10%"), unsafe_allow_html=True)
        st.caption("Model B terkalibrasi (Brier 0,023); probabilitas bisa dibaca sebagai perkiraan peluang. "
                   "Kategori memakai ambang absolut 2% dan 10%.")
        st.caption("Model B · AUC 0,833 · PR-AUC 0,251 vs baseline 2,7% · 5% flock berisiko menangkap 44% grade C")


def main():
    st.title("Prediktor Unggas — Prima System")
    st.caption("Dua model: risiko grade C saat chick-in dan peringatan dini ≤21 hari. Model mortalitas dari skor lesi tidak dipakai (AUC 0,645).")
    tabs = st.tabs(["Grade C (chick-in)", "Peringatan dini (≤21 hari)", "Tentang model"])
    with tabs[0]:
        page_a()
    with tabs[1]:
        page_b()
    with tabs[2]:
        about_block()


if __name__ == "__main__":
    main()

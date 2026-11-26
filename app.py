"""Sistem prediktor unggas — Streamlit (model A terkalibrasi + model B).

  A. grade C saat chick-in — regresi logistik + kalibrasi Platt (Brier 0,021)
  B. peringatan dini grade C dari performa harian <=21 hari — HistGradientBoosting

Model mortalitas dari skor lesi (C) tidak dipakai: AUC 0,645.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

APP = Path(__file__).resolve().parent
MODELS_DIR = APP / "models"
SCHEMA = json.loads((APP / "schema.json").read_text())

st.set_page_config(page_title="Prediktor Unggas — Prima System", page_icon="🐔", layout="wide")

CSS = """
<style>
.block-container { padding-top: 2.2rem; max-width: 1180px; }
h1, h2, h3 { color: #1f3864; }
.kpi { background: #f4f7fb; border: 1px solid #dbe4ee; border-radius: 10px; padding: 14px 16px; }
.kpi .label { color: #6b7a8c; font-size: 0.82rem; }
.kpi .value { color: #1f3864; font-size: 1.9rem; font-weight: 700; line-height: 1.1; }
.tag { display: inline-block; padding: 4px 10px; border-radius: 999px; font-size: .82rem; font-weight: 700; }
.low { background: #e6f4ea; color: #1e6b3a; }
.mid { background: #fff4e0; color: #8a5a00; }
.high { background: #fbe9ec; color: #a3243b; }
.note { color: #6b7a8c; font-size: .86rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def model_path(name: str) -> Path:
    p = MODELS_DIR / name
    if not p.exists():
        raise FileNotFoundError(name)
    return p


@st.cache_resource(show_spinner=False)
def load_models():
    try:
        a = joblib.load(model_path("grade_placement_calibrated.joblib"))
        kind = "calibrated"
    except FileNotFoundError:
        a = joblib.load(model_path("grade_placement.joblib"))
        kind = "base"
    return {"A": a, "A_kind": kind, "B": joblib.load(model_path("grade_early.joblib"))}


@st.cache_data(show_spinner=False)
def load_metrics():
    p = APP.parent / "reports" / "training_results.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


CAT_A = ["Area", "Location", "Customer Group", "Customer Name", "Sub Customer", "House",
         "Fdg", "Plant", "Company Name", "Pre Starter", "Starter", "Finisher", "Product Group", "DOC"]
NUM_A = ["log_chick_in"]
CAT_B = ["Area", "Hou", "DOC", "Cust. Group", "Product Group", "Plant"]
NUM_B = ["Age-d", "Body Weight", "Mort %", "Act FCR", "Feed Intake", "Indeks", "n_obs"]


def kpi(label: str, value: str, tag: tuple[str, str] | None = None) -> str:
    badge = f"<div style='margin-top:6px'><span class='tag {tag[1]}'>{tag[0]}</span></div>" if tag else ""
    return f"<div class='kpi'><div class='label'>{label}</div><div class='value'>{value}</div>{badge}</div>"


def base_pipeline(model):
    if hasattr(model, "calibrated_classifiers_"):
        return model.calibrated_classifiers_[0].estimator
    return model


def contributions(model, X: pd.DataFrame, top: int = 8) -> pd.DataFrame:
    pipe = base_pipeline(model)
    prep = pipe.named_steps["prep"]
    lr = pipe.named_steps[list(pipe.named_steps)[-1]]
    if not hasattr(lr, "coef_"):
        return pd.DataFrame()
    Xt = prep.transform(X)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    vals = Xt.ravel() * lr.coef_.ravel()
    names = [str(n).replace("cat__", "").replace("num__", "") for n in prep.get_feature_names_out()]
    df = pd.DataFrame({"fitur": names, "kontribusi": vals})
    df = df[df["kontribusi"].abs() > 1e-6]
    return df.reindex(df["kontribusi"].abs().sort_values(ascending=False).index).head(top)


def cost_threshold(ratio: int) -> float:
    return 1 / (1 + ratio)


def about_block():
    m = load_metrics()
    cal = SCHEMA["model_a"]["calibrated"]
    st.subheader("Cara pakai")
    st.markdown(
        "1. Pilih tab sesuai data yang tersedia.\n"
        "2. Isi form (nilai awal = median data latih) atau unggah CSV untuk prediksi massal.\n"
        "3. Baca probabilitas, keputusan pada ambang biaya, dan faktor penentu.\n"
        "4. Hasil = prioritas kunjungan, bukan keputusan otomatis."
    )
    st.subheader("Metrik model (out-of-fold)")
    def num(x, d=3):
        return f"{x:.{d}f}".replace(".", ",")
    rows = [
        ["A — pra-kalibrasi", num(cal["metrics"]["auc_before"]), num(cal["metrics"]["pr_auc_before"]), "2,48%",
         num(cal["metrics"]["brier_before"]), num(cal["metrics"]["ece_before"])],
        ["A — Platt (dipakai)", num(cal["metrics"]["auc"]), num(cal["metrics"]["pr_auc"]), "2,48%",
         num(cal["metrics"]["brier_after"]), num(cal["metrics"]["ece_after"])],
        ["A — uji temporal 2026", num(cal["metrics"]["auc_temporal"]), "0,155", "1,48%",
         num(cal["metrics"]["brier_temporal"]), "0,006"],
    ]
    if m:
        b = m["grade_early"]["hgb"]
        rows.append(["B — peringatan dini ≤21 hari", num(b["auc"]), num(b["pr_auc"]),
                     num(b["prevalence_pct"], 2) + "%", num(b["brier"]), "—"])
    st.table(pd.DataFrame(rows, columns=["Model", "AUC", "PR-AUC", "Prevalensi", "Brier", "ECE"]))
    amb = ", ".join(f"R={r} → {cost_threshold(r)*100:.1f}%".replace(".", ",") for r in (10, 20, 50))
    st.markdown(
        "**Ambang biaya model A:** kunjungan = 1, gagal panen = R → ambang = 1/(1+R). " + amb + "."
    )
    scen = pd.DataFrame(cal["cost_scenarios"])
    scen_fmt = pd.DataFrame({
        "R (gagal:kunjungan)": scen["cost_ratio_fail_per_visit"].astype(int),
        "ambang": [f"{t*100:.1f}%".replace(".", ",") for t in scen["threshold"]],
        "flock ditandai": [f"{v:,}".replace(",", ".") for v in scen["flagged"]],
        "grade C tertangkap": [f"{v:,}".replace(",", ".") for v in scen["tp"]],
        "salah tanda": [f"{v:,}".replace(",", ".") for v in scen["fp"]],
        "net benefit": [f"{v:.4f}".replace(".", ",") for v in scen["net_benefit"]],
        "kunjungan/kasus": [f"{v:.1f}".replace(".", ",") for v in scen["visits_per_case"]],
    })
    st.table(scen_fmt)
    st.info(
        "Model A terkalibrasi: probabilitas dapat dibaca sebagai perkiraan peluang. "
        "PR-AUC 0,243 (turun dari 0,291) adalah konsekuensi kalibrasi — peringkat ekstrem dipadatkan; "
        "AUC hampir tidak berubah. Model C (mortalitas) tidak dipakai."
    )
    st.markdown(
        "<div class='note'>Sumber: Prima System 2021–2026 · 10.320 panen · 2.888 flock harian. "
        "Validasi grouped by farm + uji temporal 2025→2026.</div>", unsafe_allow_html=True)


def page_a():
    st.subheader("Prediksi grade C — saat chick-in")
    st.caption("Dipakai sebelum/saat DOC masuk. Model terkalibrasi (Platt). Tidak memakai FCR, mortalitas, atau umur panen.")
    s = SCHEMA["model_a"]
    # tanpa st.form: pilihan Location/Customer ikut berubah saat Area/Group berganti
    c1, c2, c3 = st.columns(3)
    area = c1.selectbox("Area", s["cat"]["Area"],
                        index=s["cat"]["Area"].index(s["defaults"]["Area"]) if s["defaults"]["Area"] in s["cat"]["Area"] else 0)
    locs = s["localities"].get(area) or s["cat"]["Location"]
    default_loc = s["defaults"].get("Location")
    # key per area: saat Area berganti, Lokasi tidak menyimpan nilai dari area lama
    location = c2.selectbox("Lokasi", locs, key=f"loc_{area}",
                            index=locs.index(default_loc) if default_loc in locs else 0)
    plants = s.get("plant_map", {}).get(area) or s["cat"]["Plant"]
    plant = c3.selectbox("Plant", plants, key=f"plant_{area}")
    c4, c5, c6 = st.columns(3)
    cust_group = c4.selectbox("Customer Group", s["cat"]["Customer Group"])
    names = s.get("cust_map", {}).get(cust_group) or s["cat"]["Customer Name"]
    customer = c5.selectbox("Customer Name", names, key=f"cust_{cust_group}")
    subs = s.get("sub_map", {}).get(customer) or s["cat"]["Sub Customer"]
    subcust = c6.selectbox("Sub Customer", subs, key=f"sub_{customer}")
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
                                value=int(s["numeric"]["Chick In"]["default"]), step=500)
    ratio = st.radio("Rasio biaya (gagal panen : kunjungan)", [10, 20, 50], index=1, horizontal=True,
                     help="Ambang keputusan = 1/(1+R). Sesuaikan dengan biaya riil di lapangan.")

    row = {"Area": area, "Location": location, "Customer Group": cust_group, "Customer Name": customer,
           "Sub Customer": subcust, "House": house, "Fdg": fdg, "Plant": plant, "Company Name": company,
           "Pre Starter": pre, "Starter": starter, "Finisher": finisher, "Product Group": pgroup,
           "DOC": doc, "Chick In": chick_in}
    X = pd.DataFrame([row])
    X["log_chick_in"] = np.log1p(X["Chick In"])
    model = load_models()["A"]
    p = float(model.predict_proba(X[CAT_A + NUM_A])[0, 1])
    thr = cost_threshold(ratio)
    decision = ("Kunjungi — prioritas", "high") if p >= thr else ("Tidak prioritas", "low")
    scen = next((c for c in SCHEMA["model_a"]["calibrated"]["cost_scenarios"]
                 if c["cost_ratio_fail_per_visit"] == ratio), None)
    k1, k2, k3 = st.columns([1.1, 1, 1.4])
    k1.markdown(kpi("Probabilitas grade C", f"{p*100:.1f}%"), unsafe_allow_html=True)
    k2.markdown(kpi("Keputusan", decision[0], decision), unsafe_allow_html=True)
    detail = f"ambang {thr*100:.1f}% (R={ratio}:1)"
    if scen:
        detail += f" · historis {scen['visits_per_case']} kunjungan/kasus"
    k3.markdown(kpi("Ambang biaya", detail), unsafe_allow_html=True)
    st.caption("Probabilitas terkalibrasi (Platt): Brier 0,021, ECE 0,007. "
               "Keputusan memakai ambang biaya, bukan kategori persentil. Hasil dihitung ulang tiap input berubah.")
    contrib = contributions(model, X[CAT_A + NUM_A])
    if not contrib.empty:
        st.markdown("**Faktor paling menentukan untuk input ini**")
        st.bar_chart(contrib.set_index("fitur")["kontribusi"], horizontal=True, color="#2f6f8f")
        st.caption("Nilai positif menaikkan risiko, negatif menurunkan.")

    with st.expander("Prediksi massal dari CSV"):
        st.markdown("Kolom wajib: " + ", ".join(CAT_A + ["Chick In"]))
        tpl = pd.DataFrame([{c: s["defaults"].get(c, "") for c in CAT_A} | {"Chick In": 7000}])
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
                df["keputusan_R20"] = np.where(pr >= cost_threshold(20), "Kunjungi", "Tidak prioritas")
                st.dataframe(df.sort_values("prob_grade_c", ascending=False).head(200),
                             hide_index=True, use_container_width=True)
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
        label, cls = ("Tinggi", "high") if p >= 0.10 else ("Sedang", "mid") if p >= 0.02 else ("Rendah", "low")
        c1, c2, c3 = st.columns([1.1, 1, 1.4])
        c1.markdown(kpi("Probabilitas grade C", f"{p*100:.1f}%", (label, cls)), unsafe_allow_html=True)
        c2.markdown(kpi("Kategori risiko", label), unsafe_allow_html=True)
        c3.markdown(kpi("Ambang kategori", "sedang ≥ 2% · tinggi ≥ 10%"), unsafe_allow_html=True)
        st.caption("Model B (HistGradientBoosting) — probabilitas terkalibrasi baik (Brier 0,023); ambang absolut 2% dan 10%.")
        st.caption("Model B · AUC 0,833 · PR-AUC 0,251 vs baseline 2,7% · 5% flock berisiko menangkap 44% grade C")


def main():
    st.title("Prediktor Unggas — Prima System")
    st.caption("Model A terkalibrasi (chick-in) dan model B (peringatan dini ≤21 hari). "
               "Model mortalitas dari skor lesi tidak dipakai (AUC 0,645).")
    tabs = st.tabs(["Grade C (chick-in)", "Peringatan dini (≤21 hari)", "Tentang model"])
    with tabs[0]:
        page_a()
    with tabs[1]:
        page_b()
    with tabs[2]:
        about_block()


if __name__ == "__main__":
    main()

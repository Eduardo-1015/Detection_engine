"""
Analises complementares pedidas na revisao do manuscrito (revPMR).

Le o DATA_FINAL.csv, calcula a qualidade dos dados no ficheiro original,
aplica a preparacao de preparar_dados.py (timestamps invalidos, duplicados,
dropna) e corre o ICU Alert Engine UMA vez sobre os dados preparados, com
instrumentacao adicional (sem alterar a logica do motor), e calcula:

  1. Qualidade dos dados: missingness, exclusoes por limites de
     plausibilidade (por variavel), SpO2 < 50 %, timestamps duplicados,
     distribuicao dos intervalos entre observacoes            [C479, C500, C505]
  2. Carga de alertas por doente e por internamento (mediana, IQR, maximo,
     proporcao sem alertas, alertas por doente-dia)             [C570, C574]
  3. Correspondencia alerta -> resolucao por episodio, duracoes e numero
     de resets por interrupcao de monitorizacao                 [C1170]
  4. Ablacao dos confirmadores (isolados, leave-one-out, contribuicao
     unica) e correlacao entre indices e variaveis primarias    [C1183, C552]
  5. Cenarios de limiar de perigo absoluto (severity floor)     [C1134, C1167]
  6. Analise temporal por periodos de calendario                [C1191]

Saidas:
  latex/generated/results_macros.tex   -> valores usados no texto (\res{...})
  latex/generated/tab_*.tex            -> tabelas incluidas no manuscrito
  latex/generated/analise_revisao.json -> todos os resultados em bruto

Uso:
  python3 analise_revisao.py                    # usa ../DATA_FINAL.csv
  python3 analise_revisao.py caminho/para.csv
"""
from __future__ import annotations

import itertools
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from preparar_dados import ler_csv, limpar
from icu_alert_engine import (
    AlertConfig, AlertEngine, Condition, EventKind,
    _CONFIRMER_SI, _CONFIRMER_RDR, _CONFIRMER_FCSPO2,
    _CONFIRMER_PASPO2, _CONFIRMER_PAMSPO2,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(os.path.dirname(SCRIPT_DIR), "DATA_FINAL.csv")
OUT_DIR = os.path.join(SCRIPT_DIR, "latex", "generated")

# Periodos de calendario para a analise temporal (inclusivos)
PERIODS = [(2008, 2012), (2013, 2016), (2017, 2020), (2021, 2025)]

# Cenarios de limiar de perigo absoluto (sobre o valor extremo do alerta)
DANGER_SCENARIOS = [
    ("SpO$_2$ $<$ 85\\% or HR $\\geq$ 150 bpm", 85.0, 150.0),
    ("SpO$_2$ $<$ 80\\% or HR $\\geq$ 150 bpm", 80.0, 150.0),
    ("SpO$_2$ $<$ 80\\% or HR $\\geq$ 130 bpm", 80.0, 130.0),
]

CONF_KEYS = {
    "SHOCK_INDEX": _CONFIRMER_SI,
    "RR_SPO2_RATIO": _CONFIRMER_RDR,
    "FC_SPO2_RATIO": _CONFIRMER_FCSPO2,
    "PAS_SPO2_RATIO": _CONFIRMER_PASPO2,
    "PAS_X_SPO2": _CONFIRMER_PAMSPO2,
}
CONF_BY_COND = {
    Condition.TACHYCARDIA: ["SHOCK_INDEX", "FC_SPO2_RATIO", "PAS_SPO2_RATIO"],
    Condition.HYPOXEMIA: ["RR_SPO2_RATIO", "FC_SPO2_RATIO", "PAS_X_SPO2"],
}
REDUCED = {Condition.TACHYCARDIA: "SHOCK_INDEX", Condition.HYPOXEMIA: "RR_SPO2_RATIO"}
PRETTY = {
    "SHOCK_INDEX": "Shock Index (HR/SBP $>$ 1.0)",
    "RR_SPO2_RATIO": "RR/SpO$_2$ ($>$ 0.30)",
    "FC_SPO2_RATIO": "HR/SpO$_2$ ($>$ 1.10)",
    "PAS_SPO2_RATIO": "SBP/SpO$_2$ ($<$ 1.0)",
    "PAS_X_SPO2": "SBP$\\times$SpO$_2$ ($<$ 9,000)",
}
COND_NAME = {Condition.TACHYCARDIA: "Tachycardia", Condition.HYPOXEMIA: "Hypoxaemia"}

VARS = {
    "FREQ_CARDIACA": ("Heart rate (bpm)", "hr_valid_min", "hr_valid_max"),
    "SPO2": ("SpO$_2$ (\\%)", "spo2_valid_min", "spo2_valid_max"),
    "PRESSAO_ARTERIAL_SISTOLICA": ("Systolic BP (mmHg)", "sbp_valid_min", "sbp_valid_max"),
    "FREQ_RESPIRATORIA": ("Respiratory rate (breaths/min)", "rr_valid_min", "rr_valid_max"),
}


# ---------------------------------------------------------------------------
# Motor instrumentado (a logica do motor NAO e alterada)
# ---------------------------------------------------------------------------
class InstrumentedEngine(AlertEngine):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.alert_conf_states: list[dict] = []   # estado de TODOS os confirmadores no alerta
        self.gap_resets = Counter()               # resets de episodios ja alertados
        self.gap_reset_episodes = set()           # (pid, cond, breach_start)
        self.gap_resets_any = Counter()           # qualquer reset com breach em curso

    def _update(self, patient_id, timestamp, condition, value, valid_min, valid_max,
                is_breach, threshold, higher_is_worse, confirmers=None):
        s = self._state_for(patient_id, condition)
        try:
            v = None if value is None or pd.isna(value) else float(value)
        except (TypeError, ValueError):
            v = None
        if (v is not None and valid_min <= v <= valid_max and s.last_valid_ts is not None
                and (timestamp - s.last_valid_ts).total_seconds() > self.config.max_gap_seconds):
            if s.consecutive_count > 0:
                self.gap_resets_any[condition] += 1
            if s.alert_fired:
                self.gap_resets[condition] += 1
                self.gap_reset_episodes.add((patient_id, condition, s.breach_start))
        emitted = super()._update(patient_id, timestamp, condition, value, valid_min,
                                  valid_max, is_breach, threshold, higher_is_worse, confirmers)
        for ev in emitted:
            if ev.kind == EventKind.ALERT:
                st = {}
                for name in CONF_BY_COND[condition]:
                    cs = self._state.get(patient_id, {}).get(CONF_KEYS[name])
                    st[name] = bool(cs is not None and cs.sustained)
                self.alert_conf_states.append(st)
        return emitted


# ---------------------------------------------------------------------------
# Utilitarios
# ---------------------------------------------------------------------------
def fint(x) -> str:
    return f"{int(round(x)):,}"


def fpct(num, den, d=1) -> str:
    return "---" if not den else f"{100.0 * num / den:.{d}f}\\%"


def fnum(x, d=1) -> str:
    return "---" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.{d}f}"


def q(series, p):
    return float(np.nanpercentile(series, p)) if len(series) else float("nan")


def med_iqr(series, d=0) -> str:
    s = pd.Series(series).dropna()
    if s.empty:
        return "---"
    return f"{fnum(q(s,50),d)} ({fnum(q(s,25),d)}--{fnum(q(s,75),d)})"


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={k: v for k, v in {"PATIENT_ID": "patient_id",
                                             "RTDATA_TIME": "timestamp"}.items()
                            if k in df.columns and v not in df.columns})
    df["patient_id"] = df["patient_id"].astype(str)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for c in VARS:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def valid(series, lo, hi):
    return series.where((series >= lo) & (series <= hi))


def indices(df, cfg):
    hr = valid(df["FREQ_CARDIACA"], cfg.hr_valid_min, cfg.hr_valid_max)
    sp = valid(df["SPO2"], cfg.spo2_valid_min, cfg.spo2_valid_max)
    sbp = valid(df["PRESSAO_ARTERIAL_SISTOLICA"], cfg.sbp_valid_min, cfg.sbp_valid_max)
    rr = valid(df["FREQ_RESPIRATORIA"], cfg.rr_valid_min, cfg.rr_valid_max)
    out = pd.DataFrame({
        "HR": hr, "SpO2": sp, "SBP": sbp, "RR": rr,
        "SHOCK_INDEX": valid(hr / sbp, cfg.si_valid_min, cfg.si_valid_max),
        "RR_SPO2_RATIO": valid(rr / sp, cfg.rdr_valid_min, cfg.rdr_valid_max),
        "FC_SPO2_RATIO": valid(hr / sp, cfg.fc_spo2_valid_min, cfg.fc_spo2_valid_max),
        "PAS_SPO2_RATIO": valid(sbp / sp, cfg.pas_spo2_valid_min, cfg.pas_spo2_valid_max),
        "PAS_X_SPO2": valid(sbp * sp, cfg.pas_x_spo2_valid_min, cfg.pas_x_spo2_valid_max),
    })
    return out


def write_table(name, body):
    with open(os.path.join(OUT_DIR, name), "w", encoding="utf-8") as f:
        f.write(body)


# ---------------------------------------------------------------------------
# Principal
# ---------------------------------------------------------------------------
def main(path: str) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    cfg = AlertConfig()   # configuracao por omissao = configuracao do manuscrito
    R: dict = {}          # macros para o LaTeX
    J: dict = {}          # resultados em bruto

    print(f"A ler {path} ...")
    raw_src = ler_csv(path)
    df = normalise(raw_src)          # ficheiro original (qualidade dos dados)
    has_adm = "ADMISSION_ID" in df.columns

    # ------------------------------------------------------------------ 1
    n_rows, n_pat = len(df), df["patient_id"].nunique()
    n_raw_rows = n_rows
    R["raw-obs"] = fint(n_rows)
    R["raw-patients"] = fint(n_pat)
    R["raw-admissions"] = fint(df["ADMISSION_ID"].nunique()) if has_adm else "n/a"
    ts_ok = df["timestamp"].notna()
    R["ts-invalid"] = fint((~ts_ok | (df["timestamp"] < pd.Timestamp("2008-01-01"))
                            | (df["timestamp"] > pd.Timestamp("2025-12-31 23:59:59"))).sum())

    rows_q = []
    for col, (label, lo_k, hi_k) in VARS.items():
        lo, hi = getattr(cfg, lo_k), getattr(cfg, hi_k)
        s = df[col]
        n_missing = int(s.isna().sum())
        n_low = int((s < lo).sum())
        n_high = int((s > hi).sum())
        pat_novalid = int((~((s >= lo) & (s <= hi))).groupby(df["patient_id"]).all().sum())
        rows_q.append((label, n_missing, n_low, n_high, pat_novalid, f"{lo:g}--{hi:g}"))
        J[f"quality_{col}"] = dict(missing=n_missing, below=n_low, above=n_high,
                                   patients_without_valid=pat_novalid)
    sp = df["SPO2"]
    R["spo2-zero"] = fint((sp == 0).sum())
    R["spo2-1to49"] = fint(((sp > 0) & (sp < 50)).sum())
    R["spo2-1to49-pat"] = fint(df.loc[(sp > 0) & (sp < 50), "patient_id"].nunique())

    dup_mask = df.duplicated(["patient_id", "timestamp"], keep=False)
    dup_extra = int(df.duplicated(["patient_id", "timestamp"]).sum())
    R["dup-rows"] = fint(dup_extra)
    R["dup-pct"] = fpct(dup_extra, n_rows, 2)
    R["dup-patients"] = fint(df.loc[dup_mask, "patient_id"].nunique())

    # ------------------------------------------------------------------ limpeza
    df_clean, flow = limpar(raw_src)
    body = ["\\begin{tabular}{lrrrr}", "\\toprule",
            "Step & Rows removed & Rows remaining & Patients & ICU admissions \\\\", "\\midrule"]
    labels = ["Source file", "Invalid timestamp or outside 2008--2025",
              "Duplicate patient--timestamp record", "Missing HR, SpO$_2$, SBP, or RR"]
    for lab, st in zip(labels, flow):
        body.append(f"{lab} & {fint(st['removidas']) if st['removidas'] else '---'} & {fint(st['linhas'])} & "
                    f"{fint(st['doentes'])} & {fint(st['internamentos']) if st['internamentos'] is not None else 'n/a'} \\\\")
    body += ["\\bottomrule", "\\end{tabular}"]
    write_table("tab_flow.tex", "\n".join(body) + "\n")
    R["flow-ts-rows"] = fint(flow[1]["removidas"])
    R["flow-dup-rows"] = fint(flow[2]["removidas"])
    R["flow-na-rows"] = fint(flow[3]["removidas"])
    R["flow-na-pct"] = fpct(flow[3]["removidas"], flow[2]["linhas"])
    R["flow-pat-excluded"] = fint(flow[0]["doentes"] - flow[3]["doentes"])
    J["flow"] = flow

    df = normalise(df_clean)         # dados limpos: tudo o que se segue
    n_rows, n_pat = len(df), df["patient_id"].nunique()
    R["n-obs"] = fint(n_rows)
    R["n-patients"] = fint(n_pat)
    R["n-admissions"] = fint(df["ADMISSION_ID"].nunique()) if has_adm else "n/a"
    R["date-first"] = df["timestamp"].min().strftime("%d %B %Y")
    R["date-last"] = df["timestamp"].max().strftime("%d %B %Y")
    if has_adm:
        adm_per_pat = df.groupby("patient_id")["ADMISSION_ID"].nunique()
        R["pct-readmitted"] = fpct((adm_per_pat > 1).sum(), n_pat)

    def interval_stats(frame, prefix):
        dd = frame.dropna(subset=["timestamp"]).sort_values(["patient_id", "timestamp"], kind="mergesort")
        gg = dd.groupby("patient_id")["timestamp"].diff().dt.total_seconds().dropna()
        gp = gg[gg > 0]
        R[f"{prefix}interval-median-min"] = fnum(q(gp, 50) / 60, 1)
        R[f"{prefix}interval-iqr-min"] = f"{fnum(q(gp,25)/60,1)}--{fnum(q(gp,75)/60,1)}"
        R[f"{prefix}interval-pct-le5"] = fpct((gp <= 330).sum(), len(gp))
        R[f"{prefix}interval-pct-gt15"] = fpct((gp > cfg.max_gap_seconds).sum(), len(gp))
        return dd, gg

    interval_stats(normalise(raw_src), "raw-")
    d, gaps = interval_stats(df, "")

    mon = gaps[(gaps > 0) & (gaps <= cfg.max_gap_seconds)]
    pat_days = mon.groupby(d.loc[mon.index, "patient_id"]).sum() / 86400.0
    pat_days = pat_days.reindex(df["patient_id"].unique(), fill_value=0.0)
    total_days = float(pat_days.sum())
    R["patient-days"] = fint(total_days)

    body = ["\\begin{tabular}{lrrrrr}", "\\toprule",
            "Variable & Bounds & \\shortstack{Missing,\\\\n (\\%)} & \\shortstack{Below\\\\bound, n} & \\shortstack{Above\\\\bound, n} & \\shortstack{Patients with no\\\\valid value, n} \\\\",
            "\\midrule"]
    for label, nm, nl, nh, pv, b in rows_q:
        body.append(f"{label} & {b} & {fint(nm)} ({fpct(nm, n_raw_rows)}) & {fint(nl)} & {fint(nh)} & {fint(pv)} \\\\")
    body += ["\\bottomrule", "\\end{tabular}"]
    write_table("tab_quality.tex", "\n".join(body) + "\n")

    # ------------------------------------------------------------------ run
    print("A correr o motor instrumentado (pode demorar) ...")
    eng = InstrumentedEngine(cfg, on_alert=lambda e: None)
    events = eng.process_dataframe(df)
    alerts = [e for e in events if e.kind == EventKind.ALERT]
    resols = [e for e in events if e.kind == EventKind.RESOLUTION]
    assert len(alerts) == len(eng.alert_conf_states)
    A = pd.DataFrame([dict(pid=e.patient_id, cond=e.condition, ts=e.timestamp,
                           start=e.started_at, value=e.value, confirmed=e.confirmed,
                           first_conf=e.confirmer_name) for e in alerts])
    S = pd.DataFrame(eng.alert_conf_states).astype("boolean").fillna(False).astype(bool)
    A = pd.concat([A.reset_index(drop=True), S.reset_index(drop=True)], axis=1)
    Rz = pd.DataFrame([dict(pid=e.patient_id, cond=e.condition, start=e.started_at,
                            dur=e.duration_seconds) for e in resols])
    R["n-alerts"] = fint(len(A))
    R["n-resolutions"] = fint(len(Rz))
    R["n-confirmed"] = fint(A["confirmed"].sum())
    n_all = len(A)
    R["n-unconfirmed"] = fint(n_all - A["confirmed"].sum())
    R["pct-confirmed"] = fpct(A["confirmed"].sum(), n_all)
    R["pct-unconfirmed"] = fpct(n_all - A["confirmed"].sum(), n_all)
    red_all = 0
    ev_rows, cf_rows = [], []
    for cond in CONF_BY_COND:
        tag = "tachy" if cond == Condition.TACHYCARDIA else "hypox"
        a_ = A[A["cond"] == cond]
        n_c = len(a_)
        conf = int(a_["confirmed"].sum())
        red = int(a_[REDUCED[cond]].sum())
        red_all += red
        n_res = int((Rz["cond"] == cond).sum()) if len(Rz) else 0
        R[f"{tag}-alerts"] = fint(n_c)
        R[f"{tag}-share"] = fpct(n_c, n_all)
        R[f"{tag}-conf-n"] = fint(conf)
        R[f"{tag}-conf-pct"] = fpct(conf, n_c)
        R[f"{tag}-unconf-n"] = fint(n_c - conf)
        R[f"{tag}-unconf-pct"] = fpct(n_c - conf, n_c)
        R[f"{tag}-res-n"] = fint(n_res)
        R[f"{tag}-red-n"] = fint(red)
        R[f"{tag}-red-pct"] = fpct(red, n_c)
        R[f"{tag}-red-unconf-n"] = fint(n_c - red)
        R[f"{tag}-red-unconf-pct"] = fpct(n_c - red, n_c)
        dpp = (100.0 * (conf - red) / n_c) if n_c else 0.0
        R[f"{tag}-diff-pp"] = f"{dpp:.1f}"
        nm = COND_NAME[cond]
        ev_rows += [f"{nm} & Alert, confirmed & Critical & {fint(conf)} & {fpct(conf, n_c)} \\\\",
                    f"{nm} & Alert, unconfirmed & High & {fint(n_c-conf)} & {fpct(n_c-conf, n_c)} \\\\",
                    f"{nm} & Resolution & Info & {fint(n_res)} & --- \\\\"]
        cf_rows += [f"{nm}, confirmed & {fint(red)} ({fpct(red, n_c)}) & {fint(conf)} ({fpct(conf, n_c)}) & $+{dpp:.1f}$ pp \\\\",
                    f"{nm}, unconfirmed & {fint(n_c-red)} ({fpct(n_c-red, n_c)}) & {fint(n_c-conf)} ({fpct(n_c-conf, n_c)}) & $-{dpp:.1f}$ pp \\\\"]
    R["red-conf-n"] = fint(red_all)
    R["red-conf-pct"] = fpct(red_all, n_all)
    dpp_all = (100.0 * (A["confirmed"].sum() - red_all) / n_all) if n_all else 0.0
    R["diff-pp"] = f"{dpp_all:.1f}"
    write_table("tab_events.tex", "\n".join(
        ["\\begin{tabular}{lllrr}", "\\toprule",
         "Condition & Event type & Severity & N & \\% of condition alerts \\\\", "\\midrule"]
        + ev_rows +
        ["\\midrule",
         f"Total & Alerts & --- & {fint(n_all)} & --- \\\\",
         f"Total & Confirmed alerts & --- & {fint(A['confirmed'].sum())} & {fpct(A['confirmed'].sum(), n_all)} \\\\",
         f"Total & Resolutions & --- & {fint(len(Rz))} & --- \\\\",
         "\\bottomrule", "\\end{tabular}"]) + "\n")
    write_table("tab_config.tex", "\n".join(
        ["\\begin{tabular}{lrrr}", "\\toprule",
         "Classification & \\shortstack{One confirmer\\\\per condition} & \\shortstack{Three confirmers\\\\per condition} & \\shortstack{Absolute\\\\difference} \\\\",
         "\\midrule"] + cf_rows +
        ["\\midrule",
         f"Overall confirmed & {fint(red_all)} ({fpct(red_all, n_all)}) & {fint(A['confirmed'].sum())} ({fpct(A['confirmed'].sum(), n_all)}) & $+{dpp_all:.1f}$ pp \\\\",
         "\\bottomrule", "\\end{tabular}"]) + "\n")

    # ------------------------------------------------------------------ 2
    per_pat = A.groupby("pid").size().reindex(df["patient_id"].unique(), fill_value=0)
    R["pat-zero-pct"] = fpct((per_pat == 0).sum(), len(per_pat))
    R["pat-zero-n"] = fint((per_pat == 0).sum())
    R["pat-median"] = fnum(q(per_pat, 50), 0)
    R["pat-iqr"] = f"{fnum(q(per_pat,25),0)}--{fnum(q(per_pat,75),0)}"
    R["pat-max"] = fint(per_pat.max())
    with_alert = per_pat[per_pat > 0]
    R["pat-withalert-median"] = fnum(q(with_alert, 50), 0) if len(with_alert) else "---"
    R["pat-withalert-iqr"] = (f"{fnum(q(with_alert,25),0)}--{fnum(q(with_alert,75),0)}"
                             if len(with_alert) else "---")
    top = per_pat.sort_values(ascending=False)
    k = max(1, int(np.ceil(0.10 * len(top))))
    R["top10-share"] = fpct(top.iloc[:k].sum(), top.sum())
    R["alerts-per-patday"] = fnum(len(A) / total_days, 2) if total_days else "---"
    rate = (per_pat / pat_days.replace(0, np.nan)).dropna()
    R["patday-median"] = fnum(q(rate, 50), 2)
    R["patday-iqr"] = f"{fnum(q(rate,25),2)}--{fnum(q(rate,75),2)}"

    burden_rows = [("Patients", per_pat)]
    if has_adm:
        key = df.drop_duplicates(["patient_id", "timestamp"])[["patient_id", "timestamp", "ADMISSION_ID"]]
        Aa = A.merge(key, left_on=["pid", "ts"], right_on=["patient_id", "timestamp"], how="left")
        per_adm = Aa.groupby("ADMISSION_ID").size().reindex(df["ADMISSION_ID"].dropna().unique(), fill_value=0)
        burden_rows.append(("ICU admissions", per_adm))
        R["adm-zero-pct"] = fpct((per_adm == 0).sum(), len(per_adm))
        R["adm-median"] = fnum(q(per_adm, 50), 0)
        R["adm-iqr"] = f"{fnum(q(per_adm,25),0)}--{fnum(q(per_adm,75),0)}"
        R["adm-max"] = fint(per_adm.max())
    body = ["\\begin{tabular}{lrrrrr}", "\\toprule",
            "Unit of analysis & N & \\shortstack{No alerts,\\\\n (\\%)} & \\shortstack{Median\\\\(IQR)} & Maximum & \\shortstack{Share of alerts\\\\in top 10\\%} \\\\",
            "\\midrule"]
    for lab, s in burden_rows:
        t = s.sort_values(ascending=False)
        kk = max(1, int(np.ceil(0.10 * len(t))))
        body.append(f"{lab} & {fint(len(s))} & {fint((s==0).sum())} ({fpct((s==0).sum(), len(s))}) & "
                    f"{med_iqr(s)} & {fint(s.max())} & {fpct(t.iloc[:kk].sum(), t.sum())} \\\\")
    for cond in CONF_BY_COND:
        s = A[A["cond"] == cond].groupby("pid").size().reindex(per_pat.index, fill_value=0)
        body.append(f"\\quad {COND_NAME[cond]} alerts per patient & {fint(len(s))} & "
                    f"{fint((s==0).sum())} ({fpct((s==0).sum(), len(s))}) & {med_iqr(s)} & {fint(s.max())} & --- \\\\")
    body += ["\\bottomrule", "\\end{tabular}"]
    write_table("tab_burden.tex", "\n".join(body) + "\n")

    # ------------------------------------------------------------------ 3
    body = ["\\begin{tabular}{lrrrrr}", "\\toprule",
            "Condition & Alerts & \\shortstack{Resolved,\\\\n (\\%)} & \\shortstack{Reset by gap,\\\\n (\\%)} & \\shortstack{Open at end of\\\\record, n (\\%)} & \\shortstack{Duration of resolved\\\\episodes, min,\\\\median (IQR)} \\\\",
            "\\midrule"]
    for cond in CONF_BY_COND:
        a = A[A["cond"] == cond]
        r = Rz[Rz["cond"] == cond] if len(Rz) else Rz
        rk = set(zip(r["pid"], r["start"])) if len(r) else set()
        gk = {(p, s) for (p, c, s) in eng.gap_reset_episodes if c == cond}
        keys = list(zip(a["pid"], a["start"]))
        n_res = sum(k_ in rk for k_ in keys)
        n_gap = sum((k_ not in rk) and (k_ in gk) for k_ in keys)
        n_open = len(keys) - n_res - n_gap
        durs = (r["dur"] / 60.0) if len(r) else pd.Series(dtype=float)
        tag = "tachy" if cond == Condition.TACHYCARDIA else "hypox"
        R[f"{tag}-resolved-pct"] = fpct(n_res, len(keys))
        R[f"{tag}-gapreset-n"] = fint(n_gap)
        R[f"{tag}-gapreset-pct"] = fpct(n_gap, len(keys))
        R[f"{tag}-open-pct"] = fpct(n_open, len(keys))
        R[f"{tag}-dur-median"] = fnum(q(durs, 50), 0)
        R[f"{tag}-dur-iqr"] = f"{fnum(q(durs,25),0)}--{fnum(q(durs,75),0)}"
        body.append(f"{COND_NAME[cond]} & {fint(len(keys))} & {fint(n_res)} ({fpct(n_res,len(keys))}) & "
                    f"{fint(n_gap)} ({fpct(n_gap,len(keys))}) & {fint(n_open)} ({fpct(n_open,len(keys))}) & {med_iqr(durs)} \\\\")
        J[f"episodes_{cond.value}"] = dict(alerts=len(keys), resolved=n_res, gap_reset=n_gap, open=n_open)
    body += ["\\bottomrule", "\\end{tabular}"]
    write_table("tab_episodes.tex", "\n".join(body) + "\n")
    R["gap-resets-any"] = fint(sum(eng.gap_resets_any.values()))
    R["gap-resets-alerted"] = fint(sum(eng.gap_resets.values()))

    # ------------------------------------------------------------------ 4
    body = ["\\begin{tabular}{llrr}", "\\toprule",
            "Condition & Confirmer set & \\shortstack{Confirmed,\\\\n (\\%)} & \\shortstack{Confirmed only by\\\\this signal, n} \\\\",
            "\\midrule"]
    for cond, names in CONF_BY_COND.items():
        a = A[A["cond"] == cond]
        n = len(a)
        tag = "tachy" if cond == Condition.TACHYCARDIA else "hypox"
        first = True
        for size in (1, 2, 3):
            for combo in itertools.combinations(names, size):
                m = a[list(combo)].any(axis=1)
                if size == 1:
                    others = [x for x in names if x not in combo]
                    uniq = int((a[combo[0]] & ~a[others].any(axis=1)).sum())
                    lab = PRETTY[combo[0]] + (" (reduced configuration)" if combo[0] == REDUCED[cond] else "")
                    R[f"{tag}-only-{combo[0].lower().replace('_','')}"] = fint(uniq)
                    R[f"{tag}-alone-{combo[0].lower().replace('_','')}"] = fpct(m.sum(), n)
                    extra = fint(uniq)
                else:
                    lab = " + ".join(PRETTY[x].split(" (")[0] for x in combo)
                    if size == 3:
                        lab = "All three (extended configuration)"
                    extra = "---"
                cname = COND_NAME[cond] if first else ""
                first = False
                body.append(f"{cname} & {lab} & {fint(m.sum())} ({fpct(m.sum(), n)}) & {extra} \\\\")
        # verificacao C552: reduzida = confirmador de referencia isolado
        R[f"{tag}-reduced-check"] = fpct(a[REDUCED[cond]].sum(), n)
        R[f"{tag}-n-sustained-ge2"] = fpct((a[names].sum(axis=1) >= 2).sum(), n)
        body.append("\\midrule")
    body[-1] = "\\bottomrule"
    body.append("\\end{tabular}")
    write_table("tab_ablation.tex", "\n".join(body) + "\n")

    X = indices(df, cfg)
    if len(X) > 1_000_000:
        X = X.sample(1_000_000, random_state=42)
    cols = ["SHOCK_INDEX", "FC_SPO2_RATIO", "PAS_SPO2_RATIO", "RR_SPO2_RATIO", "PAS_X_SPO2"]
    base = ["HR", "SpO2", "SBP", "RR"]
    rho = X[base + cols].corr(method="spearman", min_periods=30)
    body = ["\\begin{tabular}{lrrrr}", "\\toprule",
            "Index & HR & SpO$_2$ & SBP & RR \\\\", "\\midrule"]
    for c in cols:
        body.append(PRETTY[c].split(" (")[0] + " & " +
                    " & ".join(fnum(rho.loc[c, b], 2) for b in base) + " \\\\")
    body += ["\\bottomrule", "\\end{tabular}"]
    write_table("tab_corr.tex", "\n".join(body) + "\n")
    R["rho-fcspo2-hr"] = fnum(rho.loc["FC_SPO2_RATIO", "HR"], 2)
    R["rho-fcspo2-spo2"] = fnum(rho.loc["FC_SPO2_RATIO", "SpO2"], 2)
    R["rho-pasxspo2-spo2"] = fnum(rho.loc["PAS_X_SPO2", "SpO2"], 2)
    R["rho-pasxspo2-sbp"] = fnum(rho.loc["PAS_X_SPO2", "SBP"], 2)
    R["rho-si-hr"] = fnum(rho.loc["SHOCK_INDEX", "HR"], 2)
    R["rho-rrspo2-rr"] = fnum(rho.loc["RR_SPO2_RATIO", "RR"], 2)
    R["rho-passpo2-sbp"] = fnum(rho.loc["PAS_SPO2_RATIO", "SBP"], 2)
    J["spearman"] = rho.round(3).to_dict()

    # ------------------------------------------------------------- exemplos
    EXAMPLES = [("ex1", Condition.TACHYCARDIA, "2025-02-28", 121),
                ("ex2", Condition.TACHYCARDIA, "2025-01-16", 105),
                ("ex3", Condition.HYPOXEMIA, "2024-09-20", 88),
                ("ex4", Condition.HYPOXEMIA, "2025-02-20", 74)]
    ex_found = []
    for key, cond, day, val in EXAMPLES:
        m = A[(A["cond"] == cond) & (pd.to_datetime(A["ts"]).dt.strftime("%Y-%m-%d") == day)
              & (A["value"].round() == val)]
        ok = len(m) > 0
        ex_found.append(f"{key}={'sim' if ok else 'NAO'}"
                        + (f" (confirmado={bool(m['confirmed'].iloc[0])})" if ok else ""))
    R["examples-check"] = "; ".join(ex_found)

    # ------------------------------------------------------------------ 5
    body = ["\\begin{tabular}{lrrr}", "\\toprule",
            "Scenario & \\shortstack{Unconfirmed alerts\\\\upgraded, n} & \\shortstack{Critical alerts,\\\\n (\\%)} & \\shortstack{High-severity\\\\alerts, n (\\%)} \\\\",
            "\\midrule"]
    n = len(A)
    body.append(f"Current rule (confirmation only) & --- & {fint(A['confirmed'].sum())} ({fpct(A['confirmed'].sum(), n)}) & "
                f"{fint((~A['confirmed']).sum())} ({fpct((~A['confirmed']).sum(), n)}) \\\\")
    for i, (lab, sp_thr, hr_thr) in enumerate(DANGER_SCENARIOS):
        danger = (((A["cond"] == Condition.HYPOXEMIA) & (A["value"] < sp_thr)) |
                  ((A["cond"] == Condition.TACHYCARDIA) & (A["value"] >= hr_thr)))
        up = int((danger & ~A["confirmed"]).sum())
        crit = int((danger | A["confirmed"]).sum())
        body.append(f"{lab} & {fint(up)} & {fint(crit)} ({fpct(crit, n)}) & {fint(n-crit)} ({fpct(n-crit, n)}) \\\\")
        R[f"danger{i+1}-upgraded"] = fint(up)
        R[f"danger{i+1}-critical-pct"] = fpct(crit, n)
    hx = A[(A["cond"] == Condition.HYPOXEMIA) & ~A["confirmed"]]
    R["unconf-hypox-lt80"] = fint((hx["value"] < 80).sum())
    R["unconf-hypox-lt85"] = fint((hx["value"] < 85).sum())
    tc = A[(A["cond"] == Condition.TACHYCARDIA) & ~A["confirmed"]]
    R["unconf-tachy-ge150"] = fint((tc["value"] >= 150).sum())
    body += ["\\bottomrule", "\\end{tabular}"]
    write_table("tab_danger.tex", "\n".join(body) + "\n")

    # ------------------------------------------------------------------ 6
    raw_n = normalise(raw_src)
    raw_n["_year"] = raw_n["timestamp"].dt.year
    df["_year"] = df["timestamp"].dt.year
    A["_year"] = pd.to_datetime(A["ts"]).dt.year
    body = ["\\begin{tabular}{lrrrrrrrr}", "\\toprule",
            "Period & Patients & Observations & \\shortstack{Median\\\\interval, min} & \\shortstack{SpO$_2$\\\\missing} & \\shortstack{RR\\\\missing} & "
            "\\shortstack{Alerts per\\\\1,000 obs.} & \\shortstack{Tachycardia\\\\confirmed} & \\shortstack{Hypoxaemia\\\\confirmed} \\\\", "\\midrule"]
    per_rows = []
    for (y0, y1) in PERIODS:
        dm = df[(df["_year"] >= y0) & (df["_year"] <= y1)]
        am = A[(A["_year"] >= y0) & (A["_year"] <= y1)]
        rm = raw_n[(raw_n["_year"] >= y0) & (raw_n["_year"] <= y1)]
        if dm.empty:
            continue
        dd = dm.sort_values(["patient_id", "timestamp"], kind="mergesort")
        g = dd.groupby("patient_id")["timestamp"].diff().dt.total_seconds()
        g = g[g > 0]
        at = am[am["cond"] == Condition.TACHYCARDIA]
        ah = am[am["cond"] == Condition.HYPOXEMIA]
        body.append(f"{y0}--{y1} & {fint(dm['patient_id'].nunique())} & {fint(len(dm))} & "
                    f"{fnum(q(g,50)/60,1) if len(g) else '---'} & {fpct(rm['SPO2'].isna().sum(), len(rm))} & "
                    f"{fpct(rm['FREQ_RESPIRATORIA'].isna().sum(), len(rm))} & "
                    f"{fnum(1000*len(am)/len(dm),2)} & {fpct(at['confirmed'].sum(), len(at))} & "
                    f"{fpct(ah['confirmed'].sum(), len(ah))} \\\\")
        per_rows.append(dict(period=f"{y0}-{y1}", patients=int(dm['patient_id'].nunique()),
                             obs=len(dm), alerts=len(am)))
    body += ["\\bottomrule", "\\end{tabular}"]
    write_table("tab_temporal.tex", "\n".join(body) + "\n")
    J["temporal"] = per_rows

    # ------------------------------------------------------------------ saida
    with open(os.path.join(OUT_DIR, "results_macros.tex"), "w", encoding="utf-8") as f:
        f.write("% Gerado automaticamente por analise_revisao.py -- nao editar a mao\n")
        for k_, v_ in R.items():
            f.write(f"\\expandafter\\def\\csname res@{k_}\\endcsname{{{v_}}}\n")
    J["macros"] = R
    with open(os.path.join(OUT_DIR, "analise_revisao.json"), "w", encoding="utf-8") as f:
        json.dump(J, f, indent=2, default=str, ensure_ascii=False)

    print("\nResumo:")
    for k_ in ("raw-obs", "raw-patients", "flow-ts-rows", "flow-dup-rows", "flow-na-rows",
               "n-obs", "n-patients", "n-admissions", "n-alerts", "n-confirmed", "pct-confirmed",
               "n-resolutions", "tachy-alerts", "tachy-conf-pct", "hypox-alerts", "hypox-conf-pct",
               "red-conf-pct", "examples-check",
               "tachy-reduced-check", "hypox-reduced-check", "pat-median", "pat-iqr",
               "pat-zero-pct", "dup-rows", "gap-resets-alerted"):
        print(f"  {k_:<24} {R.get(k_)}")
    print(f"\nFicheiros escritos em {OUT_DIR}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV)

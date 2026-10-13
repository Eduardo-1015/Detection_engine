"""
Preparacao do dataset antes de correr o ICU Alert Engine.

Passos (aplicados por esta ordem, com contagem do que e removido em cada um):
  1. Timestamps invalidos: remove linhas cujo RTDATA_TIME nao e uma data
     valida ou esta fora do periodo do estudo (2008-01-01 a 2025-12-31),
     p.ex. as linhas com data 1970-01-01.
  2. Duplicados: remove linhas com o mesmo PATIENT_ID e RTDATA_TIME,
     mantendo a primeira ocorrencia na ordem original do ficheiro
     (regra deterministica).
  3. Valores em falta (dropna): remove linhas sem valor em qualquer uma das
     quatro variaveis usadas pelo motor (FC, SpO2, PAS, FR).

Uso como modulo (demo.py e analise_revisao.py):
    from preparar_dados import carregar_limpo
    df, relatorio = carregar_limpo("../DATA_FINAL.csv")

Uso na linha de comandos (grava ../DATA_FINAL_LIMPO.csv):
    python3 preparar_dados.py
"""
from __future__ import annotations

import os
import sys

import pandas as pd

VARIAVEIS = ["FREQ_CARDIACA", "SPO2", "PRESSAO_ARTERIAL_SISTOLICA", "FREQ_RESPIRATORIA"]
INICIO = pd.Timestamp("2008-01-01")
FIM = pd.Timestamp("2025-12-31 23:59:59")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_ORIGINAL = os.path.join(os.path.dirname(SCRIPT_DIR), "DATA_FINAL.csv")
CSV_LIMPO = os.path.join(os.path.dirname(SCRIPT_DIR), "DATA_FINAL_LIMPO.csv")


def ler_csv(path: str) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    dtype = {c: str for c in ("PATIENT_ID", "patient_id", "ADMISSION_ID") if c in header.columns}
    return pd.read_csv(path, dtype=dtype or None, low_memory=False)


def _estado(df: pd.DataFrame, pid: str, adm: str | None) -> dict:
    return {
        "linhas": int(len(df)),
        "doentes": int(df[pid].nunique()),
        "internamentos": int(df[adm].nunique()) if adm else None,
    }


def limpar(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Aplica os tres passos e devolve (df_limpo, relatorio)."""
    pid = "PATIENT_ID" if "PATIENT_ID" in df.columns else "patient_id"
    ts = "RTDATA_TIME" if "RTDATA_TIME" in df.columns else "timestamp"
    adm = "ADMISSION_ID" if "ADMISSION_ID" in df.columns else None

    df = df.copy()
    df[pid] = df[pid].astype(str)
    for c in VARIAVEIS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["_ts"] = pd.to_datetime(df[ts], errors="coerce")

    rel = [dict(passo="Ficheiro original", removidas=0, **_estado(df, pid, adm))]

    n0 = len(df)
    df = df[df["_ts"].notna() & (df["_ts"] >= INICIO) & (df["_ts"] <= FIM)]
    rel.append(dict(passo="Timestamps invalidos ou fora de 2008-2025",
                    removidas=n0 - len(df), **_estado(df, pid, adm)))

    n0 = len(df)
    df = df.drop_duplicates(subset=[pid, "_ts"], keep="first")
    rel.append(dict(passo="Duplicados (mesmo doente e timestamp)",
                    removidas=n0 - len(df), **_estado(df, pid, adm)))

    n0 = len(df)
    presentes = [c for c in VARIAVEIS if c in df.columns]
    df = df.dropna(subset=presentes)
    rel.append(dict(passo="Valores em falta em FC, SpO2, PAS ou FR (dropna)",
                    removidas=n0 - len(df), **_estado(df, pid, adm)))

    df = (df.sort_values([pid, "_ts"], kind="mergesort")
            .drop(columns="_ts")
            .reset_index(drop=True))
    return df, rel


def carregar_limpo(path: str = CSV_ORIGINAL) -> tuple[pd.DataFrame, list[dict]]:
    return limpar(ler_csv(path))


def imprimir(rel: list[dict]) -> None:
    print(f"\n{'Passo':<52}{'Removidas':>11}{'Linhas':>11}{'Doentes':>9}{'Intern.':>9}")
    for r in rel:
        print(f"{r['passo']:<52}{r['removidas']:>11,}{r['linhas']:>11,}{r['doentes']:>9,}"
              f"{(r['internamentos'] or 0):>9,}")


if __name__ == "__main__":
    origem = sys.argv[1] if len(sys.argv) > 1 else CSV_ORIGINAL
    df, rel = carregar_limpo(origem)
    imprimir(rel)
    df.to_csv(CSV_LIMPO, index=False)
    print(f"\nFicheiro limpo gravado em {CSV_LIMPO}")

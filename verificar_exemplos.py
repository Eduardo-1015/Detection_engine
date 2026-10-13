"""
Verifica os 4 exemplos citados nas secções 4.4 e 4.5 do artigo, usando os
dados preparados (preparar_dados.py).

Para cada exemplo mostra: doente, hora do alerta, valor principal,
intervalo entre as duas leituras, classificação com 3 confirmadores e com
1 confirmador, e o valor e o estado (em breach sustentado ou não) de cada
confirmador no momento do alerta.

Para ser rápido, só processa os registos do dia do exemplo e do dia
anterior. Como o motor faz reset a qualquer intervalo acima de 15 minutos,
isto não altera o resultado.

Uso:  python3 verificar_exemplos.py
"""
import pandas as pd

from icu_alert_engine import AlertConfig, AlertEngine, Condition, EventKind
from preparar_dados import carregar_limpo, CSV_ORIGINAL
from analise_revisao import CONF_BY_COND, CONF_KEYS, REDUCED

EXEMPLOS = [
    ("Taquicardia confirmada", Condition.TACHYCARDIA, "2025-02-28", 121),
    ("Taquicardia nao confirmada", Condition.TACHYCARDIA, "2025-01-16", 105),
    ("Hipoxemia confirmada", Condition.HYPOXEMIA, "2024-09-20", 88),
    ("Hipoxemia SpO2 74%", Condition.HYPOXEMIA, "2025-02-20", 74),
]


class Motor(AlertEngine):
    """Guarda, em cada alerta, o valor e o estado de todos os confirmadores."""
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.registos = []

    def _update(self, patient_id, timestamp, condition, value, *args, **kwargs):
        emitted = super()._update(patient_id, timestamp, condition, value, *args, **kwargs)
        for ev in emitted:
            if ev.kind == EventKind.ALERT:
                conf = {}
                for nome in CONF_BY_COND[condition]:
                    cs = self._state.get(patient_id, {}).get(CONF_KEYS[nome])
                    conf[nome] = (None if cs is None else cs.last_value,
                                  bool(cs is not None and cs.sustained))
                self.registos.append((ev, conf))
        return emitted


def main():
    print("A preparar os dados ...")
    df, _ = carregar_limpo(CSV_ORIGINAL)
    df["_ts"] = pd.to_datetime(df["RTDATA_TIME"])

    for titulo, cond, dia, valor in EXEMPLOS:
        d = pd.Timestamp(dia)
        sub = df[(df["_ts"] >= d - pd.Timedelta(days=1)) & (df["_ts"] < d + pd.Timedelta(days=1))]
        eng = Motor(AlertConfig(), on_alert=lambda e: None)
        eng.process_dataframe(sub.drop(columns="_ts"))
        encontrados = [(ev, c) for ev, c in eng.registos
                       if ev.condition == cond and ev.timestamp.strftime("%Y-%m-%d") == dia
                       and round(ev.value) == valor]
        print("\n" + "=" * 72)
        print(f"{titulo}  ({dia}, valor {valor})")
        if not encontrados:
            print("  NAO ENCONTRADO nos dados preparados")
            continue
        for ev, c in encontrados:
            ref = REDUCED[cond]
            print(f"  Doente: {ev.patient_id}   hora do alerta: {ev.timestamp}")
            print(f"  Valor principal (pior): {ev.value:.1f}   leituras: {ev.consecutive_samples}"
                  f"   intervalo: {ev.duration_seconds/60:.1f} min")
            print(f"  3 confirmadores: {'CONFIRMADO (critical)' if ev.confirmed else 'nao confirmado (high)'}"
                  f"   registado: {ev.confirmer_name}")
            print(f"  1 confirmador ({ref}): "
                  f"{'CONFIRMADO (critical)' if c[ref][1] else 'nao confirmado (high)'}")
            for nome, (v, sust) in c.items():
                vtxt = "sem valor" if v is None else f"{v:.2f}"
                print(f"     {nome:<16} valor {vtxt:<10} em breach sustentado: {'sim' if sust else 'nao'}")


if __name__ == "__main__":
    main()

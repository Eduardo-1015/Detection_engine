"""Demo: limpa o DATA_FINAL.csv (preparar_dados.py), corre o motor e
imprime os alertas.
Dispara apenas os dois alarmes primarios (TACHYCARDIA, HYPOXEMIA).
Para cada alerta, mostra se algum dos confirmadores associados estava em
breach sustentado no momento do disparo.
"""
import json
import os
from icu_alert_engine import AlertConfig, AlertEngine, EventKind, Condition
from preparar_dados import carregar_limpo, imprimir


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
CSV = os.path.join(PARENT_DIR, "DATA_FINAL.csv")


# Em producao, este callback enviaria push/MQTT/webhook.
push_log = []
def on_alert(event):
    push_log.append(event.to_push_payload())

engine = AlertEngine(
    AlertConfig(
        hr_high_bpm=100,
        spo2_low_pct=90,
        shock_index_high=1.0,
        rr_spo2_ratio_high=0.30,
        persistence_samples=2,
        max_gap_seconds=900,
    ),
    on_alert=on_alert,
)

df, relatorio = carregar_limpo(CSV)
imprimir(relatorio)
events = engine.process_dataframe(df)

print("\n" + "=" * 70)
print(f"Demo concluida: {len(events)} eventos emitidos.")
print("=" * 70)

print("\nPayloads enviados para push:\n")
for p in push_log:
    print(json.dumps(p, indent=2, default=str, ensure_ascii=False))
    print("-" * 60)

n_alerts = sum(1 for e in events if e.kind == EventKind.ALERT)
n_confirmed = sum(1 for e in events if e.kind == EventKind.ALERT and e.confirmed)
n_res = sum(1 for e in events if e.kind == EventKind.RESOLUTION)

print("\nResumo:")
print(f"  Alarmes:               {n_alerts}")
print(f"     - Confirmados:      {n_confirmed}")
print(f"     - Sem confirmacao:  {n_alerts - n_confirmed}")
print(f"  Resolucoes:            {n_res}")

print("\nPor condicao:")
from collections import Counter
by_cond = Counter(
    (e.condition.value, e.kind.value, "confirmado" if (e.kind == EventKind.ALERT and e.confirmed) else "-")
    for e in events
)
for (cond, kind, conf), n in sorted(by_cond.items()):
    print(f"  {cond:<12} {kind:<11} {conf:<12} {n}")

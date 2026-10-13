"""
ICU Alert Engine
================

Motor de detecao de eventos clinicos em UCI a partir de sinais vitais continuos.

Filosofia de alarme
-------------------
O motor dispara DOIS tipos de alarme primario, ambos baseados numa unica
variavel vital. A "sustentacao" e definida em **leituras consecutivas** (e
nao em segundos), o que torna o motor adequado a dados de UCI tipicamente
amostrados a cada 5 minutos -- mas tambem funciona em fluxo continuo
(basta aumentar o numero de leituras exigidas).

    - TACHYCARDIA  : FREQ_CARDIACA > 100 bpm em >= N leituras consecutivas
    - HYPOXEMIA    : SPO2 < 90 % em >= N leituras consecutivas

Default: N = 2 (= ~10 min de evidencia para amostragem 5/5 min).

Estes alarmes podem ainda ser "confirmados" por dois indices compostos que
correm em background sem disparar eventos proprios:

    - Shock Index (FC/PAS)      confirma TACHYCARDIA
    - Racio FR/SpO2             confirma HYPOXEMIA

Um alarme primario fica marcado como `confirmed=True` quando, no instante em
que dispara, o confirmador correspondente esta ele proprio em breach
sustentado (>= 30 s acima do seu limiar). Sem PAS, o confirmador de Shock
Index simplesmente nao avalia -- a TACHYCARDIA dispara na mesma, apenas nao
confirmada. O mesmo para FR.

Entrada
-------
CSV ou pandas.DataFrame com colunas:
    timestamp                    (ISO 8601 ou parseavel por pandas.to_datetime)
    patient_id                   (str)
    FREQ_CARDIACA                (bpm,   pode ser NaN)
    SPO2                         (%,     pode ser NaN)
    PRESSAO_ARTERIAL_SISTOLICA   (mmHg,  opcional -- alimenta o confirmador de SI)
    FREQ_RESPIRATORIA            (rpm,   opcional -- alimenta o confirmador de FR/SpO2)

Uso minimo
----------
    >>> engine = AlertEngine()
    >>> events = engine.process_csv("vitals.csv")
    >>> for e in events: print(e)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Callable, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("icu_alerts")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Tipos publicos
# ---------------------------------------------------------------------------
class Condition(str, Enum):
    """So as duas condicoes que disparam alarme."""
    TACHYCARDIA = "TACHYCARDIA"
    HYPOXEMIA = "HYPOXEMIA"


class EventKind(str, Enum):
    ALERT = "ALERT"
    RESOLUTION = "RESOLUTION"


# Chaves internas para os confirmadores (estado mantido lado a lado com as
# condicoes mas sem emitir eventos publicos).
_CONFIRMER_SI     = "_CONF_SHOCK_INDEX"
_CONFIRMER_RDR    = "_CONF_RR_SPO2_RATIO"
_CONFIRMER_FCSPO2 = "_CONF_FC_SPO2_RATIO"   # FC/SpO2 -- distress combinado
_CONFIRMER_PASPO2 = "_CONF_PAS_SPO2_RATIO"  # PAS/SpO2 -- hipotensao relativa
_CONFIRMER_PAMSPO2 = "_CONF_PAS_X_SPO2"     # PAS x SpO2 -- compromisso combinado


# Aliases de colunas suportados em CSVs/DataFrames clinicos.
# A chave e o nome alternativo que pode aparecer no ficheiro;
# o valor e o nome canonico que o motor usa internamente.
_COLUMN_ALIASES = {
    "PATIENT_ID":  "patient_id",
    "RTDATA_TIME": "timestamp",
}


@dataclass(frozen=True)
class AlertEvent:
    """Evento emitido pelo motor. Pode ser ligado a push/MQTT/webhook.

    Campos de confirmacao (so preenchidos em eventos ALERT):
        confirmed              -- True se o confirmador estava em breach sustentado
        confirmer_name         -- "SHOCK_INDEX" ou "RR_SPO2_RATIO"
        confirmer_value        -- valor atual do indice no momento do alerta
        confirmer_threshold    -- limiar do indice
    """
    kind: EventKind
    condition: Condition
    patient_id: str
    started_at: datetime          # inicio do periodo de breach
    timestamp: datetime           # instante em que o evento foi emitido
    duration_seconds: float       # duracao do breach ate `timestamp`
    consecutive_samples: int = 0  # numero de leituras consecutivas em breach
    value: float = 0.0            # valor do alarme primario (pior durante o breach)
    threshold: float = 0.0        # limiar do alarme primario
    message: str = ""
    confirmed: bool = False
    confirmer_name: Optional[str] = None
    confirmer_value: Optional[float] = None
    confirmer_threshold: Optional[float] = None

    def to_push_payload(self) -> dict:
        """Payload pronto a entregar a um servico de push/notificacao."""
        payload = {
            "patient_id": self.patient_id,
            "type": f"{self.condition.value}_{self.kind.value}",
            "timestamp": self.timestamp.isoformat(),
            "duration_s": round(self.duration_seconds, 1),
            "duration_min": round(self.duration_seconds / 60.0, 1),
            "consecutive_samples": self.consecutive_samples,
            "value": self.value,
            "threshold": self.threshold,
            "title": self._title(),
            "body": self.message,
            "severity": self._severity(),
        }
        if self.kind == EventKind.ALERT:
            payload["confirmed"] = self.confirmed
            if self.confirmer_name is not None:
                payload["confirmer"] = {
                    "name": self.confirmer_name,
                    "value": self.confirmer_value,
                    "threshold": self.confirmer_threshold,
                }
        return payload

    def _severity(self) -> str:
        if self.kind == EventKind.RESOLUTION:
            return "info"
        return "critical" if self.confirmed else "high"

    def _title(self) -> str:
        pt = {
            Condition.TACHYCARDIA: "Taquicardia",
            Condition.HYPOXEMIA: "Hipoxemia",
        }[self.condition]
        if self.kind == EventKind.RESOLUTION:
            return f"Resolvido: {pt} - doente {self.patient_id}"
        suffix = " CONFIRMADA" if self.confirmed else ""
        return f"ALERTA: {pt} sustentada{suffix} - doente {self.patient_id}"


# ---------------------------------------------------------------------------
# Configuracao
# ---------------------------------------------------------------------------
@dataclass
class AlertConfig:
    # Limiares dos alarmes primarios
    hr_high_bpm: float = 100.0
    spo2_low_pct: float = 90.0

    # Limiares dos confirmadores (silenciosos)
    shock_index_high: float = 1.0       # SI = FC/PAS; > 1.0 sugere choque
    rr_spo2_ratio_high: float = 0.30    # FR/SpO2; > 0.30 sugere distress respiratorio

    # Persistencia minima: numero de LEITURAS CONSECUTIVAS em breach antes
    # de disparar (mesma regra para primarios e confirmadores).
    # Para amostragem 5/5 min, 2 leituras = ~10 min de evidencia.
    persistence_samples: int = 2

    # Bounds fisiologicos -- valores fora destes intervalos sao tratados
    # como artefactos de sensor e ignorados (nem disparam nem cancelam).
    hr_valid_min: float = 20.0
    hr_valid_max: float = 250.0
    spo2_valid_min: float = 50.0
    spo2_valid_max: float = 100.0
    sbp_valid_min: float = 30.0
    sbp_valid_max: float = 250.0
    rr_valid_min: float = 4.0
    rr_valid_max: float = 60.0
    # Novos confirmadores
    fc_spo2_ratio_high: float = 1.10     # FC/SpO2 > 1.10 indica distress combinado
    pas_spo2_ratio_low: float = 1.0      # PAS/SpO2 < 1.0 indica hipotensao relativa
    pas_x_spo2_low: float = 9000.0       # PAS x SpO2 < 9000 indica compromisso combinado

    # Bounds dos indices derivados (sanidade pos-calculo)
    si_valid_min: float = 0.1
    si_valid_max: float = 5.0
    rdr_valid_min: float = 0.04
    rdr_valid_max: float = 1.5
    fc_spo2_valid_min: float = 0.5
    fc_spo2_valid_max: float = 3.0
    pas_spo2_valid_min: float = 0.3
    pas_spo2_valid_max: float = 3.0
    pas_x_spo2_valid_min: float = 2000.0
    pas_x_spo2_valid_max: float = 30000.0

    # Tolerancia para "gaps" no sinal: se o intervalo entre duas amostras
    # consecutivas (validas) exceder isto, considera-se que houve perda de
    # monitorizacao e o estado de breach e reiniciado silenciosamente.
    # Default 15 min (= 900s): tolera uma leitura em falta em amostragem 5/5min.
    max_gap_seconds: float = 900.0


# ---------------------------------------------------------------------------
# Estado interno
# ---------------------------------------------------------------------------
@dataclass
class _BreachState:
    breach_start: Optional[datetime] = None  # ts da primeira leitura do breach actual
    consecutive_count: int = 0               # nr de leituras consecutivas em breach
    last_valid_ts: Optional[datetime] = None
    last_value: Optional[float] = None
    peak_value: Optional[float] = None       # pior valor durante o breach
    sustained: bool = False                  # consecutive_count >= persistence_samples
    alert_fired: bool = False                # True se o ALERT primario ja foi emitido
    # Snapshot capturado no momento da resolucao (para construir o evento)
    last_episode_start: Optional[datetime] = None
    last_episode_end: Optional[datetime] = None
    last_episode_peak: Optional[float] = None
    last_episode_count: int = 0


# ---------------------------------------------------------------------------
# Motor
# ---------------------------------------------------------------------------
AlertCallback = Callable[[AlertEvent], None]


def _default_callback(event: AlertEvent) -> None:
    """Default: regista no logger. Substituir em producao por push/MQTT/etc."""
    level = logging.WARNING if event.kind == EventKind.ALERT else logging.INFO
    logger.log(
        level,
        "%s | patient=%s | value=%.1f (limiar=%.1f) | duracao=%.1fs",
        event._title(),
        event.patient_id,
        event.value,
        event.threshold,
        event.duration_seconds,
    )


class AlertEngine:
    """Motor de detecao com state machine por (doente, condicao)."""

    def __init__(
        self,
        config: Optional[AlertConfig] = None,
        on_alert: Optional[AlertCallback] = None,
    ) -> None:
        self.config = config or AlertConfig()
        self.on_alert: AlertCallback = on_alert or _default_callback
        # state[patient_id][key] -> _BreachState
        # key e Condition (alarmes primarios) ou string (confirmadores silenciosos)
        self._state: dict[str, dict[object, _BreachState]] = {}
        self.events: list[AlertEvent] = []   # historico (so eventos primarios)

    # ------------------------------------------------------------------
    # API publica
    # ------------------------------------------------------------------
    def process_sample(
        self,
        patient_id: str,
        timestamp: datetime,
        FREQ_CARDIACA: Optional[float] = None,
        SPO2: Optional[float] = None,
        PRESSAO_ARTERIAL_SISTOLICA: Optional[float] = None,
        FREQ_RESPIRATORIA: Optional[float] = None,
    ) -> list[AlertEvent]:
        """Processa uma leitura. Devolve eventos primarios (ALERT/RESOLUTION).

        Os confirmadores (Shock Index e FR/SpO2) sao actualizados em silencio
        e o seu estado e consultado quando um alarme primario dispara, para
        marcar `confirmed=True` no AlertEvent.
        """
        # ---- 1) Confirmadores silenciosos (sem emissao) ----
        # Shock Index (FC/PAS) -- existente
        shock_index = self._safe_ratio(
            FREQ_CARDIACA, PRESSAO_ARTERIAL_SISTOLICA,
            num_bounds=(self.config.hr_valid_min, self.config.hr_valid_max),
            den_bounds=(self.config.sbp_valid_min, self.config.sbp_valid_max),
        )
        self._track_silent(
            patient_id, timestamp, _CONFIRMER_SI, shock_index,
            self.config.si_valid_min, self.config.si_valid_max,
            lambda v: v > self.config.shock_index_high,
            higher_is_worse=True,
        )

        # FR/SpO2 -- existente
        rr_spo2 = self._safe_ratio(
            FREQ_RESPIRATORIA, SPO2,
            num_bounds=(self.config.rr_valid_min, self.config.rr_valid_max),
            den_bounds=(self.config.spo2_valid_min, self.config.spo2_valid_max),
        )
        self._track_silent(
            patient_id, timestamp, _CONFIRMER_RDR, rr_spo2,
            self.config.rdr_valid_min, self.config.rdr_valid_max,
            lambda v: v > self.config.rr_spo2_ratio_high,
            higher_is_worse=True,
        )

        # FC/SpO2 -- novo: sobe quando FC sobe e/ou SpO2 desce
        fc_spo2 = self._safe_ratio(
            FREQ_CARDIACA, SPO2,
            num_bounds=(self.config.hr_valid_min, self.config.hr_valid_max),
            den_bounds=(self.config.spo2_valid_min, self.config.spo2_valid_max),
        )
        self._track_silent(
            patient_id, timestamp, _CONFIRMER_FCSPO2, fc_spo2,
            self.config.fc_spo2_valid_min, self.config.fc_spo2_valid_max,
            lambda v: v > self.config.fc_spo2_ratio_high,
            higher_is_worse=True,
        )

        # PAS/SpO2 -- novo: valor BAIXO indica hipotensao relativa (inverso)
        pas_spo2 = self._safe_ratio(
            PRESSAO_ARTERIAL_SISTOLICA, SPO2,
            num_bounds=(self.config.sbp_valid_min, self.config.sbp_valid_max),
            den_bounds=(self.config.spo2_valid_min, self.config.spo2_valid_max),
        )
        self._track_silent(
            patient_id, timestamp, _CONFIRMER_PASPO2, pas_spo2,
            self.config.pas_spo2_valid_min, self.config.pas_spo2_valid_max,
            lambda v: v < self.config.pas_spo2_ratio_low,
            higher_is_worse=False,
        )

        # PAS x SpO2 -- novo: produto; valor BAIXO indica compromisso combinado
        pas_x_spo2: Optional[float] = None
        if (PRESSAO_ARTERIAL_SISTOLICA is not None and SPO2 is not None):
            try:
                if not (pd.isna(PRESSAO_ARTERIAL_SISTOLICA) or pd.isna(SPO2)):
                    sbp_f = float(PRESSAO_ARTERIAL_SISTOLICA)
                    spo2_f = float(SPO2)
                    if (self.config.sbp_valid_min <= sbp_f <= self.config.sbp_valid_max
                            and self.config.spo2_valid_min <= spo2_f <= self.config.spo2_valid_max):
                        pas_x_spo2 = sbp_f * spo2_f
            except (TypeError, ValueError):
                pass
        self._track_silent(
            patient_id, timestamp, _CONFIRMER_PAMSPO2, pas_x_spo2,
            self.config.pas_x_spo2_valid_min, self.config.pas_x_spo2_valid_max,
            lambda v: v < self.config.pas_x_spo2_low,
            higher_is_worse=False,
        )

        # ---- 2) Alarmes primarios (emissao + consulta de confirmadores) ----
        # Logica OR: basta um confirmador estar em breach sustentado para confirmar.
        emitted: list[AlertEvent] = []
        emitted += self._update(
            patient_id, timestamp, Condition.TACHYCARDIA, FREQ_CARDIACA,
            self.config.hr_valid_min, self.config.hr_valid_max,
            lambda v: v > self.config.hr_high_bpm,
            self.config.hr_high_bpm, higher_is_worse=True,
            confirmers=[
                (_CONFIRMER_SI,     "SHOCK_INDEX",    self.config.shock_index_high),
                (_CONFIRMER_FCSPO2, "FC_SPO2_RATIO",  self.config.fc_spo2_ratio_high),
                (_CONFIRMER_PASPO2, "PAS_SPO2_RATIO", self.config.pas_spo2_ratio_low),
            ],
        )
        emitted += self._update(
            patient_id, timestamp, Condition.HYPOXEMIA, SPO2,
            self.config.spo2_valid_min, self.config.spo2_valid_max,
            lambda v: v < self.config.spo2_low_pct,
            self.config.spo2_low_pct, higher_is_worse=False,
            confirmers=[
                (_CONFIRMER_RDR,    "RR_SPO2_RATIO", self.config.rr_spo2_ratio_high),
                (_CONFIRMER_FCSPO2, "FC_SPO2_RATIO", self.config.fc_spo2_ratio_high),
                (_CONFIRMER_PAMSPO2,"PAS_X_SPO2",    self.config.pas_x_spo2_low),
            ],
        )
        return emitted

    def process_dataframe(self, df: pd.DataFrame) -> list[AlertEvent]:
        """Processa um DataFrame ordenado por timestamp.

        Aceita os nomes canonicos `timestamp` e `patient_id`, mas tambem os
        nomes que aparecem em ficheiros clinicos reais: `RTDATA_TIME` (sera
        renomeado para `timestamp`) e `PATIENT_ID` (sera renomeado para
        `patient_id`). Colunas adicionais (PH, HCO3, TROPONINA, DESCRICAO,
        etc.) sao ignoradas em silencio.
        """
        df = df.copy()

        # Aplicar aliases de colunas conhecidos (case-sensitive, so renomeia se
        # o alvo nao existir ja para evitar colisoes).
        rename_map = {
            src: tgt for src, tgt in _COLUMN_ALIASES.items()
            if src in df.columns and tgt not in df.columns
        }
        if rename_map:
            df = df.rename(columns=rename_map)

        required = {"timestamp", "patient_id"}
        missing = required - set(df.columns)
        if missing:
            aliases_str = ", ".join(f"{s} -> {t}" for s, t in _COLUMN_ALIASES.items())
            raise ValueError(
                f"Colunas em falta: {missing}. Esperadas: 'timestamp', "
                f"'patient_id'. Aliases suportados: {aliases_str}."
            )

        # Garantir que patient_id e sempre string (preserva IDs gigantes como
        # 1220000000003067552 que de outra forma o pandas poderia ler como float).
        df["patient_id"] = df["patient_id"].astype(str)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        emitted: list[AlertEvent] = []
        for row in df.itertuples(index=False):
            ts = row.timestamp.to_pydatetime() if hasattr(row.timestamp, "to_pydatetime") else row.timestamp
            fc = getattr(row, "FREQ_CARDIACA", None)
            spo2 = getattr(row, "SPO2", None)
            pas = getattr(row, "PRESSAO_ARTERIAL_SISTOLICA", None)
            fr = getattr(row, "FREQ_RESPIRATORIA", None)
            emitted += self.process_sample(row.patient_id, ts, fc, spo2, pas, fr)
        return emitted

    def process_csv(self, path: str) -> list[AlertEvent]:
        # Detectar qual variante do ID esta presente para a forcar como
        # string (preserva IDs gigantes como 1220000000003067552).
        header = pd.read_csv(path, nrows=0)
        dtype = {}
        for col in ("PATIENT_ID", "patient_id"):
            if col in header.columns:
                dtype[col] = str
        df = pd.read_csv(path, dtype=dtype or None)
        return self.process_dataframe(df)

    def reset(self) -> None:
        self._state.clear()
        self.events.clear()

    # ------------------------------------------------------------------
    # Interno
    # ------------------------------------------------------------------
    def _state_for(self, patient_id: str, key) -> _BreachState:
        return self._state.setdefault(patient_id, {}).setdefault(key, _BreachState())

    @staticmethod
    def _safe_ratio(
        num: Optional[float],
        den: Optional[float],
        num_bounds: tuple[float, float],
        den_bounds: tuple[float, float],
    ) -> Optional[float]:
        """Calcula num/den so se ambos forem validos; senao devolve None."""
        for v in (num, den):
            if v is None:
                return None
            try:
                if pd.isna(v):
                    return None
            except (TypeError, ValueError):
                return None
        try:
            num_f, den_f = float(num), float(den)
        except (TypeError, ValueError):
            return None
        if not (num_bounds[0] <= num_f <= num_bounds[1]):
            return None
        if not (den_bounds[0] <= den_f <= den_bounds[1]):
            return None
        if den_f == 0:
            return None
        return num_f / den_f

    def _advance_state(
        self,
        s: _BreachState,
        timestamp: datetime,
        value: Optional[float],
        valid_min: float,
        valid_max: float,
        is_breach: Callable[[float], bool],
        higher_is_worse: bool,
    ) -> Optional[str]:
        """Avanca a maquina de estados para uma leitura. Devolve uma de:
            None        -- nenhuma transicao relevante
            'sustained' -- contagem consecutiva acaba de atingir persistence_samples
            'resolved'  -- valor regressou ao normal apos breach sustentado
        Tambem actualiza last_value, peak_value, etc.
        """
        # Valor invalido -> ignorar (nao avanca contagem, nao reseta)
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            return None
        try:
            v = float(value)
        except (TypeError, ValueError):
            return None
        if not (valid_min <= v <= valid_max):
            return None

        # Gap excessivo entre leituras validas -> reset silencioso
        if s.last_valid_ts is not None:
            gap = (timestamp - s.last_valid_ts).total_seconds()
            if gap > self.config.max_gap_seconds:
                s.breach_start = None
                s.consecutive_count = 0
                s.peak_value = None
                s.sustained = False
                s.alert_fired = False

        s.last_valid_ts = timestamp
        s.last_value = v
        breach = is_breach(v)

        if breach:
            if s.consecutive_count == 0:
                s.breach_start = timestamp
                s.peak_value = v
            else:
                if s.peak_value is None:
                    s.peak_value = v
                elif higher_is_worse and v > s.peak_value:
                    s.peak_value = v
                elif not higher_is_worse and v < s.peak_value:
                    s.peak_value = v
            s.consecutive_count += 1
            if not s.sustained and s.consecutive_count >= self.config.persistence_samples:
                s.sustained = True
                return "sustained"
        else:
            was_sustained = s.sustained
            if was_sustained:
                s.last_episode_start = s.breach_start
                s.last_episode_end = timestamp
                s.last_episode_peak = s.peak_value
                s.last_episode_count = s.consecutive_count
            s.breach_start = None
            s.consecutive_count = 0
            s.peak_value = None
            s.sustained = False
            s.alert_fired = False
            if was_sustained:
                return "resolved"
        return None

    def _track_silent(
        self, patient_id, timestamp, key, value, valid_min, valid_max,
        is_breach, higher_is_worse,
    ) -> None:
        """Tracking de breach sem emissao de eventos (para confirmadores)."""
        s = self._state_for(patient_id, key)
        self._advance_state(s, timestamp, value, valid_min, valid_max, is_breach, higher_is_worse)

    def _update(
        self,
        patient_id: str,
        timestamp: datetime,
        condition: Condition,
        value: Optional[float],
        valid_min: float,
        valid_max: float,
        is_breach: Callable[[float], bool],
        threshold: float,
        higher_is_worse: bool,
        confirmers: Optional[list] = None,  # list of (key, name, threshold)
    ) -> list[AlertEvent]:
        """Processa um alarme primario e emite eventos.

        Quando emite um ALERT, percorre a lista de confirmadores por ordem e
        marca `confirmed=True` no primeiro que estiver em breach sustentado
        (logica OR). O nome, valor e limiar desse confirmador sao registados
        no AlertEvent.
        """
        s = self._state_for(patient_id, condition)
        transition = self._advance_state(
            s, timestamp, value, valid_min, valid_max, is_breach, higher_is_worse
        )
        emitted: list[AlertEvent] = []

        if transition == "sustained" and not s.alert_fired:
            # Percorrer confirmadores -- primeiro em breach sustentado vence
            confirmed = False
            confirmer_name = None
            confirmer_value = None
            confirmer_threshold = None
            for conf_key, conf_name, conf_thr in (confirmers or []):
                cs = self._state.get(patient_id, {}).get(conf_key)
                if cs is not None and cs.sustained:
                    confirmed = True
                    confirmer_name = conf_name
                    confirmer_value = cs.last_value
                    confirmer_threshold = conf_thr
                    break
            duration = (timestamp - s.breach_start).total_seconds()
            peak = s.peak_value if s.peak_value is not None else float(value)
            ev = AlertEvent(
                kind=EventKind.ALERT,
                condition=condition,
                patient_id=patient_id,
                started_at=s.breach_start,
                timestamp=timestamp,
                duration_seconds=duration,
                consecutive_samples=s.consecutive_count,
                value=peak,
                threshold=threshold,
                message=self._format_message(
                    condition, peak, threshold, duration, s.consecutive_count,
                    confirmed, confirmer_name, confirmer_value, confirmer_threshold,
                ),
                confirmed=confirmed,
                confirmer_name=confirmer_name,
                confirmer_value=confirmer_value,
                confirmer_threshold=confirmer_threshold,
            )
            s.alert_fired = True
            emitted.append(ev)
            self._emit(ev)

        elif transition == "resolved":
            # Usa snapshot capturado em _advance_state antes do reset
            start = s.last_episode_start or timestamp
            end = s.last_episode_end or timestamp
            duration = (end - start).total_seconds()
            ev = AlertEvent(
                kind=EventKind.RESOLUTION,
                condition=condition,
                patient_id=patient_id,
                started_at=start,
                timestamp=timestamp,
                duration_seconds=duration,
                consecutive_samples=s.last_episode_count,
                value=s.last_value if s.last_value is not None else 0.0,
                threshold=threshold,
                message=(
                    f"{condition.value} resolvida apos {s.last_episode_count} leituras "
                    f"em breach (~{duration/60:.0f} min). Valor actual {s.last_value:.1f}."
                ),
            )
            emitted.append(ev)
            self._emit(ev)
        return emitted

    def _emit(self, event: AlertEvent) -> None:
        self.events.append(event)
        try:
            self.on_alert(event)
        except Exception:
            logger.exception("Callback on_alert falhou para evento %s", event)

    @staticmethod
    def _format_message(
        condition: Condition,
        value: float,
        threshold: float,
        duration: float,
        consecutive: int,
        confirmed: bool,
        confirmer_name: Optional[str],
        confirmer_value: Optional[float],
        confirmer_threshold: Optional[float],
    ) -> str:
        # Apresentar duracao em min para episodios > 60s (amostragem em min)
        if duration >= 60:
            tempo = f"{duration / 60:.0f} min"
        else:
            tempo = f"{duration:.0f}s"
        leituras = f"{consecutive} leituras consecutivas"
        if condition == Condition.TACHYCARDIA:
            base = (
                f"FC acima de {threshold:.0f} bpm em {leituras} "
                f"(~{tempo}); pico {value:.0f} bpm."
            )
        else:  # HYPOXEMIA
            base = (
                f"SpO2 abaixo de {threshold:.0f}% em {leituras} "
                f"(~{tempo}); minimo {value:.0f}%."
            )
        if confirmed and confirmer_value is not None and confirmer_threshold is not None:
            pretty = {
                "SHOCK_INDEX":    "Shock Index (FC/PAS)",
                "RR_SPO2_RATIO":  "racio FR/SpO2",
                "FC_SPO2_RATIO":  "racio FC/SpO2",
                "PAS_SPO2_RATIO": "racio PAS/SpO2",
                "PAS_X_SPO2":     "produto PAS×SpO2",
            }.get(confirmer_name or "", confirmer_name or "confirmador")
            return (
                f"{base} CONFIRMADA por {pretty} {confirmer_value:.2f} "
                f"(limiar {confirmer_threshold:.2f}). Avaliar com urgencia."
            )
        if confirmer_value is not None and confirmer_threshold is not None:
            pretty = {
                "SHOCK_INDEX":    "Shock Index",
                "RR_SPO2_RATIO":  "FR/SpO2",
                "FC_SPO2_RATIO":  "FC/SpO2",
                "PAS_SPO2_RATIO": "PAS/SpO2",
                "PAS_X_SPO2":     "PAS×SpO2",
            }.get(confirmer_name or "", confirmer_name or "")
            return (
                f"{base} Sem confirmacao por {pretty} (actual {confirmer_value:.2f}, "
                f"limiar {confirmer_threshold:.2f}). Avaliar doente."
            )
        return f"{base} Confirmador indisponivel (faltam dados). Avaliar doente."


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="ICU Alert Engine")
    parser.add_argument("csv", help="CSV com colunas timestamp,patient_id,FREQ_CARDIACA,SPO2,PRESSAO_ARTERIAL_SISTOLICA,FREQ_RESPIRATORIA")
    parser.add_argument("--hr-high", type=float, default=100.0)
    parser.add_argument("--spo2-low", type=float, default=90.0)
    parser.add_argument("--si-high", type=float, default=1.0, help="Limiar do confirmador Shock Index")
    parser.add_argument("--rdr-high", type=float, default=0.30, help="Limiar do confirmador FR/SpO2")
    parser.add_argument("--persistence", type=int, default=2,
                        help="Numero de leituras consecutivas em breach para disparar (default 2)")
    parser.add_argument("--max-gap-min", type=float, default=15.0,
                        help="Gap maximo entre leituras antes de reset (minutos, default 15)")
    parser.add_argument("--json", action="store_true", help="Imprime eventos em JSON")
    args = parser.parse_args()

    cfg = AlertConfig(
        hr_high_bpm=args.hr_high,
        spo2_low_pct=args.spo2_low,
        shock_index_high=args.si_high,
        rr_spo2_ratio_high=args.rdr_high,
        persistence_samples=args.persistence,
        max_gap_seconds=args.max_gap_min * 60.0,
    )
    eng = AlertEngine(cfg)
    evs = eng.process_csv(args.csv)
    if args.json:
        print(json.dumps([e.to_push_payload() for e in evs], indent=2, default=str))
    else:
        n_a = sum(1 for e in evs if e.kind == EventKind.ALERT)
        n_c = sum(1 for e in evs if e.kind == EventKind.ALERT and e.confirmed)
        n_r = sum(1 for e in evs if e.kind == EventKind.RESOLUTION)
        print(f"\n{n_a} alarmes ({n_c} confirmados), {n_r} resolucoes.")

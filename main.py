# streamlit_app.py  
# Run: streamlit run streamlit_app.py  
#  
# Disclaimer:  
# This is a synthetic probabilistic scenario model for exploration and education.  
# It is not a forecast, prophecy, or a substitute for real data, domain expertise,  
# or causal identification.  
 
from __future__ import annotations  
 
import math  
import random  
from dataclasses import asdict, dataclass  
from typing import Dict, List, Tuple, Optional  
 
import pandas as pd  
import altair as alt  
import streamlit as st  
 
 
# -----------------------------  
# Helpers  
# -----------------------------  
def sigmoid(x: float) -> float:  
    # stable sigmoid  
    if x >= 0:  
        z = math.exp(-x)  
        return 1.0 / (1.0 + z)  
    else:  
        z = math.exp(x)  
        return z / (1.0 + z)  
 
 
def softmax(xs: List[float]) -> List[float]:  
    m = max(xs)  
    exps = [math.exp(x - m) for x in xs]  
    s = sum(exps)  
    return [e / s for e in exps]  
 
 
def clamp(x: float, lo: float, hi: float) -> float:  
    return max(lo, min(hi, x))  
 
 
def normal(rng: random.Random, mu: float, sigma: float) -> float:  
    # Box-Muller  
    u1 = max(1e-12, rng.random())  
    u2 = max(1e-12, rng.random())  
    z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)  
    return mu + sigma * z  
 
 
# -----------------------------  
# State definitions  
# -----------------------------  
@dataclass  
class IranInternalState:  
    """  
    Variables are scaled approximately to [-1, +1].  
 
    Interpretation (stylized):  
    - economic_stress: higher => worse economy (stress/constraints)  
    - protest_pressure: higher => more protest/strike pressure  
    - repression: higher => more coercive repression  
    - elite_cohesion: higher => stronger elite cohesion (more stability)  
    - info_blackout: higher => stronger information controls/blackouts  
    """  
    economic_stress: float  
    protest_pressure: float  
    repression: float  
    elite_cohesion: float  
    info_blackout: float  
 
 
@dataclass  
class ExternalActions:  
    """  
    External action intensities, mostly [0,1] except iraq_syria_spillover in [-1,1].  
    """  
    sanctions_enforcement: float  
    escalation_risk: float  
    israel_strike_pressure: float  
    gulf_restraint_pressure: float  
    russia_support: float  
    china_support: float  
    europe_pressure: float  
    turkey_mediation: float  
    iraq_syria_spillover: float  
 
 
# -----------------------------  
# Actor game module (Quantal Response / Logit choice)  
# -----------------------------  
class ActorGame:  
    """  
    Each month, each actor chooses among discrete actions using quantal response:  
 
        P(a) = exp(λ U(a)) / Σ_a' exp(λ U(a'))  
 
    where λ (rationality) controls how close choices are to best-response:  
    - small λ => noisy/mixed actions  
    - large λ => nearly deterministic argmax utility  
    """  
 
    def __init__(self, rng: random.Random, rationality: float = 2.2):  
        self.rng = rng  
        self.lam = rationality  
 
    def _sample_action(self, utilities: Dict[str, float]) -> str:  
        actions = list(utilities.keys())  
        probs = softmax([self.lam * utilities[a] for a in actions])  
        r = self.rng.random()  
        cum = 0.0  
        for a, p in zip(actions, probs):  
            cum += p  
            if r <= cum:  
                return a  
        return actions[-1]  
 
    def step(self, iran: IranInternalState, scenario: Dict[str, float]) -> ExternalActions:  
        """  
        Scenario levers are in [0,1].  
        """  
        us_restraint = scenario.get("us_restraint", 0.55)  
        gulf_anti_war = scenario.get("gulf_anti_war", 0.70)  
        israel_risk = scenario.get("israel_risk", 0.55)  
        russia_alignment = scenario.get("russia_alignment", 0.60)  
        china_oil_priority = scenario.get("china_oil_priority", 0.55)  
        europe_hr_pressure = scenario.get("europe_hr_pressure", 0.60)  
        turkey_balancing = scenario.get("turkey_balancing", 0.55)  
        iraq_syria_instability = scenario.get("iraq_syria_instability", 0.55)  
 
        unrest = clamp(iran.protest_pressure, -1, 1)  
        econ = clamp(iran.economic_stress, -1, 1)  
 
        # --- US choices ---  
        us_utils = {  
            "enforce_sanctions": 0.7 * (0.3 + econ + 0.4 * unrest) + 0.2 * (1 - us_restraint),  
            "deescalate":        0.6 * us_restraint + 0.2 * (1 - israel_risk) - 0.2 * iraq_syria_instability,  
            "escalate_pressure": 0.5 * (1 - us_restraint) + 0.3 * israel_risk - 0.3 * gulf_anti_war,  
        }  
        us_action = self._sample_action(us_utils)  
 
        # --- Israel choices ---  
        isr_utils = {  
            "high_pressure":   0.8 * israel_risk + 0.2 * (1 - gulf_anti_war) + 0.1 * (1 - us_restraint),  
            "covert_pressure": 0.5 * israel_risk + 0.2 * us_restraint + 0.1 * unrest,  
            "cautious":        0.6 * (1 - israel_risk) + 0.3 * gulf_anti_war,  
        }  
        israel_action = self._sample_action(isr_utils)  
 
        # --- GCC choices ---  
        gcc_utils = {  
            "push_restraint": 0.9 * gulf_anti_war + 0.2 * iraq_syria_instability,  
            "hedge":          0.4 + 0.2 * israel_risk + 0.2 * us_restraint,  
            "support_pressure": 0.3 * (1 - gulf_anti_war) + 0.2 * israel_risk,  
        }  
        gcc_action = self._sample_action(gcc_utils)  
 
        # --- Europe/UK choices ---  
        eu_utils = {  
            "sanctions_hr": 0.9 * europe_hr_pressure + 0.2 * unrest,  
            "diplomacy":    0.4 + 0.2 * us_restraint - 0.1 * israel_risk,  
            "low_profile":  0.5 * (1 - europe_hr_pressure) + 0.1 * econ,  
        }  
        eu_action = self._sample_action(eu_utils)  
 
        # --- Russia choices ---  
        ru_utils = {  
            "support_iran": 0.9 * russia_alignment + 0.2 * (1 - us_restraint),  
            "neutral":      0.5 * (1 - russia_alignment) + 0.1 * gulf_anti_war,  
            "tradeoff":     0.4 + 0.2 * russia_alignment + 0.1 * econ,  
        }  
        ru_action = self._sample_action(ru_utils)  
 
        # --- China choices ---  
        cn_utils = {  
            "keep_oil":        0.9 * china_oil_priority + 0.2 * econ,  
            "cautious":        0.5 + 0.2 * us_restraint,  
            "align_sanctions": 0.4 * (1 - china_oil_priority) + 0.2 * (1 - us_restraint),  
        }  
        cn_action = self._sample_action(cn_utils)  
 
        # --- Turkey choices ---  
        tr_utils = {  
            "mediate":       0.8 * turkey_balancing + 0.2 * us_restraint,  
            "hedge":         0.5 + 0.2 * iraq_syria_instability,  
            "pressure_iran": 0.3 * (1 - turkey_balancing) + 0.1 * unrest,  
        }  
        tr_action = self._sample_action(tr_utils)  
 
        # --- Iraq/Syria spillover baseline ---  
        spill = clamp(  
            0.6 * (2 * iraq_syria_instability - 1) + 0.2 * israel_risk + 0.2 * (1 - us_restraint),  
            -1, 1  
        )  
 
        # Map discrete actions -> numeric intensities  
        sanctions_enforcement = 0.0  
        escalation_risk = 0.0  
        israel_strike_pressure = 0.0  
        gulf_restraint_pressure = 0.0  
        russia_support = 0.0  
        china_support = 0.0  
        europe_pressure = 0.0  
        turkey_mediation = 0.0  
 
        # US  
        if us_action == "enforce_sanctions":  
            sanctions_enforcement += 0.8  
            escalation_risk += 0.1  
        elif us_action == "deescalate":  
            sanctions_enforcement += 0.2  
            escalation_risk -= 0.4  
        else:  
            sanctions_enforcement += 0.5  
            escalation_risk += 0.5  
 
        # Israel  
        if israel_action == "high_pressure":  
            israel_strike_pressure += 0.9  
            escalation_risk += 0.4  
        elif israel_action == "covert_pressure":  
            israel_strike_pressure += 0.6  
            escalation_risk += 0.2  
        else:  
            israel_strike_pressure += 0.2  
            escalation_risk -= 0.1  
 
        # GCC  
        if gcc_action == "push_restraint":  
            gulf_restraint_pressure += 0.9  
            escalation_risk -= 0.3  
        elif gcc_action == "hedge":  
            gulf_restraint_pressure += 0.4  
        else:  
            gulf_restraint_pressure += 0.1  
            escalation_risk += 0.1  
 
        # EU/UK  
        if eu_action == "sanctions_hr":  
            europe_pressure += 0.8  
            sanctions_enforcement += 0.3  
        elif eu_action == "diplomacy":  
            europe_pressure += 0.4  
        else:  
            europe_pressure += 0.1  
 
        # Russia  
        if ru_action == "support_iran":  
            russia_support += 0.8  
        elif ru_action == "tradeoff":  
            russia_support += 0.4  
        else:  
            russia_support += 0.1  
 
        # China  
        if cn_action == "keep_oil":  
            china_support += 0.8  
        elif cn_action == "cautious":  
            china_support += 0.5  
        else:  
            china_support += 0.2  
            sanctions_enforcement += 0.1  
 
        # Turkey  
        if tr_action == "mediate":  
            turkey_mediation += 0.8  
            escalation_risk -= 0.2  
        elif tr_action == "hedge":  
            turkey_mediation += 0.4  
        else:  
            turkey_mediation += 0.2  
            escalation_risk += 0.1  
 
        # Normalize to sensible ranges  
        sanctions_enforcement = clamp(sanctions_enforcement / 1.2, 0.0, 1.0)  
        escalation_risk = clamp((escalation_risk + 0.5) / 1.5, 0.0, 1.0)  
        israel_strike_pressure = clamp(israel_strike_pressure, 0.0, 1.0)  
        gulf_restraint_pressure = clamp(gulf_restraint_pressure, 0.0, 1.0)  
        russia_support = clamp(russia_support, 0.0, 1.0)  
        china_support = clamp(china_support, 0.0, 1.0)  
        europe_pressure = clamp(europe_pressure, 0.0, 1.0)  
        turkey_mediation = clamp(turkey_mediation, 0.0, 1.0)  
 
        return ExternalActions(  
            sanctions_enforcement=sanctions_enforcement,  
            escalation_risk=escalation_risk,  
            israel_strike_pressure=israel_strike_pressure,  
            gulf_restraint_pressure=gulf_restraint_pressure,  
            russia_support=russia_support,  
            china_support=china_support,  
            europe_pressure=europe_pressure,  
            turkey_mediation=turkey_mediation,  
            iraq_syria_spillover=spill,  
        )  
 
 
# -----------------------------  
# Coupled internal dynamics + hazard  
# -----------------------------  
@dataclass  
class ModelParams:  
    """  
    Hazard model coefficients and internal dynamics parameters.  
    These defaults are illustrative (not calibrated).  
    """  
    # Base monthly fall hazard (log-odds). More negative => lower baseline.  
    base_logit: float = -5.2  
 
    # Internal drivers (log-odds weights)  
    w_econ: float = 1.30  
    w_protest: float = 1.65  
    w_repression: float = 0.55  
    w_elite: float = -1.70  
    w_blackout: float = 0.25  
 
    # Interactions  
    w_protest_x_repression: float = 0.55  
    w_econ_x_elite: float = 0.35  
 
    # External action weights (log-odds)  
    w_sanctions: float = 0.55  
    w_escalation: float = 0.30  
    w_israel_pressure: float = 0.15  
    w_gulf_restraint: float = -0.20  
    w_russia_support: float = -0.45  
    w_china_support: float = -0.35  
    w_europe_pressure: float = 0.25  
    w_turkey_mediation: float = -0.10  
    w_spillover: float = 0.25  
 
    # Internal state dynamics (monthly)  
    econ_drift: float = 0.03  
    econ_noise: float = 0.10  
 
    protest_drift: float = 0.00  
    protest_noise: float = 0.13  
 
    repression_drift: float = 0.01  
    repression_noise: float = 0.08  
 
    elite_drift: float = 0.00  
    elite_noise: float = 0.06  
 
    blackout_drift: float = 0.00  
    blackout_noise: float = 0.07  
 
 
def hazard_probability(  
    iran: IranInternalState, ext: ExternalActions, p: ModelParams  
) -> Tuple[float, Dict[str, float], float]:  
    """  
    Returns:  
      - monthly hazard probability h in [0,1]  
      - contributions dict of additive log-odds terms  
      - total logit (log-odds)  
    """  
    econ = clamp(iran.economic_stress, -1, 1)  
    protest = clamp(iran.protest_pressure, -1, 1)  
    repression = clamp(iran.repression, -1, 1)  
    elite = clamp(iran.elite_cohesion, -1, 1)  
    blackout = clamp(iran.info_blackout, -1, 1)  
 
    sanctions = clamp(ext.sanctions_enforcement, 0, 1)  
    escalation = clamp(ext.escalation_risk, 0, 1)  
    israel = clamp(ext.israel_strike_pressure, 0, 1)  
    gulf_restraint = clamp(ext.gulf_restraint_pressure, 0, 1)  
    russia = clamp(ext.russia_support, 0, 1)  
    china = clamp(ext.china_support, 0, 1)  
    europe = clamp(ext.europe_pressure, 0, 1)  
    turkey = clamp(ext.turkey_mediation, 0, 1)  
    spill = clamp(ext.iraq_syria_spillover, -1, 1)  
 
    c: Dict[str, float] = {}  
    c["base"] = p.base_logit  
    c["econ"] = p.w_econ * econ  
    c["protest"] = p.w_protest * protest  
    c["repression"] = p.w_repression * repression  
    c["elite_cohesion"] = p.w_elite * elite  
    c["info_blackout"] = p.w_blackout * blackout  
 
    c["protest_x_repression"] = p.w_protest_x_repression * (protest * repression)  
    c["econ_x_elite"] = p.w_econ_x_elite * (econ * (-elite))  
 
    c["sanctions_enforcement"] = p.w_sanctions * sanctions  
    c["escalation_risk"] = p.w_escalation * escalation  
    c["israel_pressure"] = p.w_israel_pressure * israel  
    c["gulf_restraint"] = p.w_gulf_restraint * gulf_restraint  
    c["russia_support"] = p.w_russia_support * russia  
    c["china_support"] = p.w_china_support * china  
    c["europe_pressure"] = p.w_europe_pressure * europe  
    c["turkey_mediation"] = p.w_turkey_mediation * turkey  
    c["spillover"] = p.w_spillover * spill  
 
    logit = sum(c.values())  
    return sigmoid(logit), c, logit  
 
 
def evolve_internal_state(  
    rng: random.Random, iran: IranInternalState, ext: ExternalActions, p: ModelParams  
) -> IranInternalState:  
    """  
    Monthly internal evolution with stylized couplings + noise.  
    """  
    econ = iran.economic_stress  
    protest = iran.protest_pressure  
    repression = iran.repression  
    elite = iran.elite_cohesion  
    blackout = iran.info_blackout  
 
    # Economic stress  
    econ += p.econ_drift  
    econ += 0.25 * ext.sanctions_enforcement  
    econ -= 0.18 * ext.russia_support  
    econ -= 0.14 * ext.china_support  
    econ += normal(rng, 0, p.econ_noise)  
    econ = clamp(econ, -1, 1)  
 
    # Protest pressure  
    protest += p.protest_drift  
    protest += 0.35 * econ  
    protest += 0.18 * ext.iraq_syria_spillover  
    protest -= 0.22 * repression  
    protest += 0.10 * ext.europe_pressure  
    protest += normal(rng, 0, p.protest_noise)  
    protest = clamp(protest, -1, 1)  
 
    # Repression  
    repression += p.repression_drift  
    repression += 0.35 * protest  
    repression += 0.06 * ext.escalation_risk  
    repression += normal(rng, 0, p.repression_noise)  
    repression = clamp(repression, -1, 1)  
 
    # Elite cohesion  
    elite += p.elite_drift  
    elite -= 0.22 * econ  
    elite -= 0.18 * protest  
    elite += 0.10 * ext.escalation_risk  # "rally" effect in this calibration  
    elite += 0.08 * ext.russia_support  
    elite += normal(rng, 0, p.elite_noise)  
    elite = clamp(elite, -1, 1)  
 
    # Info blackout  
    blackout += p.blackout_drift  
    blackout += 0.25 * protest + 0.20 * repression  
    blackout += normal(rng, 0, p.blackout_noise)  
    blackout = clamp(blackout, -1, 1)  
 
    return IranInternalState(  
        economic_stress=econ,  
        protest_pressure=protest,  
        repression=repression,  
        elite_cohesion=elite,  
        info_blackout=blackout,  
    )  
 
 
# -----------------------------  
# Monte Carlo simulation  
# -----------------------------  
@dataclass  
class SimulationResult:  
    months: int  
    sims: int  
    seed: int  
    scenario: Dict[str, float]  
    params: ModelParams  
    rationality: float  
    init_state: IranInternalState  
 
    p_fall_within_horizon: float  
    avg_monthly_hazard: float  
 
    # month-level outputs  
    survival_curve: List[float]          # length months+1, S(0)=1  
    mean_hazard_alive: List[float]       # length months, mean h_t conditional on alive at t  
    falls_per_month: List[int]           # length months, count of falls in month t (1-indexed)  
 
    # distribution outputs  
    fall_months: List[int]               # 1..months for falls; empty if none  
    driver_importance: List[Tuple[str, float]]  # mean abs log-odds contribution per alive month  
 
    # sample path (first simulation)  
    sample_path: List[Tuple[IranInternalState, ExternalActions, float, float]]  # (state, ext, hazard, logit)  
 
 
def _simulate_impl(  
    months: int,  
    sims: int,  
    seed: int,  
    scenario: Dict[str, float],  
    params: ModelParams,  
    rationality: float,  
    init_state: IranInternalState,  
) -> SimulationResult:  
    rng = random.Random(seed)  
    game = ActorGame(rng=rng, rationality=rationality)  
 
    falls = 0  
    total_hazard = 0.0  
 
    alive_counts = [0 for _ in range(months)]  
    hazards_sum_alive = [0.0 for _ in range(months)]  
    falls_per_month = [0 for _ in range(months)]  
    survival_curve_counts = [0 for _ in range(months + 1)]  # survivors after t months  
    survival_curve_counts[0] = sims  
 
    fall_months: List[int] = []  
 
    contrib_abs_sums: Dict[str, float] = {}  
    contrib_count_alive = 0  
 
    sample_path: List[Tuple[IranInternalState, ExternalActions, float, float]] = []  
 
    for s in range(sims):  
        iran = init_state  
        alive = True  
 
        for t in range(months):  
            if not alive:  
                break  
 
            alive_counts[t] += 1  
 
            ext = game.step(iran, scenario=scenario)  
            h, contribs, logit = hazard_probability(iran, ext, params)  
 
            hazards_sum_alive[t] += h  
            total_hazard += h  
 
            for k, v in contribs.items():  
                contrib_abs_sums[k] = contrib_abs_sums.get(k, 0.0) + abs(v)  
            contrib_count_alive += 1  
 
            if s == 0:  
                sample_path.append((iran, ext, h, logit))  
 
            # fall draw  
            if rng.random() < h:  
                alive = False  
                falls += 1  
                falls_per_month[t] += 1  
                fall_months.append(t + 1)  
                # survivors after t+1 months do not include this sim  
                # (do nothing for survival_curve_counts beyond 0; we fill later)  
                break  
 
            # evolve to next month  
            iran = evolve_internal_state(rng, iran, ext, params)  
 
        # If survived all months, it's a survivor through horizon.  
 
    # Build survival curve counts from falls_per_month  
    survivors = sims  
    survival_curve_counts[0] = survivors  
    for t in range(months):  
        survivors -= falls_per_month[t]  
        survival_curve_counts[t + 1] = survivors  
 
    survival_curve = [c / float(sims) for c in survival_curve_counts]  
    mean_hazard_alive = [  
        (hazards_sum_alive[t] / alive_counts[t]) if alive_counts[t] > 0 else 0.0  
        for t in range(months)  
    ]  
 
    p_fall = falls / float(sims)  
    avg_hazard = total_hazard / float(sims * months)  
 
    driver_importance = sorted(  
        ((k, v / float(contrib_count_alive)) for k, v in contrib_abs_sums.items()),  
        key=lambda kv: kv[1],  
        reverse=True,  
    )  
 
    return SimulationResult(  
        months=months,  
        sims=sims,  
        seed=seed,  
        scenario=scenario,  
        params=params,  
        rationality=rationality,  
        init_state=init_state,  
        p_fall_within_horizon=p_fall,  
        avg_monthly_hazard=avg_hazard,  
        survival_curve=survival_curve,  
        mean_hazard_alive=mean_hazard_alive,  
        falls_per_month=falls_per_month,  
        fall_months=fall_months,  
        driver_importance=driver_importance,  
        sample_path=sample_path,  
    )  
 
 
@st.cache_data(show_spinner=False)  
def run_simulation_cached(  
    months: int,  
    sims: int,  
    seed: int,  
    scenario: Dict[str, float],  
    params_dict: Dict[str, float],  
    rationality: float,  
    init_state_dict: Dict[str, float],  
) -> SimulationResult:  
    # Rehydrate dataclasses inside cached function  
    params = ModelParams(**params_dict)  
    init_state = IranInternalState(**init_state_dict)  
    # Ensure scenario is clamped  
    scenario2 = {k: clamp(float(v), 0.0, 1.0) for k, v in scenario.items()}  
    return _simulate_impl(  
        months=months,  
        sims=sims,  
        seed=seed,  
        scenario=scenario2,  
        params=params,  
        rationality=rationality,  
        init_state=init_state,  
    )  
 
 
# -----------------------------  
# UI: parameter definitions (descriptions + "how it changes outcome")  
# -----------------------------  
SCENARIO_PARAM_DOCS = {  
    "us_restraint": {  
        "label": "US restraint",  
        "desc": "Preference/constraint toward de-escalation vs coercive escalation.",  
        "effect": "Higher → more de-escalation choices, typically lower escalation risk and (often) lower hazard via fewer shocks.",  
    },  
    "gulf_anti_war": {  
        "label": "Gulf anti-war pressure",  
        "desc": "How strongly GCC states push for restraint/de-escalation.",  
        "effect": "Higher → more restraint pressure, typically lowers escalation risk and can reduce hazard via fewer escalatory shocks.",  
    },  
    "israel_risk": {  
        "label": "Israel perceived risk/threat",  
        "desc": "Perceived urgency/threat driving Israeli pressure actions.",  
        "effect": "Higher → higher Israeli pressure and escalation risk; may increase hazard directly and via spillover/repression dynamics.",  
    },  
    "russia_alignment": {  
        "label": "Russia alignment/backstop willingness",  
        "desc": "Russia's propensity to support Iran diplomatically/economically.",  
        "effect": "Higher → more Russia support which buffers economy/elite cohesion; typically reduces hazard.",  
    },  
    "china_oil_priority": {  
        "label": "China oil/trade priority",  
        "desc": "China’s willingness to maintain trade/oil channels.",  
        "effect": "Higher → more China support; buffers economic stress; typically reduces hazard.",  
    },  
    "europe_hr_pressure": {  
        "label": "Europe human-rights / sanctions pressure",  
        "desc": "EU/UK propensity to apply sanctions and pressure linked to unrest/human rights.",  
        "effect": "Higher → more Europe pressure/sanctions enforcement; tends to raise economic stress and hazard.",  
    },  
    "turkey_balancing": {  
        "label": "Turkey mediation/balancing",  
        "desc": "Turkey’s preference to mediate/hedge rather than pressure.",  
        "effect": "Higher → more mediation; can reduce escalation risk; modest hazard reduction in this calibration.",  
    },  
    "iraq_syria_instability": {  
        "label": "Iraq/Syria instability baseline",  
        "desc": "Baseline proxy/spillover instability affecting regional dynamics.",  
        "effect": "Higher → higher spillover term; raises protest pressure and hazard via regional destabilization.",  
    },  
}  
 
# A concise guide for the hazard coefficients (sign meaning)  
HAZARD_PARAM_DOCS = {  
    "base_logit": ("Base log-odds", "More negative → lower baseline monthly hazard (more stable baseline)."),  
    "w_econ": ("Weight: economic stress", "Higher → economy contributes more to hazard (worse economy raises hazard more)."),  
    "w_protest": ("Weight: protest pressure", "Higher → protests contribute more to hazard."),  
    "w_repression": ("Weight: repression", "Higher → repression raises hazard more (in this calibration)."),  
    "w_elite": ("Weight: elite cohesion", "More negative (or larger magnitude) → cohesion reduces hazard more strongly."),  
    "w_blackout": ("Weight: info blackout", "Higher → blackouts raise hazard more (treated as stress signal)."),  
    "w_protest_x_repression": ("Interaction: protest × repression", "Higher → repression during high protest backfires more (raises hazard)."),  
    "w_econ_x_elite": ("Interaction: econ × (−elite)", "Higher → economic stress eroding cohesion raises hazard more."),  
    "w_sanctions": ("Weight: sanctions enforcement", "Higher → sanctions raise hazard more (via economic pressure)."),  
    "w_escalation": ("Weight: escalation risk", "Higher → escalation shocks raise hazard more."),  
    "w_israel_pressure": ("Weight: Israel pressure", "Higher → Israel pressure raises hazard more (small by default)."),  
    "w_gulf_restraint": ("Weight: Gulf restraint", "More negative → Gulf restraint reduces hazard more."),  
    "w_russia_support": ("Weight: Russia support", "More negative → Russia support reduces hazard more."),  
    "w_china_support": ("Weight: China support", "More negative → China support reduces hazard more."),  
    "w_europe_pressure": ("Weight: Europe pressure", "Higher → Europe pressure raises hazard more."),  
    "w_turkey_mediation": ("Weight: Turkey mediation", "More negative → mediation reduces hazard more."),  
    "w_spillover": ("Weight: spillover", "Higher → regional spillover raises hazard more."),  
}  
 
 
# -----------------------------  
# Plotting helpers  
# -----------------------------  
def _chart_survival(res: SimulationResult) -> alt.Chart:  
    df = pd.DataFrame({  
        "Month": list(range(0, res.months + 1)),  
        "Survival probability": res.survival_curve,  
    })  
    return (  
        alt.Chart(df)  
        .mark_line(point=True)  
        .encode(  
            x=alt.X("Month:Q", title="Month"),  
            y=alt.Y("Survival probability:Q", title="Estimated survival P(no fall by month)", scale=alt.Scale(domain=[0, 1])),  
            tooltip=["Month", alt.Tooltip("Survival probability:Q", format=".3f")],  
        )  
        .properties(height=260)  
    )  
 
 
def _chart_mean_hazard(res: SimulationResult) -> alt.Chart:  
    df = pd.DataFrame({  
        "Month": list(range(1, res.months + 1)),  
        "Mean hazard (alive)": res.mean_hazard_alive,  
    })  
    return (  
        alt.Chart(df)  
        .mark_line(point=True)  
        .encode(  
            x=alt.X("Month:Q", title="Month"),  
            y=alt.Y("Mean hazard (alive):Q", title="Mean monthly hazard (conditional on survival)", scale=alt.Scale(zero=True)),  
            tooltip=["Month", alt.Tooltip("Mean hazard (alive):Q", format=".5f")],  
        )  
        .properties(height=260)  
    )  
 
 
def _chart_falls_hist(res: SimulationResult) -> alt.Chart:  
    df = pd.DataFrame({  
        "Month": list(range(1, res.months + 1)),  
        "Falls": res.falls_per_month,  
    })  
    return (  
        alt.Chart(df)  
        .mark_bar()  
        .encode(  
            x=alt.X("Month:O", title="Month (1-indexed)"),  
            y=alt.Y("Falls:Q", title="Count of falls in month"),  
            tooltip=["Month", "Falls"],  
        )  
        .properties(height=260)  
    )  
 
 
def _chart_driver_importance(res: SimulationResult, top_k: int = 12) -> alt.Chart:  
    df = pd.DataFrame(res.driver_importance[:top_k], columns=["Driver term", "Mean |log-odds contribution|"])  
    return (  
        alt.Chart(df)  
        .mark_bar()  
        .encode(  
            x=alt.X("Mean |log-odds contribution|:Q", title="Mean absolute contribution to log-odds (per alive month)"),  
            y=alt.Y("Driver term:N", title=None, sort="-x"),  
            tooltip=["Driver term", alt.Tooltip("Mean |log-odds contribution|:Q", format=".4f")],  
        )  
        .properties(height=320)  
    )  
 
 
def _df_sample_path(res: SimulationResult) -> pd.DataFrame:  
    rows = []  
    for i, (st_i, ex_i, h_i, logit_i) in enumerate(res.sample_path, start=1):  
        rows.append({  
            "Month": i,  
            "Hazard": h_i,  
            "Logit": logit_i,  
            "Economic stress": st_i.economic_stress,  
            "Protest pressure": st_i.protest_pressure,  
            "Repression": st_i.repression,  
            "Elite cohesion": st_i.elite_cohesion,  
            "Info blackout": st_i.info_blackout,  
            "Sanctions enforcement": ex_i.sanctions_enforcement,  
            "Escalation risk": ex_i.escalation_risk,  
            "Israel pressure": ex_i.israel_strike_pressure,  
            "Gulf restraint": ex_i.gulf_restraint_pressure,  
            "Russia support": ex_i.russia_support,  
            "China support": ex_i.china_support,  
            "Europe pressure": ex_i.europe_pressure,  
            "Turkey mediation": ex_i.turkey_mediation,  
            "Spillover": ex_i.iraq_syria_spillover,  
        })  
    return pd.DataFrame(rows)  
 
 
def _chart_sample_internals(df: pd.DataFrame) -> alt.Chart:  
    melt = df.melt(  
        id_vars=["Month"],  
        value_vars=["Economic stress", "Protest pressure", "Repression", "Elite cohesion", "Info blackout"],  
        var_name="Internal variable",  
        value_name="Value",  
    )  
    return (  
        alt.Chart(melt)  
        .mark_line(point=False)  
        .encode(  
            x=alt.X("Month:Q", title="Month"),  
            y=alt.Y("Value:Q", title="Internal state (scaled)", scale=alt.Scale(domain=[-1, 1])),  
            color=alt.Color("Internal variable:N", title="Internal variable"),  
            tooltip=["Month", "Internal variable", alt.Tooltip("Value:Q", format=".3f")],  
        )  
        .properties(height=300)  
    )  
 
 
def _chart_sample_externals(df: pd.DataFrame) -> alt.Chart:  
    melt = df.melt(  
        id_vars=["Month"],  
        value_vars=[  
            "Sanctions enforcement", "Escalation risk", "Israel pressure", "Gulf restraint",  
            "Russia support", "China support", "Europe pressure", "Turkey mediation"  
        ],  
        var_name="External action",  
        value_name="Value",  
    )  
    return (  
        alt.Chart(melt)  
        .mark_line(point=False)  
        .encode(  
            x=alt.X("Month:Q", title="Month"),  
            y=alt.Y("Value:Q", title="External action intensity", scale=alt.Scale(domain=[0, 1])),  
            color=alt.Color("External action:N", title="External action"),  
            tooltip=["Month", "External action", alt.Tooltip("Value:Q", format=".3f")],  
        )  
        .properties(height=300)  
    )  
 
 
def _chart_sample_hazard(df: pd.DataFrame) -> alt.Chart:  
    return (  
        alt.Chart(df)  
        .mark_line(point=True)  
        .encode(  
            x=alt.X("Month:Q", title="Month"),  
            y=alt.Y("Hazard:Q", title="Monthly hazard P(fall this month | alive)", scale=alt.Scale(domain=[0, max(0.02, float(df["Hazard"].max() * 1.15))])),  
            tooltip=["Month", alt.Tooltip("Hazard:Q", format=".5f"), alt.Tooltip("Logit:Q", format=".3f")],  
        )  
        .properties(height=240)  
    )  
 
 
# -----------------------------  
# Streamlit App  
# -----------------------------  
st.set_page_config(page_title="Regime Fall Probability Model (Monte Carlo)", layout="wide")  
 
st.title("Iran Regime Fall Probability Model")  
st.caption(  
    "Synthetic scenario model: internal dynamics + external actor game + hazard model, estimated via Monte Carlo."  
)  
 
tabs = st.tabs(["1) Model (formal explanation)", "2) Simulation (interactive)", "3) Notes & limitations"])  
 
 
# =============================  
# Tab 1: Formal explanation  
# =============================  
with tabs[0]:  
    st.subheader("Mathematical structure")  
 
    # 1) State, actions, and time
    st.markdown(r"### 1) State, actions, and time")
    st.markdown(r"We simulate monthly time steps $t = 0,1,\dots,T-1$.")
 
    st.write("**Internal state (scaled):**")
    st.latex(r'''
    x_t =
    \begin{bmatrix}
    e_t \\ p_t \\ r_t \\ c_t \\ b_t
    \end{bmatrix}
    =
    \begin{bmatrix}
    \text{economic stress} \\
    \text{protest pressure} \\
    \text{repression} \\
    \text{elite cohesion} \\
    \text{info blackout}
    \end{bmatrix},
    \qquad x_{t,i} \in [-1,1]
    ''')
 
    st.write("**External action vector (intensities):**")
    st.latex(r'''
    a_t =
    \begin{bmatrix}
    s_t \\ q_t \\ i_t \\ g_t \\ u_t \\ \chi_t \\ \epsilon_t \\ \tau_t \\ \omega_t
    \end{bmatrix}
    =
    \begin{bmatrix}
    \text{sanctions enforcement} \\
    \text{escalation risk} \\
    \text{Israel pressure} \\
    \text{Gulf restraint pressure} \\
    \text{Russia support} \\
    \text{China support} \\
    \text{Europe pressure} \\
    \text{Turkey mediation} \\
    \text{Iraq/Syria spillover}
    \end{bmatrix}
    ''')
 
    st.markdown(r"In the code, most components are in $[0,1]$ except spillover $\omega_t \in [-1,1]$.")
 
    # 2) External actor “game”
    st.markdown(r"""
    ### 2) External actor “game”: Quantal response (logit choice)
    Each actor $j$ chooses a discrete action $A_{j,t}$ from a small menu.
    Given utilities $U_j(A_{j,t}=a \mid x_t, \theta)$, we use **quantal response**:
    $$
    \Pr(A_{j,t} = a) = \frac{\exp(\lambda \, U_j(a))}{\sum_{a'} \exp(\lambda \, U_j(a'))}
    $$
    * $\lambda > 0$ is a **rationality** parameter (higher $\lambda$ makes choices closer to "always pick max utility").
    * Scenario levers (all in $[0,1]$) shift these utilities (e.g., US restraint, Israel risk).
 
    Chosen discrete actions are then mapped into numeric intensities $a_t$ (e.g., “enforce sanctions” increases $s_t$).
    """)
 
    # 3) Monthly hazard
    st.markdown(r"""
    ### 3) Monthly hazard (probability of regime fall)
    Conditioning on survival up to month $t$, the probability of “fall” during month $t$ is:
    $$
    h_t = \Pr(\text{fall at } t \mid \text{alive at } t) = \sigma(\eta_t), \quad \sigma(z)=\frac{1}{1+e^{-z}}
    $$
 
    The **log-odds** (logit) is additive:
    $$
    \eta_t = \beta_0 + \sum_k \beta_k z_{k,t} + \sum_{m} \beta_m \, \phi_m(x_t)
    $$
    where $z_{k,t}$ are internal/external features and $\phi_m(\cdot)$ are interaction terms. Concretely:
    * **Internal terms:** $e_t, p_t, r_t, c_t, b_t$
    * **Interactions:** $p_t r_t$ and $e_t(-c_t)$
    * **External terms:** $s_t, q_t, i_t, g_t, u_t, \chi_t, \epsilon_t, \tau_t, \omega_t$
 
    Importance metric:
    $$
    \text{importance}(k) = \mathbb{E}\left[\left|\text{contribution}_{k,t}\right|\right]
    $$
    """)
 
    # 4) Stochastic internal dynamics
    st.markdown(r"""
    ### 4) Stochastic internal dynamics
    The internal state evolves as:
    $$
    x_{t+1} = \mathrm{clip}\left(x_t + \text{drift} + \text{coupling}(a_t) + \varepsilon_t,\ -1,\ +1\right)
    $$
    with Gaussian noise $\varepsilon_t \sim \mathcal{N}(0, \Sigma)$.
 
    **Dynamics examples:**
    * Sanctions increase economic stress; Russia/China support buffer it.
    * Protest increases with economic stress/spillover; decreases with repression.
    * Repression rises with protests.
    * Elite cohesion erodes with stress and protests; can increase with escalation (“rally around the flag”).
    """)
 
    # 5) Monte Carlo estimation
    st.markdown(r"""
    ### 5) Monte Carlo estimation
    We simulate $N$ independent trajectories and estimate:
    $$
    \widehat{P}(\text{fall within } T) = \frac{1}{N}\sum_{n=1}^N \mathbf{1}\{\text{fall occurs by } T\}
    $$
    We also estimate a survival curve $S(t)$ by tracking the population of "surviving" runs over time.
    """)
 
    st.subheader("Interpretation guide (what the sliders mean)")  
    st.markdown(  
        "The simulation is sensitive to both (i) **scenario levers** that shape external actions "  
        "and (ii) **hazard weights** and **dynamic couplings** that define how states translate into risk."  
    )  
    st.markdown("Below are the scenario levers and their intended directional effects:")  
 
    doc_rows = []  
    for k, v in SCENARIO_PARAM_DOCS.items():  
        doc_rows.append({  
            "Parameter": k,  
            "Meaning": v["desc"],  
            "Directional effect (in this model)": v["effect"],  
        })  
    st.dataframe(pd.DataFrame(doc_rows), width='stretch', hide_index=True)  
 
 
# =============================  
# Tab 2: Simulation  
# =============================  
with tabs[1]:  
    st.subheader("Run an interactive simulation")  
 
    colL, colR = st.columns([1.05, 1.0], gap="large")  
 
    with colL:  
        st.markdown("### Inputs")  
 
        with st.form("sim_form", clear_on_submit=False):  
            st.markdown("#### Core settings")  
            c1, c2, c3, c4 = st.columns(4)  
            with c1:  
                months = st.number_input("Horizon (months)", min_value=1, max_value=60, value=12, step=1,  
                                         help="Number of monthly steps T to simulate.")  
            with c2:  
                sims = st.number_input("Monte Carlo sims", min_value=200, max_value=100000, value=20000, step=500,  
                                       help="Number of simulated trajectories. More = smoother estimates, slower runtime.")  
            with c3:  
                seed = st.number_input("Random seed", min_value=0, max_value=10_000_000, value=7, step=1,  
                                       help="Reproducible randomness seed.")  
            with c4:  
                rationality = st.slider("Game rationality (λ)", min_value=0.2, max_value=10.0, value=2.2, step=0.1,  
                                        help="Higher λ makes external actors choose higher-utility actions more deterministically.")  
 
            st.markdown("#### Initial internal state (x₀)")  
            st.caption("All are scaled to [-1, +1].")  
            i1, i2, i3, i4, i5 = st.columns(5)  
            with i1:  
                init_econ = st.slider("econ₀", -1.0, 1.0, 0.35, 0.01,  
                                      help="Higher = worse economic conditions at start; increases protests and hazard in this model.")  
            with i2:  
                init_protest = st.slider("protest₀", -1.0, 1.0, 0.20, 0.01,  
                                         help="Higher = greater protest pressure; increases hazard directly and via repression.")  
            with i3:  
                init_repress = st.slider("repression₀", -1.0, 1.0, 0.25, 0.01,  
                                         help="Higher = stronger repression; suppresses protests short-run but can backfire through interaction.")  
            with i4:  
                init_elite = st.slider("elite₀", -1.0, 1.0, 0.35, 0.01,  
                                       help="Higher = stronger elite cohesion; reduces hazard (stabilizing) in this model.")  
            with i5:  
                init_blackout = st.slider("blackout₀", -1.0, 1.0, 0.10, 0.01,  
                                          help="Higher = stronger information controls; treated here as a stress signal, mildly increasing hazard.")  
 
            st.markdown("#### Scenario levers (shape external actions)")  
            s_cols1 = st.columns(4)  
            us_restraint = s_cols1[0].slider(  
                SCENARIO_PARAM_DOCS["us_restraint"]["label"], 0.0, 1.0, 0.55, 0.01,  
                help=SCENARIO_PARAM_DOCS["us_restraint"]["desc"] + " " + SCENARIO_PARAM_DOCS["us_restraint"]["effect"]  
            )  
            gulf_anti_war = s_cols1[1].slider(  
                SCENARIO_PARAM_DOCS["gulf_anti_war"]["label"], 0.0, 1.0, 0.70, 0.01,  
                help=SCENARIO_PARAM_DOCS["gulf_anti_war"]["desc"] + " " + SCENARIO_PARAM_DOCS["gulf_anti_war"]["effect"]  
            )  
            israel_risk = s_cols1[2].slider(  
                SCENARIO_PARAM_DOCS["israel_risk"]["label"], 0.0, 1.0, 0.55, 0.01,  
                help=SCENARIO_PARAM_DOCS["israel_risk"]["desc"] + " " + SCENARIO_PARAM_DOCS["israel_risk"]["effect"]  
            )  
            iraq_syria_instability = s_cols1[3].slider(  
                SCENARIO_PARAM_DOCS["iraq_syria_instability"]["label"], 0.0, 1.0, 0.55, 0.01,  
                help=SCENARIO_PARAM_DOCS["iraq_syria_instability"]["desc"] + " " + SCENARIO_PARAM_DOCS["iraq_syria_instability"]["effect"]  
            )  
 
            s_cols2 = st.columns(4)  
            russia_alignment = s_cols2[0].slider(  
                SCENARIO_PARAM_DOCS["russia_alignment"]["label"], 0.0, 1.0, 0.60, 0.01,  
                help=SCENARIO_PARAM_DOCS["russia_alignment"]["desc"] + " " + SCENARIO_PARAM_DOCS["russia_alignment"]["effect"]  
            )  
            china_oil_priority = s_cols2[1].slider(  
                SCENARIO_PARAM_DOCS["china_oil_priority"]["label"], 0.0, 1.0, 0.55, 0.01,  
                help=SCENARIO_PARAM_DOCS["china_oil_priority"]["desc"] + " " + SCENARIO_PARAM_DOCS["china_oil_priority"]["effect"]  
            )  
            europe_hr_pressure = s_cols2[2].slider(  
                SCENARIO_PARAM_DOCS["europe_hr_pressure"]["label"], 0.0, 1.0, 0.60, 0.01,  
                help=SCENARIO_PARAM_DOCS["europe_hr_pressure"]["desc"] + " " + SCENARIO_PARAM_DOCS["europe_hr_pressure"]["effect"]  
            )  
            turkey_balancing = s_cols2[3].slider(  
                SCENARIO_PARAM_DOCS["turkey_balancing"]["label"], 0.0, 1.0, 0.55, 0.01,  
                help=SCENARIO_PARAM_DOCS["turkey_balancing"]["desc"] + " " + SCENARIO_PARAM_DOCS["turkey_balancing"]["effect"]  
            )  
 
            st.markdown("#### Model parameters (hazard + dynamics)")  
            st.caption("Adjust these only if you want to change the model’s assumed causal structure/strengths (not just the scenario).")  
 
            with st.expander("Hazard coefficients (log-odds weights)", expanded=False):  
                # Start with defaults; let user edit a subset or all.  
                default_params = ModelParams()  
 
                # Build in a stable order for UI  
                hazard_fields = [  
                    "base_logit",  
                    "w_econ", "w_protest", "w_repression", "w_elite", "w_blackout",  
                    "w_protest_x_repression", "w_econ_x_elite",  
                    "w_sanctions", "w_escalation", "w_israel_pressure", "w_gulf_restraint",  
                    "w_russia_support", "w_china_support", "w_europe_pressure", "w_turkey_mediation", "w_spillover",  
                ]  
 
                p_ui: Dict[str, float] = {}  
                for f in hazard_fields:  
                    label, effect = HAZARD_PARAM_DOCS.get(f, (f, ""))  
                    v0 = getattr(default_params, f)  
                    # Ranges: widen for exploration; keep base_logit separate  
                    if f == "base_logit":  
                        p_ui[f] = st.slider(  
                            f"{label} ({f})", -10.0, -1.0, float(v0), 0.1,  
                            help=effect  
                        )  
                    else:  
                        p_ui[f] = st.slider(  
                            f"{label} ({f})", -3.0, 3.0, float(v0), 0.05,  
                            help=effect  
                        )  
 
            with st.expander("Internal dynamics (drift + noise)", expanded=False):  
                d_fields = [  
                    "econ_drift", "econ_noise",  
                    "protest_drift", "protest_noise",  
                    "repression_drift", "repression_noise",  
                    "elite_drift", "elite_noise",  
                    "blackout_drift", "blackout_noise",  
                ]  
                d_ui: Dict[str, float] = {}  
                for f in d_fields:  
                    v0 = getattr(default_params, f)  
                    if f.endswith("_noise"):  
                        d_ui[f] = st.slider(  
                            f"{f}", 0.0, 0.5, float(v0), 0.01,  
                            help="Standard deviation of monthly Gaussian noise for this variable."  
                        )  
                    else:  
                        d_ui[f] = st.slider(  
                            f"{f}", -0.2, 0.2, float(v0), 0.005,  
                            help="Deterministic monthly drift term for this variable."  
                        )  
 
            use_cache = st.checkbox(  
                "Use caching (faster for repeated runs with same inputs)",  
                value=True,  
                help="When enabled, identical runs reuse previous results."  
            )  
 
            submitted = st.form_submit_button("Run simulation")  
 
        scenario = {  
            "us_restraint": clamp(us_restraint, 0, 1),  
            "gulf_anti_war": clamp(gulf_anti_war, 0, 1),  
            "israel_risk": clamp(israel_risk, 0, 1),  
            "russia_alignment": clamp(russia_alignment, 0, 1),  
            "china_oil_priority": clamp(china_oil_priority, 0, 1),  
            "europe_hr_pressure": clamp(europe_hr_pressure, 0, 1),  
            "turkey_balancing": clamp(turkey_balancing, 0, 1),  
            "iraq_syria_instability": clamp(iraq_syria_instability, 0, 1),  
        }  
 
        if "p_ui" not in locals():  
            p_ui = asdict(ModelParams())  
        else:  
            # Merge hazard + dynamics edits with defaults  
            merged = asdict(ModelParams())  
            merged.update(p_ui)  
            merged.update(d_ui)  
            p_ui = merged  
 
        init_state_dict = asdict(IranInternalState(  
            economic_stress=float(init_econ),  
            protest_pressure=float(init_protest),  
            repression=float(init_repress),  
            elite_cohesion=float(init_elite),  
            info_blackout=float(init_blackout),  
        ))  
 
    with colR:  
        st.markdown("### Outputs")  
 
        if submitted:  
            with st.spinner("Simulating trajectories..."):  
                if use_cache:  
                    res = run_simulation_cached(  
                        months=int(months),  
                        sims=int(sims),  
                        seed=int(seed),  
                        scenario=scenario,  
                        params_dict=p_ui,  
                        rationality=float(rationality),  
                        init_state_dict=init_state_dict,  
                    )  
                else:  
                    res = _simulate_impl(  
                        months=int(months),  
                        sims=int(sims),  
                        seed=int(seed),  
                        scenario=scenario,  
                        params=ModelParams(**p_ui),  
                        rationality=float(rationality),  
                        init_state=IranInternalState(**init_state_dict),  
                    )  
 
            st.session_state["last_result"] = res  
 
        res: Optional[SimulationResult] = st.session_state.get("last_result")  
 
        if res is None:  
            st.info("Set parameters and click **Run simulation**.")  
        else:  
            # Summary metrics + simple CI for p  
            p = res.p_fall_within_horizon  
            n = res.sims  
            se = math.sqrt(max(1e-12, p * (1 - p) / n))  
            ci_lo = max(0.0, p - 1.96 * se)  
            ci_hi = min(1.0, p + 1.96 * se)  
 
            m1, m2, m3 = st.columns(3)  
            m1.metric("P(fall within horizon)", f"{p:.4f}", help="Monte Carlo estimate: falls / sims")  
            m2.metric("95% approx. CI", f"[{ci_lo:.4f}, {ci_hi:.4f}]", help="Normal approximation to binomial uncertainty.")  
            m3.metric("Avg monthly hazard", f"{res.avg_monthly_hazard:.6f}", help="Average of h_t over all sims and months.")  
 
            # Additional distribution summaries  
            if len(res.fall_months) > 0:  
                df_fm = pd.Series(res.fall_months)  
                st.caption(  
                    f"Falls observed in {len(res.fall_months)} / {res.sims} runs. "  
                    f"Median fall month (conditional on fall): {int(df_fm.median())}."  
                )  
            else:  
                st.caption("No falls observed under this configuration (within the simulated horizon).")  
 
            st.markdown("#### Storytelling plots")  
            cA, cB = st.columns(2)  
            with cA:  
                st.altair_chart(_chart_survival(res), width='stretch')  
                st.caption("Estimated survival curve: fraction of runs that remain ‘alive’ after each month.")  
            with cB:  
                st.altair_chart(_chart_mean_hazard(res), width='stretch')  
                st.caption("Mean monthly hazard among runs that are still alive at that month.")  
 
            cC, cD = st.columns(2)  
            with cC:  
                st.altair_chart(_chart_falls_hist(res), width='stretch')  
                st.caption("Histogram of simulated fall timing (counts by month).")  
            with cD:  
                st.altair_chart(_chart_driver_importance(res, top_k=12), width='stretch')  
                st.caption("Top drivers by mean absolute log-odds contribution (a descriptive importance metric, not causal proof).")  
 
            st.markdown("#### Sample trajectory (first simulation)")  
            df_path = _df_sample_path(res)  
 
            st.altair_chart(_chart_sample_hazard(df_path), width='stretch')  
            st.caption("Hazard for the sample trajectory. (This is one draw; do not over-interpret.)")  
 
            cE, cF = st.columns(2)  
            with cE:  
                st.altair_chart(_chart_sample_internals(df_path), width='stretch')  
                st.caption("Internal state path (scaled to [-1,1]).")  
            with cF:  
                st.altair_chart(_chart_sample_externals(df_path), width='stretch')  
                st.caption("External action intensities over time (mostly [0,1]).")  
 
            with st.expander("Show sample trajectory table (first run)", expanded=False):  
                st.dataframe(df_path, width='stretch')  
 
            with st.expander("Show inputs used for this run", expanded=False):  
                st.json({  
                    "months": res.months,  
                    "sims": res.sims,  
                    "seed": res.seed,  
                    "rationality": res.rationality,  
                    "scenario": res.scenario,  
                    "init_state": asdict(res.init_state),  
                    "params": asdict(res.params),  
                })  
 
 
# =============================  
# Tab 3: Notes & limitations  
# =============================  
with tabs[2]:  
    st.subheader("Notes, limitations, and responsible use")  
 
    st.markdown("""  
**What this model is**  
- A *scenario sandbox* that combines (i) a stochastic internal state evolution, (ii) a simplified probabilistic “game” for external actors, and (iii) a logistic hazard model that converts state/action signals into a monthly probability of a discrete “fall” event.  
 
**What this model is not**  
- Not a calibrated forecast.  
- Not a causal identification strategy.  
- Not a substitute for high-quality domain expertise or data.  
 
**Key limitations**  
- Variables are synthetic and scaled to convenient ranges.  
- Utility functions and couplings are hand-specified and can embed strong assumptions.  
- The hazard model uses linear log-odds with a small set of interactions; real systems can be nonlinear, regime-dependent, and nonstationary.  
- “Driver importance” here is *mean absolute log-odds contribution*; it is not causal attribution.  
 
**How to make it more empirical**  
- Replace the internal state evolution with observed time series (economy, protests, repression proxies).  
- Estimate hazard coefficients with historical regime instability datasets (or a carefully defined event).  
- Calibrate the actor module using historical action frequencies or expert-elicited priors.  
 
**Practical tip**  
- Use the dashboard to compare *relative* differences across scenarios while holding modeling assumptions fixed.  
""")  
   
# -----------------------------  
# Footer  
# -----------------------------  
 
AUTHOR_NAME = "Ilia Teimouri"  
COPYRIGHT_YEAR = "2026"  
COPYRIGHT_HOLDER = ""  
GITHUB_URL = "https://github.com/iteimouri/ir-scenarios"  
 
st.markdown("---")  
st.markdown(  
    f"""  
    <div style="text-align: center; color: #6b7280; font-size: 0.9rem; padding: 0.5rem 0;">  
        <strong>Author:</strong> {AUTHOR_NAME} |  
        <strong>©</strong> {COPYRIGHT_YEAR} All rights reserved.  
        <br/>  
        <a href="{GITHUB_URL}" target="_blank" rel="noopener noreferrer" style="text-decoration: none;">  
            Contribute on GitHub  
        </a>  
    </div>  
    """,  
    unsafe_allow_html=True,  
)  
# img_path = r"State_flag_of_Iran_(1964–1980).svg.webp"    
# left, mid, right = st.columns([1, 0.01, 1])
# with mid:  
#     st.image(img_path, width=50)

import json
import re
import math
import random

# ===== Offline-calibrated constants (from training data) =====
BR_LO = 1.8040
BR_HI = 3.8443


# Consumption 常量（数据驱动）
CONS_POOR = 0.5740
CONS_RICH = 0.2282
CONS_STD = 0.1480
CONS_MARGIN = 0.75 * CONS_STD 

def range_reward(x: float, low: float, high: float) -> float:
    width = max(high - low, 1e-6)
    
    if x < low:
        return -min((low - x) / width, 1.0)
    elif x > high:
        return -min((x - high) / width, 1.0)
    else:
        mid = 0.5 * (low + high)
        half = max(0.5 * width, 1e-6)
        return 0.5 + 0.5 * (1.0 - abs(x - mid) / half)


def parse_action(response: str):
    try:
        text = response.replace('```json', '').replace('```', '').strip()
        json_match = re.search(r'\{[^}]+\}', text)
        if json_match:
            return json.loads(json_match.group())
        return None
    except:
        return None


def _to_float_or_none(x):
    if x is None:
        return None
    try:
        x = float(x)
    except:
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


_DEBUG_COUNT = 0
_DEFAULT_VALUE_COUNT = 0
_SEED_SET = False


def compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs) -> float:
    global _DEBUG_COUNT, _DEFAULT_VALUE_COUNT, _SEED_SET
    if not _SEED_SET:
        random.seed(42)
        _SEED_SET = True
    _DEBUG_COUNT += 1

    if _DEBUG_COUNT <= 5:
        try:
            with open('/workspace/reward_debug.log', 'a') as f:
                f.write(f"\n=== DEBUG {_DEBUG_COUNT} ===\n")
                f.write(f"solution_str: {repr(solution_str)[:500]}\n")
                f.write(f"extra_info: {repr(extra_info)[:300]}\n")
        except:
            pass

    if data_source != "econ_agent":
        return 0.0

    reward = 0.0

    action = parse_action(solution_str)
    if action is None:
        return -1.0

    work = action.get("work")
    consumption = action.get("consumption")

    if work is None or consumption is None:
        return -0.8

    try:
        work = float(work)
        consumption = float(consumption)
    except:
        return -0.8

    if not (0 <= work <= 1):
        reward -= 0.3
    if not (0 <= consumption <= 1):
        reward -= 0.3

    work = max(0.0, min(1.0, work))
    consumption = max(0.0, min(1.0, consumption))

    if isinstance(extra_info, str):
        try:
            extra_info = json.loads(extra_info)
        except:
            extra_info = {}
    elif extra_info is None:
        extra_info = {}

    income   = _to_float_or_none(extra_info.get('income', 0))   or 0.0
    lump_sum = _to_float_or_none(extra_info.get('lump_sum', 0)) or 0.0
    tax_paid = _to_float_or_none(extra_info.get('tax_paid', 0)) or 0.0
    wealth   = _to_float_or_none(extra_info.get('wealth', 0))   or 0.0

    dpi = _to_float_or_none(extra_info.get('dpi', None))
    if dpi is None:
        dpi = income + lump_sum - tax_paid
    dpi = float(dpi)

    dpi_amt = max(dpi, 0.0)

    buffer_ratio = _to_float_or_none(extra_info.get('buffer_ratio', None))
    if buffer_ratio is None:
        if dpi > 1e-6:
            cash_on_hand = wealth + dpi
            buffer_ratio = cash_on_hand / (dpi + 1e-8)
        else:
            buffer_ratio = 1.0
    buffer_ratio = max(0.0, min(10.0, float(buffer_ratio)))

    unemp = _to_float_or_none(extra_info.get('unemployment_rate', None))
    gdp_g = _to_float_or_none(extra_info.get('gdp_growth', None))
    infl  = _to_float_or_none(extra_info.get('price_inflation', None))

    regime = extra_info.get("regime", None)
    if regime is None:
        regime = "normal"
        _DEFAULT_VALUE_COUNT += 1
        if _DEFAULT_VALUE_COUNT <= 10:
            try:
                with open('/workspace/reward_debug.log', 'a') as f:
                    f.write(f"[WARNING] Missing regime at sample {_DEBUG_COUNT}\n")
            except:
                pass

    regime_strength = _to_float_or_none(extra_info.get("regime_strength", None))
    if regime_strength is None:
        regime_strength = 0.15
        _DEFAULT_VALUE_COUNT += 1
        if _DEFAULT_VALUE_COUNT <= 10:
            try:
                with open('/workspace/reward_debug.log', 'a') as f:
                    f.write(f"[WARNING] Missing regime_strength at sample {_DEBUG_COUNT}\n")
            except:
                pass

    regime_strength = max(0.0, min(1.0, regime_strength))

    saving_rate = 1.0 - consumption
    sr_reward = range_reward(saving_rate, 0.014, 0.60)

    br = buffer_ratio

# ===== Work reward: penalize poor-laziness only =====
    den = max(BR_HI - BR_LO, 1e-6)
    alpha = (br - BR_LO) / den
    alpha = max(0.0, min(1.0, alpha))

    work_pen = 0.0

    # 数值稳定
    if work < 0.02 or work > 0.98:
        work_pen -= 0.5

    # 只对"穷 + 很低 work"给连续负梯度
    # alpha≈0 → 穷人；alpha≈1 → 富人（不管）
    poor_strength = max(0.0, 1.0 - alpha)

    lazy_gap = 0.4 - work          # 只在 work < 0.4 时生效
    if lazy_gap > 0:
        work_pen -= poor_strength * (lazy_gap / 0.4)

    work_reward = work_pen 


    overconsume_pen = 0.0
    if consumption > 0.90:
        overconsume_pen -= 0.20 * (consumption - 0.90) / 0.10

    penalty_raw = overconsume_pen
    cons_target = CONS_POOR + alpha * (CONS_RICH - CONS_POOR)

    # regime shift (data-scaled by CONS_MARGIN, not a hand-picked multiplier)
    cons_shift = CONS_MARGIN * regime_strength
    if regime == "recession":
        cons_target -= cons_shift
    elif regime == "boom":
        cons_target += cons_shift
    cons_target = max(0.0, min(1.0, cons_target))
    
    cons_err = (consumption - cons_target) / max(CONS_MARGIN, 1e-6)
    cons_r = 1.0 - min(cons_err * cons_err, 2.0)  # cap
    cons_center = 2.0 * consumption - 1.0
    cons_r -= 0.05 * (1.0 - cons_center * cons_center)

    action_struct = cons_r

    extreme_pen = 0.0
    if work < 0.05 or work > 0.95:
        extreme_pen -= 0.10
    if consumption < 0.05 or consumption > 0.95:
        extreme_pen -= 0.10

    guard_parts = []
    guard_w = []

    if unemp is not None:
        guard_parts.append(range_reward(unemp, 0.02, 0.20))
        guard_w.append(0.40)
    if gdp_g is not None:
        guard_parts.append(range_reward(gdp_g, -5.0, 10.0))
        guard_w.append(0.35)
    if infl is not None:
        guard_parts.append(range_reward(infl, -2.0, 8.0))
        guard_w.append(0.25)

    if guard_w:
        wsum = sum(guard_w)
        guard = sum(p * w for p, w in zip(guard_parts, guard_w)) / wsum
    else:
        guard = 0.0

    guard = max(-1.0, min(1.0, guard))

    macro_reward = 0.9 * action_struct + 0.1 * guard

    W_SR = 0.15
    W_WORK = 0.35
    W_MACRO = 0.35
    W_PEN = 0.15

    w_sr = W_SR
    w_work = W_WORK 
    w_macro = W_MACRO * (0.5 + 0.5 * regime_strength) 

    penalty_term = max(-1.0, min(0.0, penalty_raw / 0.20))

    w_pen = W_PEN

    w_sum = w_sr + w_work + w_macro + w_pen + 1e-8

    reward += (w_sr / w_sum) * sr_reward
    reward += (w_work / w_sum) * work_reward
    reward += (w_macro / w_sum) * macro_reward
    reward += (w_pen / w_sum) * penalty_term

    if _DEBUG_COUNT <= 20:
        try:
            with open('/workspace/reward_debug.log', 'a') as f:
                f.write(
                    f"[{_DEBUG_COUNT:03d}] "
                    f"sr={saving_rate:.3f} sr_r={sr_reward:.3f} | "
                    f"buf={buffer_ratio:.2f} work={work:.2f} "
                    f"alpha={alpha:.2f} poor_str={poor_strength:.2f} work_r={work_reward:.3f} | "
                    f"cons={consumption:.2f} cons_r={cons_r:.2f} regime={regime} rs={regime_strength:.2f} | "
                    f"total={reward:.3f}\n"
                )
        except:
            pass

    if _DEBUG_COUNT % 1000 == 0 and _DEFAULT_VALUE_COUNT > 0:
        try:
            with open('/workspace/reward_debug.log', 'a') as f:
                f.write(
                    f"[STATS] Processed {_DEBUG_COUNT} samples, default value triggered {_DEFAULT_VALUE_COUNT} times "
                    f"({_DEFAULT_VALUE_COUNT/_DEBUG_COUNT*100:.2f}%)\n"
                )
        except:
            pass

    reward = max(-1.0, min(1.0, reward))
    return reward
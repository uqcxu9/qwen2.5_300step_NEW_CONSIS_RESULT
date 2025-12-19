# prepare_verl_data.py
import pickle as pkl
import pandas as pd
import os
import json
import numpy as np

# ==================== 宏观变量计算（与 reward_function.py 口径一致）====================
WINDOW = 24  # 2年滚动窗口

def compute_unemployment_rate(states, t, num_agents):
    """计算 t 时刻的失业率（基于实际就业状态，而非劳动意愿）"""
    if t < 0 or t >= len(states):
        return None
    unemployed = 0
    valid_count = 0
    for i in range(num_agents):
        agent_state = states[t].get(str(i))
        if agent_state is None:
            continue
        valid_count += 1
        job_status = agent_state.get("endogenous", {}).get("job", None)
        if job_status == "Unemployment":
            unemployed += 1
    if valid_count == 0:
        return None
    return unemployed / valid_count


def compute_gdp_growth(states, t, num_agents):
    """计算 GDP 年同比增长率（百分比）"""
    if t < 12 or t >= len(states):
        return None
    
    def get_total_income(states, t_idx):
        if t_idx < 0 or t_idx >= len(states):
            return None
        total = 0.0
        for i in range(num_agents):
            agent_state = states[t_idx].get(str(i), {})
            income = agent_state.get('income', {})
            if isinstance(income, dict):
                total += income.get('Coin', 0)
            else:
                total += float(income) if income else 0
        return total
    
    gdp_now = get_total_income(states, t)
    gdp_12m_ago = get_total_income(states, t - 12)
    
    if gdp_now is None or gdp_12m_ago is None or gdp_12m_ago < 1e-6:
        return None
    return (gdp_now - gdp_12m_ago) / gdp_12m_ago * 100


def compute_price_inflation(dense_log, t):
    """计算价格通胀率（年化百分比）"""
    world = dense_log.get("world", [])
    prices = [None] * len(world)
    for i, w in enumerate(world):
        if isinstance(w, dict):
            prices[i] = w.get("Price")
    
    if not prices or t < 0 or t >= len(prices):
        return None
    
    if t < 12:
        # 前12个月用月环比年化
        if t >= 1:
            p_now = prices[t]
            p_prev = prices[t - 1]
            if p_now is None or p_prev is None:
                return None
            if p_prev > 1e-6:
                return (p_now - p_prev) / p_prev * 12 * 100
        return None
    
    p_now = prices[t]
    p_12m_ago = prices[t - 12]
    if p_now is None or p_12m_ago is None:
        return None
    if p_12m_ago < 1e-6:
        return None
    return (p_now - p_12m_ago) / p_12m_ago * 100


def compute_regime(states, t, num_agents, window=WINDOW):
    unemp_history, gdp_history = [], []
    for tt in range(max(1, t - window + 1), t + 1):
        u = compute_unemployment_rate(states, tt, num_agents)  
        g = compute_gdp_growth(states, tt, num_agents)
        if u is not None:
            unemp_history.append(u)
        if g is not None:
            gdp_history.append(g)
    
    unemp_now = compute_unemployment_rate(states, t, num_agents)  
    gdp_now = compute_gdp_growth(states, t, num_agents)
    unemp_now = 0.05 if unemp_now is None else unemp_now
    gdp_now = 2.0 if gdp_now is None else gdp_now

    if len(unemp_history) >= 3:
        u_lo, u_hi = np.quantile(unemp_history, 0.20), np.quantile(unemp_history, 0.80)
    else:
        u_lo, u_hi = 0.04, 0.08
    
    if len(gdp_history) >= 3:
        g_lo, g_hi = np.quantile(gdp_history, 0.20), np.quantile(gdp_history, 0.80)
    else:
        g_lo, g_hi = -1.0, 5.0

    is_recession = (unemp_now >= u_hi) or (gdp_now <= g_lo)
    is_boom = (unemp_now <= u_lo) and (gdp_now >= g_hi)
    
    if is_recession and not is_boom:
        regime = "recession"
    elif is_boom and not is_recession:
        regime = "boom"
    else:
        regime = "normal"

    def clip01(x):
        return max(0.0, min(1.0, x))
    
    if regime == "normal":
        regime_strength = 0.15
    else:
        if regime == "recession":
            s_u = (unemp_now - u_hi) / (u_hi - u_lo + 1e-6) if u_hi > u_lo else 0.0
            s_g = (g_lo - gdp_now) / (g_hi - g_lo + 1e-6) if g_hi > g_lo else 0.0
        else:  # boom
            s_u = (u_lo - unemp_now) / (u_hi - u_lo + 1e-6) if u_hi > u_lo else 0.0
            s_g = (gdp_now - g_hi) / (g_hi - g_lo + 1e-6) if g_hi > g_lo else 0.0
        # 0.5×s_g + 0.5×s_u，然后 0.2 + 0.8×combined
        combined = 0.5 * clip01(s_g) + 0.5 * clip01(s_u)
        regime_strength = clip01(0.2 + 0.8 * combined)
    
    return regime, regime_strength

SYSTEM_PROMPT = """You are an economic decision-making agent. Based on your current financial situation, you need to make two decisions:
1. work: A value between 0 and 1 representing your labor supply (0 = no work, 1 = full-time work)
2. consumption: A value between 0 and 1 representing the proportion of your disposable income to consume

You MUST respond with a valid JSON object in this exact format:
{"work": <float 0-1>, "consumption": <float 0-1>}

Do not include any other text or explanation, only output the JSON."""

def prepare_verl_dataset(data_dir, output_dir, num_agents=100):
    
    dense_log_path = f"{data_dir}/dense_log.pkl"
    with open(dense_log_path, 'rb') as f:
        dense_log = pkl.load(f)
    
    states = dense_log['states']
    actions = dense_log['actions']
    periodic_tax = dense_log['PeriodicTax']
    prompts = dense_log.get('prompts', [])  
    
    samples = []
    
    errors = {
        'missing_state': 0,
        'missing_prompt': 0,
        'missing_action': 0,
        'invalid_action_format': 0,
        'invalid_skill': 0,
    }
    
    print("预计算宏观指标...")
    macro_cache = {}
    for t in range(0, len(actions)):
        if t == 0:
            macro_cache[t] = {
                'unemployment_rate': None,
                'gdp_growth': None,
                'price_inflation': None,
                'regime': "normal",
                'regime_strength': 0.15,
            }
        else:
            unemp = compute_unemployment_rate(states, t-1, num_agents)
            gdp_g = compute_gdp_growth(states, t-1, num_agents)
            infl = compute_price_inflation(dense_log, t-1)
            regime, regime_strength = compute_regime(states, t-1, num_agents)
            macro_cache[t] = {
                'unemployment_rate': unemp,
                'gdp_growth': gdp_g,
                'price_inflation': infl,
                'regime': regime,
                'regime_strength': regime_strength,
            }
    regime_counts = {r: sum(1 for m in macro_cache.values() if m['regime'] == r) for r in ['normal', 'boom', 'recession']}
    print(f"Regime 分布: {regime_counts}")
    
    for t in range(0, len(actions)):
        for agent_id in range(num_agents):
            agent_id_str = str(agent_id)
            
            # 检查 state 是否存在
            if agent_id_str not in states[t]:
                errors['missing_state'] += 1
                continue
            
            # 检查 prompt 是否存在 - 不返回默认值，记录错误
            if t >= len(prompts) or agent_id_str not in prompts[t]:
                errors['missing_prompt'] += 1
                continue
            
            raw_prompt = prompts[t][agent_id_str]
            if not raw_prompt or not isinstance(raw_prompt, str) or len(raw_prompt) < 100:
                errors['missing_prompt'] += 1
                continue
            
            state = states[t][agent_id_str]
            
            # 检查必要的 state 字段
            try:
                income = state['income']['Coin']
                wealth = state['inventory']['Coin']
                skill = state['skill']
            except (KeyError, TypeError):
                errors['missing_state'] += 1
                continue
            
            tax_info = periodic_tax[t].get(agent_id_str, {})
            tax_paid = tax_info.get('tax_paid', 0)
            lump_sum = tax_info.get('lump_sum', 0)
            
            dpi = income + lump_sum - tax_paid
            dpi_amt = max(dpi, 0.0)
            cash_on_hand = wealth + dpi_amt
            buffer_ratio = cash_on_hand / (dpi_amt + 1e-8) if dpi_amt > 1e-6 else 1.0

            # 检查 action 是否存在 - 不返回默认值，记录错误
            action_data = actions[t].get(agent_id_str)
            if action_data is None:
                errors['missing_action'] += 1
                continue

            if isinstance(action_data, dict):
                work = action_data.get('SimpleLabor')
                cons_idx = action_data.get('SimpleConsumption')
                if work is None or cons_idx is None:
                    errors['invalid_action_format'] += 1
                    continue
            elif isinstance(action_data, (list, tuple)) and len(action_data) >= 2:
                work = action_data[0]
                cons_idx = action_data[1]
            else:
                errors['invalid_action_format'] += 1
                continue

            consumption_prop = float(np.clip(cons_idx * 0.02, 0.0, 1.0))

            
            # 构建 verl 期望的 chat 格式 prompt
            chat_prompt = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": raw_prompt}
            ]
            
            # ✅ buffer_ratio clip 到 [0, 10]，与 reward_function.py 一致
            buffer_ratio = max(0.0, min(10.0, buffer_ratio))
            
            # ✅ 获取预计算的宏观指标
            macro = macro_cache[t]
            
            job_status = state.get("endogenous", {}).get("job", "Unknown")
            gt_employed = 1 if job_status != "Unemployment" else 0
            
            extra_info_dict = {
                "timestep": t,
                "agent_id": agent_id,
                # 微观变量
                "income": float(income),
                "wealth": float(wealth),
                "dpi": float(dpi),
                "buffer_ratio": float(buffer_ratio),
                "tax_paid": float(tax_paid),
                "lump_sum": float(lump_sum),
                "skill": float(skill),
                # ✅ 劳动相关（区分意愿 vs 结果）
                "gt_labor_supply": int(work),      # 劳动意愿（SimpleLabor）
                "gt_employed": gt_employed,         # 实际就业状态（0/1）
                "job_status": job_status,           # 就业状态字符串
                "gt_consumption": float(consumption_prop),
                # 宏观变量
                "unemployment_rate": macro['unemployment_rate'],
                "gdp_growth": macro['gdp_growth'],
                "price_inflation": macro['price_inflation'],
                "regime": macro['regime'],
                "regime_strength": macro['regime_strength'],
            }
            
            samples.append({
                "prompt": chat_prompt,  # chat 格式的列表
                "data_source": "econ_agent",
                "extra_info": extra_info_dict,  # 字典格式（verl 要求）
                "reward_model": {"ground_truth": ""},  # verl 需要这个字段
                "ability": "economic_decision",  # 任务类型
            })
    
    # 打印错误统计
    print(f"\n=== 数据处理统计 ===")
    print(f"成功样本数: {len(samples)}")
    print(f"错误统计:")
    for err_type, count in errors.items():
        if count > 0:
            print(f"  - {err_type}: {count}")
    total_errors = sum(errors.values())
    print(f"  总错误数: {total_errors}")
    
    if len(samples) == 0:
        raise ValueError("没有有效样本！请检查数据源。")
    
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(samples)
    
    # ✅ 按 agent 划分：随机选 5 个 agent 作为验证集
    all_agents = list(range(num_agents))
    np.random.seed(42)
    val_agents = set(np.random.choice(all_agents, size=5, replace=False))
    train_agents = set(all_agents) - val_agents
    
    print(f"\n验证集 agent: {sorted(val_agents)}")
    print(f"训练集 agent 数量: {len(train_agents)}")
    
    # 按 agent_id 划分
    train_df = df[df['extra_info'].apply(lambda x: x['agent_id'] in train_agents)]
    val_df_full = df[df['extra_info'].apply(lambda x: x['agent_id'] in val_agents)]
    
    # ✅ 验证集只取 200 个样本
    val_size = min(200, len(val_df_full))
    val_df = val_df_full.sample(n=val_size, random_state=42)
    
    train_df.to_parquet(f"{output_dir}/train.parquet", index=False)
    val_df.to_parquet(f"{output_dir}/val.parquet", index=False)
    
    print(f"\n✅ Saved {len(train_df)} training, {len(val_df)} validation samples")
    print(f"   验证集来自 {len(val_df_full)} 个样本中随机抽取 {val_size} 个")
    print(f"   输出目录: {output_dir}")
    
    # 验证数据格式
    print(f"\n=== 数据格式验证 ===")
    print(f"prompt 类型: {type(samples[0]['prompt'])}")
    print(f"prompt 长度: {len(samples[0]['prompt'])} messages")
    print(f"prompt[0] role: {samples[0]['prompt'][0]['role']}")
    print(f"prompt[1] role: {samples[0]['prompt'][1]['role']}")
    print(f"user content 前100字符: {samples[0]['prompt'][1]['content'][:100]}...")
    
    return train_df, val_df


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    
    data_dir = os.path.join(base_dir, "data/gpt-3-noperception-reflection-1-100agents-240months")
    # 输出到 verl_dataset_small 目录（与配置文件匹配）
    output_dir = os.path.join(base_dir, "data/verl_dataset_small")
    
    prepare_verl_dataset(data_dir, output_dir)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy.integrate import solve_ivp

# Data
city_list = ["Cambridge", "South Cambridgeshire", "Huntingdonshire"]

df = pd.read_csv("ODWP01EW_LTLA.csv")
df = df[df["Place of work indicator (4 categories) code"] == 3]
df = df[
    (df["Lower tier local authorities label"].isin(city_list)) &
    (df["LTLA of workplace label"].isin(city_list))
]
flow_matrix = (
    df.groupby(
        ["Lower tier local authorities label", "LTLA of workplace label"]
    )["Count"].sum().unstack(fill_value=0)
)

pop_df = pd.read_excel("mye24tablesuk.xlsx", sheet_name="MYE4", skiprows=7)[["Name", "Mid-2024"]]
pop_dict = pop_df[pop_df["Name"].isin(city_list)].set_index("Name")["Mid-2024"].to_dict()

w_matrix = flow_matrix.div([pop_dict[i] for i in flow_matrix.index], axis=0)
flow_matrix.to_csv("commuting_matrix_counts_core3.csv")
w_matrix.to_csv("commuting_matrix_ratio_core3.csv")

# Visualization
plt.figure(figsize=(6, 5))
sns.heatmap(w_matrix, annot=True, fmt=".5f", cmap="YlOrRd",
            cbar_kws={'label': 'Commuting ratio w_{n<-m}'})
plt.title("Commuting Ratio Matrix (w_{n<-m})")
plt.xlabel("Workplace (destination)"); plt.ylabel("Residence (origin)")
plt.tight_layout(); plt.show()

# SIR with balanced commuting
W_raw = pd.read_csv("commuting_matrix_ratio_core3.csv", index_col=0)
W_raw = W_raw.reindex(index=city_list, columns=city_list)
W = W_raw.T.copy()

N_vec = np.array([pop_dict[c] for c in city_list], dtype=float)
W_rate = W.copy()
np.fill_diagonal(W_rate.values, 0.0)
Wmat, N0, K = W_rate.values.astype(float), N_vec.copy(), W_rate.shape[0]

B = Wmat * N0[np.newaxis, :]
C = (N0[:, np.newaxis]) * Wmat.T
A = B - C
_, _, vh = np.linalg.svd(A)
s_cols = np.abs(vh[-1, :]) + 1e-12
s_cols /= s_cols.mean()

W_bal = W_rate.copy()
for m in range(K): W_bal.iloc[:, m] *= s_cols[m]
for _ in range(100):
    inflow = W_bal.values @ N0
    outflow = W_bal.values.sum(axis=0) * N0
    corr = inflow / (outflow + 1e-12)
    for m in range(K): W_bal.iloc[:, m] *= corr[m]

alpha = 1 / 6.7
R0 = 1.65
beta = R0 * alpha
beta_lock_factor = 0.75
lock_start_day = 45
T_days = 365

I0_vec = np.array([10.0, 0.0, 0.0])
R0_vec = np.zeros(3)
S0_vec = N_vec - I0_vec - R0_vec
y0 = np.concatenate([S0_vec, I0_vec, R0_vec])

def sir_rhs_with_beta(b, y):
    S, I, R = y[0:3], y[3:6], y[6:9]
    dS = -b * S * I / N_vec
    dI =  b * S * I / N_vec - alpha * I
    dR =  alpha * I
    Wv = W_bal.values
    for n in range(3):
        dS[n] += np.sum(Wv[n, :] * S) - np.sum(Wv[:, n] * S[n])
        dI[n] += np.sum(Wv[n, :] * I) - np.sum(Wv[:, n] * I[n])
        dR[n] += np.sum(Wv[n, :] * R) - np.sum(Wv[:, n] * R[n])
    return np.concatenate([dS, dI, dR])

def rhs_baseline(t, y): return sir_rhs_with_beta(beta, y)
def rhs_lockdown(t, y):
    return sir_rhs_with_beta(beta if t <= lock_start_day else beta * beta_lock_factor, y)

t_eval = np.linspace(0, T_days, T_days + 1)
sol_base = solve_ivp(rhs_baseline, (0, T_days), y0, t_eval=t_eval)
sol_lock = solve_ivp(rhs_lockdown, (0, T_days), y0, t_eval=t_eval)

def unpack(sol): return sol.y[0:3, :], sol.y[3:6, :], sol.y[6:9, :]
S_b, I_b, R_b = unpack(sol_base)
S_l, I_l, R_l = unpack(sol_lock)

# Plots: I, S, R
colors = ["tab:blue", "tab:orange", "tab:green"]

plt.figure(figsize=(9,6))
for k, c in enumerate(city_list):
    plt.plot(t_eval, I_b[k], color=colors[k], linestyle='-',  label=f"{c} — Baseline")
    plt.plot(t_eval, I_l[k], color=colors[k], linestyle='--', label=f"{c} — Lockdown")
plt.axvline(x=lock_start_day, color='gray', linestyle=':', linewidth=1)
plt.title("Infected — Baseline vs Lockdown (β×0.75 after Day 45)")
plt.xlabel("Day"); plt.ylabel("Infected (persons)")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()

plt.figure(figsize=(9,6))
for k, c in enumerate(city_list):
    plt.plot(t_eval, S_b[k], color=colors[k], linestyle='-',  label=f"{c} — Baseline")
    plt.plot(t_eval, S_l[k], color=colors[k], linestyle='--', label=f"{c} — Lockdown")
plt.axvline(x=lock_start_day, color='gray', linestyle=':', linewidth=1)
plt.title("Susceptible — Baseline vs Lockdown")
plt.xlabel("Day"); plt.ylabel("Susceptible (persons)")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()

plt.figure(figsize=(9,6))
for k, c in enumerate(city_list):
    plt.plot(t_eval, R_b[k], color=colors[k], linestyle='-',  label=f"{c} — Baseline")
    plt.plot(t_eval, R_l[k], color=colors[k], linestyle='--', label=f"{c} — Lockdown")
plt.axvline(x=lock_start_day, color='gray', linestyle=':', linewidth=1)
plt.title("Recovered — Baseline vs Lockdown")
plt.xlabel("Day"); plt.ylabel("Recovered (persons)")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()

# Peak stats 
def peak_stats(I_mat):
    out = []
    for k, c in enumerate(city_list):
        idx = np.argmax(I_mat[k])
        out.append({"City": c, "Peak_day": t_eval[idx],
                    "Peak_I": I_mat[k, idx], "Peak_Prevalence": I_mat[k, idx] / N_vec[k]})
    return pd.DataFrame(out)

df_peak_base = peak_stats(I_b).rename(columns={"Peak_day":"Peak_day_baseline","Peak_I":"Peak_I_baseline","Peak_Prevalence":"Peak_prev_baseline"})
df_peak_lock = peak_stats(I_l).rename(columns={"Peak_day":"Peak_day_lockdown","Peak_I":"Peak_I_lockdown","Peak_Prevalence":"Peak_prev_lockdown"})
df_peaks = df_peak_base.merge(df_peak_lock, on="City")
df_peaks["Peak_drop_%"] = 100 * (1 - df_peaks["Peak_I_lockdown"]/df_peaks["Peak_I_baseline"])
df_peaks["Peak_delay_days"] = df_peaks["Peak_day_lockdown"] - df_peaks["Peak_day_baseline"]
print(df_peaks)

# Error analysis 
h_list = [4, 2, 1, 0.5, 0.25]
errors = []
ref_h = 0.125
t_ref = np.arange(0, T_days + ref_h, ref_h)
sol_ref = solve_ivp(rhs_baseline, (0, T_days), y0, t_eval=t_ref, method="RK45", max_step=ref_h)
I_ref = sol_ref.y[3:6, :]

for h in h_list:
    t_eval_h = np.arange(0, T_days + h/2, h)
    t_eval_h = t_eval_h[t_eval_h <= T_days]
    sol_h = solve_ivp(rhs_baseline, (0, T_days), y0, t_eval=t_eval_h, method="RK45", max_step=h)
    I_h = sol_h.y[3:6, :]
    I_interp = np.zeros_like(I_ref)
    for k in range(3):
        I_interp[k, :] = np.interp(t_ref, t_eval_h, I_h[k, :])
    errors.append(np.max(np.abs(I_interp - I_ref)))

plt.figure(figsize=(6,4))
plt.loglog(h_list, errors, 'o-', label="Observed")
plt.loglog(h_list, np.array(h_list)**2 / h_list[0]**2 * errors[0], 'k--', label="~O(h²)")
plt.xlabel("Step size h (days)"); plt.ylabel("Max |I_h - I_ref|"); plt.title("Truncation Error")
plt.grid(True, which='both', ls='--', alpha=0.6); plt.legend(); plt.show()

def richardson_extrapolation(rhs_func, h, p=2):
    t_h  = np.arange(0, T_days + h, h);  t_h  = t_h[t_h <= T_days]
    t_h2 = np.arange(0, T_days + h/2, h/2); t_h2 = t_h2[t_h2 <= T_days]
    I_h  = solve_ivp(rhs_func, (0, T_days), y0, t_eval=t_h,  method="RK45", max_step=h).y[3:6, :]
    I_h2 = solve_ivp(rhs_func, (0, T_days), y0, t_eval=t_h2, method="RK45", max_step=h/2).y[3:6, :]
    I_h2i = np.vstack([np.interp(t_h, t_h2, I_h2[k, :]) for k in range(3)])
    I_rich = (2**p * I_h2i - I_h) / (2**p - 1)
    return t_h, I_h, I_h2i, I_rich

# Richardson Extrapolation — Error vs. reference
h = 2.0
t_h, I_h, I_h2i, I_rich = richardson_extrapolation(rhs_baseline, h, p=2)

# reference
ref_h = 0.125
t_ref = np.arange(0, T_days + ref_h, ref_h)
sol_ref = solve_ivp(rhs_baseline, (0, T_days), y0, t_eval=t_ref, method="RK45", max_step=ref_h)
I_ref = sol_ref.y[3:6, :]

I_ref_interp = np.vstack([np.interp(t_h, t_ref, I_ref[k, :]) for k in range(3)])

# compute errors
err_h    = I_h    - I_ref_interp
err_h2   = I_h2i  - I_ref_interp
err_rich = I_rich - I_ref_interp


plt.figure(figsize=(11,7))
for k, city in enumerate(city_list):
    plt.plot(t_h, err_h[k],    linestyle="--",  label=f"{city} — h={h} error")
    plt.plot(t_h, err_h2[k],   linestyle="-.",  label=f"{city} — h/2={h/2} error")
    plt.plot(t_h, err_rich[k], linestyle="-",   label=f"{city} — Richardson error")

plt.axhline(0, color="black", linewidth=0.8)
plt.title("Richardson Extrapolation — Error vs. reference")
plt.xlabel("Day")
plt.ylabel("Error in infected")
plt.grid(True, alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()


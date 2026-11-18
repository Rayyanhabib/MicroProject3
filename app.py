import json
import lightgbm as lgb
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.optimize import brentq

st.set_page_config(page_title="Housing Market Lab", page_icon="🏠", layout="wide")


@st.cache_resource
def load():
    with open("tool_models.json") as f:
        a = json.load(f)
    a["town_table"] = pd.DataFrame(a["town_table"])
    a["tract_table"] = pd.DataFrame(a["tract_table"])
    a["booster"] = lgb.Booster(model_str=a["gbm_model"])
    return a


def lin_pred(m, vals, town):
    return m["intercept"] + sum(c * vals[n] for n, c in zip(m["num"], m["coef"])) + m["town_coef"].get(town, 0.0)


def logit_pred(m, vals, town):
    z = m["intercept"] + sum(c * (vals[n] - mu) / sd for n, c, mu, sd in zip(m["num"], m["coef"], m["mean"], m["scale"]))
    return 1 / (1 + np.exp(-(z + m["town_coef"].get(town, 0.0))))


A = load()
T = A["town_table"].set_index("town")
M = A["metrics"]
HC = A["hed_coefs"]
money = lambda v: f"${v:,.0f}"
BLUE, RED, GREEN, GOLD, GREY = "#457b9d", "#e63946", "#2a9d8f", "#e9c46a", "#999999"

st.title("🏠 Housing Market Lab: Cook County")
st.caption("Home sales are real (Cook County Assessor, 2013-2019). Variables marked SIM (income, crime, schools, rent, days on "
           "market and others) are simulated estimates, so exogenous effects illustrate the mechanism and are not findings.")

# ---------------- sidebar ----------------
sb = st.sidebar
sb.header("The home")
town = sb.selectbox("Town (location is given to the buyer)", list(T.index),
                    format_func=lambda t: f"Town {t}{' (Chicago)' if T.loc[t,'chicago'] else ''}  ·  {int(T.loc[t,'sales']):,} sales")
d = A["defaults"]
sqft = sb.number_input("Building size A (sq ft)", 400, 10000, int(round(d["sqft"], -1)), step=50)
land = sb.number_input("Lot size (sq ft)", 500, 100000, int(round(d["land"], -2)), step=250)
beds = sb.number_input("Bedrooms", 1, 10, 3)
fb = sb.number_input("Full bathrooms", 1, 8, 2)
hb = sb.number_input("Half bathrooms", 0, 4, 0)
age = sb.number_input("Age (years)", 1, 170, d["age"])
tw = T.loc[town]
year_cols = [c for c in T.columns if str(c).startswith("y20")]

row = pd.DataFrame([{
    "building_square_feet": float(sqft), "land_square_feet": float(land), "bedrooms": float(beds),
    "full_baths": float(fb), "half_baths": float(hb), "age": float(age), "sale_year": float(A["tool_year"]),
    "log_sqft": np.log(sqft), "log_land": np.log(max(land, 1)), "log_age": np.log(age),
    "log_income": np.log(tw.income), "log_crime": np.log(tw.crime), "school_score": tw.school,
    "dist_loop": tw.dist, "dist_ohare": tw.dist_ohare, "tax_rate_pct": tw.tax, "unemployment": tw.unemployment,
    "pct_bachelors": tw.bachelors, "walk_score": tw.walk, "transit_min": tw.transit,
    "o_hare_noise": 0.0, "floodplain": 0.0, "road_proximity": 0.0,
    "sim_median_hh_income": tw.income, "sim_crime_per_1000": tw.crime, "town": str(town)}])
vals = row.iloc[0].to_dict()
log1 = float(lin_pred(A["m1"], vals, str(town))); P0 = float(np.exp(log1))
lo, hi = float(np.exp(log1 + A["resid_lo"])), float(np.exp(log1 + A["resid_hi"]))
gx = row[A["gcols"]].copy(); gx["town"] = pd.Categorical(gx["town"], categories=A["towns"])
Pg = float(np.exp(A["booster"].predict(gx)[0]))
p_above = float(logit_pred(A["m2"], vals, str(town)))
sb.divider(); sb.metric("Predicted price (endogenous)", money(P0))

# ---------------- micro-model objects (professor's framework) ----------------
# Budget:  p_A * A + p_L * L = Budget.   A = area (sq ft).  L = standard of living (L index, 0-100).
pA = HC["log_sqft"] * P0 / sqft          # implicit price of area: $ per extra sq ft at this home
pL = HC["L_index"] * P0                   # implicit price of standard of living: $ per L point
A0, L0 = float(sqft), float(tw.L)
B0 = pA * A0 + pL * L0                    # cost of this home's bundle at implicit prices

tabs = st.tabs(["1. Price predictor", "2. Exogenous vs endogenous", "3. Consumer choice (A and L)",
                "4. Income & substitution", "5. Supply & demand", "6. Subsidy & tax incidence"])

# ======================= TAB 1 =======================
with tabs[0]:
    c1, c2, c3 = st.columns(3)
    c1.metric("Model 1: linear regression", money(P0)); c1.caption(f"80% range {money(lo)} to {money(hi)}")
    c2.metric("Model 2: logistic regression", f"{p_above:.0%}")
    c2.caption(f"chance of selling above the median ({money(A['median_price'])})")
    c3.metric("LightGBM second opinion", money(Pg), f"{(Pg-P0)/P0:+.0%} vs Model 1", delta_color="off")
    c3.caption("Inside Model 1's range: models agree." if lo <= Pg <= hi else "Outside Model 1's range: check comparable sales.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Standard of living L (0-100)", f"{tw.L:.0f}"); c2.metric("Price per sq ft", money(P0 / sqft))
    c3.metric("Price-to-income", f"{P0/tw.income:.1f}x"); c4.metric("Typical days on market (SIM)", f"{tw.days_on_market:.0f}")
    st.caption("Open tab 2 for every exogenous and endogenous variable behind this estimate.")
    with st.expander("Model quality (held-out sales)"):
        m1, m2, mg, mr = M["model1"], M["model2"], M["gbm"], M["real_only"]
        st.write(f"- Model 1: typical error {m1['median_pct_error']:.0%}; R² on log price {m1['log_r2']:.2f}; 80% range held {m1['range_coverage']:.0%} of the time.\n"
                 f"- Model 2: {m2['accuracy']:.0%} correct, AUC {m2['auc']:.2f}.\n- LightGBM: typical error {mg['median_pct_error']:.0%}.\n"
                 f"- Same linear model with only real variables: R² {mr['log_r2']:.2f}. Part of the gain from SIM variables is built in, "
                 "because they were constructed partly from home prices.\n"
                 "- Only buyer choices and exogenous variables predict price. Endogenous variables (days on market, list price, rent) are excluded on purpose.")

# ======================= TAB 2 =======================
with tabs[1]:
    st.subheader(f"Variables behind the price in Town {town}")
    st.markdown("**Exogenous** variables are given to buyers and sellers. **Endogenous** variables are determined inside the market. "
                "Tag: Real, SIM (simulated), or Real-ish.")
    ex = [("Mortgage rate (30-yr)", f"{A['mortgage_rate']:.2f}%", "Real-ish"), ("Property tax rate", f"{tw.tax:.2f}%", "SIM"),
          ("Land already developed", f"{tw.developed:.0f}%", "SIM"), ("Median household income", money(tw.income), "SIM"),
          ("Poverty rate", f"{tw.poverty:.1f}%", "SIM"), ("Unemployment rate", f"{tw.unemployment:.1f}%", "SIM"),
          ("Adults with a bachelor's degree", f"{tw.bachelors:.0f}%", "SIM"),
          ("Crimes per 1,000 (violent / property)", f"{tw.crime:.0f} ({tw.violent:.0f} / {tw.property_crime:.0f})", "SIM"),
          ("School score (1-10)", f"{tw.school:.1f}", "SIM"), ("Walk score", f"{tw.walk:.0f}", "SIM"),
          ("Minutes to transit", f"{tw.transit:.0f}", "SIM"),
          ("Distance to Loop / O'Hare", f"{tw.dist:.1f} / {tw.dist_ohare:.1f} mi", "Real"),
          ("Share in floodplain / O'Hare noise / near road", f"{tw.flood_share:.1%} / {tw.noise_share:.1%} / {tw.road_share:.1%}", "Real"),
          ("Standard of living index L (from the above)", f"{tw.L:.0f}", "SIM (constructed)")]
    en = [("Predicted price (Model 1)", money(P0), "Model"), ("Price per sq ft (implicit price of area)", money(P0 / sqft), "Real-based"),
          ("Price-to-income", f"{P0/tw.income:.1f}x", "SIM-based"), ("Homes sold per year (approx.)", f"{tw[year_cols].mean()/0.3:,.0f}", "Real"),
          ("Median price, avg yearly change", f"{money(tw.median_price)}, {tw.yoy:+.1f}%/yr", "Real"),
          ("Days on market", f"{tw.days_on_market:.0f}", "SIM"), ("Sale vs list price", f"{tw.over_asking:+.1f}%", "SIM"),
          ("Monthly rent (the rental alternative)", money(tw.rent), "SIM"), ("Gross rent yield", f"{tw.rent_yield:.1f}%", "SIM")]
    c1, c2 = st.columns(2)
    c1.markdown("##### Exogenous (given)")
    c1.dataframe(pd.DataFrame(ex, columns=["Variable", "Value", "Tag"]), hide_index=True)
    c2.markdown("##### Endogenous (market outcomes)")
    c2.dataframe(pd.DataFrame(en, columns=["Variable", "Value", "Tag"]), hide_index=True)
    st.markdown("##### Buyer's choices")
    st.write(f"Area **{sqft:,} sq ft**, lot {land:,} sq ft, {beds} bed, {fb} full / {hb} half bath, {age} years old.")
    tt = A["tract_table"]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.scatter(tt.L, tt.ppsf, s=8, alpha=.35, color=BLUE)
    cf = np.polyfit(tt.L, tt.ppsf, 1); xs = np.linspace(tt.L.min(), tt.L.max(), 50)
    ax.plot(xs, np.polyval(cf, xs), color=RED); ax.scatter([tw.L], [tw.median_ppsf], c="k", s=70, zorder=5, label=f"Town {town}")
    ax.set_xlabel("Standard of living L (SIM index)"); ax.set_ylabel("Median price per sq ft ($)")
    ax.set_title("Price of area rises with the standard of living (each dot = a tract)"); ax.legend(); st.pyplot(fig)
    st.divider()
    st.subheader("Exogenous shocks move willingness to pay")
    with st.expander("Effect sizes used (ASSUMPTIONS, editable; replace with estimates from real data)"):
        e_inc = st.number_input("Income elasticity of willingness to pay", value=0.50, step=0.05)
        e_cr = st.number_input("Crime elasticity of price", value=-0.10, step=0.01)
        b_sc = st.number_input("% price change per +1 school score point", value=2.0, step=0.5) / 100
        b_tx = st.number_input("% price change per +1 pp property tax rate", value=-6.0, step=0.5) / 100
        b_rt = st.number_input("% price change per +1 pp mortgage rate", value=-6.0, step=0.5) / 100
    s1, s2 = st.columns(2)
    d_inc = s1.slider("Area income change (%)", -30, 30, 0) / 100
    d_cr = s1.slider("Crime rate change (%)", -50, 50, 0) / 100
    d_sc = s2.slider("School score change (points)", -3.0, 3.0, 0.0, 0.5)
    d_tx = s2.slider("Property tax rate change (pp)", -1.0, 1.0, 0.0, 0.1)
    d_rt = s2.slider("Mortgage rate change (pp)", -2.0, 3.0, 0.0, 0.25)
    parts = {"Income": e_inc * np.log(1 + d_inc), "Crime": e_cr * np.log(1 + d_cr), "Schools": b_sc * d_sc,
             "Property tax": b_tx * d_tx, "Mortgage rate": b_rt * d_rt}
    m_shock = float(np.exp(sum(parts.values())))
    c1, c2 = st.columns(2)
    c1.metric("Willingness-to-pay change", f"{m_shock-1:+.1%}")
    c2.metric("Price if supply were fixed", money(P0 * m_shock), money(P0 * (m_shock - 1)))
    fig, ax = plt.subplots(figsize=(6, 2.6)); vals = [(np.exp(v) - 1) * 100 for v in parts.values()]
    ax.barh(list(parts), vals, color=[GREEN if v >= 0 else RED for v in vals]); ax.axvline(0, color="k", lw=.8)
    ax.set_xlabel("% change in willingness to pay"); st.pyplot(fig)
    st.caption("Tab 5 shows how much of this shift becomes a higher price once supply responds.")

# ======================= TAB 3 =======================
with tabs[2]:
    st.subheader("Consumer choice: area A and standard of living L")
    st.latex(r"p_A\,A + p_L\,L = \text{Budget}")
    st.markdown(f"**A** = area (sq ft), **L** = standard of living (the L index: safety, schools, income, walkability, access). "
                f"Implicit prices from the hedonic regression: **p_A = {money(pA)} per sq ft**, **p_L = {money(pL)} per L point**. "
                f"The home you entered, (A, L) = ({A0:,.0f}, {L0:.0f}), costs **{money(B0)}** at these prices.")
    kind = st.radio("Preferences", ["U = A · L  (Cobb-Douglas)", "U = A + L  (perfect substitutes)",
                                    "U = A + ln(L)  (quasi-linear)", "Extra: perfect complements"], horizontal=True)
    mult = st.slider("Budget as a multiple of that cost", 0.5, 1.5, 1.0, 0.05)
    B = B0 * mult; Amax, Lmax = B / pA, B / pL
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.plot([0, Lmax], [Amax, 0], "k", label=f"Budget = {money(B)}")
    gl = np.linspace(Lmax * .02, Lmax * 1.05, 300)
    if kind.startswith("U = A · L"):
        al = st.slider("α: weight on area (professor's U = A·L is α = 0.5)", 0.1, 0.9, 0.5, 0.05)
        A_s, L_s = al * B / pA, (1 - al) * B / pL
        u0 = A_s ** al * L_s ** (1 - al)
        for f in (1.0, 1.3): ax.plot(gl, (u0 * f / gl ** (1 - al)) ** (1 / al), "--", color=BLUE, alpha=.9 if f == 1 else .4)
        note = "Cobb-Douglas: a fixed share of the budget goes to each good. A and L are imperfect complements, so the buyer balances both."
    elif kind.startswith("U = A + L"):
        gm = st.slider("γ: sq ft of area the buyer would swap for 1 point of L (U = A + γL)", 5, 150, 30)
        be = pL / pA
        if gm < be: A_s, L_s = Amax, 0.0
        else: A_s, L_s = 0.0, Lmax
        u0 = A_s + gm * L_s
        for f in (1.0, 1.4): ax.plot([0, u0 * f / gm], [u0 * f, 0], "--", color=BLUE, alpha=.9 if f == 1 else .4)
        note = (f"Perfect substitutes: the buyer spends everything on the cheaper good per unit of utility. Break-even γ* = p_L / p_A = {be:.0f} sq ft per point. "
                "Below it the buyer takes the biggest home and the lowest standard of living; above it, the reverse.")
    elif kind.startswith("U = A + ln"):
        th0 = max(round(L0 * pL / pA, -1), 10)
        th = st.slider("θ: how strongly the buyer values L (in sq ft)", int(th0 * .2), int(th0 * 3), int(th0), step=max(int(th0 * .05), 1))
        L_s = min(th * pA / pL, Lmax); A_s = max((B - pL * L_s) / pA, 0.0)
        u0 = A_s + th * np.log(max(L_s, 1e-6))
        for f in (0, 0.25 * Amax): ax.plot(gl, (u0 + f) - th * np.log(gl), "--", color=BLUE, alpha=.9 if f == 0 else .4)
        note = (f"Quasi-linear: L* = θ·p_A / p_L = {th*pA/pL:.0f}, which does not depend on the budget. A raise goes entirely into more area A "
                "(zero income effect on L).")
    else:
        r = st.slider("Sq ft of area wanted per point of L", 5, 150, int(max(A0 / max(L0, 1), 5)))
        A_s = B / (pA + pL / r); L_s = A_s / r
        for f in (1.0, 1.3): ax.plot([L_s * f, L_s * f, Lmax * 1.1], [Amax * 1.1, A_s * f, A_s * f], "--", color=BLUE, alpha=.9 if f == 1 else .4)
        note = "Perfect complements: the buyer will not trade A for L. The optimum sits at the corner of the L-shape, and a price change is a pure income effect."
    ax.scatter([L_s], [A_s], c=RED, s=80, zorder=5, label="Optimum")
    ax.scatter([L0], [A0], c="k", marker="s", s=60, zorder=5, label="Home you entered")
    ax.set_xlabel("Standard of living L (points)"); ax.set_ylabel("Area A (sq ft)")
    ax.set_xlim(0, Lmax * 1.1); ax.set_ylim(0, Amax * 1.1); ax.legend(loc="upper right")
    ax.set_title("Budget line, indifference curves and optimum"); st.pyplot(fig)
    st.markdown(f"**Optimal bundle:** {A_s:,.0f} sq ft and L = {L_s:.0f}, spending {money(pA*A_s)} on area and {money(pL*L_s)} on standard of living.")
    if A_s < A0 * 0.95 and L_s > L0 * 1.05: st.success("Smaller home with a better standard of living (more security): the professor's first case.")
    elif A_s > A0 * 1.05 and L_s < L0 * 0.95: st.success("Larger home with a lower standard of living (less security): the professor's second case.")
    st.info(note)
    st.caption("Dashed curves are indifference curves; the outer one is a higher utility level the budget cannot reach.")

# ======================= TAB 4 =======================
with tabs[3]:
    st.subheader("Income and substitution effects of a price rise")
    c1, c2, c3 = st.columns(3)
    kind2 = c1.radio("Preferences ", ["U = A · L", "U = A + ln(L)", "Perfect complements"])
    which = c2.radio("Which price rises?", ["Area (p_A)", "Standard of living (p_L)"])
    rise = c3.slider("Price increase (%)", 5, 100, 30) / 100
    mult2 = st.slider("Budget as a multiple of the home's cost ", 0.5, 1.5, 1.0, 0.05)
    B = B0 * mult2
    ia = 0 if which.startswith("Area") else 1
    p0 = np.array([pA, pL]); p1 = p0.copy(); p1[ia] *= 1 + rise
    if kind2 == "U = A · L":
        al = st.slider("α: weight on area  ", 0.1, 0.9, 0.5, 0.05); sh = np.array([al, 1 - al])
        q0 = sh * B / p0; q1 = sh * B / p1
        Bc = B * (p1[ia] / p0[ia]) ** sh[ia]; qc = sh * Bc / p1
        u0 = q0[0] ** al * q0[1] ** (1 - al)
        curve = lambda gl: (u0 / gl ** (1 - al)) ** (1 / al)
    elif kind2 == "U = A + ln(L)":
        th0 = max(round(L0 * pL / pA, -1), 10)
        th = st.slider("θ  ", int(th0 * .2), int(th0 * 3), int(th0), step=max(int(th0 * .05), 1))
        def qm(p, Bd):
            Ls = min(th * p[0] / p[1], Bd / p[1]); return np.array([max((Bd - p[1] * Ls) / p[0], 0.0), Ls])
        q0, q1 = qm(p0, B), qm(p1, B)
        Lc = q1[1]; qc = np.array([q0[0] + th * np.log(q0[1] / Lc), Lc])
        u0 = q0[0] + th * np.log(q0[1]); curve = lambda gl: u0 - th * np.log(gl)
    else:
        r = st.slider("Sq ft wanted per point of L ", 5, 150, int(max(A0 / max(L0, 1), 5)))
        def qm(p, Bd):
            a = Bd / (p[0] + p[1] / r); return np.array([a, a / r])
        q0, q1 = qm(p0, B), qm(p1, B); qc = q0.copy(); curve = None
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.plot([0, B / pL], [B / pA, 0], "k", label="Old budget")
    ax.plot([0, B / p1[1]], [B / p1[0], 0], "k:", label="New budget")
    gl = np.linspace(max(min(q0[1], q1[1]) * .3, .5), B / min(pL, p1[1]) * 1.0, 300)
    if curve is not None:
        yy = curve(gl); ax.plot(gl, np.where((yy > 0) & (yy < B / pA * 1.3), yy, np.nan), "--", color=BLUE, label="Original indifference curve")
    pts = {"Start": (q0, GREEN), "Compensated": (qc, GOLD), "End": (q1, RED)}
    for k, (q, c) in pts.items(): ax.scatter([q[1]], [q[0]], c=c, s=80, zorder=5, label=k)
    ax.set_xlabel("Standard of living L (points)"); ax.set_ylabel("Area A (sq ft)"); ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    ax.legend(fontsize=8); st.pyplot(fig)
    tab = pd.DataFrame({"Good": ["Area A (sq ft)", "Standard of living L (points)"],
                        "Total change": q1 - q0, "Substitution effect": qc - q0, "Income effect": q1 - qc}).set_index("Good").round(1)
    st.dataframe(tab)
    msg = {"U = A · L": "Both effects appear: the buyer substitutes away from the pricier good and also feels poorer.",
           "U = A + ln(L)": "L has no income effect (L* does not depend on the budget), so its whole change is substitution. A absorbs the income effect.",
           "Perfect complements": "No substitution effect: the compensated bundle is the original corner. All of the change is the income effect."}[kind2]
    st.info(msg)
    st.caption("Compensated bundle = what the buyer would choose at the new prices if given just enough extra budget to keep the original utility (Hicks).")

# ======================= TAB 5 =======================
def Ps(x, es): return P0 * x ** (1 / es)
def Pd(x, ed, m=1.0): return P0 * m * x ** (-1 / ed)

Q0 = float(tw[year_cols].mean() / 0.3)

with tabs[4]:
    st.subheader("Supply and demand for homes in this town")
    st.markdown("**Demand shifters (exogenous):** income, mortgage rates, crime, schools, taxes. "
                "**Supply shifters (exogenous):** land available, zoning and construction costs. **Endogenous:** price and quantity.")
    c1, c2 = st.columns(2)
    ed = c1.slider("Demand elasticity |ε_d| (assumption)", 0.2, 3.0, 1.0, 0.1, key="ed")
    es = c2.slider(f"Supply elasticity ε_s (town default {tw.supply_el:.2f}, from land developed {tw.developed:.0f}%)", 0.1, 3.0,
                   float(min(max(tw.supply_el, 0.1), 3.0)), 0.05, key="es")
    use = st.checkbox("Apply the exogenous shock from tab 2", value=True)
    man = st.slider("Or set the demand shift manually (% change in willingness to pay)", -30, 30, 10) / 100
    m = m_shock if use else 1 + man
    xq = m ** (es * ed / (es + ed)); Pn = P0 * m ** (ed / (es + ed))
    fig, ax = plt.subplots(figsize=(6.6, 4.4)); xs = np.linspace(.5, 1.6, 200)
    ax.plot(Q0 * xs, Ps(xs, es), color="#e76f51", label="Supply"); ax.plot(Q0 * xs, Pd(xs, ed), color=BLUE, label="Demand (old)")
    ax.plot(Q0 * xs, Pd(xs, ed, m), color=BLUE, ls="--", label="Demand (after shock)")
    ax.scatter([Q0], [P0], c="k", zorder=5); ax.scatter([Q0 * xq], [Pn], c=RED, zorder=5)
    ax.set_xlabel("Homes sold per year (approx.)"); ax.set_ylabel("Price ($)"); ax.legend(); st.pyplot(fig)
    c1, c2, c3 = st.columns(3)
    c1.metric("New equilibrium price", money(Pn), f"{Pn/P0-1:+.1%}"); c2.metric("New quantity", f"{Q0*xq:,.0f}", f"{xq-1:+.1%}")
    c3.metric("Share of the shock that becomes price", f"{ed/(es+ed):.0%}")
    st.info("When supply is inelastic (little developable land), most of a demand increase becomes a higher price instead of more homes. "
            "Pass-through = ε_d / (ε_s + ε_d). Elasticities are assumptions; verify a published metro estimate before citing one.")
    st.caption(f"Baseline quantity ≈ {Q0:,.0f} homes a year (sample sales ÷ 0.3, averaged over 2013-2019). It only scales the axis.")

# ======================= TAB 6 =======================
with tabs[5]:
    st.subheader("Who gets the benefit of a subsidy (or pays a tax)?")
    st.caption("Uses the elasticities set in tab 5.")
    pol = st.radio("Policy", ["Homebuyer subsidy (per home)", "Tax paid by buyers (per home)"], horizontal=True)
    amt = st.slider("Amount per home ($)", 1_000, 40_000, 10_000, 1_000)
    if pol.startswith("Home"):
        x1 = brentq(lambda x: Ps(x, es) - (Pd(x, ed) + amt), 0.01, 6); Ps1 = Ps(x1, es); Pb1 = Ps1 - amt
        to_sell, to_buy, approx = Ps1 - P0, P0 - Pb1, ed / (es + ed)
    else:
        x1 = brentq(lambda x: Pd(x, ed) - Ps(x, es) - amt, 0.01, 6); Ps1 = Ps(x1, es); Pb1 = Ps1 + amt
        to_sell, to_buy, approx = P0 - Ps1, Pb1 - P0, es / (es + ed)
    Q1 = Q0 * x1
    fig, ax = plt.subplots(figsize=(6.6, 4.4)); xs = np.linspace(.6, 1.5, 200)
    ax.plot(Q0 * xs, Ps(xs, es), color="#e76f51", label="Supply"); ax.plot(Q0 * xs, Pd(xs, ed), color=BLUE, label="Demand")
    ax.axhline(P0, color=GREY, lw=.6, ls=":"); ax.vlines(Q1, min(Pb1, Ps1), max(Pb1, Ps1), color=RED, lw=3, label="Policy wedge")
    ax.scatter([Q1, Q1], [Pb1, Ps1], c=RED, zorder=5); ax.scatter([Q0], [P0], c="k", zorder=5)
    ax.set_xlabel("Homes sold per year (approx.)"); ax.set_ylabel("Price ($)"); ax.legend(); st.pyplot(fig)
    c1, c2, c3 = st.columns(3)
    if pol.startswith("Home"):
        c1.metric("Price sellers receive", money(Ps1), f"{money(to_sell)} higher")
        c2.metric("Net price buyers pay", money(Pb1), money(-to_buy), delta_color="inverse")
        c3.metric("Sellers capture", f"{to_sell/amt:.0%}", f"small-subsidy formula {approx:.0%}", delta_color="off")
        st.info(f"With ε_s = {es:.2f} and |ε_d| = {ed:.1f}, sellers capture about {to_sell/amt:.0%} of the subsidy through higher prices and buyers keep "
                f"{to_buy/amt:.0%}. The less elastic supply is, the more of the subsidy ends up with sellers.")
        st.write(f"Total cost ≈ **{money(amt*Q1)}** a year; deadweight loss ≈ **{money(0.5*amt*(Q1-Q0))}** (homes sold {Q1/Q0-1:+.1%}).")
    else:
        c1.metric("Price buyers pay", money(Pb1), f"{money(to_buy)} higher", delta_color="inverse")
        c2.metric("Price sellers receive", money(Ps1), money(-to_sell), delta_color="inverse")
        c3.metric("Buyers bear", f"{to_buy/amt:.0%}", f"small-tax formula {approx:.0%}", delta_color="off")
        st.info(f"Buyers bear about {to_buy/amt:.0%} of the tax and sellers {to_sell/amt:.0%}. With inelastic supply, sellers bear more of a tax.")
        st.write(f"Revenue ≈ **{money(amt*Q1)}**; deadweight loss ≈ **{money(0.5*amt*(Q0-Q1))}** (homes sold {Q1/Q0-1:+.1%}).")

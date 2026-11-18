# Trains the tool models from model_dataset.csv and saves tool_models.joblib
# Run locally or in Colab:  python train_tool_models.py
import json, sklearn, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, roc_auc_score

TOOL_YEAR = 2019
d = pd.read_csv('model_dataset.csv')
for c in ['o_hare_noise', 'floodplain', 'road_proximity']:
    d[c] = d[c].fillna(0)
d['town'] = d['town_code'].astype(int).astype(str)
d['log_sqft'] = np.log(d.building_square_feet); d['log_land'] = np.log(d.land_square_feet.clip(lower=1))
d['log_age'] = np.log(d.age); d['log_price'] = np.log(d.sale_price)
d['log_income'] = np.log(d.sim_median_hh_income); d['log_crime'] = np.log(d.sim_crime_per_1000)
d = d.rename(columns={'sim_school_score': 'school_score', 'sim_property_tax_rate_pct': 'tax_rate_pct',
                      'distance_to_loop_miles': 'dist_loop', 'distance_to_ohare_miles': 'dist_ohare',
                      'sim_unemployment_rate_pct': 'unemployment', 'sim_pct_bachelors': 'pct_bachelors',
                      'sim_walk_score': 'walk_score', 'sim_transit_minutes': 'transit_min'})

# Only buyer-choice features + EXOGENOUS variables predict price. Endogenous variables (days on market,
# list price, rent, town-year sales/prices) are deliberately left out.
CHAR = ['log_sqft', 'bedrooms', 'full_baths', 'half_baths', 'log_age', 'log_land', 'sale_year']
EXO = ['log_income', 'log_crime', 'school_score', 'dist_loop', 'dist_ohare', 'tax_rate_pct', 'unemployment',
       'pct_bachelors', 'walk_score', 'transit_min', 'o_hare_noise', 'floodplain', 'road_proximity']
towns = sorted(d.town.unique(), key=int)

def pre(num, scale=False, loc=True):
    t = [('num', Pipeline([('sc', StandardScaler())]) if scale else 'passthrough', num)]
    if loc: t.append(('cat', OneHotEncoder(handle_unknown='ignore', categories=[towns]), ['town']))
    return ColumnTransformer(t)

y = d.log_price
tr, te = train_test_split(d.index, test_size=0.20, random_state=42)
fit, cal = train_test_split(tr, test_size=0.15, random_state=42)
lin = lambda num, loc=True: Pipeline([('p', pre(num, loc=loc)), ('r', LinearRegression())])
cols_pred = CHAR + EXO + ['town']

m = lin(CHAR + EXO).fit(d.loc[fit, cols_pred], y[fit])
res = y[cal] - m.predict(d.loc[cal, cols_pred]); lo_r, hi_r = float(res.quantile(.1)), float(res.quantile(.9))
model1 = lin(CHAR + EXO).fit(d.loc[tr, cols_pred], y[tr])
# Micro hedonic model: price on home features + the standard-of-living index L (no location dummies)
HED = CHAR + ['L_index']
hed = lin(HED, loc=False).fit(d.loc[tr, HED], y[tr])
hed_coefs = dict(zip(HED, map(float, hed.named_steps['r'].coef_)))
real_only = lin(CHAR).fit(d.loc[tr, CHAR + ['town']], y[tr])
med = float(y[tr].median()); yc = (y > med).astype(int)
model2 = Pipeline([('p', pre(CHAR + EXO, scale=True)), ('c', LogisticRegression(max_iter=4000))]).fit(d.loc[tr, cols_pred], yc[tr])
names = model2.named_steps['p'].get_feature_names_out()
odds = {n.replace('num__', ''): float(np.exp(c)) for n, c in zip(names, model2.named_steps['c'].coef_[0]) if n.startswith('num__')}
gcols = ['building_square_feet', 'land_square_feet', 'bedrooms', 'full_baths', 'half_baths', 'age', 'sale_year',
         'sim_median_hh_income', 'sim_crime_per_1000', 'school_score', 'dist_loop', 'dist_ohare', 'tax_rate_pct',
         'unemployment', 'pct_bachelors', 'walk_score', 'transit_min', 'o_hare_noise', 'floodplain', 'road_proximity', 'town']
def gframe(x):
    x = x[gcols].copy(); x['town'] = pd.Categorical(x['town'], categories=towns); return x
gbm = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
                        colsample_bytree=0.8, random_state=42, verbose=-1).fit(gframe(d.loc[tr]), y[tr])

act = np.exp(y[te])
def pm(lp):
    p = np.exp(lp)
    return dict(log_r2=r2_score(y[te], lp), r2_dollars=r2_score(act, p), mae_dollars=mean_absolute_error(act, p),
                median_pct_error=float(np.median(np.abs(p - act) / act)))
p1 = model1.predict(d.loc[te, cols_pred]); pg = gbm.predict(gframe(d.loc[te]))
ph = hed.predict(d.loc[te, HED]); pr = real_only.predict(d.loc[te, CHAR + ['town']])
prob = model2.predict_proba(d.loc[te, cols_pred])[:, 1]
cov = float(((act >= np.exp(p1 + lo_r)) & (act <= np.exp(p1 + hi_r))).mean())
metrics = {'model1': {**pm(p1), 'range_coverage': cov}, 'hedonic': pm(ph), 'real_only': pm(pr), 'gbm': pm(pg),
           'model2': dict(accuracy=accuracy_score(yc[te], prob >= .5), auc=roc_auc_score(yc[te], prob)),
           'n_train': len(tr), 'n_test': len(te)}
for k, v in metrics.items(): print(k, {a: round(b, 3) for a, b in v.items()} if isinstance(v, dict) else v)

# Town table (exogenous conditions + endogenous market outcomes) for the website
agg = dict(sales=('sale_price', 'size'), median_price=('sale_price', 'median'), median_ppsf=('price_per_sqft', 'median'),
    price_to_income=('price_to_income', 'median'), days_on_market=('sim_days_on_market', 'median'),
    over_asking=('sim_over_asking_pct', 'median'), rent=('sim_median_monthly_rent', 'median'), rent_yield=('sim_gross_rent_yield_pct', 'median'),
    yoy=('town_year_yoy_price_change_pct', 'mean'),
    income=('sim_median_hh_income', 'median'), poverty=('sim_poverty_rate_pct', 'median'), unemployment=('unemployment', 'median'),
    bachelors=('pct_bachelors', 'median'), crime=('sim_crime_per_1000', 'median'), violent=('sim_violent_crime_per_1000', 'median'),
    property_crime=('sim_property_crime_per_1000', 'median'), school=('school_score', 'median'), walk=('walk_score', 'median'),
    transit=('transit_min', 'median'), dist=('dist_loop', 'median'), dist_ohare=('dist_ohare', 'median'), L=('L_index', 'median'),
    tax=('tax_rate_pct', 'first'), developed=('sim_land_developed_pct', 'first'), supply_el=('sim_supply_elasticity', 'first'),
    chicago=('is_chicago_town', 'max'), flood_share=('floodplain', 'mean'), noise_share=('o_hare_noise', 'mean'), road_share=('road_proximity', 'mean'))
T = d.groupby('town').agg(**agg).reset_index()
yc_ = d.pivot_table(index='town', columns='sale_year', values='sale_price', aggfunc='size', fill_value=0)
T = T.merge(yc_.reset_index(), on='town'); T['town'] = T.town.astype(str)
T = T.sort_values('sales', ascending=False).reset_index(drop=True)
tract = d.groupby('tract_geoid').agg(L=('L_index', 'first'), ppsf=('price_per_sqft', 'median'), price=('sale_price', 'median'),
                                     n=('sale_price', 'size')).reset_index()
tract = tract[tract.n >= 10]

# ---------- Save as plain JSON (no pickles), so it loads on any Python / pandas / scikit-learn version ----------
def cleanrec(df):
    out = []
    for r in df.to_dict('records'):
        out.append({(f'y{k}' if isinstance(k, (int, np.integer)) else str(k)): (float(v) if isinstance(v, (int, float, np.integer, np.floating, np.bool_)) else v)
                    for k, v in r.items()})
    return out

def lin_export(pipe, num):
    r = pipe.named_steps['r']; k = len(num)
    return dict(num=num, intercept=float(r.intercept_), coef=[float(c) for c in r.coef_[:k]],
                town_coef={t: float(c) for t, c in zip(towns, r.coef_[k:])} if len(r.coef_) > k else {})

m1 = lin_export(model1, CHAR + EXO)
sc = model2.named_steps['p'].named_transformers_['num'].named_steps['sc']; c2 = model2.named_steps['c']; k2 = len(CHAR + EXO)
m2 = dict(num=CHAR + EXO, mean=[float(v) for v in sc.mean_], scale=[float(v) for v in sc.scale_], intercept=float(c2.intercept_[0]),
          coef=[float(v) for v in c2.coef_[0][:k2]], town_coef={t: float(v) for t, v in zip(towns, c2.coef_[0][k2:])})
artifact = dict(m1=m1, m2=m2, hed=lin_export(hed, HED), gbm_model=gbm.booster_.model_to_string(), hed_coefs=hed_coefs,
                resid_lo=lo_r, resid_hi=hi_r, median_price=float(np.exp(med)), towns=towns,
                town_table=cleanrec(T), tract_table=cleanrec(tract), tool_year=TOOL_YEAR,
                mortgage_rate=float(d[d.sale_year == TOOL_YEAR].mortgage_rate_30y.iloc[0]),
                CHAR=CHAR, EXO=EXO, gcols=gcols, m2_odds=odds,
                metrics={k: ({a: float(b) for a, b in v.items()} if isinstance(v, dict) else float(v)) for k, v in metrics.items()},
                defaults=dict(sqft=float(d.building_square_feet.median()), land=float(d.land_square_feet.median()), age=int(d.age.median())))
with open('tool_models.json', 'w') as f: json.dump(artifact, f)
print('Saved tool_models.json')
print('hedonic coefs', {k: round(v, 4) for k, v in hed_coefs.items()})

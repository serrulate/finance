from pathlib import Path
import os
from datetime import date, timedelta
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Finance Budget App", layout="wide")

# ------------------------------------------------------------------------------
# Password gate (robust: secrets -> env -> dev fallback)
# ------------------------------------------------------------------------------
PASSWORD = None
try:
    PASSWORD = st.secrets.get("general", {}).get("app_password", None)
except Exception:
    PASSWORD = None

if PASSWORD is None:
    PASSWORD = os.environ.get("APP_PASSWORD")

DEV_FALLBACK = None  # set to a string like "dev" if you want a local fallback
if PASSWORD is None and DEV_FALLBACK:
    st.warning("No password configured; using development fallback.")
    PASSWORD = DEV_FALLBACK

if PASSWORD is None:
    st.error("Password is not configured. Set [general].app_password in Secrets (or APP_PASSWORD env).")
    st.stop()

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    pw = st.text_input("Enter password:", type="password")
    if pw == PASSWORD:
        st.session_state.authenticated = True
        st.success("Access granted!")
        st.rerun()
    else:
        st.stop()

# ------------------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = APP_DIR.parent / "Data" / "finance_table.xlsx"  # your default file

# ------------------------------------------------------------------------------
# IO & cleaning helpers
# ------------------------------------------------------------------------------
@st.cache_data
def read_excel_any(path_or_bytes):
    xl = pd.ExcelFile(path_or_bytes)
    lower = [s.lower() for s in xl.sheet_names]
    sheet = xl.sheet_names[lower.index("table")] if "table" in lower else xl.sheet_names[0]
    return pd.read_excel(path_or_bytes, sheet_name=sheet)

def stdcols(df):
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    if "limit" not in df.columns:
        df["limit"] = 0
    if "adjust" not in df.columns:
        df["adjust"] = 1
    if "category" not in df.columns:
        df["category"] = "uncategorized"
    return df

def coerce_types(df):
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["limit"]  = pd.to_numeric(df.get("limit", 0), errors="coerce").fillna(0).astype(int)
    df["adjust"] = pd.to_numeric(df.get("adjust", 1), errors="coerce").fillna(1).astype(int)
    df["category"] = df["category"].astype(str)
    return df

def prune_and_split(df):
    """
    - Drop AMEXPAYMENT rows (case-insensitive)
    - Split Income vs Expenses by Category == INCOME
    - Ensure 'month'
    - If expenses are negative overall, flip to positive
    """
    df = df[~df["category"].str.upper().eq("AMEXPAYMENT")].copy()

    df_inc = df[df["category"].str.upper().eq("INCOME")].copy()
    df_exp = df[~df["category"].str.upper().eq("INCOME")].copy()

    if not df_exp.empty and float(df_exp["amount"].sum(skipna=True)) < 0:
        df_exp["amount"] = df_exp["amount"].abs()

    for d in (df_inc, df_exp):
        if "date" in d.columns:
            d["month"] = d["date"].dt.to_period("M")

    df_inc = df_inc.dropna(subset=["amount"])
    df_exp = df_exp.dropna(subset=["amount"])
    return df_exp, df_inc

# ------------------------------------------------------------------------------
# Date helpers
# ------------------------------------------------------------------------------
def get_data_bounds(df_all):
    dates = df_all["date"].dropna()
    if dates.empty:
        today = date.today()
        return today, today
    return dates.min().date(), dates.max().date()

def preset_range(preset: str, min_d: date, max_d: date):
    if preset == "All data":
        return min_d, max_d
    if preset == "Last 12 months":
        return max_d - timedelta(days=365), max_d
    if preset == "Year-to-date":
        return date(max_d.year, 1, 1), max_d
    if preset == "Last calendar year":
        y = max_d.year - 1
        return date(y, 1, 1), date(y, 12, 31)
    if preset == "Fiscal year (Oct–Sep)":
        y = max_d.year
        fy_start = date(y-1, 10, 1) if max_d.month < 10 else date(y, 10, 1)
        fy_end = date(fy_start.year + 1, 9, 30)
        return fy_start, min(fy_end, max_d)
    return min_d, max_d

def clamp_to_window(d: date, start: date, end: date):
    if d < start:
        return start, True
    if d > end:
        return end, True
    return d, False

def prev_year_safe(d: date) -> date:
    """Return the same month/day in the previous year, with a safe Feb 29 fallback."""
    try:
        return d.replace(year=d.year - 1)
    except ValueError:
        if d.month == 2 and d.day == 29:
            return d.replace(year=d.year - 1, day=28)
        return d - timedelta(days=365)

# ------------------------------------------------------------------------------
# Computations
# ------------------------------------------------------------------------------
def monthly_income_vs_expenses(exp_df: pd.DataFrame, inc_df: pd.DataFrame):
    m_exp = (exp_df.groupby("month", as_index=False)
             .agg(expenses=("amount","sum"))) if not exp_df.empty else pd.DataFrame(columns=["month","expenses"])
    m_inc = (inc_df.groupby("month", as_index=False)
             .agg(income=("amount","sum"))) if not inc_df.empty else pd.DataFrame(columns=["month","income"])
    out = pd.merge(m_inc.assign(month=m_inc["month"].astype(str)),
                   m_exp.assign(month=m_exp["month"].astype(str)),
                   on="month", how="outer").fillna(0)
    if out.empty:
        return out
    out["net"] = out["income"] - out["expenses"]
    out["savings_rate"] = np.where(out["income"]>0, out["net"]/out["income"], np.nan)
    return out.sort_values("month")

def monthly_actuals_by_category(exp_df: pd.DataFrame):
    if exp_df.empty:
        return pd.DataFrame(columns=["month","category","actual_spent"])
    m = (exp_df.groupby(["month","category"], as_index=False)
         .agg(actual_spent=("amount","sum")))
    m["month"] = m["month"].astype(str)
    return m

def monthly_proposed_from_seasonality_goals(goal_table: pd.DataFrame, exp_df: pd.DataFrame):
    """
    Spread each category’s 'goal_budget' across observed months (in the current window),
    using historical expense shares; even split if no history.
    Expects goal_table with columns: ['category','goal_budget'] (no TOTAL row).
    """
    if goal_table.empty or exp_df.empty:
        return pd.DataFrame(columns=["month","category","proposed_budget"])

    hist = (exp_df.groupby(["category","month"], as_index=False)
            .agg(spent=("amount","sum")))
    if hist.empty:
        return pd.DataFrame(columns=["month","category","proposed_budget"])

    hist["month"] = hist["month"].astype(str)
    cat_tot = hist.groupby("category", as_index=False)["spent"].sum().rename(columns={"spent":"cattotal"})
    hist = hist.merge(cat_tot, on="category", how="left")
    hist["share"] = np.where(hist["cattotal"]>0, hist["spent"]/hist["cattotal"], 0.0)
    hist["share"] = hist.groupby("category")["share"].transform(lambda s: s/(s.sum() if s.sum()>0 else 1))

    months = sorted(hist["month"].unique())
    rows = []
    for _, r in goal_table.iterrows():
        cat = r["category"]
        annual_goal = float(r["goal_budget"])
        shares = hist[hist["category"]==cat][["month","share"]]
        if shares.empty:
            if months:
                for mo in months:
                    rows.append({"month": mo, "category": cat, "proposed_budget": annual_goal/12.0})
        else:
            for _, s in shares.iterrows():
                rows.append({"month": s["month"], "category": cat, "proposed_budget": annual_goal * float(s["share"])})

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.groupby(["month","category"], as_index=False)["proposed_budget"].sum()
    return out

def build_budget_window_comparison(exp_now: pd.DataFrame, exp_prev: pd.DataFrame, reduction_pct: float):
    """
    Returns a table with:
      category, total_spent, total_spent_last_year, goal_budget, delta_goal, delta_actual
    - goal_budget = total_spent_last_year * (1 - reduction_pct)
    - delta_goal  = total_spent - goal_budget
    - delta_actual = total_spent - total_spent_last_year
    Includes a final TOTAL row. Deltas rounded to 2 decimals.
    """
    cols = ["category","total_spent","total_spent_last_year","goal_budget","delta_goal","delta_actual"]
    if exp_now.empty and exp_prev.empty:
        return pd.DataFrame(columns=cols)

    now = (exp_now.groupby("category", as_index=False)
                 .agg(total_spent=("amount","sum"))) if not exp_now.empty else pd.DataFrame(columns=["category","total_spent"])

    prev = (exp_prev.groupby("category", as_index=False)
                 .agg(total_spent_last_year=("amount","sum"))) if not exp_prev.empty else pd.DataFrame(columns=["category","total_spent_last_year"])

    merged = pd.merge(now, prev, on="category", how="outer").fillna(0.0)

    merged["goal_budget"] = merged["total_spent_last_year"] * (1.0 - float(reduction_pct))
    merged["delta_goal"] = merged["total_spent"] - merged["goal_budget"]
    merged["delta_actual"] = merged["total_spent"] - merged["total_spent_last_year"]

    out = merged[["category","total_spent","total_spent_last_year","goal_budget","delta_goal","delta_actual"]].copy()

    totals = pd.DataFrame({
        "category": ["TOTAL"],
        "total_spent": [out["total_spent"].sum()],
        "total_spent_last_year": [out["total_spent_last_year"].sum()],
        "goal_budget": [out["goal_budget"].sum()],
        "delta_goal": [out["delta_goal"].sum()],
        "delta_actual": [out["delta_actual"].sum()]
    })

    out = pd.concat([out, totals], ignore_index=True)

    out["delta_goal"] = out["delta_goal"].round(2)
    out["delta_actual"] = out["delta_actual"].round(2)
    return out

# ------------------------------------------------------------------------------
# UI — Data load
# ------------------------------------------------------------------------------
st.title("💸 Finance Budget App")

st.write(
    "Defaults to **Data/finance_table.xlsx**. "
    "You can upload another Excel to test."
)

with st.sidebar:
    st.header("Data")
    upl = st.file_uploader("Upload Excel (.xlsx) [optional]", type=["xlsx"])

try:
    if upl is not None:
        raw = read_excel_any(upl)
        src = f"Uploaded: {upl.name}"
    else:
        raw = read_excel_any(DEFAULT_DATA)
        src = f"Default: {DEFAULT_DATA.name}"
except Exception as e:
    st.error(f"Failed to read Excel: {e}")
    st.stop()

raw = stdcols(raw)
raw = coerce_types(raw)

if raw["date"].dropna().empty:
    st.error("No valid dates found. Check the 'Date' column.")
    st.stop()

st.caption(src)

# ------------------------------------------------------------------------------
# Sidebar — Date controls and tracking dates
# ------------------------------------------------------------------------------
min_date, max_date = get_data_bounds(raw)

with st.sidebar:
    st.header("Date Range")
    preset = st.radio(
        "Preset",
        ["All data","Last 12 months","Year-to-date","Last calendar year","Fiscal year (Oct–Sep)","Custom"],
        index=0
    )

    if preset == "Custom":
        start_d, end_d = st.slider(
            "Custom range",
            min_value=min_date, max_value=max_date,
            value=(min_date, max_date)
        )
    else:
        start_d, end_d = preset_range(preset, min_date, max_date)

    # Tracking window (START & END)
    st.markdown("---")
    st.header("Tracking")
    start_track = st.date_input(
        "Start_Tracking_Date",
        value=start_d, min_value=min_date, max_value=max_date,
        help="Tables/charts accumulate from this date."
    )
    end_track = st.date_input(
        "End_Tracking_Date",
        value=end_d, min_value=min_date, max_value=max_date,
        help="Tables/charts accumulate up to this date."
    )

    if end_track < start_track:
        st.info("End_Tracking_Date was before Start_Tracking_Date. Swapping the two.")
        start_track, end_track = end_track, start_track

    start_track_clamped, clamped_s = clamp_to_window(start_track, start_d, end_d)
    end_track_clamped, clamped_e = clamp_to_window(end_track, start_d, end_d)
    if clamped_s or clamped_e:
        st.info(f"Tracking dates adjusted to window: {start_track_clamped} → {end_track_clamped}")

    st.markdown("---")
    reduction_pct = st.slider("Target expense reduction (%)", 0, 50, 15, 1) / 100.0
    st.caption("INCOME excluded from expense budgeting; AMEXPAYMENT removed. "
               "Budget Plan compares the current tracked window vs the same dates last year.")

# ------------------------------------------------------------------------------
# Apply analysis window & tracking slices
# ------------------------------------------------------------------------------
# First, analysis window
raw_win = raw[(raw["date"].dt.date >= start_d) & (raw["date"].dt.date <= end_d)].copy()

# Current tracked slice
mask_now = (raw_win["date"].dt.date >= start_track_clamped) & (raw_win["date"].dt.date <= end_track_clamped)
raw_tracked = raw_win.loc[mask_now].copy()

# Previous-year tracked slice (same day range, prior year)
start_prev = prev_year_safe(start_track_clamped)
end_prev = prev_year_safe(end_track_clamped)
mask_prev = (raw["date"].dt.date >= start_prev) & (raw["date"].dt.date <= end_prev)
raw_prev_tracked = raw.loc[mask_prev].copy()

# Split / prune AMEXPAYMENT & INCOME
ExpTracked, IncTracked = prune_and_split(raw_tracked)
ExpPrev, IncPrev = prune_and_split(raw_prev_tracked)

if ExpTracked.empty and IncTracked.empty:
    st.warning("No rows in the selected current tracking range.")
    st.stop()

# ------------------------------------------------------------------------------
# Metrics (from current tracked slice)
# ------------------------------------------------------------------------------
total_income = float(IncTracked["amount"].sum()) if not IncTracked.empty else 0.0
total_expenses_actual = float(ExpTracked["amount"].sum()) if not ExpTracked.empty else 0.0
net_total = total_income - total_expenses_actual
savings_rate = (net_total/total_income) if total_income > 0 else np.nan

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Total Income ({start_track_clamped} → {end_track_clamped})", f"${total_income:,.0f}")
c2.metric(f"Total Expenses ({start_track_clamped} → {end_track_clamped})", f"${total_expenses_actual:,.0f}")
c3.metric("Net (Income − Expenses)", f"${net_total:,.0f}")
c4.metric("Savings Rate", f"{savings_rate:.1%}" if pd.notnull(savings_rate) else "—")

# ------------------------------------------------------------------------------
# Income vs Expenses — Monthly (current tracked)
# ------------------------------------------------------------------------------
st.markdown("### Income vs Expenses — Monthly")
m_ie = monthly_income_vs_expenses(ExpTracked, IncTracked)
if m_ie.empty:
    st.info("No monthly data in the selected/track range.")
else:
    fig_ie = px.bar(m_ie, x="month", y=["income","expenses"], barmode="group")
    st.plotly_chart(fig_ie, use_container_width=True)
    fig_net = px.line(m_ie, x="month", y="net", markers=True)
    st.plotly_chart(fig_net, use_container_width=True)

# ------------------------------------------------------------------------------
# Annual Expense Budget Plan — Window vs Previous Year
# ------------------------------------------------------------------------------
st.markdown("### Annual Expense Budget Plan — Window vs Previous Year")
bp_compare = build_budget_window_comparison(ExpTracked, ExpPrev, reduction_pct=reduction_pct)

if bp_compare.empty:
    st.info("No expense categories to show for the selected windows.")
else:
    display = bp_compare.copy()
    money_cols = ["total_spent","total_spent_last_year","goal_budget","delta_goal","delta_actual"]
    for c in money_cols:
        display[c] = display[c].round(2)
    st.dataframe(display.sort_values("category"), use_container_width=True)

# ------------------------------------------------------------------------------
# Monthly Expenses by Category — Actuals / Proposed (current tracked)
# ------------------------------------------------------------------------------
st.markdown("### Monthly Expenses by Category")
view_mode = st.radio("View", ["Actuals","Proposed"], horizontal=True)

monthly_actuals = monthly_actuals_by_category(ExpTracked)

# Build the per-category goals (exclude TOTAL row)
goals_no_total = pd.DataFrame(columns=["category","goal_budget"])
if not bp_compare.empty:
    goals_no_total = bp_compare[bp_compare["category"]!="TOTAL"][["category","goal_budget"]].copy()

monthly_proposed = monthly_proposed_from_seasonality_goals(goals_no_total, ExpTracked)

if view_mode == "Actuals":
    if monthly_actuals.empty:
        st.info("No expenses found for the selected/track range.")
    else:
        fig_a = px.bar(monthly_actuals, x="month", y="actual_spent", color="category", barmode="stack")
        st.plotly_chart(fig_a, use_container_width=True)
        st.markdown("**Data (Actuals)**")
        st.dataframe(monthly_actuals.sort_values(["month","category"]), use_container_width=True)
else:
    if monthly_proposed.empty:
        st.info("No proposed monthly budget could be generated (insufficient expense history in the range).")
    else:
        fig_p = px.bar(monthly_proposed, x="month", y="proposed_budget", color="category", barmode="stack")
        st.plotly_chart(fig_p, use_container_width=True)
        if not monthly_actuals.empty:
            var = pd.merge(
                monthly_actuals.rename(columns={"actual_spent":"Actual"}),
                monthly_proposed.rename(columns={"proposed_budget":"Proposed"}),
                on=["month","category"], how="outer"
            ).fillna(0)
            var["Variance (A−P)"] = (var["Actual"] - var["Proposed"]).round(2)
            st.markdown("**Variance Table (Actual vs Proposed)**")
            st.dataframe(var.sort_values(["month","category"]), use_container_width=True)

# ------------------------------------------------------------------------------
# Downloads
# ------------------------------------------------------------------------------
st.markdown("### Export CSV")
if not m_ie.empty:
    st.download_button(
        "Download Monthly Income vs Expenses",
        m_ie.to_csv(index=False).encode("utf-8"),
        file_name="monthly_income_vs_expenses.csv",
        mime="text/csv"
    )
if not bp_compare.empty:
    st.download_button(
        "Download Annual Expense Budget Plan (Window vs Previous Year)",
        bp_compare.to_csv(index=False).encode("utf-8"),
        file_name="annual_expense_budget_window_vs_prev_year.csv",
        mime="text/csv"
    )
if not monthly_actuals.empty:
    st.download_button(
        "Download Monthly Expenses by Category (Actuals)",
        monthly_actuals.to_csv(index=False).encode("utf-8"),
        file_name="monthly_expenses_by_category_actuals.csv",
        mime="text/csv"
    )
if not monthly_proposed.empty:
    st.download_button(
        "Download Monthly Expenses by Category (Proposed)",
        monthly_proposed.to_csv(index=False).encode("utf-8"),
        file_name="monthly_expenses_by_category_proposed.csv",
        mime="text/csv"
    )

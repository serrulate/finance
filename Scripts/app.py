from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from datetime import date, timedelta

# --- Simple password gate using Streamlit secrets ---
PASSWORD = st.secrets["general"]["app_password"]

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    pwd = st.text_input("Enter password:", type="password")
    if pwd == PASSWORD:
        st.session_state.authenticated = True
        st.success("Access granted!")
    else:
        st.stop()




st.set_page_config(page_title="Finance Budget App", layout="wide")

# ------------------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = APP_DIR.parent / "Data" / "finance_table.xlsx"  # your default file

# ------------------------------------------------------------------------------
# Helpers: IO & cleaning
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
    - Split Income vs Expenses (Category == INCOME)
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
# Date presets & filtering
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

# ------------------------------------------------------------------------------
# Budget logic — simplified (adjust=0 fixed, others reduced by slider)
# ------------------------------------------------------------------------------
def compute_budget_simplified(exp_df: pd.DataFrame, reduction_pct: float):
    """
    For each category (expenses only):
      - Total_Spent = sum since Start_Tracking_Date (already filtered before call)
      - Budgeted_Spent = Total_Spent if adjust==0 else Total_Spent*(1 - reduction_pct)
      - Delta = Budgeted_Spent - Total_Spent (rounded to 2 decimals for display)
    Returns (budget_table_with_totals, base_table_no_totals)
    """
    if exp_df.empty:
        columns = ["category","total_spent","budgeted_spent","delta"]
        return pd.DataFrame(columns=columns), pd.DataFrame(columns=columns)

    # Aggregate by category
    base = (exp_df.groupby("category", as_index=False)
            .agg(total_spent=("amount","sum"),
                 adjust=("adjust","max")))

    # Apply simplified rule
    factor = 1.0 - float(reduction_pct)
    base["budgeted_spent"] = np.where(base["adjust"]==0,
                                      base["total_spent"],
                                      base["total_spent"] * factor)
    base["delta"] = base["budgeted_spent"] - base["total_spent"]

    out = base[["category","total_spent","budgeted_spent","delta"]].copy()

    # Totals row
    total_row = pd.DataFrame({
        "category": ["TOTAL"],
        "total_spent": [out["total_spent"].sum()],
        "budgeted_spent": [out["budgeted_spent"].sum()],
        "delta": [out["delta"].sum()]
    })
    out_totals = pd.concat([out, total_row], ignore_index=True)

    # Round delta to 2 decimals for display
    out_totals["delta"] = out_totals["delta"].round(2)
    out["delta"] = out["delta"].round(2)

    return out_totals, out

def monthly_actuals_by_category(exp_df: pd.DataFrame):
    if exp_df.empty:
        return pd.DataFrame(columns=["month","category","actual_spent"])
    m = (exp_df.groupby(["month","category"], as_index=False)
         .agg(actual_spent=("amount","sum")))
    m["month"] = m["month"].astype(str)
    return m

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

def monthly_proposed_from_seasonality(budget_table_no_totals: pd.DataFrame, exp_df: pd.DataFrame):
    """
    Spread each category’s annual *budgeted_spent* across observed months using
    historical expense shares; even split if no history.
    Expects a budget table WITHOUT the TOTAL row.
    """
    if budget_table_no_totals.empty or exp_df.empty:
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
    for _, r in budget_table_no_totals.iterrows():
        cat = r["category"]
        annual_budget = float(r["budgeted_spent"])
        shares = hist[hist["category"]==cat][["month","share"]]
        if shares.empty:
            if months:
                for mo in months:
                    rows.append({"month": mo, "category": cat, "proposed_budget": annual_budget/12.0})
        else:
            for _, s in shares.iterrows():
                rows.append({"month": s["month"], "category": cat, "proposed_budget": annual_budget * float(s["share"])})

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.groupby(["month","category"], as_index=False)["proposed_budget"].sum()
    return out

# ------------------------------------------------------------------------------
# UI — Inputs
# ------------------------------------------------------------------------------
st.title("💸 Finance Budget App")

st.write(
    "Defaults to **Data/finance_table.xlsx**. "
    "Upload another Excel (same columns) to test."
)

with st.sidebar:
    st.header("Data")
    upl = st.file_uploader("Upload Excel (.xlsx) [optional]", type=["xlsx"])

# Load & clean
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
# Sidebar — Date Presets, Custom Range, Start_Tracking_Date, Reduction
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

    st.markdown("---")
    st.header("Tracking")
    start_track = st.date_input(
        "Start_Tracking_Date",
        value=start_d, min_value=min_date, max_value=max_date,
        help="Annual Budget Plan totals begin on this date."
    )
    st.markdown("---")
    reduction_pct = st.slider("Target expense reduction (%)", 0, 50, 15, 1) / 100.0
    st.caption("Rules: INCOME excluded from expense budget; AMEXPAYMENT dropped; categories with adjust=0 are fixed; all other categories reduced by the slider.")

# Clamp Start_Tracking_Date to window
start_track_clamped, clamped = clamp_to_window(start_track, start_d, end_d)
if clamped:
    st.info(f"Start_Tracking_Date adjusted to be within the analysis window: {start_track_clamped}")

# ------------------------------------------------------------------------------
# Apply filters (window + tracking) and split
# ------------------------------------------------------------------------------
raw_win = raw[(raw["date"].dt.date >= start_d) & (raw["date"].dt.date <= end_d)].copy()
mask_track = (raw_win["date"].dt.date >= start_track_clamped)
raw_tracked = raw_win.loc[mask_track].copy()

ExpTracked, IncTracked = prune_and_split(raw_tracked)

if ExpTracked.empty and IncTracked.empty:
    st.warning("No rows in the selected range starting from Start_Tracking_Date.")
    st.stop()

# ------------------------------------------------------------------------------
# Metrics (from TRACKED slice)
# ------------------------------------------------------------------------------
total_income = float(IncTracked["amount"].sum()) if not IncTracked.empty else 0.0
total_expenses_actual = float(ExpTracked["amount"].sum()) if not ExpTracked.empty else 0.0
net_total = total_income - total_expenses_actual
savings_rate = (net_total/total_income) if total_income > 0 else np.nan

col1, col2, col3, col4 = st.columns(4)
col1.metric(f"Total Income ({preset})", f"${total_income:,.0f}")
col2.metric(f"Total Expenses ({preset})", f"${total_expenses_actual:,.0f}")
col3.metric("Net (Income − Expenses)", f"${net_total:,.0f}")
col4.metric("Savings Rate", f"{savings_rate:.1%}" if pd.notnull(savings_rate) else "—")

# ------------------------------------------------------------------------------
# Income vs Expenses — Monthly (TRACKED)
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
# Annual Expense Budget Plan — Simplified (TRACKED)
# ------------------------------------------------------------------------------
st.markdown("### Annual Expense Budget Plan (Simplified)")
bp_with_totals, bp_no_totals = compute_budget_simplified(ExpTracked, reduction_pct=reduction_pct)
if bp_with_totals.empty:
    st.info("No expense categories to show.")
else:
    # nicer formatting
    show = bp_with_totals.copy()
    money_cols = ["total_spent","budgeted_spent"]
    for c in money_cols:
        show[c] = show[c].round(2)
    st.dataframe(show.sort_values("category"), use_container_width=True)

# ------------------------------------------------------------------------------
# Monthly Expenses by Category — Actuals / Proposed (TRACKED)
# ------------------------------------------------------------------------------
st.markdown("### Monthly Expenses by Category")
view_mode = st.radio("View", ["Actuals","Proposed"], horizontal=True)

monthly_actuals = monthly_actuals_by_category(ExpTracked)
monthly_proposed = monthly_proposed_from_seasonality(bp_no_totals, ExpTracked)

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
        # Optional variance table if you also want to compare to actuals
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
if not bp_with_totals.empty:
    st.download_button(
        "Download Annual Expense Budget Plan (Simplified)",
        bp_with_totals.to_csv(index=False).encode("utf-8"),
        file_name="annual_expense_budget_simplified.csv",
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

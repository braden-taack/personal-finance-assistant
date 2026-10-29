"""Streamlit dashboard. Run with:  streamlit run app.py"""
from __future__ import annotations

import json
import uuid
from datetime import date

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import agent as agent_mod
import agent_tools
import classifier
import db
import importer
import llm
import pipeline
import reports
from config import EMPLOYER_CATEGORY, LLM_BASE_URL, LLM_THRESHOLD, ML_THRESHOLD, PAYCHECK_PERIODS, SHARED_SPLIT

st.set_page_config(page_title="Finance Tracker", page_icon="💰", layout="wide")
db.init_db()

PAGE = st.sidebar.radio(
    "Go to",
    ["Dashboard", "Budgets", "Paychecks", "Shared expenses", "Import CSV", "Review queue", "Transactions",
     "Categories & Model"],
)
q = len(db.fetch_review_queue())
if q:
    st.sidebar.warning(f"{q} transaction(s) need review")


@st.cache_data(ttl=15, show_spinner=False)
def llm_models() -> list[dict]:
    return llm.list_models()


def sidebar_llm() -> None:
    """LLM status + model selector. The choice is saved; an unavailable choice falls back to the loaded model."""
    models = llm_models()
    if not models:
        st.sidebar.caption(f"LLM: 🔴 offline ({LLM_BASE_URL})")
        return
    chosen = pipeline.llm_model()
    effective = llm.resolve_model(chosen, models)
    ids = [m["id"] for m in models]
    loaded = {m["id"] for m in models if m["loaded"]}
    pick = st.sidebar.selectbox("LLM model", ids, index=ids.index(effective), key="llm_model_select",
                                format_func=lambda i: f"{i} ●" if i in loaded else i,
                                help="● = loaded in LM Studio. Others are loaded on first use.")
    if pick != effective:
        db.set_setting("llm_model", pick)
        effective = pick
    if chosen != effective and pick == llm.resolve_model(chosen, models):
        st.sidebar.caption(f"{chosen} isn't available in LM Studio, so the loaded model is used.")
    st.sidebar.caption(f"LLM: 🟢 reachable ({LLM_BASE_URL}) · {pipeline.llm_parallel()} parallel requests")


sidebar_llm()

THIS_MONTH = date.today().strftime("%Y-%m")
PERIOD_LABEL = {"mid": "Mid Month Paycheck", "end": "End of Month Paycheck"}
def money(x: float) -> str:
    return f"-${-x:,.2f}" if x < -0.004 else f"${abs(x):,.2f}"


MONEY = money  # Styler.format accepts a callable


def month_options() -> list[str]:
    with db.connect() as c:
        planned = {r[0] for r in c.execute("SELECT month FROM budgets UNION SELECT month FROM paychecks")}
    return sorted(set(db.transaction_months()) | planned | {THIS_MONTH}, reverse=True)


def color_difference(v: float) -> str:
    if v > 0.005:
        return "color: #1a7f37; font-weight: 600"
    if v < -0.005:
        return "color: #cf222e; font-weight: 600"
    return ""


def bold_total(row: pd.Series) -> list[str]:
    return ["font-weight: 700" if row.name == reports.TOTAL else "" for _ in row]


def paycheck_lines(months: list[str], period: str) -> pd.DataFrame:
    """One paycheck's lines; for several months, the amounts are summed line by line."""
    parts = [db.effective_paychecks(m) for m in months]
    lines = pd.concat([p[p["period"] == period] for p in parts if not p.empty]) if any(
        not p.empty for p in parts) else pd.DataFrame(columns=["category", "subcategory", "amount", "sort_order"])
    if lines.empty:
        return lines[["category", "subcategory", "amount"]]
    return (lines.groupby(["category", "subcategory"], as_index=False, sort=False)
            .agg(amount=("amount", "sum"), sort_order=("sort_order", "min"))
            .sort_values("sort_order")[["category", "subcategory", "amount"]])


def show_paycheck(lines: pd.DataFrame, title: str) -> None:
    st.markdown(f"**{title}**")
    if lines.empty:
        st.caption("No paycheck saved — see the Paychecks page.")
        return
    t = db.paycheck_totals(lines)
    view = lines.rename(columns={"category": "Category", "subcategory": "Subcategory", "amount": "Amount"})
    view = pd.concat([view, pd.DataFrame([{"Category": "Total", "Subcategory": "Take-home", "Amount": t["take_home"]}])],
                     ignore_index=True)
    st.dataframe(view.style.format({"Amount": MONEY}).apply(
        lambda r: ["font-weight: 700" if r["Category"] == "Total" else "" for _ in r], axis=1),
        hide_index=True, width="stretch")
    st.caption(f"{EMPLOYER_CATEGORY} lines are not part of take-home; they only count toward 401k Saved.")


# ============================================================ Dashboard
def page_dashboard():
    st.title("Dashboard")
    opts = month_options()
    years = sorted({int(m[:4]) for m in opts}, reverse=True)

    c1, c2, c3 = st.columns([1, 1, 2])
    mode = c1.radio("View", ["Month", "Year"], horizontal=True)
    if mode == "Month":
        month = c2.selectbox("Month", opts, index=opts.index(THIS_MONTH))
        months, label, year = [month], month, int(month[:4])
    else:
        year = c2.selectbox("Year", years, index=years.index(int(THIS_MONTH[:4])))
        months, label = reports.months_of_year(year), str(year)

    tx = reports.expenses_for(months)
    summary = reports.summary_by_category(months, tx)
    t1 = reports.table1(months, summary)
    pay = reports.paycheck_summary(months)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Take-home pay", money(pay["take_home"]))
    m2.metric("Spent (my share)", money(summary.loc[reports.TOTAL, "Actual"]))
    m3.metric("Net Leftover", money(t1.loc["Actual", "Net Leftover"]))
    m4.metric("Transactions", len(tx))

    left, right = st.columns([3, 2])
    with left:
        st.subheader(f"Summary by Category — {label}")
        st.dataframe(summary.style.format(MONEY).map(color_difference, subset=["Difference"]).apply(bold_total, axis=1),
                     width="stretch", height=(len(summary) + 1) * 35 + 3)
        st.caption(f"Actual counts shared transactions at {SHARED_SPLIT:.0%} (my share).")
    with right:
        st.subheader("Table 1")
        st.dataframe(t1.style.format(MONEY), width="stretch")
        st.caption("Net Leftover = take-home of both paychecks − all categories (incl. Investing & Savings). "
                   "Liquid Saved = Investing + Savings. 401k Saved = employee 401K + employer match.")
        pie = summary.drop(index=reports.TOTAL)
        pie = pie[pie["Actual"] > 0].reset_index()
        if not pie.empty:
            fig = px.pie(pie, names="Category", values="Actual", hole=0.4, title="Spending by category")
            fig.update_layout(showlegend=True, height=420, margin=dict(t=40, b=0))
            st.plotly_chart(fig, width="stretch")

    a, b = st.columns(2)
    with a:
        show_paycheck(paycheck_lines(months, "mid"), PERIOD_LABEL["mid"] + (f" ({len(months)} months)" if mode == "Year" else ""))
    with b:
        show_paycheck(paycheck_lines(months, "end"), PERIOD_LABEL["end"] + (f" ({len(months)} months)" if mode == "Year" else ""))

    st.subheader(f"Budget vs actual — {label}")
    bars = summary.drop(index=reports.TOTAL)
    bars = bars[(bars["Budget"] != 0) | (bars["Actual"] != 0)]
    figb = go.Figure()
    figb.add_bar(name="Budget", x=bars.index, y=bars["Budget"])
    figb.add_bar(name="Actual", x=bars.index, y=bars["Actual"])
    figb.update_layout(barmode="group", height=380, margin=dict(t=20))
    st.plotly_chart(figb, width="stretch")

    st.subheader(f"Transactions — {label}")
    if tx.empty:
        st.caption("No transactions in this period. Use **Import CSV** to add some.")
    else:
        view = tx[["date", "bank", "description", "category", "amount", "effective", "shared"]].copy()
        view["date"] = view["date"].dt.strftime("%Y-%m-%d")
        view["shared"] = view["shared"].astype(bool)
        st.dataframe(view.rename(columns={"effective": "my share"}), hide_index=True, width="stretch",
                     column_config={"amount": st.column_config.NumberColumn(format="dollar"),
                                    "my share": st.column_config.NumberColumn(format="dollar")})

    # ---- trends (all data, share-adjusted)
    st.subheader("Monthly trend")
    allexp = db.fetch_expenses()
    if allexp.empty:
        st.caption("No transactions yet.")
    else:
        allexp["category"] = allexp["category"].fillna("Uncategorized")
        trend = allexp.groupby(["month", "category"], as_index=False)["effective"].sum()
        cats_all = [c for c in db.get_categories() if c in set(trend["category"])] + sorted(
            set(trend["category"]) - set(db.get_categories()))
        cats_sel = st.multiselect("Categories", cats_all, default=cats_all[:5])
        if cats_sel:
            st.plotly_chart(px.line(trend[trend["category"].isin(cats_sel)].sort_values("month"),
                                    x="month", y="effective", color="category", markers=True,
                                    labels={"effective": "my share ($)"}), width="stretch")
        tm = sorted(allexp["month"].unique())
        tot = pd.DataFrame({"month": tm,
                            "Spent (my share)": allexp.groupby("month")["effective"].sum().reindex(tm).values,
                            "Take-home": [reports.paycheck_summary([m])["take_home"] for m in tm]})
        st.plotly_chart(px.line(tot.melt("month", var_name="series", value_name="amount"), x="month", y="amount",
                                color="series", markers=True, title="Take-home vs spending"), width="stretch")

    # ---- Sankey (always the chosen year)
    st.subheader(f"Money flow — {year}")
    sk = reports.sankey(year)
    if not sk:
        st.caption("No paycheck or spending data for this year.")
    else:
        fig = go.Figure(go.Sankey(
            node=dict(label=sk["labels"], pad=18, thickness=18),
            link=dict(source=sk["source"], target=sk["target"], value=sk["value"]),
        ))
        fig.update_layout(height=560, margin=dict(t=20))
        st.plotly_chart(fig, width="stretch")
        st.caption("Gross pay → taxes, other deductions, 401k and take-home; take-home → categories (my share) "
                   "and whatever is left over.")


# ============================================================ Budgets
def page_budgets():
    st.title("Budgets")
    cats = db.get_categories()
    if not cats:
        st.info("Add categories first (Categories & Model page).")
        return

    month = st.text_input("Month (YYYY-MM)", value=THIS_MONTH)
    current = db.fetch_budgets(month)
    existing = dict(zip(current["category"], current["amount"]))
    if current.empty:
        eff = db.effective_budget(month)
        if not eff.empty:
            st.caption(f"No budget saved for {month}. Prefilled from {eff['month'].iloc[0]}.")
            existing = dict(zip(eff["category"], eff["amount"]))

    editor_df = pd.DataFrame({"category": cats, "amount": [float(existing.get(c, 0.0)) for c in cats]})
    edited = st.data_editor(editor_df, hide_index=True, disabled=["category"], width="stretch", key=f"bud_{month}",
                            height=(len(cats) + 1) * 35 + 3,
                            column_config={"amount": st.column_config.NumberColumn(format="dollar", min_value=0.0)})
    st.metric("Total monthly budget", money(edited["amount"].sum()))
    if st.button("Save budget for this month", type="primary"):
        for _, r in edited.iterrows():
            db.set_budget(month, r["category"], float(r["amount"]))
        st.success(f"Saved budget for {month}.")
        st.rerun()

    st.divider()
    st.subheader("Budget history")
    hist = db.fetch_budgets()
    if hist.empty:
        st.caption("Nothing saved yet.")
    else:
        pivot = hist.pivot(index="month", columns="category", values="amount").sort_index(ascending=False)
        pivot = pivot[[c for c in cats if c in pivot.columns] + [c for c in pivot.columns if c not in cats]]
        pivot["TOTAL"] = pivot.sum(axis=1)
        st.dataframe(pivot.style.format(MONEY, na_rep="—"), width="stretch")
        sel = st.multiselect("Show categories", [c for c in cats if c in set(hist["category"])],
                             default=[c for c in cats if c in set(hist["category"])][:5], key="bh")
        if sel:
            st.plotly_chart(px.line(hist[hist["category"].isin(sel)].sort_values("month"),
                                    x="month", y="amount", color="category", markers=True,
                                    title="Budget amounts over time"), width="stretch")

    with st.expander("Copy a month's budget forward"):
        a, b = st.columns(2)
        src = a.text_input("From (YYYY-MM)")
        dst = b.text_input("To (YYYY-MM)")
        if st.button("Copy") and src and dst:
            n = db.copy_budget_forward(src, dst)
            st.success(f"Copied {n} rows (existing rows in {dst} left untouched).")


# ============================================================ Paychecks
def page_paychecks():
    st.title("Paychecks")
    st.caption("Two paychecks per month. A month without saved paychecks uses the most recent earlier month.")
    month = st.text_input("Month (YYYY-MM)", value=THIS_MONTH)
    saved = db.fetch_paychecks(month)
    lines = saved
    if saved.empty:
        lines = db.effective_paychecks(month)
        if lines.empty:
            st.warning("No paychecks saved yet.")
            return
        st.caption(f"No paychecks saved for {month}. Prefilled from {lines['month'].iloc[0]}.")

    edited = {}
    cols = st.columns(2)
    for col, period in zip(cols, PAYCHECK_PERIODS):
        with col:
            st.subheader(PERIOD_LABEL[period])
            p = lines[lines["period"] == period][["category", "subcategory", "amount"]].reset_index(drop=True)
            e = st.data_editor(p, hide_index=True, disabled=["category", "subcategory"], width="stretch",
                               key=f"pay_{month}_{period}", height=(len(p) + 1) * 35 + 3,
                               column_config={"amount": st.column_config.NumberColumn(format="dollar", min_value=0.0)})
            t = db.paycheck_totals(e)
            st.metric("Total (take-home)", money(t["take_home"]))
            st.caption(f"Gross {money(t['gross'])} − deductions {money(t['deductions'])}. "
                       f"401k: {money(t['retirement'])} + employer match {money(t['employer_match'])} "
                       f"({EMPLOYER_CATEGORY} lines are not part of take-home).")
            edited[period] = e.assign(period=period)
    both = sum(db.paycheck_totals(e)["take_home"] for e in edited.values())
    st.metric("Take-home for the month", money(both))
    if st.button("Save paychecks for this month", type="primary"):
        db.save_paychecks(month, pd.concat(edited.values(), ignore_index=True))
        st.success(f"Saved paychecks for {month}.")
        st.rerun()

    with st.expander("Copy a month's paychecks forward"):
        a, b = st.columns(2)
        src = a.text_input("From (YYYY-MM)", key="pay_src")
        dst = b.text_input("To (YYYY-MM)", key="pay_dst")
        if st.button("Copy", key="pay_copy") and src and dst:
            n = db.copy_paychecks_forward(src, dst)
            st.success(f"Copied {n} lines (existing lines in {dst} left untouched).")


# ============================================================ Shared expenses
def page_shared():
    st.title("Shared expenses")
    st.caption(f"Mark transactions you split with your roommate. Shared rows count at {SHARED_SPLIT:.0%} in your "
               f"budget; your roommate owes the other {1 - SHARED_SPLIT:.0%}. Nothing is marked automatically.")
    opts = month_options()
    month = st.selectbox("Month", opts, index=opts.index(THIS_MONTH))

    tx = reports.expenses_for([month])
    st.subheader(f"Transactions — {month}")
    if tx.empty:
        st.info("No transactions this month.")
    else:
        f1, f2, f3, f4 = st.columns([2, 2, 2, 1])
        cat_f = f1.multiselect("Category", [c for c in db.get_categories() if c in set(tx["category"])]
                               + sorted(set(tx["category"]) - set(db.get_categories())))
        bank_f = f2.multiselect("Bank", sorted(tx["bank"].dropna().unique()))
        text_f = f3.text_input("Search description")
        only_shared = f4.checkbox("Shared only")
        view = tx
        if cat_f:
            view = view[view["category"].isin(cat_f)]
        if bank_f:
            view = view[view["bank"].isin(bank_f)]
        if text_f:
            view = view[view["description"].str.contains(text_f, case=False, na=False)]
        if only_shared:
            view = view[view["shared"] == 1]

        hints = reports.shared_merchant_hints()
        select_all = st.checkbox(f"Select all {len(view)} shown")
        ed = pd.DataFrame({
            "Select": select_all,
            "Shared": view["shared"].astype(bool).values,
            "Date": view["date"].dt.strftime("%Y-%m-%d").values,
            "Bank": view["bank"].values,
            "Description": view["description"].values,
            "Category": view["category"].values,
            "Amount": view["amount"].values,
            "My share": view["effective"].values,
            "Settled": view["shared_settled"].astype(bool).values,
            "Hint": ["shared before" if classifier.clean(d) in hints and not s else ""
                     for d, s in zip(view["description"], view["shared"])],
        }, index=view["id"].values)
        key = f"shared_{month}_{hash((tuple(cat_f), tuple(bank_f), text_f, only_shared, select_all))}"
        out = st.data_editor(
            ed, key=key, width="stretch", hide_index=True,
            disabled=["Date", "Bank", "Description", "Category", "Amount", "My share", "Settled", "Hint"],
            column_config={"Amount": st.column_config.NumberColumn(format="dollar"),
                           "My share": st.column_config.NumberColumn(format="dollar"),
                           "Hint": st.column_config.TextColumn(help="Merchant was marked shared in another month. "
                                                                    "Just a hint — nothing is changed.")})
        sel = out.index[out["Select"]].tolist()
        b1, b2, b3, b4, b5 = st.columns(5)
        if b1.button("Save Shared checkboxes", type="primary"):
            changed = out["Shared"] != ed["Shared"]
            db.set_shared(out.index[changed & out["Shared"]], True)
            db.set_shared(out.index[changed & ~out["Shared"]], False)
            st.rerun()
        if b2.button(f"Mark {len(sel)} selected shared", disabled=not sel):
            db.set_shared(sel, True)
            st.rerun()
        if b3.button(f"Unmark {len(sel)} selected", disabled=not sel):
            db.set_shared(sel, False)
            st.rerun()
        if b4.button("Settle selected", disabled=not sel):
            db.set_settled(sel, True)
            st.rerun()
        if b5.button("Unsettle selected", disabled=not sel):
            db.set_settled(sel, False)
            st.rerun()
        st.caption("Settling only applies to rows already marked shared (save checkboxes first).")

    st.divider()
    st.subheader("What my roommate owes")
    total_owed = reports.still_owed()
    view_mode = st.radio("View", ["Month", "Year"], horizontal=True, key="rm_view")
    fmt = {c: MONEY for c in ["Shared total", "Roommate owes", "Settled", "Still owed", "Running still owed"]}

    if view_mode == "Month":
        table = reports.roommate_by_category(month)
        m1, m2, m3 = st.columns(3)
        m1.metric(f"Still owed for {month}", money(table.loc[reports.TOTAL, "Still owed"]))
        m2.metric("Carried from earlier months", money(reports.still_owed(before=month)))
        m3.metric("Total still owed by roommate across all months", money(total_owed))
        st.dataframe(table.style.format(fmt).apply(bold_total, axis=1), width="stretch")

        items = reports.itemized(month)
        if items.empty:
            st.caption("No shared transactions this month.")
            return
        for cat in table.drop(index=reports.TOTAL).index:
            rows = items[items["category"] == cat]
            with st.expander(f"{cat} — {len(rows)} item(s), roommate owes {money(rows['roommate'].sum())}"):
                st.dataframe(rows.drop(columns=["id", "category"]).rename(columns={"roommate": "roommate's half"}),
                             hide_index=True, width="stretch",
                             column_config={"amount": st.column_config.NumberColumn("full amount", format="dollar"),
                                            "roommate's half": st.column_config.NumberColumn(format="dollar")})

        s1, s2, s3 = st.columns([2, 1, 1])
        settle_cat = s1.selectbox("Category to settle", table.drop(index=reports.TOTAL).index, key="settle_cat")
        if s2.button("Mark category settled"):
            n = db.set_settled(items.loc[(items["category"] == settle_cat) & ~items["settled"], "id"], True)
            st.success(f"Settled {n} row(s).")
            st.rerun()
        if s3.button("Mark whole month settled"):
            n = db.set_settled(items.loc[~items["settled"], "id"], True)
            st.success(f"Settled {n} row(s).")
            st.rerun()

        c1, c2 = st.columns(2)
        if c1.button("Copy request message"):
            st.session_state["rm_msg"] = month
        c2.download_button("Download itemized CSV", items.drop(columns=["id"]).to_csv(index=False),
                           file_name=f"shared-expenses-{month}.csv", mime="text/csv")
        if st.session_state.get("rm_msg") == month:   # rebuilt every run, so it never shows settled amounts
            st.code(reports.request_message(month), language=None)
            st.caption("Use the copy icon in the box's corner, then paste into Venmo or a text.")
    else:
        years = sorted({int(m[:4]) for m in month_options()}, reverse=True)
        year = st.selectbox("Year", years, index=years.index(int(THIS_MONTH[:4])), key="rm_year")
        st.metric("Total still owed by roommate across all months", money(total_owed))
        by_month = reports.roommate_by_month(year)
        st.dataframe(by_month.style.format(fmt), width="stretch")
        st.caption("Running still owed carries unsettled balances forward, including from earlier years.")
        breakdown = reports.roommate_year_breakdown(year)
        if not breakdown.empty:
            st.markdown("**By category, month by month**")
            st.dataframe(breakdown.style.format(fmt), hide_index=True, width="stretch")


# ============================================================ Import
def page_import():
    st.title("Import CSV")
    tab_csv, tab_manual = st.tabs(["Upload CSV", "Manual entry"])
    with tab_csv:
        import_csv_tab()
    with tab_manual:
        manual_entry_tab()


def manual_entry_tab():
    st.caption("Add a single transaction by hand.")
    banks = db.get_banks()
    with st.form("manual_tx", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)
        d = c1.date_input("Date", value=date.today())
        amount = c2.number_input("Amount", min_value=0.0, step=1.0, format="%.2f",
                                 help="Enter a positive number; use Kind to mark income.")
        kind = c3.selectbox("Kind", ["expense", "income"])
        desc = st.text_input("Description")
        c4, c5, c6 = st.columns(3)
        bank = c4.selectbox("Bank", banks, index=None, placeholder="(none)")
        cats = db.get_categories(kind=None)
        cat = c5.selectbox("Category", cats, index=None, placeholder="(auto-categorize)")
        shared = c6.checkbox("Shared")
        s1, s2 = st.columns(2)
        share_mode = s1.radio("Shared portion", ["Whole expense", "Percent", "Amount"], horizontal=True,
                              help="Only used when Shared is ticked: how much of the expense is split.")
        share_val = s2.number_input("Percent (0-100) or amount shared", min_value=0.0, step=1.0, format="%.2f",
                                    help="Ignored for 'Whole expense'.")
        submitted = st.form_submit_button("Add transaction", type="primary")
    if not submitted:
        return
    if not desc.strip():
        st.error("Description is required.")
        return
    if amount <= 0:
        st.error("Amount must be greater than zero.")
        return
    shared_amount = None
    if shared and share_mode != "Whole expense":
        shared_amount = amount * share_val / 100 if share_mode == "Percent" else share_val
        if share_mode == "Percent" and share_val > 100 or shared_amount > amount:
            st.error("The shared portion can't be more than the whole expense.")
            return
        shared_amount = round(shared_amount, 2)
    row = pd.DataFrame([{"date": d.isoformat(), "bank": bank, "description": desc.strip(), "amount": amount,
                         "kind": kind, "account": "", "category": cat, "shared": int(shared),
                         "shared_amount": shared_amount}])
    done = pipeline.categorize_frame(row, use_llm=bool(cat is None and st.session_state.get("imp_llm", False)))
    if cat:
        done["cat_source"], done["cat_confidence"], done["needs_review"] = "manual", 1.0, 0
    db.insert_transactions(done.to_dict("records"), f"manual-{date.today().isoformat()}-{uuid.uuid4().hex[:6]}")
    msg = f"Added {desc.strip()} ({money(amount)}) on {d.isoformat()}"
    got = done.at[0, "category"]
    st.success(msg + (f" as {got}." if isinstance(got, str) and got else ". It needs review."))


def import_csv_tab():
    st.caption("Required columns: date, description and amount (or separate Debit/Credit). Duplicates are checked "
               "against your database and within the file before anything is saved.")
    f = st.file_uploader("CSV file (bank export, or the Transactions table exported from Numbers)", type=["csv"])
    if not f:
        return
    file_key = f"{f.name}-{f.size}"
    if st.session_state.get("imp_file") != file_key:
        for k in ("imp_clean", "imp_dups", "imp_transfers", "rm_msg"):
            st.session_state.pop(k, None)
        st.session_state["imp_file"] = file_key

    raw = importer.read_csv(f)
    st.write("Preview", raw.head())
    cols = list(raw.columns)
    guess = importer.guess_mapping(cols)

    # ---- bank confirmation
    banks = db.get_banks()
    auto_bank, why = importer.guess_bank(f.name, raw, guess)
    bank = st.selectbox("Bank — applies to every row", banks, index=banks.index(auto_bank) if auto_bank else None,
                        placeholder="Choose…", key=f"bank_{file_key}")
    st.caption(f"Pre-selected from: {why}." if auto_bank else "Couldn't detect the bank — please choose.")
    if not bank:
        return
    settings = db.get_bank(bank)
    saved = settings["mapping"] if importer.saved_mapping_fits(settings["mapping"], cols) else {}
    # A saved mapping decides the required/amount columns; optional Category, Bank, Shared and Settled columns are
    # picked up from this file's headers when the saved mapping doesn't name them.
    defaults = {k: saved.get(k) or (guess.get(k) if not saved or k in ("category", "bank", "shared", "settled") else None)
                for k in importer.FIELDS}
    if saved:
        st.caption(f"Using the saved column mapping for {bank}.")

    # ---- column mapping (keyed by bank so switching bank reloads its defaults)
    opts = ["(none)"] + cols

    def sel(container, label, key):
        default = opts.index(defaults[key]) if defaults.get(key) in opts else 0
        v = container.selectbox(label, opts, index=default, key=f"map_{file_key}_{bank}_{key}")
        return None if v == "(none)" else v

    split_default = bool(defaults.get("debit") and defaults.get("credit") and not defaults.get("amount"))
    layout = st.radio("Amount layout", ["One amount column", "Separate Debit and Credit columns"],
                      index=1 if split_default else 0, horizontal=True, key=f"layout_{file_key}_{bank}")
    c = st.columns(5)
    m = {"date": sel(c[0], "Date column", "date"), "description": sel(c[1], "Description column", "description")}
    if layout.startswith("One"):
        m["amount"] = sel(c[2], "Amount column", "amount")
        charges_negative = st.checkbox(f"{bank} exports charges as negative numbers",
                                       value=settings["charges_negative"], key=f"neg_{file_key}_{bank}")
    else:
        m["debit"] = sel(c[2], "Debit column (charges)", "debit")
        m["credit"] = sel(c[2], "Credit column (refunds/payments)", "credit")
        charges_negative = False
        st.caption("Amount = Debit − Credit, so charges are positive and refunds negative.")
    m["category"] = sel(c[3], "Category column (optional)", "category")
    m["bank"] = sel(c[4], "Bank column (optional)", "bank")
    c2 = st.columns(5)
    m["shared"] = sel(c2[0], "Shared column (optional)", "shared")
    m["settled"] = sel(c2[1], "Settled column (optional)", "settled")
    if m["bank"]:
        st.info(f"The file has a Bank column: each row uses its own value. Blank or unknown values "
                f"(anything not in {', '.join(banks)}) fall back to {bank}.")
    remember = st.checkbox(f"Remember this mapping and sign convention for {bank}", value=not saved,
                           key=f"remember_{file_key}_{bank}")
    use_llm = st.checkbox("Use LM Studio fallback", value=True)

    amount_ok = m.get("amount") or (m.get("debit") and m.get("credit"))
    if not (m["date"] and m["description"] and amount_ok):
        st.warning("Pick date, description and amount (or debit + credit) columns.")
        return

    if st.button("Analyze file"):
        if remember:
            db.save_bank(bank, m, charges_negative)
        clean = importer.normalize(raw, m, bank=bank, charges_negative=charges_negative)
        st.session_state["imp_clean"] = clean
        st.session_state["imp_dups"] = importer.find_duplicates(clean)
        st.session_state["imp_transfers"] = importer.flag_transfers(clean)
        st.session_state["imp_llm"] = use_llm

    clean = st.session_state.get("imp_clean")
    if clean is None:
        return
    dups, transfers = st.session_state["imp_dups"], st.session_state["imp_transfers"]
    missing = clean.index[clean["date_missing"] == 1].tolist()
    st.success(f"{len(clean)} rows · {len(dups)} possible duplicate(s) · {len(transfers)} payment/transfer(s) · "
               f"{len(missing)} missing date(s) · {int(clean['shared'].sum())} shared "
               f"({int(clean['shared_settled'].sum())} settled) · "
               f"banks: {clean['bank'].value_counts().to_dict()}")

    bad_bank = clean[clean["bank_invalid"] == 1]
    if m.get("bank") and not bad_bank.empty:
        st.warning(f"{len(bad_bank)} row(s) had a blank or unknown Bank value and will use {bank}.")
    unknown = clean.loc[clean["category"].isna() & (clean["category_raw"] != ""), "category_raw"].unique()
    if len(unknown):
        st.info(f"Category values that aren't budget categories go through the ML → LLM → review cascade: "
                f"{', '.join(map(str, unknown[:15]))}{'…' if len(unknown) > 15 else ''}")

    # ---- blank dates: fix before import
    fixes: dict[int, str] = {}
    exclude_undated = False
    if missing:
        st.subheader("Rows missing a date — fill them in")
        fix_df = pd.DataFrame({"date": pd.Series([pd.NaT] * len(missing), index=missing, dtype="datetime64[ns]"),
                               "original": clean.loc[missing, "date_raw"].values,
                               "description": clean.loc[missing, "description"].values,
                               "amount": clean.loc[missing, "amount"].values}, index=missing)
        fixed = st.data_editor(fix_df, key=f"datefix_{file_key}", disabled=["original", "description", "amount"],
                               width="stretch", column_config={"date": st.column_config.DateColumn(format="YYYY-MM-DD")})
        fixes = {i: pd.Timestamp(d).strftime("%Y-%m-%d") for i, d in fixed["date"].items() if pd.notna(d)}
        exclude_undated = st.checkbox("Leave out rows I haven't dated yet", value=False)

    # ---- card payments / transfers
    decisions: dict[int, str] = {}
    transfer_skip: set[int] = set()
    if transfers:
        st.subheader("Likely card payments / transfers")
        st.caption("These usually double-count against the card purchases they pay off. "
                   "Investing / Savings transfers are not flagged; they map to those categories.")
        t_bulk = st.radio("Default", ["Skip them", "Import anyway"], horizontal=True, key="tr_bulk")
        for i in transfers:
            r = clean.loc[i]
            a, b = st.columns([3, 2])
            a.markdown(f"{r['date'] or '(no date)'} · {r['bank']} · **{r['description']}** · {money(r['amount'])}")
            ch = b.radio("Action", ["Skip", "Import"], index=0 if t_bulk == "Skip them" else 1, key=f"tr_{i}",
                         horizontal=True, label_visibility="collapsed")
            if ch == "Skip":
                transfer_skip.add(i)

    # ---- duplicates
    if dups:
        st.subheader("Possible duplicates — choose what to do")
        bulk = st.radio("Default for all", ["Skip duplicates", "Import anyway"], horizontal=True)
        for d in dups:
            i = d["new_idx"]
            row = clean.loc[i]
            mt = d["match"]
            with st.container(border=True):
                a, b = st.columns(2)
                a.markdown(f"**New:** {row['date']} · {row['bank']} · {row['description']} · {money(row['amount'])}")
                b.markdown(f"**{'Already in DB' if d['match_source'] == 'db' else 'Earlier in this file'}:** "
                           f"{mt['date']} · {mt['bank'] or '—'} · {mt['description']} · {money(mt['amount'])}")
                st.caption(f"similarity {d['score']:.0%}" + (" · exact match" if d["exact"] else ""))
                choice = st.radio("Action", ["Skip duplicates", "Import anyway"],
                                  index=0 if bulk == "Skip duplicates" else 1,
                                  key=f"dup_{i}", horizontal=True, label_visibility="collapsed")
                decisions[i] = "skip" if choice.startswith("Skip") else "import"

    still_missing = [i for i in missing if i not in fixes]
    blocked = bool(still_missing) and not exclude_undated
    if blocked:
        st.warning(f"{len(still_missing)} row(s) still need a date (or tick “Leave out rows I haven't dated yet”).")
    if st.button("Categorize & import", type="primary", disabled=blocked):
        skip = {i for i, d in decisions.items() if d == "skip"} | transfer_skip
        if exclude_undated:
            skip |= set(still_missing)
        keep = clean.drop(index=list(skip)).copy()
        for i, d in fixes.items():
            if i in keep.index:
                keep.at[i, "date"] = d
        with st.spinner("Categorizing…"):
            done = pipeline.categorize_frame(keep, use_llm=st.session_state["imp_llm"])
        batch = f"{date.today().isoformat()}-{uuid.uuid4().hex[:6]}"
        n = db.insert_transactions(done.to_dict("records"), batch)
        with db.connect() as con:
            for d in dups:
                r = clean.loc[d["new_idx"]]
                con.execute("INSERT INTO dedupe_decisions(new_desc,new_date,new_amount,new_bank,existing_id,decision) "
                            "VALUES(?,?,?,?,?,?)",
                            (r["description"], r["date"], r["amount"], r["bank"],
                             d["match"]["ref"] if d["match_source"] == "db" else None,
                             decisions.get(d["new_idx"])))
        st.success(f"Imported {n} rows ({len(skip)} skipped). "
                   f"Sources: {done['cat_source'].value_counts(dropna=False).to_dict()}. "
                   f"{int(done['needs_review'].sum())} need review."
                   + (f" LLM: {done.attrs['llm_model']}." if done.attrs.get("llm_model") else ""))
        for k in ("imp_clean", "imp_dups", "imp_transfers"):
            st.session_state.pop(k, None)


# ============================================================ Review
def page_review():
    st.title("Review queue")
    qd = db.fetch_review_queue()
    if qd.empty:
        st.success("Nothing to review 🎉")
        return
    cats = db.get_categories()
    st.caption(f"{len(qd)} item(s). Labels you confirm here become training data.")
    for _, r in qd.iterrows():
        with st.container(border=True):
            a, b, c, d = st.columns([4, 2, 3, 1])
            a.markdown(f"**{r['description']}**  \n{r['date']} · {r['bank'] or '—'} · {money(r['amount'])}")
            guess = r["category"] if r["category"] in cats else None
            src = {"llm": "LLM", "ml": "ML model"}.get(r["cat_source"], r["cat_source"])
            b.caption(f"Best guess: {guess or '—'}" + (f" ({src}, {r['cat_confidence']:.0%})"
                                                      if guess and pd.notna(r["cat_confidence"]) else ""))
            choice = c.selectbox("Category", cats, index=cats.index(guess) if guess else None,
                                 key=f"rv_{r['id']}", placeholder="Choose…", label_visibility="collapsed")
            if d.button("Save", key=f"sv_{r['id']}") and choice:
                db.set_category(int(r["id"]), choice)
                st.rerun()
    st.info("After labeling a batch, retrain on the **Categories & Model** page.")


# ============================================================ Transactions
def page_transactions():
    st.title("Transactions")
    tx = db.fetch_transactions()
    if tx.empty:
        st.info("No data.")
        return
    cats = db.get_categories()
    f1, f2, f3, f4 = st.columns(4)
    cat = f1.selectbox("Category", ["All", "Uncategorized"] + cats)
    bank = f2.selectbox("Bank", ["All", "(none)"] + db.get_banks())
    text = f3.text_input("Search description")
    src = f4.selectbox("Source", ["All", "ml", "llm", "manual", "import", "rule"])
    if cat == "Uncategorized":
        tx = tx[tx["category"].isna()]
    elif cat != "All":
        tx = tx[tx["category"] == cat]
    if bank == "(none)":
        tx = tx[tx["bank"].isna() | (tx["bank"] == "")]
    elif bank != "All":
        tx = tx[tx["bank"] == bank]
    if text:
        tx = tx[tx["description"].str.contains(text, case=False, na=False)]
    if src != "All":
        tx = tx[tx["cat_source"] == src]
    tx = tx.assign(effective=db.effective_amount(tx))
    if (tx["kind"] == "income").any():
        st.caption("Rows with kind = income were imported before paychecks existed; they are excluded from budgets.")

    ver = st.session_state.get("tx_ver", 0)   # bumped after every save so stale editor state is dropped
    select_all = st.checkbox(f"Select all {len(tx)} shown")
    banks = sorted((set(db.get_banks()) | set(tx["bank"].dropna())) - {""})
    cat_opts = cats + sorted(set(tx["category"].dropna()) - set(cats))   # keep retired categories visible
    ed = pd.DataFrame({
        "Select": select_all,
        "Date": tx["date"].dt.date.values,
        "Bank": tx["bank"].values,
        "Description": tx["description"].values,
        "Category": tx["category"].values,
        "Amount": tx["amount"].values,
        "Kind": tx["kind"].values,
        "Shared": tx["shared"].astype(bool).values,
        "Shared amount": tx["shared_amount"].values,
        "Settled": tx["shared_settled"].astype(bool).values,
        "My share": tx["effective"].values,
        "Source": tx["cat_source"].values,
        "Confidence": tx["cat_confidence"].values,
    }, index=tx["id"].values)
    key = f"tx_{ver}_{hash((cat, bank, text, src, select_all))}"
    out = st.data_editor(
        ed, key=key, hide_index=True, width="stretch", height=520, num_rows="fixed",
        disabled=["My share", "Source", "Confidence"],
        column_config={
            "Date": st.column_config.DateColumn(format="YYYY-MM-DD", required=True),
            "Bank": st.column_config.SelectboxColumn(options=banks),
            "Description": st.column_config.TextColumn(required=True),
            "Category": st.column_config.SelectboxColumn(options=cat_opts),
            "Amount": st.column_config.NumberColumn(format="dollar", required=True),
            "Kind": st.column_config.SelectboxColumn(options=["expense", "income"], required=True),
            "Shared amount": st.column_config.NumberColumn(
                format="dollar", min_value=0.0,
                help="Optional: the part of the amount split with your roommate. Blank = the whole amount. "
                     "Entering one marks the row shared."),
            "My share": st.column_config.NumberColumn(format="dollar"),
            "Confidence": st.column_config.NumberColumn(format="percent"),
        })

    fields = {"Date": "date", "Bank": "bank", "Description": "description", "Category": "category",
              "Amount": "amount", "Kind": "kind", "Shared": "shared", "Shared amount": "shared_amount",
              "Settled": "shared_settled"}

    def changed_cells() -> dict[int, dict]:
        edits: dict[int, dict] = {}
        for col, field in fields.items():
            diff = ~((out[col] == ed[col]) | (out[col].isna() & ed[col].isna()))
            for tid in out.index[diff]:
                v = out.at[tid, col]
                v = None if pd.isna(v) else v
                if field == "date" and v is not None:
                    v = pd.Timestamp(v).date().isoformat()
                edits.setdefault(int(tid), {})[field] = v
        for tid, e in edits.items():   # a shared amount implies the row is shared
            if e.get("shared_amount") is not None and "shared" not in e:
                e["shared"] = True
        return edits

    edits = changed_cells()
    sel = [int(i) for i in out.index[out["Select"]]]
    if st.button(f"Save {len(edits)} edited row(s)", type="primary", disabled=not edits):
        bad = [t for t, e in edits.items()
               if ("description" in e and not str(e["description"] or "").strip())
               or any(k in e and e[k] is None for k in ("date", "amount", "kind"))]
        over = [t for t, e in edits.items()
                if e.get("shared_amount") is not None and e["shared_amount"] > float(out.at[t, "Amount"]) + 0.004]
        if bad:
            st.error(f"Date, description, amount and kind can't be empty (transaction id {bad[0]}).")
        elif over:
            st.error(f"Shared amount can't exceed the transaction amount (transaction id {over[0]}).")
        else:
            for tid, e in edits.items():
                if "description" in e:
                    e["description"] = e["description"].strip()
                db.update_transactions([tid], e)
            st.session_state["tx_ver"] = ver + 1
            st.rerun()

    with st.expander(f"Bulk edit {len(sel)} selected", expanded=bool(sel)):
        if not sel:
            st.caption("Tick the Select boxes in the table first.")
        st.caption("Only the fields you fill in are changed; the rest are left as they are. "
                   "Unsaved cell edits above are discarded when a bulk action is applied.")
        b1, b2, b3, b4 = st.columns(4)
        b_cat = b1.selectbox("Category", cat_opts, index=None, placeholder="(no change)", key=f"bk_cat_{ver}")
        b_bank = b2.selectbox("Bank", banks, index=None, placeholder="(no change)", key=f"bk_bank_{ver}")
        b_kind = b3.selectbox("Kind", ["expense", "income"], index=None, placeholder="(no change)", key=f"bk_kind_{ver}")
        b_shared = b4.selectbox("Shared", ["(no change)", "Shared", "Not shared"], key=f"bk_sh_{ver}")
        b5, b6, b7 = st.columns(3)
        b_date = b5.date_input("Date", value=None, key=f"bk_date_{ver}")
        b_desc = b6.text_input("Description", key=f"bk_desc_{ver}", placeholder="(no change)")
        b_settled = b7.selectbox("Settled", ["(no change)", "Settled", "Not settled"], key=f"bk_se_{ver}")
        bulk = {k: v for k, v in {
            "category": b_cat, "bank": b_bank, "kind": b_kind,
            "date": b_date.isoformat() if b_date else None,
            "description": b_desc.strip() or None,
            "shared": {"Shared": True, "Not shared": False}.get(b_shared),
            "shared_settled": {"Settled": True, "Not settled": False}.get(b_settled),
        }.items() if v is not None}
        a, d = st.columns([1, 1])
        if a.button(f"Apply to {len(sel)} row(s)", type="primary", disabled=not sel or not bulk):
            db.update_transactions(sel, bulk)
            st.session_state["tx_ver"] = ver + 1
            st.rerun()
        confirm = d.checkbox("Confirm delete", key=f"bk_del_ok_{ver}")
        if d.button(f"Delete {len(sel)} row(s)", disabled=not sel or not confirm):
            db.delete_transactions(sel)
            st.session_state["tx_ver"] = ver + 1
            st.rerun()


# ============================================================ Assistant (floating chat widget)
ASSISTANT_CSS = """
<style>
.st-key-assistant_fab, .st-key-assistant_panel { position: fixed; right: 1.5rem; bottom: 1.5rem; z-index: 1000000; }
.st-key-assistant_fab { width: auto; }
.st-key-assistant_fab button { width: 3.6rem; height: 3.6rem; border-radius: 50%; font-size: 1.6rem; padding: 0;
    box-shadow: 0 4px 14px rgba(0,0,0,.35); }
.st-key-assistant_panel { width: min(430px, calc(100vw - 2rem)); max-height: calc(100vh - 3rem); overflow-y: auto;
    padding: .75rem; border-radius: 14px; border: 1px solid rgba(128,128,128,.4);
    box-shadow: 0 8px 28px rgba(0,0,0,.4); background: var(--assistant-bg, Canvas); color: CanvasText; }
.st-key-assistant_panel [data-testid="stChatInput"] { position: static; }
</style>
"""


def assistant_widget() -> None:
    """A chat bubble fixed to the bottom-right of every page. Click to expand; minimize to shrink it back."""
    st.markdown(ASSISTANT_CSS, unsafe_allow_html=True)
    ag = st.session_state.get("assistant")
    n_pending = len(ag.ctx.open_actions()) if ag else 0
    if not st.session_state.get("assistant_open"):
        with st.container(key="assistant_fab"):
            if st.button("💬" + (f" {n_pending}" if n_pending else ""), key="assistant_show",
                         help="Open the finance assistant" + (f" ({n_pending} action(s) waiting for approval)" if n_pending else "")):
                st.session_state["assistant_open"] = True
                st.rerun()
        return

    with st.container(key="assistant_panel"):
        model = llm.resolve_model(pipeline.llm_model())
        h1, h2, h3 = st.columns([5, 1, 1], vertical_alignment="center")
        h1.markdown("**Assistant**  \n" + (f"<small>{model} · temperature 0</small>" if model else ""),
                    unsafe_allow_html=True)
        if h2.button("🗑", key="assistant_new", help="Start a new conversation"):
            st.session_state.pop("assistant", None)
            st.session_state.pop("assistant_log", None)
            st.rerun()
        if h3.button("—", key="assistant_hide", help="Shrink back to the icon"):
            st.session_state["assistant_open"] = False
            st.rerun()
        if not model:
            st.error(f"The LLM server isn't reachable at {LLM_BASE_URL}. Start LM Studio's server.")
            return
        if ag is None:
            ag = st.session_state["assistant"] = agent_mod.Agent(model)
            st.session_state["assistant_log"] = []
        ag.model = model
        log = st.session_state["assistant_log"]

        def show_message(m):
            with st.chat_message(m["role"]):
                st.markdown(m["content"].replace("$", "\\$"))   # two bare $ would render as LaTeX math
                if m.get("staged"):
                    st.info("Proposed only. Nothing has changed until you approve it under Pending actions.")
                if m.get("events"):
                    st.caption("Checks: " + " ".join(m["events"]))
                if m.get("ungrounded"):
                    st.warning("Couldn't verify these figures against the database: " + ", ".join(m["ungrounded"]))
                if m.get("calls"):
                    with st.expander(f"Data used ({len(m['calls'])} tool call(s))"):
                        for c in m["calls"]:
                            st.markdown(f"**{c.name}** `{json.dumps(c.args)}`")
                            st.json(c.result, expanded=False)

        pending = ag.ctx.open_actions()
        question = None
        with st.container(height=380 if not pending else 260, border=False):
            if not log:
                st.caption("Ask about your money or ask for changes. It answers only from your database and "
                           "never changes anything until you approve. Try: *How much did I spend on groceries each "
                           "month this year?* or *Am I over budget this month?*")
            for m in log:
                show_message(m)
            question = st.session_state.pop("assistant_question", None)
            if question:
                with st.chat_message("assistant"), st.spinner("Looking at your data…"):
                    turn = ag.ask(question)
                log.append({"role": "assistant", "content": turn.answer, "calls": turn.calls,
                            "ungrounded": turn.ungrounded, "staged": turn.staged, "events": turn.events})

        if question:
            st.rerun()   # redraw with the new answer and any pending actions

        for a in pending:
            with st.container(border=True):
                st.markdown(f"**Pending:** {a['summary']}")
                if a["preview"]:
                    st.dataframe(pd.DataFrame(a["preview"]), hide_index=True, width="stretch")
                    if a["rows_affected"] and a["rows_affected"] > len(a["preview"]):
                        st.caption(f"Showing {len(a['preview'])} of {a['rows_affected']} rows.")
                ok_col, no_col = st.columns(2)
                if ok_col.button("Approve", key=f"ap_{a['id']}", type="primary", width="stretch"):
                    ok, msg = agent_tools.apply_action(a)
                    ag.note(f"Action {a['id']} ({a['summary']}) was approved and "
                            + (f"applied. {msg}" if ok else f"FAILED: {msg}"))
                    log.append({"role": "assistant", "content": ("✅ " if ok else "❌ ") + f"{a['summary']}: {msg}"})
                    st.rerun()
                if no_col.button("Reject", key=f"rj_{a['id']}", width="stretch"):
                    agent_tools.reject_action(a)
                    ag.note(f"Action {a['id']} ({a['summary']}) was rejected by the user; nothing changed.")
                    log.append({"role": "assistant", "content": f"Rejected: {a['summary']}"})
                    st.rerun()

        typed = st.chat_input("Ask about your finances…", key="assistant_input")
        if typed:
            log.append({"role": "user", "content": typed})
            st.session_state["assistant_question"] = typed
            st.rerun()


# ============================================================ Categories & Model
def page_model():
    st.title("Categories & Model")
    left, right = st.columns(2)
    with left:
        st.subheader("Categories")
        cats = db.category_table()
        cats["is_savings"] = cats["is_savings"].astype(bool)
        cats["active"] = cats["active"].astype(bool)
        edited = st.data_editor(cats, hide_index=True, width="stretch", disabled=["name", "kind", "active"],
                                key="cat_editor",
                                column_config={"is_savings": st.column_config.CheckboxColumn(
                                    "savings", help="Counts toward Liquid Saved (e.g. Investing, Savings)")})
        if st.button("Save savings flags"):
            for _, r in edited[edited["is_savings"] != cats["is_savings"]].iterrows():
                db.set_category_savings(r["name"], bool(r["is_savings"]))
            st.rerun()
        with st.form("addcat", clear_on_submit=True):
            n = st.text_input("New category")
            sv = st.checkbox("Savings category")
            if st.form_submit_button("Add") and n.strip():
                db.add_category(n.strip(), "expense", sv)
                st.rerun()
        names = db.get_categories(kind=None, active_only=False)
        if names:
            with st.form("rencat"):
                old = st.selectbox("Rename", names)
                new = st.text_input("New name")
                if st.form_submit_button("Rename") and new.strip():
                    db.rename_category(old, new.strip())
                    st.rerun()
            dis = st.selectbox("Retire a category (kept on old transactions)", [""] + db.get_categories())
            if dis and st.button("Retire"):
                db.deactivate_category(dis)
                st.rerun()

        st.subheader("Banks")
        st.caption("Saved CSV column mapping and sign convention per bank, used to pre-fill the Import page.")
        bank = st.selectbox("Bank", db.get_banks())
        b = db.get_bank(bank)
        with st.form(f"bank_{bank}"):
            mp = b["mapping"]
            split = st.checkbox("Separate Debit and Credit columns", value=bool(mp.get("debit")))
            vals = {}
            for k, label in [("date", "Date column"), ("description", "Description column"),
                             ("amount", "Amount column (one-column layout)"), ("debit", "Debit column"),
                             ("credit", "Credit column"), ("category", "Category column"), ("bank", "Bank column")]:
                vals[k] = st.text_input(label, value=mp.get(k) or "").strip() or None
            neg = st.checkbox("Charges are exported as negative numbers (one-column layout)",
                              value=b["charges_negative"])
            if st.form_submit_button("Save bank settings"):
                if split:
                    vals["amount"] = None
                else:
                    vals["debit"] = vals["credit"] = None
                db.save_bank(bank, vals, neg and not split)
                st.success(f"Saved {bank}.")
        with st.form("addbank", clear_on_submit=True):
            nb = st.text_input("Add a bank (name)")
            if st.form_submit_button("Add bank") and nb.strip():
                db.save_bank(nb, {}, False)
                st.rerun()
    with right:
        st.subheader("Classifier")
        llm_step = (f"LLM threshold **{LLM_THRESHOLD:.0%}**" if LLM_THRESHOLD <= 1
                    else "LLM **suggests only** (its guess pre-fills the review queue)")
        st.write(f"ML threshold **{ML_THRESHOLD:.0%}** → {llm_step} → manual review. "
                 "(Set via `FT_ML_THRESHOLD` / `FT_LLM_THRESHOLD`.)")
        par = st.number_input("Parallel LLM requests", min_value=1, max_value=16, value=pipeline.llm_parallel(),
                              help="How many uncertain rows are sent to LM Studio at once. LM Studio only runs them "
                                   "concurrently when its Max Concurrent Predictions setting is at least this high.")
        if par != pipeline.llm_parallel():
            db.set_setting("llm_parallel", int(par))
            st.rerun()
        lab = db.fetch_labeled()
        st.metric("Labeled rows available", len(lab))
        st.caption("Training uses manual + imported labels only, so the model never trains on its own guesses.")
        if st.button("Retrain model", type="primary"):
            try:
                with st.spinner("Training…"):
                    m = classifier.train()
                acc = f"{m['cv_accuracy']:.1%}" if m["cv_accuracy"] is not None else "n/a (too few examples/class)"
                st.success(f"Trained on {m['rows']} rows, {m['categories']} categories. CV accuracy: {acc}")
            except ValueError as e:
                st.error(str(e))
        if st.button("Re-run categorizer on all uncategorized / flagged rows"):
            with db.connect() as con:
                rows = pd.read_sql_query(
                    "SELECT * FROM transactions WHERE kind='expense' AND (category IS NULL OR needs_review=1)", con)
            if rows.empty:
                st.info("Nothing to redo.")
            else:
                rows["category"] = None
                out = pipeline.categorize_frame(rows)
                with db.connect() as con:
                    for _, r in out.iterrows():
                        con.execute(
                            "UPDATE transactions SET category=?, cat_source=?, cat_confidence=?, needs_review=? WHERE id=?",
                            (r["category"], r["cat_source"], r["cat_confidence"], int(r["needs_review"]), int(r["id"])))
                st.success(f"Re-processed {len(out)} rows; {int(out['needs_review'].sum())} still need review.")


{
    "Dashboard": page_dashboard,
    "Budgets": page_budgets,
    "Paychecks": page_paychecks,
    "Shared expenses": page_shared,
    "Import CSV": page_import,
    "Review queue": page_review,
    "Transactions": page_transactions,
    "Categories & Model": page_model,
}[PAGE]()
assistant_widget()

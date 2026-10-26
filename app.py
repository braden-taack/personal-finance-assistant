"""Streamlit dashboard. Run with:  streamlit run app.py"""
from __future__ import annotations

import uuid
from datetime import date

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import classifier
import db
import importer
import llm
import pipeline
from config import LLM_BASE_URL, LLM_THRESHOLD, ML_THRESHOLD

st.set_page_config(page_title="Finance Tracker", page_icon="💰", layout="wide")
db.init_db()

PAGE = st.sidebar.radio(
    "Go to",
    ["Dashboard", "Budgets", "Import CSV", "Review queue", "Transactions", "Categories & Model"],
)
q = len(db.fetch_review_queue())
if q:
    st.sidebar.warning(f"{q} transaction(s) need review")
st.sidebar.caption(f"LLM: {'🟢 reachable' if llm.is_available() else '🔴 offline'} ({LLM_BASE_URL})")


def money(x: float) -> str:
    return f"${x:,.2f}"


# ============================================================ Dashboard
def page_dashboard():
    st.title("Dashboard")
    tx = db.fetch_transactions()
    if tx.empty:
        st.info("No transactions yet. Use **Import CSV** to add some.")
        return

    tx["year"] = tx["date"].dt.year
    tx["month"] = tx["date"].dt.strftime("%Y-%m")
    years = sorted(tx["year"].unique(), reverse=True)

    c1, c2, c3 = st.columns([1, 1, 2])
    mode = c1.radio("View", ["Month", "Year"], horizontal=True)
    year = c2.selectbox("Year", years)
    if mode == "Month":
        months = sorted(tx.loc[tx["year"] == year, "month"].unique(), reverse=True)
        month = c3.selectbox("Month", months)
        scope = tx[tx["month"] == month]
        label = month
    else:
        scope = tx[tx["year"] == year]
        label = str(year)

    exp = scope[scope["kind"] == "expense"].copy()
    inc = scope[scope["kind"] == "income"]
    exp["category"] = exp["category"].fillna("Uncategorized")

    spent, earned = exp["amount"].sum(), inc["amount"].sum()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Spent", money(spent))
    m2.metric("Income", money(earned))
    m3.metric("Net", money(earned - spent))
    m4.metric("Transactions", len(scope))

    # ---- budget vs actual (month mode only)
    if mode == "Month":
        bud = db.effective_budget(month)
        by_cat = exp.groupby("category")["amount"].sum()
        if not bud.empty:
            comp = bud.set_index("category")[["amount"]].rename(columns={"amount": "Budget"})
            comp["Spent"] = by_cat.reindex(comp.index).fillna(0)
            comp["Remaining"] = comp["Budget"] - comp["Spent"]
            st.subheader(f"Budget vs actual — {label}")
            figb = go.Figure()
            figb.add_bar(name="Budget", x=comp.index, y=comp["Budget"])
            figb.add_bar(name="Spent", x=comp.index, y=comp["Spent"])
            figb.update_layout(barmode="group", height=380, margin=dict(t=20))
            st.plotly_chart(figb, use_container_width=True)
            st.dataframe(comp.style.format("${:,.2f}"), use_container_width=True)
        else:
            st.caption("No budget set for this period yet — see the Budgets page.")

    # ---- pie + category table
    left, right = st.columns(2)
    with left:
        st.subheader(f"Spending by category — {label}")
        if exp.empty:
            st.caption("No expenses in range.")
        else:
            pie = exp.groupby("category", as_index=False)["amount"].sum()
            st.plotly_chart(px.pie(pie, names="category", values="amount", hole=0.4), use_container_width=True)
    with right:
        st.subheader("Top merchants")
        top = exp.groupby("description")["amount"].agg(["sum", "count"]).sort_values("sum", ascending=False).head(10)
        st.dataframe(top.rename(columns={"sum": "Total", "count": "Count"}).style.format({"Total": "${:,.2f}"}),
                     use_container_width=True)

    # ---- trends
    st.subheader("Monthly trend")
    allexp = tx[tx["kind"] == "expense"].copy()
    allexp["category"] = allexp["category"].fillna("Uncategorized")
    trend = allexp.groupby(["month", "category"], as_index=False)["amount"].sum()
    cats_sel = st.multiselect("Categories", sorted(trend["category"].unique()),
                              default=sorted(trend["category"].unique())[:5])
    if cats_sel:
        st.plotly_chart(
            px.line(trend[trend["category"].isin(cats_sel)].sort_values("month"),
                    x="month", y="amount", color="category", markers=True),
            use_container_width=True,
        )
    tot = tx.groupby(["month", "kind"], as_index=False)["amount"].sum().sort_values("month")
    st.plotly_chart(px.line(tot, x="month", y="amount", color="kind", markers=True,
                            title="Income vs expenses"), use_container_width=True)

    # ---- Sankey (year view always uses the chosen year)
    st.subheader(f"Income → category flow — {year}")
    yr = tx[tx["year"] == year].copy()
    yr["category"] = yr["category"].fillna("Uncategorized")
    yr_exp = yr[yr["kind"] == "expense"].groupby("category")["amount"].sum().sort_values(ascending=False)
    yr_inc = yr[yr["kind"] == "income"]["amount"].sum()
    if yr_exp.empty:
        st.caption("No expenses to show.")
    else:
        total_exp = yr_exp.sum()
        # Income node -> each category; leftover savings if income exceeds spending.
        source_total = max(yr_inc, total_exp)
        labels = [f"Income ({money(source_total)})"] + list(yr_exp.index)
        sources = [0] * len(yr_exp)
        targets = list(range(1, len(yr_exp) + 1))
        values = list(yr_exp.values)
        if yr_inc > total_exp:
            labels.append("Savings")
            sources.append(0)
            targets.append(len(labels) - 1)
            values.append(yr_inc - total_exp)
        fig = go.Figure(go.Sankey(
            node=dict(label=labels, pad=18, thickness=18),
            link=dict(source=sources, target=targets, value=values),
        ))
        fig.update_layout(height=520, margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)
        if yr_inc < total_exp:
            st.caption("Spending exceeded recorded income this year; the source node shows total spending.")


# ============================================================ Budgets
def page_budgets():
    st.title("Budgets")
    cats = db.get_categories(kind="expense")
    if not cats:
        st.info("Add expense categories first (Categories & Model page).")
        return

    today = date.today()
    month = st.text_input("Month (YYYY-MM)", value=today.strftime("%Y-%m"))
    current = db.fetch_budgets(month)
    existing = dict(zip(current["category"], current["amount"]))
    if current.empty:
        eff = db.effective_budget(month)
        if not eff.empty:
            st.caption(f"No budget saved for {month}. Prefilled from {eff['month'].iloc[0]}.")
            existing = dict(zip(eff["category"], eff["amount"]))

    editor_df = pd.DataFrame({"category": cats, "amount": [float(existing.get(c, 0.0)) for c in cats]})
    edited = st.data_editor(editor_df, hide_index=True, disabled=["category"], use_container_width=True,
                            column_config={"amount": st.column_config.NumberColumn(format="$%.2f", min_value=0.0)})
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
        pivot["TOTAL"] = pivot.sum(axis=1)
        st.dataframe(pivot.style.format("${:,.2f}", na_rep="—"), use_container_width=True)
        long = hist.copy()
        sel = st.multiselect("Show categories", sorted(long["category"].unique()),
                             default=sorted(long["category"].unique())[:5], key="bh")
        if sel:
            st.plotly_chart(px.line(long[long["category"].isin(sel)].sort_values("month"),
                                    x="month", y="amount", color="category", markers=True,
                                    title="Budget amounts over time"), use_container_width=True)

    with st.expander("Copy a month's budget forward"):
        a, b = st.columns(2)
        src = a.text_input("From (YYYY-MM)")
        dst = b.text_input("To (YYYY-MM)")
        if st.button("Copy") and src and dst:
            n = db.copy_budget_forward(src, dst)
            st.success(f"Copied {n} rows (existing rows in {dst} left untouched).")


# ============================================================ Import
def page_import():
    st.title("Import CSV")
    st.caption("Duplicates are checked against your database *and* within the file before anything is saved.")
    f = st.file_uploader("CSV file", type=["csv"])
    if not f:
        return

    raw = importer.read_csv(f)
    st.write("Preview", raw.head())
    guess = importer.guess_mapping(list(raw.columns))
    opts = ["(none)"] + list(raw.columns)

    def sel(label, key, required=False):
        default = opts.index(guess[key]) if guess.get(key) in opts else 0
        v = st.selectbox(label, opts, index=default, key=f"map_{key}")
        return None if v == "(none)" else v

    c = st.columns(5)
    with c[0]:
        m_date = sel("Date column", "date")
    with c[1]:
        m_desc = sel("Description column", "description")
    with c[2]:
        m_amt = sel("Amount column", "amount")
    with c[3]:
        m_cat = sel("Category column (optional)", "category")
    with c[4]:
        m_acct = sel("Account column (optional)", "account")
    invert = st.checkbox("My bank exports spending as negative numbers", value=False)
    use_llm = st.checkbox("Use LM Studio fallback", value=True)

    if not (m_date and m_desc and m_amt):
        st.warning("Pick date, description and amount columns.")
        return

    if st.button("Analyze file"):
        clean = importer.normalize(raw, {"date": m_date, "description": m_desc, "amount": m_amt,
                                         "category": m_cat, "account": m_acct}, invert_sign=invert)
        st.session_state["imp_clean"] = clean
        st.session_state["imp_dups"] = importer.find_duplicates(clean)
        st.session_state["imp_llm"] = use_llm

    clean = st.session_state.get("imp_clean")
    if clean is None:
        return
    dups = st.session_state["imp_dups"]
    st.success(f"{len(clean)} valid rows; {len(dups)} possible duplicate(s).")

    decisions: dict[int, str] = {}
    if dups:
        st.subheader("Possible duplicates — choose what to do")
        bulk = st.radio("Default for all", ["Skip duplicates", "Import anyway"], horizontal=True)
        for d in dups:
            i = d["new_idx"]
            row = clean.loc[i]
            m = d["match"]
            with st.container(border=True):
                a, b = st.columns(2)
                a.markdown(f"**New:** {row['date']} · {row['description']} · {money(row['amount'])}")
                b.markdown(f"**{'Already in DB' if d['match_source']=='db' else 'Earlier in this file'}:** "
                           f"{m['date']} · {m['description']} · {money(m['amount'])}")
                st.caption(f"similarity {d['score']:.0%}" + (" · exact match" if d["exact"] else ""))
                choice = st.radio("Action", ["Skip duplicates", "Import anyway"],
                                  index=0 if bulk == "Skip duplicates" else 1,
                                  key=f"dup_{i}", horizontal=True, label_visibility="collapsed")
                decisions[i] = "skip" if choice.startswith("Skip") else "import"

    if st.button("Categorize & import", type="primary"):
        skip = {i for i, d in decisions.items() if d == "skip"}
        keep = clean.drop(index=list(skip))
        with st.spinner("Categorizing…"):
            done = pipeline.categorize_frame(keep, use_llm=st.session_state["imp_llm"])
        batch = f"{date.today().isoformat()}-{uuid.uuid4().hex[:6]}"
        # make sure any imported category names exist
        for cat in done["category"].dropna().unique():
            if cat not in db.get_categories(active_only=False):
                db.add_category(cat)
        n = db.insert_transactions(done.to_dict("records"), batch)
        # log decisions for audit
        with db.connect() as con:
            for d in dups:
                r = clean.loc[d["new_idx"]]
                con.execute("INSERT INTO dedupe_decisions(new_desc,new_date,new_amount,existing_id,decision) VALUES(?,?,?,?,?)",
                            (r["description"], r["date"], r["amount"],
                             d["match"]["ref"] if d["match_source"] == "db" else None,
                             decisions.get(d["new_idx"])))
        st.success(f"Imported {n} rows ({len(skip)} skipped). "
                   f"Sources: {done['cat_source'].value_counts(dropna=False).to_dict()}. "
                   f"{int(done['needs_review'].sum())} need review.")
        for k in ("imp_clean", "imp_dups"):
            st.session_state.pop(k, None)


# ============================================================ Review
def page_review():
    st.title("Review queue")
    qd = db.fetch_review_queue()
    if qd.empty:
        st.success("Nothing to review 🎉")
        return
    cats = db.get_categories(kind="expense") + db.get_categories(kind="income")
    st.caption(f"{len(qd)} item(s). Labels you confirm here become training data.")
    for _, r in qd.iterrows():
        with st.container(border=True):
            a, b, c, d = st.columns([4, 2, 3, 1])
            a.markdown(f"**{r['description']}**  \n{r['date']} · {money(r['amount'])}")
            guess = r["category"] if r["category"] in cats else None
            b.caption(f"Best guess: {guess or '—'}" + (f" ({r['cat_confidence']:.0%})" if pd.notna(r["cat_confidence"]) else ""))
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
    f1, f2, f3 = st.columns(3)
    cat = f1.selectbox("Category", ["All", "Uncategorized"] + cats)
    text = f2.text_input("Search description")
    src = f3.selectbox("Source", ["All", "ml", "llm", "manual", "import"])
    if cat == "Uncategorized":
        tx = tx[tx["category"].isna()]
    elif cat != "All":
        tx = tx[tx["category"] == cat]
    if text:
        tx = tx[tx["description"].str.contains(text, case=False, na=False)]
    if src != "All":
        tx = tx[tx["cat_source"] == src]
    st.dataframe(tx[["id", "date", "description", "amount", "kind", "category", "cat_source", "cat_confidence", "account"]],
                 hide_index=True, use_container_width=True)
    with st.expander("Edit one transaction"):
        tid = st.number_input("Transaction id", min_value=0, step=1)
        newcat = st.selectbox("New category", cats, index=None, placeholder="Choose…")
        a, b = st.columns(2)
        if a.button("Re-categorize") and newcat and tid:
            db.set_category(int(tid), newcat)
            st.rerun()
        if b.button("Delete", type="secondary") and tid:
            db.delete_transaction(int(tid))
            st.rerun()


# ============================================================ Categories & Model
def page_model():
    st.title("Categories & Model")
    left, right = st.columns(2)
    with left:
        st.subheader("Categories")
        with db.connect() as con:
            cats = pd.read_sql_query("SELECT name, kind, active FROM categories ORDER BY name", con)
        st.dataframe(cats, hide_index=True, use_container_width=True)
        with st.form("addcat", clear_on_submit=True):
            n = st.text_input("New category")
            k = st.selectbox("Kind", ["expense", "income"])
            if st.form_submit_button("Add") and n.strip():
                db.add_category(n.strip(), k)
                st.rerun()
        names = db.get_categories(active_only=False)
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
    with right:
        st.subheader("Classifier")
        st.write(f"ML threshold **{ML_THRESHOLD:.0%}** → LLM threshold **{LLM_THRESHOLD:.0%}** → manual review. "
                 "(Set via `FT_ML_THRESHOLD` / `FT_LLM_THRESHOLD`.)")
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
                    "SELECT * FROM transactions WHERE category IS NULL OR needs_review=1", con)
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
    "Import CSV": page_import,
    "Review queue": page_review,
    "Transactions": page_transactions,
    "Categories & Model": page_model,
}[PAGE]()

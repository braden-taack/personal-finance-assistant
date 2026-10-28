"""Assistant checks on a throwaway database, with a scripted fake LLM (no LM Studio needed).

    python tests/agent_test.py

Covers: read tools against hand-computed numbers, staged writes (nothing changes until approved), validation errors
the model can recover from, the grounding check, and the agent loop (tool calls, retry on invented figures).
"""
import json
import os
import sys
import tempfile
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
tmp = Path(tempfile.mkdtemp())
os.environ["FT_DB_PATH"] = str(tmp / "t.db")
os.environ["FT_MODEL_PATH"] = str(tmp / "m.joblib")

import agent  # noqa: E402
import agent_tools as at  # noqa: E402
import db  # noqa: E402

db.init_db()
M = date.today().strftime("%Y-%m")
PREV = (date.today().replace(day=1) - __import__("datetime").timedelta(days=1)).strftime("%Y-%m")


def tx(d, desc, amt, cat, **kw):
    return {"date": d, "description": desc, "amount": amt, "bank": kw.get("bank", "AMEX"), "category": cat,
            "cat_source": kw.get("source", "manual"), "needs_review": kw.get("review", 0), "shared": kw.get("shared", 0)}


db.insert_transactions([
    tx(f"{PREV}-03", "KROGER 1", 100.0, "Groceries"),
    tx(f"{PREV}-10", "KROGER 2", 50.0, "Groceries"),
    tx(f"{PREV}-12", "CHIPOTLE", 20.0, "Restaurants"),
    tx(f"{M}-01", "KROGER 3", 80.0, "Groceries"),
    tx(f"{M}-02", "SHARED DINNER", 60.0, "Restaurants", shared=1),
    tx(f"{M}-03", "MYSTERY SHOP", 12.5, "Bars", source="llm", review=1),
], "test")
ctx = at.Context()


def call(tool_name, /, **args):
    return at.run_tool(ctx, tool_name, args)


def close(a, b):
    return abs(a - b) < 0.005


# ---- read tools: figures match hand computation
r = call("aggregate", metric="sum", category="Groceries")
assert close(r["overall"]["value"], 230.0) and r["overall"]["count"] == 3, r
r = call("aggregate", metric="sum", group_by="category", month=M)
groups = {g["group"]: g["value"] for g in r["groups"]}
assert close(groups["Restaurants"], 30.0), groups          # shared $60 counts at 50% = $30
assert close(groups["Groceries"], 80.0) and close(groups["Bars"], 12.5), groups
r = call("aggregate", metric="sum", field="amount", category="Restaurants", month=M)
assert close(r["overall"]["value"], 60.0), r
r = call("aggregate", metric="max", category="Groceries")
assert r["overall"]["transaction"]["description"] == "KROGER 1" and close(r["overall"]["value"], 100.0), r
r = call("aggregate", metric="avg", category="Groceries")
assert close(r["overall"]["value"], 230 / 3), r
r = call("aggregate", metric="min", category="Groceries")
assert close(r["overall"]["value"], 50.0), r
r = call("aggregate", metric="count", group_by="bank")
assert r["groups"][0]["count"] == 6, r
r = call("query_transactions", category="Groceries", sort_by="amount", limit=2)
assert r["matching_rows"] == 3 and r["returned_rows"] == 2 and r["rows"][0]["amount"] == 100.0, r
assert close(r["total_amount"], 230.0), r
r = call("trend", category="Groceries")
assert [m["value"] for m in r["months"]] == [150.0, 80.0] and close(r["stats"]["mean_per_month"], 115.0), r
assert r["last_month_partial"] is True and "in progress" in r["note"], r
r = call("trend", group_by="category")
assert {c["category"] for c in r["categories"]} == {"Groceries", "Restaurants", "Bars"}, r
r = call("budget_vs_actual", month=M)
bva = {c["category"]: c for c in r["categories"]}
assert close(bva["Restaurants"]["actual"], 30.0) and "Rent" in bva, r
assert close(r["total"]["difference"], r["total"]["budget"] - r["total"]["actual"]), r
r = call("list_review_queue")
assert r["total_in_queue"] == 1 and r["rows"][0]["suggested_category"] == "Bars", r
r = call("shared_summary", month=M)
assert close(r["month_total"]["roommate_owes"], 30.0) and close(r["still_owed_all_months"], 30.0), r
assert call("get_budget", month=M)["total"] > 0
assert call("get_paychecks", month=M)["month_take_home"] > 0
ov = call("get_overview")
assert ov["transactions"]["count"] == 6 and ov["review_queue_count"] == 1 and "Rent" in [c["name"] for c in ov["categories"]]

# ---- errors the model can recover from
assert "Valid categor" in call("aggregate", metric="sum", category="Groceryes")["error"]
assert "Unknown filter" in call("aggregate", metric="sum", colour="red")["error"]
assert "yyyy-mm" in call("budget_vs_actual", month="March")["error"]
assert "Unknown tool" in call("hack_database")["error"]
assert "No transaction with id" in call("update_transactions", ids=[999], changes={"category": "Gas"})["error"]
assert "at least one real filter" in call("delete_transactions", where={"kind": "all"})["error"]
assert "No transactions match" in call("update_transactions", where={"search": "zzzz"}, changes={"category": "Gas"})["error"]

# ---- writes are staged, never applied until approved
before = db.fetch_transactions()
r = call("update_transactions", where={"search": "kroger"}, changes={"category": "Fast Food"})
assert r["status"] == "pending_user_approval" and r["rows_affected"] == 3, r
assert call("update_transactions", where={"search": "kroger"}, changes={"category": "Fast Food"})["action_id"] == r["action_id"]
assert len(ctx.open_actions()) == 1                               # asking twice doesn't queue twice
assert db.fetch_transactions().equals(before)                     # database untouched
ok, msg = at.apply_action(ctx.pending[0])
assert ok and "3" in msg, msg
assert set(db.fetch_transactions().query("description.str.contains('KROGER')", engine="python")["category"]) == {"Fast Food"}
assert not at.apply_action(ctx.pending[0])[0]                     # can't apply twice

r = call("confirm_review", items=[{"id": int(db.fetch_review_queue().iloc[0]["id"])}])
assert r["status"] == "pending_user_approval"
at.apply_action(ctx.open_actions()[0])
assert db.fetch_review_queue().empty and call("list_review_queue")["total_in_queue"] == 0

# budgets: setting one month's category keeps the inherited amounts for the others
NEXT = f"{date.today().year + 1}-01"
base_total = call("get_budget", month=M)["total"]
r = call("set_budget", month=NEXT, amounts={"Groceries": 999})
assert r["preview"][0]["budget"].endswith("999.00"), r
at.apply_action(ctx.open_actions()[0])
nb = {c["category"]: c["amount"] for c in call("get_budget", month=NEXT)["categories"]}
old = {c["category"]: c["amount"] for c in call("get_budget", month=M)["categories"]}
assert nb["Groceries"] == 999 and nb["Rent"] == old["Rent"], nb
assert "Unknown category" in call("set_budget", month=NEXT, amounts={"Moon": 5})["error"]

# paycheck line
r = call("set_paycheck_line", month=M, period="mid", category="FED Taxes", subcategory="Withholding", amount=500)
assert "take_home_of_this_paycheck" in r["preview"][0], r
at.apply_action(ctx.open_actions()[0])
fed = [l for l in call("get_paychecks", month=M)["paychecks"]["mid"]["lines"] if l["subcategory"] == "Withholding" and l["category"] == "FED Taxes"]
assert fed[0]["amount"] == 500, fed
assert "Existing lines" in call("set_paycheck_line", month=M, period="mid", category="X", subcategory="Y", amount=1)["error"]

# categories
call("manage_category", action="add", name="Pets", is_savings=False)
at.apply_action(ctx.open_actions()[0])
assert "Pets" in db.get_categories()
call("manage_category", action="rename", name="Pets", new_name="Pet Care")
at.apply_action(ctx.open_actions()[0])
assert "Pet Care" in db.get_categories() and "Pets" not in db.get_categories()
assert "already exists" in call("manage_category", action="add", name="pet care")["error"]
call("delete_transactions", ids=[int(db.fetch_transactions().iloc[0]["id"])])
n = len(db.fetch_transactions())
at.apply_action(ctx.open_actions()[0])
assert len(db.fetch_transactions()) == n - 1
r = call("update_transactions", ids=[int(db.fetch_transactions().iloc[0]["id"])], changes={"amount": "abc"})
assert "number" in r["error"], r

# ---- grounding check
ev = [json.dumps({"total": 1234.56, "pct": 0.12, "date": "2026-03-15", "rows": 7})]
assert agent.check_grounding("You spent $1,234.56 on 2026-03-15 (about $1,234.6, 12%).", ev) == []
assert agent.check_grounding("You spent $1,300.00.", ev) == ["$1,300.00"]
assert agent.check_grounding("That was on 2026-04-01.", ev) == ["2026-04-01"]
assert agent.check_grounding("1. Groceries\n2. Rent\nYou have 3 items.", ev) == []
assert agent.check_grounding("Up 45%.", ev) == ["45%"]
assert agent.check_grounding("Budget of $500", ev + ["set the budget to 500"]) == []


# ---- agent loop with a scripted LLM
def scripted(replies):
    seen = []

    def chat(messages, tools=None, model=None):
        seen.append(messages)
        return replies.pop(0)
    chat.seen = seen
    return chat


def tool_call(name, **args):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}


# 1) a grounded answer goes through, with its tool calls recorded
ag = agent.Agent("fake", chat=scripted([tool_call("aggregate", metric="sum", category="Fast Food", month=PREV),
                                         {"role": "assistant", "content": "You spent $150.00 on Groceries."}]))
t = ag.ask("How much on groceries last month?")
assert t.answer == "You spent $150.00 on Groceries." and not t.ungrounded and t.calls[0].name == "aggregate", t
assert close(t.calls[0].result["overall"]["value"], 150.0)

# 2) an invented figure is sent back, then the corrected answer is accepted
chat = scripted([tool_call("aggregate", metric="sum", category="Fast Food", month=PREV),
                 {"role": "assistant", "content": "You spent $175.00 on Groceries."},
                 {"role": "assistant", "content": "You spent $150.00 on Groceries."}])
ag = agent.Agent("fake", chat=chat)
t = ag.ask("How much on groceries last month?")
assert t.answer == "You spent $150.00 on Groceries." and not t.ungrounded
assert any("$175.00" in str(m["content"]) for m in chat.seen[-1] if m["role"] == "user"), "model wasn't told what was wrong"

# 3) an answer it can never ground is shown with the warning attached
ag = agent.Agent("fake", chat=scripted([{"role": "assistant", "content": "You spent $999.99."}] * 3))
t = ag.ask("How much?")
assert t.ungrounded == ["$999.99"], t

# 4) text-form tool calls, reasoning blocks, tool errors and a proposed write
txt = ('<think>hmm</think><tool_call>{"name": "update_transactions", "arguments": {"where": {"search": "chipotle"}, '
       '"changes": {"category": "Fast Food"}}}</tool_call>')
ag = agent.Agent("fake", chat=scripted([{"role": "assistant", "content": txt},
                                         {"role": "assistant", "content": "<think>x</think>I proposed recategorizing 1 transaction."}]))
t = ag.ask("Chipotle is fast food")
assert t.calls[0].result["status"] == "pending_user_approval" and len(ag.ctx.open_actions()) == 1, t
assert t.answer == "I proposed recategorizing 1 transaction." and not t.ungrounded
assert db.fetch_transactions().query("description == 'CHIPOTLE'", engine="python")["category"].iloc[0] == "Restaurants"
ag.note("Action 1 was approved and applied. Updated 1 transaction(s).")
assert ag.messages[-1]["content"].startswith("[app notice]")

ag = agent.Agent("fake", chat=scripted([tool_call("nope"), {"role": "assistant", "content": "I can't do that."}]))
t = ag.ask("do something")
assert "Unknown tool" in t.calls[0].result["error"] and t.answer == "I can't do that."

# 5) old tool output is dropped from history on the next question; LLM failure is reported, not raised
chat = scripted([tool_call("get_overview"), {"role": "assistant", "content": "ok"}, {"role": "assistant", "content": "fine"}])
ag = agent.Agent("fake", chat=chat)
ag.ask("one")
ag.ask("two")
assert all("omitted" in m["content"] for m in chat.seen[-1] if m["role"] == "tool")


def broken(messages, tools=None, model=None):
    raise agent.llm.LLMError("offline")


t = agent.Agent("fake", chat=broken).ask("hi")
assert t.error == "offline" and "couldn't reach" in t.answer

# 5b) the reported bug: a self-computed total is rejected, then the model replies with nothing (it spent its tokens
#     thinking). The user must still get the first answer, flagged, plus a note so the model can explain truthfully.
ag = agent.Agent("fake", chat=scripted([tool_call("get_overview"),
                                         {"role": "assistant", "content": "You have 3 things worth $1,483.78 in total."},
                                         {"role": "assistant", "content": "", "reasoning_content": "hmm"},
                                         {"role": "assistant", "content": ""}]))
t = ag.ask("total?")
assert t.answer.startswith("You have 3") and t.ungrounded == ["$1,483.78"], t
assert any("empty reply" in e for e in t.events) and "[app notice] About your previous answer" in ag.messages[-1]["content"]

# an empty reply followed by a real one is recovered; all-empty gives an honest message instead of a vague one
ag = agent.Agent("fake", chat=scripted([{"role": "assistant", "content": ""}, {"role": "assistant", "content": "Hello."}]))
assert ag.ask("hi").answer == "Hello."
t = agent.Agent("fake", chat=scripted([tool_call("get_overview")] + [{"role": "assistant", "content": ""}] * 3)).ask("hi")
assert "didn't produce a readable answer" in t.answer and "Data used" in t.answer, t.answer

# the over-budget total comes from the tool, so the model never has to add it up
r = call("budget_vs_actual", month=M)
over = [c for c in r["categories"] if c["over_budget"]]
assert r["categories_over_budget"] == len(over) and close(r["total_overspent"], -sum(c["difference"] for c in over)), r

# 6) every tool's schema is valid JSON the server can take
for s in at.schemas():
    json.dumps(s)
    assert s["function"]["name"] in at.TOOLS
print(f"agent tests passed ({len(at.TOOLS)} tools)")

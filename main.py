import json, os, re
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------- 1. LOAD + CLEAN THE DATA (runs once, at startup) ----------
RATES = {"USD": 1, "EUR": 1.11, "INR": 0.01241}      # usd_per_unit from the API

rows = [json.loads(l) for l in open("ledger.ndjson") if l.strip()]
df = pd.DataFrame(rows)

# Dates have mixed timezones ("Z" and "+05:30"). Convert everything to UTC.
df["created_at"] = pd.to_datetime(df["created_at"], utc=True, format="ISO8601")
df["updated_at"] = pd.to_datetime(df["updated_at"], utc=True, format="ISO8601")

# Same order id appears twice (an old and a newer version). Keep the newest.
df = (df.sort_values("updated_at")
        .drop_duplicates("id", keep="last")
        .reset_index(drop=True))

# Convert money to USD (multiply: usd_per_unit).
df["usd"] = df["amount"] * df["currency"].map(RATES)
df["year"] = df["created_at"].dt.year
df["month"] = df["created_at"].dt.month

# ---------- 2. THE PROMPT ----------
SYSTEM = f"""You convert questions about Acme Appliances' order ledger into pandas code.

A DataFrame `df` exists (one row per order, already de-duplicated). Columns:
id, customer (e.g. 'C1169'), region (North/South/East/West/Central),
product, qty, unit_price, amount (in the order's own currency), currency,
status, created_at (UTC datetime), updated_at, usd (amount converted to USD),
year (int), month (int 1-12).

Allowed values:
- status: {sorted(df.status.unique())}
- region: {sorted(df.region.unique())}
- product: {sorted(df['product'].unique())}
- currency: {sorted(df.currency.unique())}

Business rules:
- "Revenue" / "sales" = sum of `usd` where status == 'paid'.
- "Refunds" = orders with status == 'refunded' (refund amount = their `usd`).
- status == 'void' orders are cancelled: never count them as revenue or refunds.
- Always use the `usd` column for money, never `amount`, unless asked for local currency.
- Filter months with the `year` and `month` columns.
- Use product names exactly as in the allowed values (fix capitalisation/spelling).
- "Orders"/"how many" = count of rows. "Units" = sum of `qty`.

Reply with ONLY Python code, no markdown, no comments. The last line must assign
`result` as a plain number, string, or list. Round money with round(x, 2)."""

# ---------- 3. ASK THE LLM ----------
LLM_URL = "https://aipipe.org/openai/v1/chat/completions"
LLM_KEY = os.environ.get("LLM_KEY", "")
LLM_MODEL = "gpt-4.1-mini"

def ask_llm(question, error=None):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    if error:
        msgs.append({"role": "user", "content": f"That code failed with: {error}. Fix it."})
    r = requests.post(LLM_URL, headers={"Authorization": f"Bearer {LLM_KEY}"},
                      json={"model": LLM_MODEL, "temperature": 0, "messages": msgs},
                      timeout=8)
    code = r.json()["choices"][0]["message"]["content"]
    return re.sub(r"^```(?:python)?|```$", "", code.strip(), flags=re.M).strip()

# ---------- 4. RUN THE CODE ----------
SAFE = {k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k)
        for k in ["len", "sum", "min", "max", "round", "sorted", "list", "set",
                  "dict", "float", "int", "str", "abs", "range", "zip", "enumerate"]}

def tidy(x):
    if isinstance(x, pd.DataFrame):
        x = x.iloc[:, 0] if x.shape[1] == 1 else x.to_dict("records")
    if isinstance(x, pd.Series):
        return [tidy(v) for v in x.tolist()] if x.index.equals(pd.RangeIndex(len(x))) \
               else {str(k): tidy(v) for k, v in x.items()}
    if isinstance(x, (np.floating, float)):
        return round(float(x), 2)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (pd.Index, np.ndarray)):
        x = list(x)
    if isinstance(x, (list, tuple, set)):
        return [tidy(v) for v in x]
    return x

def run(code):
    env = {"df": df.copy(), "pd": pd, "np": np}
    exec(code, {"__builtins__": SAFE}, env)
    return tidy(env["result"])

# ---------- 5. THE WEB ENDPOINT ----------
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class Q(BaseModel):
    question: str

@app.get("/")
def health():
    return {"ok": True, "rows": len(df)}

@app.post("/")
def answer(q: Q):
    err = None
    for _ in range(2):                       # one retry if the code crashes
        try:
            return {"answer": run(ask_llm(q.question, err))}
        except Exception as e:
            err = str(e)[:300]
    return {"answer": None}

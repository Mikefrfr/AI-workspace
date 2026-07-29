import pandas as pd
import numpy as np
import ollama
import json
import re
import sys
import os
from io import StringIO
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import uuid
from pathlib import Path
from rag_ingest import ingest_file
from rag_pipeline import query_rag

active_requests = {}
rag_sessions = {}   # collection → { filename, chroma_path, chat_id }

HISTORY_DIR     = Path("chat_history")
HISTORY_DIR.mkdir(exist_ok=True)

RAG_HISTORY_DIR = Path("rag_history")
RAG_HISTORY_DIR.mkdir(exist_ok=True)

UPLOAD_DIR = Path("uploads")      # ← add this
UPLOAD_DIR.mkdir(exist_ok=True)   # ← and this

app = Flask(__name__, static_folder=".")
CORS(app)

try:
    raw     = ollama.list()
    if isinstance(raw, dict):
        _models = [m.get("name") or m.get("model") for m in raw.get("models", [])]
    else:
        _models = [m.model for m in raw.models]
    MODEL = _models[0] if _models else "llama3:latest"
    print(f"[setup] default model: {MODEL}")
except Exception:
    MODEL = "llama3:latest"

app.config["MODEL"] = MODEL 
sessions = {}

from cookbook import cookbook_bp
app.config["SESSIONS"] = sessions
app.register_blueprint(cookbook_bp)


# ── File Loaders ──────────────────────────────────────────────────────────────

def save_message(chat_id, role, content):
    path = HISTORY_DIR / f"{chat_id}.json"
    history = json.loads(path.read_text()) if path.exists() else {"id": chat_id, "messages": [], "file": ""}
    history["messages"].append({
        "role": role,
        "content": content,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    path.write_text(json.dumps(history, indent=2))

def list_chats():
    chats = []
    for f in sorted(HISTORY_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        data = json.loads(f.read_text())
        chats.append({
            "id":      data["id"],
            "file":    data.get("file", ""),
            "preview": data["messages"][-1]["content"][:60] if data["messages"] else "",
            "count":   len(data["messages"]),
            "last":    data["messages"][-1]["time"] if data["messages"] else "",
        })
    # sort by last message time
    chats.sort(key=lambda x: x["last"], reverse=True)
    return chats

def rag_save_message(chat_id, role, content):
    path    = RAG_HISTORY_DIR / f"{chat_id}.json"
    history = json.loads(path.read_text()) if path.exists() else {"id": chat_id, "messages": [], "file": ""}
    history["messages"].append({
        "role":    role,
        "content": content,
        "time":    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    path.write_text(json.dumps(history, indent=2))

def list_rag_chats():
    chats = []
    for f in sorted(RAG_HISTORY_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        data = json.loads(f.read_text())
        chats.append({
            "id":      data["id"],
            "file":    data.get("file", ""),
            "preview": data["messages"][-1]["content"][:60] if data["messages"] else "",
            "count":   len(data["messages"]),
            "last":    data["messages"][-1]["time"] if data["messages"] else "",
        })
    chats.sort(key=lambda x: x["last"], reverse=True)
    return chats

def try_csv(path):
    for sep in [",", ";", "\t", "|"]:
        try:
            df = pd.read_csv(path, sep=sep)
            if df.shape[1] > 1:
                return df
        except Exception:
            pass
    raise ValueError("Could not parse CSV with any common delimiter.")

LOADERS = {
    ".csv": try_csv, ".tsv": lambda p: pd.read_csv(p, sep="\t"),
    ".txt": try_csv,
    ".xlsx": pd.read_excel, ".xls": pd.read_excel, ".xlsm": pd.read_excel,
    ".json": lambda p: pd.read_json(p),
    ".parquet": pd.read_parquet,
}

def load_file(path):
    ext = Path(path).suffix.lower()
    if ext not in LOADERS:
        raise ValueError(f"Unsupported file type '{ext}'.")
    df = LOADERS[ext](path)
    # auto-parse date columns
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().head(20).astype(str)
            if sample.str.match(r"\d{4}[.\-/]\d{2}[.\-/]\d{2}").mean() > 0.7:
                try:
                    df[col] = pd.to_datetime(df[col], infer_datetime_format=True)
                except Exception:
                    pass
    return df.reset_index(drop=True)


# ── Data Summary ──────────────────────────────────────────────────────────────

def get_summary(df, filename):
    lines = [
        f"File: {filename}  |  {len(df):,} rows × {len(df.columns)} columns",
        f"Columns: {list(df.columns)}", "",
        "DataFrame variable: df  (already loaded — do NOT reload it)", "",
        "Column details:",
    ]
    for col in df.columns:
        nulls = df[col].isna().sum()
        if pd.api.types.is_numeric_dtype(df[col]):
            lines.append(f"  {col} — min={df[col].min()}, max={df[col].max()}, mean={df[col].mean():.2f}, nulls={nulls}")
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            lines.append(f"  {col} [datetime] — {df[col].min().date()} to {df[col].max().date()}, nulls={nulls}")
        else:
            top = df[col].value_counts().head(3).index.tolist()
            lines.append(f"  {col} — sample: {top}, nulls={nulls}")
    return "\n".join(lines)


# ── Prompts ───────────────────────────────────────────────────────────────────

MAIN_PROMPT = """\
Dataset info:
{summary}

Question: "{question}"

You are an expert data analyst. Do two things in order:

1. THINKING (2-4 bullet points): briefly explain which columns you will use, what logic you will apply, and what the output will look like.

2. CODE: write the solution in ONE ```python ... ``` block immediately after your thinking.

IMPORTANT: You MUST always write a ```python ... ``` code block. Never skip the code block.

Rules:
- End with print() showing the final answer
- No plt.show(), no GUI, no reloading df
- Guard empty results before .index[0]
- ONLY use data from df — never assume, invent, or hardcode any values
- If the data is insufficient to answer the question, print('Insufficient data to answer this question') and nothing else"""


# ── Code Runner ───────────────────────────────────────────────────────────────

def extract_code(text):
    m = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else None

def auto_install(pkg):
    """pip-install a missing package; return (ok, error_msg)."""
    import subprocess
    print(f"[auto-install] installing '{pkg}'...")
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", pkg, "-q"],
        capture_output=True, text=True
    )
    ok = r.returncode == 0
    print(f"[auto-install] {'ok' if ok else 'FAILED'}: {pkg}")
    return ok, r.stderr if not ok else ""

def run_code(code, df):
    """Execute LLM code; auto-install any missing module and retry once."""
    import importlib
    ns = {"df": df, "pd": pd, "np": np, "datetime": datetime, "json": json}

    for attempt in range(2):
        buf = StringIO()
        old_out = sys.stdout
        sys.stdout = buf
        try:
            exec(code, ns)
            return buf.getvalue().strip() or "(no output)"
        except ModuleNotFoundError as e:
            sys.stdout = old_out
            missing = re.search(r"No module named '([^']+)'", str(e))
            if missing and attempt == 0:
                pkg = missing.group(1).split(".")[0]
                ok, err = auto_install(pkg)
                if ok:
                    try:
                        ns[pkg] = importlib.import_module(pkg)
                    except Exception:
                        pass
                    continue        # retry with module now available
                return f"ERROR: could not auto-install '{pkg}': {err}"
            return f"ERROR: {e}"
        except Exception as e:
            return f"ERROR: {e}"
        finally:
            sys.stdout = old_out


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/cookbook.html")
def cookbook_page():
    return send_from_directory(".", "cookbook.html")

@app.route("/cancel", methods=["POST"])
def cancel():
    chat_id = (request.json or {}).get("chat_id")
    if chat_id:
        active_requests[chat_id] = True
    return jsonify({"ok": True})

@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "No file provided"}), 400
    tmp = str(Path("uploads") / f.filename)
    f.save(tmp)
    try:
        df      = load_file(tmp)
        sid     = f.filename + str(len(df))
        summary = get_summary(df, f.filename)
        chat_id = str(uuid.uuid4())[:8]
        sessions[sid] = {"df": df, "filename": f.filename, "summary": summary, "chat_id": chat_id, "tmp_path": tmp  }
        # create the chat history file
        path = HISTORY_DIR / f"{chat_id}.json"
        path.write_text(json.dumps({"id": chat_id, "file": f.filename, "tmp_path": tmp, "messages": []}, indent=2))
        return jsonify({
            "session_id": sid,
            "chat_id":    chat_id,
            "filename":   f.filename,
            "rows":       len(df),
            "cols":       len(df.columns),
            "columns":    list(df.columns),
            "preview":    df.head(5).to_dict(orient="records"),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/chats", methods=["GET"])
def get_chats():
    return jsonify(list_chats())

@app.route("/chats/<chat_id>", methods=["GET"])
def get_chat(chat_id):
    path = HISTORY_DIR / f"{chat_id}.json"
    if not path.exists():
        return jsonify({"error": "Not found"}), 404
    return jsonify(json.loads(path.read_text()))

@app.route("/chats/<chat_id>", methods=["DELETE"])
def delete_chat(chat_id):
    path = HISTORY_DIR / f"{chat_id}.json"
    if path.exists():
        path.unlink()
    return jsonify({"ok": True})

@app.route("/chats/<chat_id>/restore", methods=["POST"])
def restore_chat(chat_id):
    path = HISTORY_DIR / f"{chat_id}.json"
    if not path.exists():
        return jsonify({"error": "Chat not found"}), 404
    data     = json.loads(path.read_text())
    tmp_path = data.get("tmp_path")
    filename = data.get("file")
    if not tmp_path or not os.path.exists(tmp_path):
        return jsonify({"error": f"Original file '{filename}' is no longer available. Please re-upload it."}), 400
    try:
        df      = load_file(tmp_path)
        sid     = filename + str(len(df))
        summary = get_summary(df, filename)
        sessions[sid] = {"df": df, "filename": filename, "summary": summary, "chat_id": chat_id, "tmp_path": tmp_path}
        return jsonify({
            "session_id": sid,
            "chat_id":    chat_id,
            "filename":   filename,
            "rows":       len(df),
            "cols":       len(df.columns),
            "columns":    list(df.columns),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/chat", methods=["POST"])
def chat():
    body     = request.json
    sid      = body.get("session_id")
    chat_id  = body.get("chat_id")
    question = body.get("question", "").strip()
    MODEL    = app.config.get("MODEL", "llama3")

    if sid not in sessions:
        return jsonify({"error": "Session not found."}), 400

    active_requests[chat_id] = False
    sess    = sessions[sid]
    df      = sess["df"]
    summary = sess["summary"]

    # Call 1 — stream response so we can cancel mid-generation
    full_text = ""
    for chunk in ollama.chat(
        model=MODEL,
        options={"temperature": 0.1, "num_predict": 600},
        messages=[{"role": "user", "content": MAIN_PROMPT.format(summary=summary, question=question)}],
        stream=True
    ):
        # check cancel flag on every chunk
        if active_requests.get(chat_id):
            active_requests.pop(chat_id, None)
            return jsonify({"cancelled": True})
        full_text += chunk["message"]["content"]

    plan = full_text.split("```")[0].strip()
    code = extract_code(full_text)

    result = None
    if code:
        result = run_code(code, df)

        for attempt in range(2):
            if not (result or "").startswith("ERROR"):
                break
            if active_requests.get(chat_id):
                active_requests.pop(chat_id, None)
                return jsonify({"cancelled": True})
            hint = ""
            err  = result.lower()
            if "m8" in err or "datetime" in err:
                hint = " HINT: Use (date_b - date_a).days"
            elif "keyerror" in err:
                hint = f" HINT: Available columns: {list(df.columns)}"
            elif "index" in err:
                hint = " HINT: Guard with: if not subset.empty: before .index[0]"

            fix_text = ""
            for chunk in ollama.chat(
                model=MODEL,
                options={"temperature": 0.1, "num_predict": 600},
                messages=[{"role": "user", "content":
                    f"This code errored:\n```python\n{code}\n```\nError: {result}{hint}\nReturn only the fixed ```python...``` block."}],
                stream=True
            ):
                if active_requests.get(chat_id):
                    active_requests.pop(chat_id, None)
                    return jsonify({"cancelled": True})
                fix_text += chunk["message"]["content"]

            code2 = extract_code(fix_text)
            if code2:
                code   = code2
                result = run_code(code2, df)

    if active_requests.get(chat_id):
        active_requests.pop(chat_id, None)
        return jsonify({"cancelled": True})

    # Call 2 — explain
    explain_text = ""
    for chunk in ollama.chat(
        model=MODEL,
        options={"temperature": 0.3, "num_predict": 500},
        messages=[{"role": "user", "content":
            f"Question: {question}\nResult: {result}\nGive a SHORT answer. If single number, ONE sentence only. Use only data from result."}],
        stream=True
    ):
        if active_requests.get(chat_id):
            active_requests.pop(chat_id, None)
            return jsonify({"cancelled": True})
        explain_text += chunk["message"]["content"]

    answer = explain_text.strip()
    active_requests.pop(chat_id, None)

    if chat_id:
        analyst_path = HISTORY_DIR / f"{chat_id}.json"
        if analyst_path.exists():
            save_message(chat_id, "user",      question)
            save_message(chat_id, "thinking",  plan)
            save_message(chat_id, "result",    str(result))
            save_message(chat_id, "assistant", answer)

    return jsonify({"thinking": plan, "result": result, "answer": answer})

@app.route("/models", methods=["GET"])
def get_models():
    try:
        raw = ollama.list()
        print("RAW OLLAMA LIST:", raw)   # check your terminal
        # try both possible response formats
        if isinstance(raw, dict):
            model_list = raw.get("models", [])
            models = [m.get("name") or m.get("model") for m in model_list]
        else:
            models = [m.model for m in raw.models]
    except Exception as e:
        print("ERROR:", e)
        models = []
    return jsonify({"models": [m for m in models if m], "current": app.config.get("MODEL", MODEL)})


@app.route("/set_model", methods=["POST"])
def set_model():
    model = (request.json or {}).get("model", "").strip()
    if not model:
        return jsonify({"error": "No model"}), 400
    global MODEL
    MODEL = model
    app.config["MODEL"] = model
    return jsonify({"ok": True, "model": model})

@app.route("/rag/upload", methods=["POST"])
def rag_upload():
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "No file"}), 400
    tmp     = str(UPLOAD_DIR / f.filename)
    f.save(tmp)
    chat_id    = str(uuid.uuid4())[:8]
    chroma_path = f"./chroma_db/{chat_id}"
    try:
        info = ingest_file(tmp, collection_name=chat_id)
        rag_sessions[chat_id] = {
            "filename":    f.filename,
            "chroma_path": chroma_path,
            "chat_id":     chat_id
        }
        path = RAG_HISTORY_DIR / f"{chat_id}.json"
        path.write_text(json.dumps({
            "id":          chat_id,
            "file":        f.filename,
            "chroma_path": chroma_path,
            "messages":    []
        }, indent=2))
        return jsonify({
            "chat_id":  chat_id,
            "filename": f.filename,
            "chunks":   info["chunks"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/rag/chat", methods=["POST"])
def rag_chat():
    body     = request.json
    chat_id  = body.get("chat_id")
    question = body.get("question", "").strip()
    model    = app.config.get("MODEL", MODEL)

    if chat_id not in rag_sessions:
        return jsonify({"error": "RAG session not found. Please upload a file first."}), 400

    chroma_path = rag_sessions[chat_id]["chroma_path"]

    try:
        result = query_rag(question, chroma_path, model=model)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    rag_save_message(chat_id, "user",      question)
    rag_save_message(chat_id, "assistant", result["answer"])

    return jsonify({
        "answer":  result["answer"],
        "sources": result["sources"]
    })

@app.route("/rag.html")
def rag_page():
    return send_from_directory(".", "rag.html")

@app.route("/rag/chats", methods=["GET"])
def get_rag_chats():
    return jsonify(list_rag_chats())

@app.route("/rag/chats/<chat_id>", methods=["GET"])
def get_rag_chat(chat_id):
    path = RAG_HISTORY_DIR / f"{chat_id}.json"
    if not path.exists():
        return jsonify({"error": "Not found"}), 404
    return jsonify(json.loads(path.read_text()))

@app.route("/rag/chats/<chat_id>", methods=["DELETE"])
def delete_rag_chat(chat_id):
    path = RAG_HISTORY_DIR / f"{chat_id}.json"
    if path.exists(): path.unlink()
    return jsonify({"ok": True})

@app.route("/rag/chats/<chat_id>/restore", methods=["POST"])
def restore_rag_chat(chat_id):
    path = RAG_HISTORY_DIR / f"{chat_id}.json"
    if not path.exists():
        return jsonify({"error": "Chat not found"}), 404
    data        = json.loads(path.read_text())
    chroma_path = data.get("chroma_path")
    filename    = data.get("file")
    if not chroma_path or not os.path.exists(chroma_path):
        return jsonify({"error": f"Chroma collection for '{filename}' no longer exists."}), 400
    rag_sessions[chat_id] = {
        "filename":    filename,
        "chroma_path": chroma_path,
        "chat_id":     chat_id
    }
    try:
        from langchain_chroma import Chroma as ChromaDB
        from langchain_huggingface import HuggingFaceEmbeddings
        db     = ChromaDB(persist_directory=chroma_path,
                          embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
        chunks = len(db.get()["ids"])
    except Exception:
        chunks = "?"
    return jsonify({
        "chat_id":  chat_id,
        "filename": filename,
        "chunks":   chunks
    })


if __name__ == "__main__":
    app.run(debug=False, port=5000)
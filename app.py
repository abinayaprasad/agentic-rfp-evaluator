"""
Agentic RFP Evaluation and Supplier Ranking - Streamlit UI.

Run locally:
    py -3.10 -m streamlit run app.py

Five screens (tabs):
    Criteria | Supplier Input | Leaderboard | Detailed Scorecard | Run Details
"""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from agents.evaluator import get_llm_config
from agents.orchestrator import (
    BatchValidationError,
    load_run_result,
    result_to_json,
    run_evaluation_batch,
    validate_uploads,
)
from agents.qa_agent import DEFAULT_TOP_K, MAX_TOP_K, answer_question_for_run
from agents.orchestrator import attach_human_review
from tools.review_tool import (
    MIN_REASON_CHARS,
    OVERRIDE_CATEGORIES,
    ReviewValidationError,
    validate_adjustment,
    validate_decision,
)
from db import database as db
from db.seed import SEED_CRITERIA, seed_if_empty

PROJECT_ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = PROJECT_ROOT / "data" / "sample_rfps"
SAMPLE_META = {
    "apex_systems.pdf": ("Apex Systems", date(2026, 10, 1), 8.5),
    "brightpath_tech.pdf": ("BrightPath Tech", date(2026, 9, 28), 5.0),
    "nexaworks.pdf": ("NexaWorks", date(2026, 10, 3), 7.0),
    "orbit_digital.pdf": ("Orbit Digital", date(2026, 10, 3), 9.0),
}
BROKEN_SAMPLE = "broken_empty.pdf"

# ---------------------------------------------------------------------------
# Bristol Myers Squibb brand palette (exact hex values from the BMS brand guide)
# ---------------------------------------------------------------------------
BMS = {
    "purple": "#BE2BBB",        # dominant brand accent (never a full field)
    "white": "#FFFFFF",         # preferred background
    "light_gray": "#F0EFEA",    # secondary background / panels
    "dark_gray": "#4F4A4A",     # body text, icons
    "dark_indigo": "#3312AF",   # selective emphasis
    "light_indigo": "#496FE3",  # active states, data visualisation
    "medium_gray": "#ADA599",   # data viz only
    "hypermint": "#2CF1B1",     # data viz only
}
CHART_COLOR = BMS["light_indigo"]
FONT_STACK = "Arial, Helvetica, 'Helvetica Neue', 'Liberation Sans', sans-serif"

BRAND_CSS = f"""
<style>
  /* Brand font on text elements only. Icon fonts are excluded below, otherwise
     Material Symbols ligatures such as "check" render as literal words. */
  html, body, .stApp, .stMarkdown, .stDataFrame, .stMetric, p, li, label, h1, h2, h3, h4,
  .stButton button, .stDownloadButton button, input, textarea, select,
  div[data-testid="stCaptionContainer"], div[data-baseweb="select"] {{
      font-family: {FONT_STACK};
  }}
  span[data-testid="stIconMaterial"], [class*="material-symbols"], .material-icons,
  span[data-testid="stExpanderToggleIcon"], span[data-testid="stStatusWidgetIcon"] {{
      font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons" !important;
      font-feature-settings: "liga";
  }}
  h1, h2, h3, h4 {{ font-family: {FONT_STACK} !important; color: {BMS["dark_gray"]}; font-weight: 700; letter-spacing: -0.01em; }}
  .bms-kv {{ margin: 0.2rem 0 1rem 0; }}
  .bms-kv .bms-chip {{ font-size: 0.9rem; padding: 0.3rem 0.8rem; }}
  h2 {{ border-left: 6px solid {BMS["purple"]}; padding-left: 0.6rem; }}

  /* Branded header band */
  .bms-header {{
      background: {BMS["white"]};
      border-top: 6px solid {BMS["purple"]};
      border-bottom: 1px solid {BMS["light_gray"]};
      padding: 1.1rem 1.4rem 0.9rem 1.4rem;
      margin: -1rem 0 1.2rem 0;
      border-radius: 0 0 10px 10px;
  }}
  .bms-header .title {{ font-size: 1.75rem; font-weight: 700; color: {BMS["dark_gray"]}; margin: 0; }}
  .bms-header .title span {{ color: {BMS["purple"]}; }}
  .bms-header .subtitle {{ color: {BMS["dark_gray"]}; opacity: 0.85; margin: 0.25rem 0 0 0; font-size: 0.95rem; }}
  .bms-chip {{
      display: inline-block; background: {BMS["light_gray"]}; color: {BMS["dark_gray"]};
      border-radius: 999px; padding: 0.15rem 0.7rem; font-size: 0.8rem; margin: 0.5rem 0.35rem 0 0;
      border: 1px solid #E3E1DA;
  }}
  .bms-chip b {{ color: {BMS["dark_indigo"]}; }}

  /* Metrics as light cards with a purple rule */
  div[data-testid="stMetric"] {{
      background: {BMS["light_gray"]}; border-radius: 10px; padding: 0.6rem 0.7rem;
      border-left: 4px solid {BMS["purple"]}; min-width: 0;
  }}
  div[data-testid="stMetricLabel"] p {{ color: {BMS["dark_gray"]}; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; }}
  /* Smaller value text that wraps instead of being cut off with an ellipsis */
  div[data-testid="stMetricValue"] {{ color: {BMS["dark_indigo"]}; font-weight: 700; font-size: 1.45rem; line-height: 1.2; }}
  div[data-testid="stMetricValue"] > div,
  div[data-testid="stMetricValue"] div[data-testid="stMarkdownContainer"] {{
      overflow: visible !important; text-overflow: clip !important; white-space: normal !important;
      overflow-wrap: anywhere;
  }}

  /* Tabs */
  button[data-baseweb="tab"] p {{ font-weight: 600; color: {BMS["dark_gray"]}; }}
  button[data-baseweb="tab"][aria-selected="true"] p {{ color: {BMS["purple"]}; }}

  /* Expanders and sidebar */
  details summary {{ font-weight: 600; color: {BMS["dark_gray"]}; }}
  section[data-testid="stSidebar"] {{ border-right: 1px solid #E3E1DA; }}
  section[data-testid="stSidebar"] h1 {{ color: {BMS["purple"]}; font-size: 1.35rem; }}

  /* Links and captions */
  a {{ color: {BMS["dark_indigo"]}; }}
  .stCaption, small {{ color: {BMS["dark_gray"]}; }}
</style>
"""

st.set_page_config(page_title="Agentic RFP Evaluator", page_icon="📊", layout="wide")
st.markdown(BRAND_CSS, unsafe_allow_html=True)


def _apply_proxy_from_secrets() -> None:
    """Local-dev convenience: if the terminal that launched Streamlit lacks the
    corporate proxy variables, take them from .streamlit/secrets.toml."""
    import os
    for key in ("HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY"):
        try:
            if key in st.secrets and not os.environ.get(key) and not os.environ.get(key.lower()):
                os.environ[key] = str(st.secrets[key])
        except Exception:
            pass


_apply_proxy_from_secrets()
seed_if_empty()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _name_from_filename(filename: str) -> str:
    stem = Path(filename).stem.replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", stem).strip().title()


def _current_result() -> dict | None:
    return st.session_state.get("current_result")


def _set_result(result: dict | None) -> None:
    st.session_state["current_result"] = result


def _review_view(result: dict) -> dict:
    """Result plus the human_review section (adjustments, adjusted ranking, decision)."""
    return attach_human_review(result)


def _reviewer() -> str:
    return str(st.session_state.get("reviewer_name", "")).strip()


def _criteria_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)[["criterion_id", "name", "description", "weight", "max_score", "is_active"]]
    df["is_active"] = df["is_active"].astype(bool)
    return df


def _supplier_rows(uploaded_files, use_samples: bool, include_broken: bool) -> list[dict]:
    """Collect pending suppliers from bundled samples and/or uploads."""
    rows: list[dict] = []
    if use_samples:
        for fname, (name, d, rating) in SAMPLE_META.items():
            path = SAMPLE_DIR / fname
            if path.exists():
                rows.append({"key": f"sample:{fname}", "filename": fname, "file_bytes": path.read_bytes(),
                             "default_name": name, "default_date": d, "default_rating": rating})
        if include_broken and (SAMPLE_DIR / BROKEN_SAMPLE).exists():
            rows.append({"key": f"sample:{BROKEN_SAMPLE}", "filename": BROKEN_SAMPLE,
                         "file_bytes": (SAMPLE_DIR / BROKEN_SAMPLE).read_bytes(),
                         "default_name": "Ghost Corp (scanned PDF)", "default_date": date(2026, 10, 2),
                         "default_rating": 6.0})
    for f in uploaded_files or []:
        rows.append({"key": f"upload:{f.name}:{f.size}", "filename": f.name, "file_bytes": f.getvalue(),
                     "default_name": _name_from_filename(f.name), "default_date": date.today(),
                     "default_rating": 5.0})
    return rows


# ---------------------------------------------------------------------------
# Sidebar: configuration + run history
# ---------------------------------------------------------------------------
cfg = get_llm_config()
with st.sidebar:
    st.title("Agentic RFP Evaluator")
    st.caption("CTMS procurement - supplier proposals scored by an LLM, ranked by deterministic rules.")

    st.subheader("LLM configuration")
    st.write(f"**Provider:** `{cfg.provider}`")
    st.write(f"**Model:** `{cfg.resolved_model()}`")
    if cfg.provider == "mock":
        st.info("Mock provider: keyword-based scoring, no API calls.")
    elif cfg.api_key:
        st.success("API key loaded from secrets.")
    else:
        st.error("No API key found. Add LLM_API_KEY to .streamlit/secrets.toml "
                 "or set LLM_PROVIDER = \"mock\".")

    with st.expander("Connection diagnostics"):
        import os as _os
        import time as _time
        from urllib.parse import urlparse as _urlparse
        _proxy = _os.environ.get("HTTPS_PROXY") or _os.environ.get("https_proxy")
        st.caption(f"Proxy seen by server: `{_urlparse(_proxy).hostname if _proxy else 'none'}`")
        if st.button("Test LLM connection", use_container_width=True):
            from agents.evaluator import make_client
            _t0 = _time.time()
            try:
                _out = make_client(cfg).complete("Return JSON only.", ['Reply with {"ok": true}'])
                st.success(f"OK in {_time.time() - _t0:.1f}s: {_out.strip()[:40]}")
            except Exception as _exc:
                st.error(f"FAILED after {_time.time() - _t0:.1f}s: {type(_exc).__name__}: {str(_exc)[:200]}")

    st.subheader("Reviewer")
    st.text_input("Your name (required to adjust scores or record a decision)", key="reviewer_name",
                  placeholder="e.g. A. Lakshmipathy")

    st.subheader("Run history")
    runs = db.list_runs()
    if runs:
        options = {f"{r['rfp_run_id']}  ({r['status']}, {r['supplier_count']} suppliers)": r["rfp_run_id"]
                   for r in runs}
        chosen = st.selectbox("Load a previous run", ["- current -"] + list(options), index=0)
        if chosen != "- current -" and st.button("Load selected run", use_container_width=True):
            loaded = load_run_result(options[chosen])
            if loaded:
                _set_result(loaded)
                st.toast(f"Loaded {loaded['rfp_run_id']}")
                st.rerun()
    else:
        st.caption("No runs yet.")

    st.divider()
    st.caption("Workflow: Setup -> Input -> Batch -> Evaluate -> Validate -> Score -> "
               "Benchmark -> Rank -> Persist -> Present")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="bms-header">
      <p class="title">Agentic <span>RFP Evaluation</span> &amp; Supplier Ranking</p>
      <p class="subtitle">Clinical Trial Management System (CTMS) procurement &middot; proposals read by an LLM,
      scored against weighted criteria, ranked by deterministic business rules.</p>
      <span class="bms-chip">Provider <b>{cfg.provider}</b></span>
      <span class="bms-chip">Model <b>{cfg.resolved_model()}</b></span>
      <span class="bms-chip">Active criteria <b>{len(db.get_active_criteria())}</b></span>
      <span class="bms-chip">Runs stored <b>{len(db.list_runs())}</b></span>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_criteria, tab_input, tab_board, tab_card, tab_run, tab_ask, tab_decision = st.tabs(
    ["1. Criteria", "2. Supplier Input", "3. Leaderboard", "4. Detailed Scorecard", "5. Run Details",
     "6. Ask the Proposals", "7. Decision"])

# ======================= 1. Criteria =======================================
with tab_criteria:
    st.header("Evaluation criteria")
    st.write("Criteria and weights are stored in SQLite. Edit weights or toggle **Active**; "
             "active weights must total **100**. The prompt and formulas read these at run time.")

    all_criteria = db.get_all_criteria()
    edited = st.data_editor(
        _criteria_df(all_criteria),
        hide_index=True,
        use_container_width=True,
        disabled=["criterion_id", "name", "description", "max_score"],
        column_config={
            "criterion_id": st.column_config.NumberColumn("ID", width="small"),
            "name": st.column_config.TextColumn("Criterion", width="medium"),
            "description": st.column_config.TextColumn("What the LLM inspects", width="large"),
            "weight": st.column_config.NumberColumn("Weight %", min_value=0, max_value=100, step=1, width="small"),
            "max_score": st.column_config.NumberColumn("Max score", width="small"),
            "is_active": st.column_config.CheckboxColumn("Active", width="small"),
        },
        key="criteria_editor",
    )
    active_total = float(edited.loc[edited["is_active"], "weight"].sum())
    c1, c2, c3 = st.columns([1, 1, 2])
    c1.metric("Active criteria", int(edited["is_active"].sum()))
    c2.metric("Active weight total", f"{active_total:.0f} %",
              delta=None if abs(active_total - 100) < 0.01 else f"{active_total - 100:+.0f} vs 100",
              delta_color="inverse")
    with c3:
        if abs(active_total - 100) < 0.01:
            st.success("Weights are valid. Evaluation is enabled.")
        else:
            st.error("Active weights must total 100 before you can evaluate.")

    b1, b2, _ = st.columns([1, 1, 3])
    if b1.button("Save changes", type="primary", use_container_width=True):
        db.bulk_update_criteria([
            {"criterion_id": int(r.criterion_id), "weight": float(r.weight), "is_active": bool(r.is_active)}
            for r in edited.itertuples()
        ])
        st.toast("Criteria saved")
        st.rerun()
    if b2.button("Restore defaults", use_container_width=True):
        by_name = {c["name"]: c for c in SEED_CRITERIA}
        db.bulk_update_criteria([
            {"criterion_id": c["criterion_id"], "weight": by_name[c["name"]]["weight"], "is_active": True}
            for c in all_criteria if c["name"] in by_name
        ])
        st.toast("Default weights restored")
        st.rerun()

# ======================= 2. Supplier Input =================================
with tab_input:
    st.header("Supplier proposals")
    st.write("Upload supplier RFP response PDFs, enter the metadata used for tie-breaks, then evaluate.")

    col_a, col_b = st.columns([2, 1])
    with col_a:
        uploaded = st.file_uploader("Upload supplier PDFs", type=["pdf"], accept_multiple_files=True)
    with col_b:
        use_samples = st.checkbox("Use bundled sample proposals (4 fictional CTMS vendors)", value=not uploaded)
        include_broken = st.checkbox("Also include the unreadable PDF (error-case demo)", value=False,
                                     disabled=not use_samples)

    rows = _supplier_rows(uploaded, use_samples, include_broken)
    uploads: list[dict] = []
    if not rows:
        st.info("Upload at least one PDF or tick the sample proposals option.")
    else:
        st.subheader(f"Supplier metadata ({len(rows)})")
        hdr = st.columns([3, 3, 2, 2])
        for h, label in zip(hdr, ["File", "Supplier name", "Submission date", "Experience rating (0-10)"]):
            h.markdown(f"**{label}**")
        for r in rows:
            c_file, c_name, c_date, c_rate = st.columns([3, 3, 2, 2])
            c_file.write(f"📄 {r['filename']}")
            name = c_name.text_input("Supplier name", value=r["default_name"], key=f"name_{r['key']}",
                                     label_visibility="collapsed")
            sub_date = c_date.date_input("Submission date", value=r["default_date"], key=f"date_{r['key']}",
                                         label_visibility="collapsed")
            rating = c_rate.slider("Experience rating", 0.0, 10.0, float(r["default_rating"]), 0.5,
                                   key=f"rate_{r['key']}", label_visibility="collapsed")
            uploads.append({"file_bytes": r["file_bytes"], "filename": r["filename"], "supplier_name": name,
                            "submission_date": sub_date, "experience_rating": rating})

    problems = validate_uploads(uploads) if uploads else []
    active = db.get_active_criteria()
    weights_ok = db.weights_are_valid(active)
    key_ok = cfg.provider == "mock" or bool(cfg.api_key)

    for p in problems:
        st.warning(p)
    if not weights_ok:
        st.error(f"Active criteria weights total {db.active_weight_total(active):.0f}; fix them on the Criteria tab.")
    if not key_ok:
        st.error("No LLM API key configured.")

    can_run = bool(uploads) and not problems and weights_ok and key_ok
    if st.button(f"Evaluate {len(uploads)} supplier{'s' if len(uploads) != 1 else ''}",
                 type="primary", disabled=not can_run):
        progress = st.progress(0, text="Starting evaluation...")
        step_log = st.container()

        def _cb(n: int, total: int, msg: str) -> None:
            # UI updates must never break the evaluation; swallow any widget error
            try:
                progress.progress(n / total, text=msg)
                step_log.caption(f"Step {n}/{total}: {msg}")
            except Exception:
                pass

        try:
            result = run_evaluation_batch(uploads, progress_callback=_cb)
            _set_result(result)
            st.success(f"Evaluation complete. Run ID **{result['rfp_run_id']}**. "
                       f"Open the Leaderboard tab to see the ranking.")
            if result["run_warnings"]:
                st.warning(f"{len(result['run_warnings'])} validation warning(s) recorded. See Run Details.")
        except BatchValidationError as exc:
            st.error(f"Cannot evaluate: {exc}")
        except Exception as exc:  # surfaced to the user; run is marked FAILED in the DB
            st.error(f"Evaluation failed: {type(exc).__name__}: {exc}")

# ======================= 3. Leaderboard ====================================
with tab_board:
    st.header("Leaderboard")
    result = _current_result()
    if not result:
        st.info("No result loaded. Run an evaluation on the Supplier Input tab or load a previous run from the sidebar.")
    else:
        st.caption(f"Run {result['rfp_run_id']} - {result['created_at']} - model {result['llm_model']}")
        board = pd.DataFrame([{
            "Rank": s["final_rank"],
            "Supplier": s["supplier_name"],
            "Absolute score": round(s["absolute_score"], 2),
            "PPI": round(s["ppi"], 2),
            "Submission date": s["submission_date"],
            "Experience rating": s["experience_rating"],
            "Warnings": len(s.get("warnings", [])),
        } for s in result["suppliers"]])

        top = result["suppliers"][0]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Suppliers evaluated", len(result["suppliers"]))
        m2.metric("Top supplier", top["supplier_name"])
        m3.metric("Top PPI", f"{top['ppi']:.2f}")
        m4.metric("Top absolute score", f"{top['absolute_score']:.1f} / 100")

        st.dataframe(board, hide_index=True, use_container_width=True,
                     column_config={
                         "PPI": st.column_config.ProgressColumn("PPI", min_value=0, max_value=100, format="%.2f"),
                         "Absolute score": st.column_config.NumberColumn(format="%.2f"),
                     })

        hr = _review_view(result)["human_review"]
        if hr["adjustments"]:
            st.caption("The table above is the original LLM-based ranking. A reviewer has adjusted "
                       f"{len(hr['adjustments'])} score(s); the ranking after adjustments is below.")
            st.subheader("Human-adjusted ranking")
            adj_board = pd.DataFrame([{
                "Rank": s["final_rank"],
                "Supplier": s["supplier_name"],
                "LLM rank": s["llm_rank"],
                "Change": ("▲ %d" % s["rank_change"]) if s["rank_change"] > 0
                          else ("▼ %d" % -s["rank_change"]) if s["rank_change"] < 0 else "–",
                "Adjusted score": round(s["absolute_score"], 2),
                "LLM score": round(s["llm_absolute_score"], 2),
                "Adjusted PPI": round(s["ppi"], 2),
                "LLM PPI": round(s["llm_ppi"], 2),
            } for s in hr["adjusted_ranking"]])
            st.dataframe(adj_board, hide_index=True, use_container_width=True,
                         column_config={"Adjusted PPI": st.column_config.ProgressColumn(
                             "Adjusted PPI", min_value=0, max_value=100, format="%.2f")})
            with st.expander("Adjustments applied"):
                for a in hr["adjustments"]:
                    cname = next((c["name"] for c in result["criteria"]
                                  if int(c["criterion_id"]) == int(a["criterion_id"])), a["criterion_id"])
                    st.write(f"- **{a['supplier_name']}** / {cname}: {a['original_score']} → "
                             f"**{a['adjusted_score']}** — {a['reason']} *({a['reviewer']})*")

        st.subheader("Peer Performance Index by supplier")
        chart_df = board[["Supplier", "PPI"]].set_index("Supplier")
        st.bar_chart(chart_df, color=CHART_COLOR, horizontal=True, height=60 + 40 * len(chart_df))

        with st.expander("How the ranking was decided"):
            st.markdown("Tie-break order: **1) higher PPI -> 2) earlier submission date -> "
                        "3) higher experience rating -> 4) supplier name A-Z**.")
            for line in result["tie_break_explanation"]:
                st.write(f"- {line}")

# ======================= 4. Detailed Scorecard =============================
with tab_card:
    st.header("Detailed scorecard")
    result = _current_result()
    if not result:
        st.info("No result loaded.")
    else:
        names = [s["supplier_name"] for s in result["suppliers"]]
        chosen = st.selectbox("Supplier", names, index=0)
        s = next(x for x in result["suppliers"] if x["supplier_name"] == chosen)

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Rank", f"#{s['final_rank']} of {len(names)}")
        k2.metric("Absolute score", f"{s['absolute_score']:.2f}")
        k3.metric("PPI", f"{s['ppi']:.2f}")
        k4.metric("Submitted", s["submission_date"])
        k5.metric("Experience", f"{s['experience_rating']:g} / 10")

        card = pd.DataFrame([{
            "Criterion": c["name"],
            "Weight %": c["weight"],
            "Score": c["score"],
            "Max": c["max_score"],
            "Benchmark": c["benchmark"],
            "Gap": c["gap"],
            "Relative %": round(c["relative_pct"], 2),
            "Weighted contribution": round(c["score"] / c["max_score"] * c["weight"], 2),
        } for c in s["criteria"]])
        st.dataframe(card, hide_index=True, use_container_width=True,
                     column_config={
                         "Relative %": st.column_config.ProgressColumn("Relative %", min_value=0, max_value=100,
                                                                       format="%.1f"),
                     })
        st.caption("Absolute score = sum of (score / max) x weight. Gap = score - benchmark (best score across "
                   "suppliers). Relative % = score / benchmark x 100. PPI = weighted average of Relative %.")

        st.subheader("Evidence and justification")
        for c in s["criteria"]:
            label = f"{c['name']} - score {c['score']}/{c['max_score']} (benchmark {c['benchmark']}, gap {c['gap']})"
            with st.expander(label, expanded=False):
                st.markdown("**Justification**")
                st.write(c["justification"])
                st.markdown("**Evidence from proposal**")
                st.info(c["evidence"])

        # ---- Human score review -------------------------------------------------
        st.subheader("Human score review")
        st.write("A reviewer may replace an LLM score for this supplier. The LLM score is kept, the "
                 "change is logged with your name and reason, and the ranking is recomputed with the "
                 "same deterministic formulas.")
        hr = _review_view(result)["human_review"]
        mine = [a for a in hr["adjustments"] if a["supplier_name"] == chosen]
        reviewer = _reviewer()
        if not reviewer:
            st.info("Enter your name in the sidebar to make adjustments.")
        crit_by_label = {f"{c['name']}  (LLM score {c['score']} / {c['max_score']})": int(c["criterion_id"])
                         for c in s["criteria"]}
        max_any = max(int(c["max_score"]) for c in s["criteria"])
        with st.form(key=f"adjust_form_{chosen}", clear_on_submit=True):
            f1, f2 = st.columns([3, 1])
            crit_label = f1.selectbox("Criterion", list(crit_by_label))
            cid = crit_by_label[crit_label]
            new_score = f2.number_input("Adjusted score", min_value=0, max_value=max_any, value=0, step=1)
            reason = st.text_area(f"Reason (required, at least {MIN_REASON_CHARS} characters)", height=80)
            submitted = st.form_submit_button("Save adjustment", type="primary", disabled=not reviewer)
        if submitted:
            try:
                crit = validate_adjustment(result, chosen, int(cid), int(new_score), reason, reviewer)
                db.save_adjustment(result["rfp_run_id"], chosen, int(cid), int(crit["score"]),
                                   int(new_score), reason, reviewer)
                st.toast(f"Adjustment saved for {chosen}")
                st.rerun()
            except ReviewValidationError as exc:
                st.error(str(exc))
        if mine:
            st.markdown("**Current adjustments for this supplier**")
            for a in mine:
                cname = next((c["name"] for c in s["criteria"] if int(c["criterion_id"]) == int(a["criterion_id"])),
                             a["criterion_id"])
                c_txt, c_btn = st.columns([6, 1])
                c_txt.write(f"- {cname}: {a['original_score']} → **{a['adjusted_score']}** — {a['reason']} "
                            f"*({a['reviewer']}, {a['created_at']})*")
                if c_btn.button("Remove", key=f"rm_adj_{chosen}_{a['criterion_id']}", disabled=not reviewer):
                    db.delete_adjustment(result["rfp_run_id"], chosen, int(a["criterion_id"]))
                    st.rerun()

        col_r, col_s = st.columns(2)
        with col_r:
            st.subheader("Risks identified")
            for r in s.get("risks", []) or ["None reported"]:
                st.write(f"- {r}")
        with col_s:
            st.subheader("Overall summary")
            st.write(s.get("overall_summary") or "Not provided")

        if s.get("warnings"):
            st.subheader("Validation warnings for this supplier")
            for w in s["warnings"]:
                st.warning(w)
        doc = s.get("document") or {}
        llm = s.get("llm") or {}
        st.caption(f"Document: {doc.get('filename', '?')} - {doc.get('page_count', '?')} pages, "
                   f"{doc.get('char_count', '?')} characters. LLM attempts: {llm.get('attempts', '?')}"
                   f"{' (skipped)' if llm.get('skipped') else ''}.")

# ======================= 5. Run Details ====================================
with tab_run:
    st.header("Run details")
    result = _current_result()
    if not result:
        st.info("No result loaded.")
    else:
        st.markdown(
            f"""
            <div class="bms-kv">
              <span class="bms-chip">RFP_RUN_ID <b>{result["rfp_run_id"]}</b></span>
              <span class="bms-chip">Status <b>{result["status"]}</b></span>
              <span class="bms-chip">Created (UTC) <b>{str(result["created_at"])[:19].replace("T", " ")}</b></span>
              <span class="bms-chip">Model <b>{result["llm_model"]}</b></span>
              <span class="bms-chip">Suppliers <b>{len(result["suppliers"])}</b></span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        view = _review_view(result)
        st.download_button(
            "Download complete result as JSON (includes human review)",
            data=result_to_json(view),
            file_name=f"{result['rfp_run_id']}.json",
            mime="application/json",
            type="primary",
        )

        st.subheader("Human review")
        hr = view["human_review"]
        for line in hr["summary"]:
            if line.startswith("OVERRIDE"):
                st.warning(line)
            else:
                st.write(f"- {line}")
        if hr["decision_history"]:
            with st.expander(f"Decision history ({len(hr['decision_history'])} version(s))"):
                for d in hr["decision_history"]:
                    tag = "OVERRIDE" if d["is_override"] else "accepted recommendation"
                    st.write(f"- v{d['version']} · {d['decided_at']} · {d['reviewer']} · "
                             f"selected **{d['selected_supplier']}** (recommended {d['recommended_supplier']}, "
                             f"basis {d['basis']}) · {tag}"
                             + (f" · {d['reason_category']}: {d['reason_text']}" if d["is_override"] else ""))

        st.subheader("Tie-break explanation")
        for line in result["tie_break_explanation"]:
            st.write(f"- {line}")

        st.subheader("Validation warnings")
        if result["run_warnings"]:
            for w in result["run_warnings"]:
                st.warning(w)
        else:
            st.success("No validation warnings. Every LLM scorecard passed schema and range checks.")

        c_left, c_right = st.columns(2)
        with c_left:
            st.subheader("Criteria snapshot used for this run")
            st.dataframe(pd.DataFrame(result["criteria"])[["criterion_id", "name", "weight", "max_score"]],
                         hide_index=True, use_container_width=True)
        with c_right:
            st.subheader("Criterion benchmarks (best score)")
            names = {str(c["criterion_id"]): c["name"] for c in result["criteria"]}
            st.dataframe(pd.DataFrame([{"Criterion": names.get(k, k), "Benchmark": v}
                                       for k, v in result["benchmarks"].items()]),
                         hide_index=True, use_container_width=True)

        if result.get("workflow_log"):
            with st.expander("Workflow log (orchestrator steps)"):
                for i, line in enumerate(result["workflow_log"], start=1):
                    st.write(f"{i}. {line}")
        if result.get("error_message"):
            st.error(result["error_message"])

# ======================= 6. Ask the Proposals (RAG) ========================
with tab_ask:
    st.header("Ask the proposals")
    st.write("Retrieval-augmented Q&A: your question is matched against the stored proposal text "
             "(BM25 over paragraph chunks), the best passages are sent to the LLM, and the answer "
             "cites the supplier for every claim. This never changes scores or ranks.")
    result = _current_result()
    if not result:
        st.info("No run loaded. Evaluate suppliers first or load a previous run from the sidebar.")
    else:
        run_id = result["rfp_run_id"]
        n_chunks = db.count_chunks(run_id)
        supplier_names = [s["supplier_name"] for s in result["suppliers"]]
        st.caption(f"Run {run_id} - {n_chunks} text chunks indexed across {len(supplier_names)} suppliers.")

        if n_chunks == 0:
            st.warning("This run has no stored proposal text (it may predate the Q&A feature or every PDF "
                       "was unreadable). Re-run the evaluation to enable questions.")
        else:
            examples = [
                "Which suppliers offer EU data residency or GDPR controls?",
                "Compare the hypercare and support SLAs offered by each supplier.",
                "Who names specific integrations with the eTMF, EDC and SAP?",
                "What does each supplier say about 21 CFR Part 11 and validation?",
                "Which proposal has the lowest five-year total cost and what assumptions apply?",
            ]
            q_col, k_col = st.columns([4, 1])
            with q_col:
                question = st.text_input("Your question", key="qa_question",
                                         placeholder=examples[0])
            with k_col:
                top_k = st.slider("Passages", 2, MAX_TOP_K, DEFAULT_TOP_K, key="qa_k")
            chosen_suppliers = st.multiselect("Limit to suppliers (optional)", supplier_names, default=[],
                                              key="qa_suppliers")
            with st.expander("Example questions"):
                for ex in examples:
                    st.write(f"- {ex}")

            if st.button("Ask", type="primary", disabled=not question.strip(), key="qa_ask"):
                with st.spinner("Retrieving passages and asking the model..."):
                    qa = answer_question_for_run(run_id, question, k=top_k,
                                                 suppliers=chosen_suppliers or None)
                st.session_state["qa_last"] = {
                    "question": qa.question, "answer": qa.answer, "passages": qa.passages,
                    "error": qa.error, "model": qa.model, "run_id": run_id,
                }

            last = st.session_state.get("qa_last")
            if last and last.get("run_id") == run_id:
                st.subheader("Answer")
                st.markdown(f"**Q:** {last['question']}")
                if last["error"] in ("NO_MATCHES", "NO_CHUNKS", "EMPTY_QUESTION"):
                    st.warning(last["answer"])
                elif last["error"]:
                    st.error(f"Model call failed: {last['error']}")
                    st.write(last["answer"])
                else:
                    st.success(last["answer"])
                    st.caption(f"Answered by {last['model']} from {len(last['passages'])} retrieved passages. "
                               f"Suppliers cited: {', '.join(dict.fromkeys(p['supplier_name'] for p in last['passages']))}.")
                if last["passages"]:
                    with st.expander(f"Retrieved passages ({len(last['passages'])})", expanded=False):
                        for i, p in enumerate(last["passages"], start=1):
                            st.markdown(f"**[{i}] {p['supplier_name']}** - chunk {p['chunk_id']}, "
                                        f"BM25 score {p['score']:.2f}")
                            st.text(p["text"])

# ======================= 7. Decision (human in the loop) ====================
with tab_decision:
    st.header("Award decision")
    st.write("The system recommends the rank-1 supplier after any reviewer score adjustments. A human "
             "records the final award. Selecting any other supplier is an **override** and must be "
             "justified; the recommendation, the selection and the reason are stored together.")
    result = _current_result()
    if not result:
        st.info("No run loaded. Evaluate suppliers first or load a previous run from the sidebar.")
    else:
        view = _review_view(result)
        hr = view["human_review"]
        reviewer = _reviewer()
        basis = "adjusted" if hr["adjustments"] else "llm"
        rec = hr["recommended_supplier"]
        names = [s["supplier_name"] for s in hr["adjusted_ranking"]]

        st.markdown(
            f"""
            <div class="bms-kv">
              <span class="bms-chip">Recommended <b>{rec}</b></span>
              <span class="bms-chip">Basis <b>{"LLM scores + reviewer adjustments" if basis == "adjusted" else "LLM scores"}</b></span>
              <span class="bms-chip">Adjustments <b>{len(hr["adjustments"])}</b></span>
              <span class="bms-chip">Decisions recorded <b>{len(hr["decision_history"])}</b></span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        rank_df = pd.DataFrame([{
            "Rank": s["final_rank"], "Supplier": s["supplier_name"],
            "Absolute score": round(s["absolute_score"], 2), "PPI": round(s["ppi"], 2),
            "LLM rank": s.get("llm_rank", s["final_rank"]),
        } for s in hr["adjusted_ranking"]])
        st.dataframe(rank_df, hide_index=True, use_container_width=True)

        if not reviewer:
            st.warning("Enter your name in the sidebar to record a decision.")

        with st.form("decision_form"):
            selected = st.radio("Select the winning supplier", names, index=names.index(rec), horizontal=True)
            st.caption("Choosing a supplier other than the recommended one is an override: pick a reason "
                       f"category and write a justification of at least {MIN_REASON_CHARS} characters.")
            category = st.selectbox("Override reason category (only needed for an override)",
                                    ["(not an override)"] + OVERRIDE_CATEGORIES)
            notes = st.text_area("Justification / notes", height=100,
                                 placeholder="Required for an override. Optional when accepting the recommendation.")
            submit = st.form_submit_button("Save decision", type="primary", disabled=not reviewer)

        if submit:
            try:
                # validate against the ranking the reviewer actually saw (adjusted when adjustments exist)
                d = validate_decision({"suppliers": hr["adjusted_ranking"]}, selected, reviewer,
                                      None if category == "(not an override)" else category, notes)
                saved = db.save_decision(result["rfp_run_id"], basis=basis, recommended_supplier=rec,
                                         selected_supplier=selected, reviewer=reviewer,
                                         reason_category=d["reason_category"], reason_text=d["reason_text"],
                                         adjustments=hr["adjustments"])
                if saved["is_override"]:
                    st.warning(f"Override recorded (v{saved['version']}): {selected} selected instead of {rec}.")
                else:
                    st.success(f"Decision recorded (v{saved['version']}): {selected} accepted as recommended.")
                st.rerun()
            except ReviewValidationError as exc:
                st.error(str(exc))

        current = hr["decision"]
        st.subheader("Current decision")
        if not current:
            st.info("No decision recorded for this run yet.")
        else:
            if current["is_override"]:
                st.warning(
                    f"**OVERRIDE** — {current['reviewer']} selected **{current['selected_supplier']}** instead of the "
                    f"recommended **{current['recommended_supplier']}** on {current['decided_at']} "
                    f"(v{current['version']}, basis: {current['basis']}).\n\n"
                    f"**Reason category:** {current['reason_category']}\n\n**Justification:** {current['reason_text']}")
            else:
                st.success(
                    f"{current['reviewer']} accepted the recommended supplier **{current['selected_supplier']}** "
                    f"on {current['decided_at']} (v{current['version']}, basis: {current['basis']})."
                    + (f"\n\n**Notes:** {current['reason_text']}" if current.get("reason_text") else ""))
            if len(hr["decision_history"]) > 1:
                with st.expander("Previous versions"):
                    for d in hr["decision_history"][1:]:
                        st.write(f"- v{d['version']} · {d['decided_at']} · {d['reviewer']} selected "
                                 f"{d['selected_supplier']} (recommended {d['recommended_supplier']})"
                                 + (f" · OVERRIDE: {d['reason_category']} — {d['reason_text']}" if d["is_override"] else ""))

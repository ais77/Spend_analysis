import streamlit as st
import pandas as pd
import sqlite3
import json
import plotly.express as px
from openai import AzureOpenAI

if "awaiting_clarification" not in st.session_state:
    st.session_state.awaiting_clarification = False

# ---------------------------------------------------------
# Streamlit page config
# ---------------------------------------------------------
st.set_page_config(page_title="USA Spending Chatbot", layout="wide")

# ---------------------------------------------------------
# Parent company → subsidiaries map
# ---------------------------------------------------------
PARENT_COMPANY_MAP = {
    "GE": [
        "GE",
        "General Electric",
        "GE Aviation",
        "GE Aerospace",
        "GE Power",
        "GE Additive",
    ],
    "SAFRAN": [
        "Safran",
        "Safran Landing Systems",
        "Safran Aerosystems",
        "Safran Electronics",
        "Safran Electronics & Defense",
        "Safran Defense",
    ],
    "CFM": [
        "CFM",
        "CFM International",
        "CFM Materials",
    ],
}

def build_parent_filter(parent_name: str) -> str:
    """Return SQL WHERE clause for a parent company and all subsidiaries."""
    terms = PARENT_COMPANY_MAP.get(parent_name.upper(), [parent_name])
    conditions = [
        f"LOWER(recipient_name) LIKE '%' || LOWER('{term}') || '%'"
        for term in terms
    ]
    return "(" + " OR ".join(conditions) + ")"

# ---------------------------------------------------------
# Load data + SQLite
# ---------------------------------------------------------
@st.cache_resource
def init_db():
    df = pd.read_pickle("embeddings_2.pkl")

    # Convert embeddings to JSON strings
    if "embedding" in df.columns:
        df["embedding"] = df["embedding"].apply(json.dumps)

    conn = sqlite3.connect("spending.db", check_same_thread=False)

    # Check if table exists
    try:
        existing = pd.read_sql_query("SELECT * FROM spending LIMIT 1", conn)

        # If schema differs, rebuild table
        if set(existing.columns) != set(df.columns):
            df.to_sql("spending", conn, index=False, if_exists="replace")

    except Exception:
        # Table does not exist → create it
        df.to_sql("spending", conn, index=False, if_exists="replace")

    return conn, df


conn, df = init_db()

@st.cache_data
def get_schema(df_in):
    cols = list(df_in.columns)
    sample = df_in.head(3).to_dict(orient="records")
    return cols, sample

COLUMNS, SAMPLE_ROWS = get_schema(df)

# ---------------------------------------------------------
# Azure OpenAI client
# ---------------------------------------------------------
client = AzureOpenAI(
    api_key=st.secrets["AZURE_OPENAI_KEY"],
    api_version="2024-12-01-preview",
    azure_endpoint="https://ease-azure-ai.openai.azure.com/",
)

# ---------------------------------------------------------
# SQL error correction
# ---------------------------------------------------------
def fix_sql(original_sql: str, error_message: str, user_query: str) -> str:
    system_prompt = (
        "You are a SQL correction assistant.\n"
        "You receive a user question, an invalid SQL query, and the SQLite error message.\n"
        "Your job is to return ONLY a corrected SQL query.\n"
        "Rules:\n"
        "- Use ONLY the table 'spending'.\n"
        "- Use ONLY these columns:\n"
        f"{COLUMNS}\n"
        "- Keep the intent of the original SQL.\n"
        "- Do NOT return explanations.\n"
    )

    msg = (
        f"User question:\n{user_query}\n\n"
        f"Original SQL:\n{original_sql}\n\n"
        f"SQLite error:\n{error_message}\n\n"
        "Return only the corrected SQL:"
    )

    response = client.chat.completions.create(
        model="gpt-4.1-2",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": msg},
        ],
    )

    return response.choices[0].message.content.strip()

def run_sql_with_correction(sql: str, user_query: str) -> pd.DataFrame:
    try:
        return pd.read_sql_query(sql, conn)
    except Exception as e:
        corrected = fix_sql(sql, str(e), user_query)
        try:
            return pd.read_sql_query(corrected, conn)
        except Exception as e2:
            st.error("SQL failed even after correction.")
            st.write("Original SQL:", sql)
            st.write("Corrected SQL:", corrected)
            st.write("Error:", str(e2))
            st.stop()

# ---------------------------------------------------------
# LLM: main assistant
# ---------------------------------------------------------


def ask_agent(user_query: str) -> str | None:
    """
    Clarifies every new query exactly once.
    If awaiting clarification, do not ask again.
    """

    # If we are already waiting for the user's clarification,
    # do NOT ask again.
    if st.session_state.awaiting_clarification:
        return None

    system_prompt = (
        "You are an Ask Agent. Your job is to determine whether the user's "
        "question is clear enough to answer.\n\n"
        "If the question is ambiguous, incomplete, or could mean multiple things, "
        "respond with a SINGLE clarifying question.\n\n"
        "If the question is already clear, respond with EXACTLY: OK\n"
        "Do not provide answers. Do not provide explanations."
    )

    response = client.chat.completions.create(
        model="gpt-4.1-2",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
    ).choices[0].message.content.strip()

    # If clear → no clarification needed
    if response == "OK":
        return None

    # Otherwise → ask once and wait for user
    st.session_state.awaiting_clarification = True
    return response


def plan_agent(user_query: str) -> dict:
    """
    Returns a structured plan describing how the main assistant should answer.
    """
    system_prompt = (
        "You are a Plan Agent. Your job is to decide the best strategy for answering "
        "the user's question.\n\n"
        "Return ONLY JSON with the following fields:\n"
        "{\n"
        "  \"mode\": \"sql | semantic | visualization\",\n"
        "  \"reason\": \"short explanation\",\n"
        "  \"notes\": \"any constraints or hints for the assistant\"\n"
        "}\n\n"
        "Rules:\n"
        "- Use 'sql' for filtering, grouping, listing, counting, aggregations.\n"
        "- Use 'visualization' only if the user explicitly asks for a chart.\n"
        "- Use 'semantic' for explanations, comparisons, correlations, reasoning.\n"
        "- Do NOT return SQL. Do NOT answer the question.\n"
    )

    response = client.chat.completions.create(
        model="gpt-4.1-2",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
    ).choices[0].message.content

    return json.loads(response)


def ask_bot(user_query: str) -> str:
    system_prompt = (
        "You are a data analysis assistant for a federal spending database.\n\n"
        "The SQL table is named 'spending'.\n\n"
        f"Valid columns:\n{COLUMNS}\n\n"
        f"Sample rows:\n{SAMPLE_ROWS}\n\n"
        "There is a Python-side mapping of parent companies:\n"
        f"{PARENT_COMPANY_MAP}\n\n"
        "When the user mentions GE, SAFRAN, or CFM, treat all subsidiaries as part of that parent.\n"
        "Use fuzzy LIKE matching for each subsidiary.\n\n"
        "Always ask the user question if the response generated answers their query\n\n"

        "DECISION RULES:\n"
        "• If the question asks for listing, filtering, grouping, counting, selecting rows, or numeric aggregation → use SQL.\n"
        "• If the user explicitly asks for a chart or visualization → return a visualization JSON block.\n"
        "• If the question asks for correlations, comparisons, explanations, reasoning, or conceptual insights → use semantic reasoning (no SQL) and generate answer in detail using the data.\n\n"

        "OUTPUT RULES:\n\n"

        "1. For visualization requests, return ONLY JSON:\n"
        "   {\n"
        "     \"sql\": \"...\",\n"
        "     \"chart\": {\n"
        "         \"chart_type\": \"bar|line|scatter|pie|histogram\",\n"
        "         \"x\": \"column_name\",\n"
        "         \"y\": \"column_name or null\",\n"
        "         \"title\": \"...\"\n"
        "     }\n"
        "   }\n\n"

        "2. For SQL queries returning multiple rows (tables), return ONLY JSON:\n"
        "   { \"sql\": \"...\", \"type\": \"table\" }\n\n"

        "3. For numeric SQL results, return ONLY JSON:\n"
        "   { \"sql\": \"...\", \"type\": \"numeric\" }\n\n"

        "4. For text + numeric SQL results, return ONLY JSON:\n"
        "   { \"sql\": \"...\", \"type\": \"text_numeric\", \"text\": \"...\" }\n\n"

        "5. For semantic reasoning (no SQL), return ONLY JSON:\n"
        "   { \"semantic\": true, \"answer\": \"...\" }\n\n"

        "RESTRICTIONS:\n"
        "• Do NOT return Python code.\n"
        "• Do NOT return explanations outside JSON.\n"
        "• Do NOT describe SQL; only return the SQL inside JSON.\n"
        "• Do NOT include backticks.\n"
    )


    # Build chat history (cleaned)
    messages = [{"role": "system", "content": system_prompt}]
    for m in st.session_state.chat_history:
        messages.append(m)
    messages.append({"role": "user", "content": user_query})

    response = client.chat.completions.create(
        model="gpt-4.1-2",
        messages=messages,
    )

    return response.choices[0].message.content

# ---------------------------------------------------------
# Chart builder
# ---------------------------------------------------------
def build_chart(chart_spec, df_result):
    chart_type = chart_spec.get("chart_type", "bar")
    x = chart_spec.get("x")
    y = chart_spec.get("y")
    color = chart_spec.get("color")
    title = chart_spec.get("title", "")

    if len(df_result) == 1 and chart_type == "pie":
        chart_type = "bar"

    if chart_type == "bar":
        return px.bar(df_result, x=x, y=y, color=color, title=title)
    if chart_type == "line":
        return px.line(df_result, x=x, y=y, color=color, title=title)
    if chart_type == "scatter":
        return px.scatter(df_result, x=x, y=y, color=color, title=title)
    if chart_type == "pie":
        return px.pie(df_result, names=x, values=y, title=title)
    if chart_type == "histogram":
        return px.histogram(df_result, x=x, title=title)

# ---------------------------------------------------------
# JSON / SQL / chart handling
# ---------------------------------------------------------
def process_llm_response(raw: str, user_query: str):
    """
    Handles all LLM responses:
    - semantic reasoning
    - SQL table results
    - numeric SQL results
    - text + numeric SQL results
    - visualization
    - plain text fallback
    """

    # Try to parse JSON
    try:
        spec = json.loads(raw)
    except Exception:
        # Plain text fallback
        with st.chat_message("assistant"):
            st.write(raw)
        st.session_state.chat_history.append({"role": "assistant", "content": raw})
        return

    # ---------------------------------------------------------
    # SEMANTIC MODE (no SQL)
    # ---------------------------------------------------------
    if spec.get("semantic") is True:
        answer = spec.get("answer", "")
        with st.chat_message("assistant"):
            st.write(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        return

    # ---------------------------------------------------------
    # TABLE MODE (SQL returning many rows)
    # ---------------------------------------------------------
    if spec.get("type") == "table":
        sql = spec["sql"]
        df_result = run_sql_with_correction(sql, user_query)

        with st.chat_message("assistant"):
            st.dataframe(df_result)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": f"Returned {len(df_result)} rows."
        })
        return

    # ---------------------------------------------------------
    # VISUALIZATION MODE
    # ---------------------------------------------------------
    if "chart" in spec:
        sql = spec["sql"]
        df_result = run_sql_with_correction(sql, user_query)
        fig = build_chart(spec["chart"], df_result)

        # Save chart
        st.session_state.chart_history.append({
            "figure": fig,
            "title": spec["chart"].get("title", "Chart"),
            "data": df_result
        })

        with st.chat_message("assistant"):
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df_result)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": "I created a chart for your request."
        })
        return

    # ---------------------------------------------------------
    # NUMERIC MODE
    # ---------------------------------------------------------
    if spec.get("type") == "numeric":
        sql = spec["sql"]
        df_result = run_sql_with_correction(sql, user_query)
        value = df_result.iloc[0, 0]

        try:
            value = float(value)
            formatted = f"{value:,.2f}"
        except:
            formatted = str(value)

        with st.chat_message("assistant"):
            st.metric("Value", formatted)
            st.dataframe(df_result)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": f"The value is {formatted}."
        })
        return

    # ---------------------------------------------------------
    # TEXT + NUMERIC MODE
    # ---------------------------------------------------------
    if spec.get("type") == "text_numeric":
        sql = spec["sql"]
        df_result = run_sql_with_correction(sql, user_query)

        entity = df_result.iloc[0, 0]
        value = df_result.iloc[0, 1]

        try:
            value = float(value)
            formatted = f"{value:,.2f}"
        except:
            formatted = str(value)

        with st.chat_message("assistant"):
            st.write(spec.get("text", ""))
            st.metric(entity, formatted)
            st.dataframe(df_result)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": spec.get("text", "")
        })
        return

    # ---------------------------------------------------------
    # FALLBACK: Unexpected JSON
    # ---------------------------------------------------------
    with st.chat_message("assistant"):
        st.json(spec)

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": "I returned structured data."
    })

# ---------------------------------------------------------
# Chat UI
# ---------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "chart_history" not in st.session_state:
    st.session_state.chart_history = []

st.title("USA Spending Chatbot")

# Show previous charts
if st.session_state.chart_history:
    st.subheader("Previous Charts")
    for item in st.session_state.chart_history:
        st.markdown(f"### {item['title']}")
        st.plotly_chart(item["figure"], use_container_width=True)

# Show chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
user_query = st.chat_input("Ask something about federal spending...")

if user_query:
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    # ---------------------------------------------------------
    # 1. ASK AGENT — clarify every query once
    # ---------------------------------------------------------
    clarification = ask_agent(user_query)

    if clarification:
        # Ask the clarifying question
        with st.chat_message("assistant"):
            st.write(clarification)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": clarification
        })

        st.stop()  # Wait for user to answer the clarifying question

    # If we reach here, the user has clarified → reset flag
    st.session_state.awaiting_clarification = False

    # ---------------------------------------------------------
    # 2. PLAN AGENT — decide how to answer
    # ---------------------------------------------------------
    plan = plan_agent(user_query)

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": f"Planning: {plan['reason']}"
    })

    # ---------------------------------------------------------
    # 3. MAIN ASSISTANT — now run the real logic
    # ---------------------------------------------------------
    raw = ask_bot(user_query)
    process_llm_response(raw, user_query)

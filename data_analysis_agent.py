import os
import re
import requests
import textwrap
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from dotenv import load_dotenv

# ===============================
# Load API Key (either .env or hardcode here)
# ===============================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or "PUT_YOUR_KEY_HERE"

if not GROQ_API_KEY or GROQ_API_KEY.strip() == "":
    st.error("❌ GROQ_API_KEY not found. Please set it in .env or in code.")
    st.stop()

# ===============================
# Helpers to fix Groq output
# ===============================
def fix_unterminated_strings(code: str) -> str:
    lines = code.split("\n")
    fixed_lines = []
    for line in lines:
        if line.count('"') % 2 == 1:
            line += '"'
        if line.count("'") % 2 == 1:
            line += "'"
        fixed_lines.append(line)
    return "\n".join(fixed_lines)

def sanitize_code(code: str) -> str:
    code = fix_unterminated_strings(code)
    code = re.sub(r'\b0+(\d+)', r'\1', code)
    code = re.sub(r'(\d+)\.(\s|\)|,)', r'\1.0\2', code)
    code = re.sub(
        r"sns\.countplot\([^)]*x\s*=\s*['\"]\w+['\"].*y\s*=\s*['\"]\w+['\"].*?\)",
        lambda m: re.sub(r"y\s*=\s*['\"]\w+['\"],?\s*", "", m.group()),
        code
    )
    return code

# ===============================
# Call Groq API
# ===============================
def call_groq(prompt: str, df: pd.DataFrame, model: str = "llama-3.1-8b-instant", max_tokens: int = 1200):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    schema = ", ".join(df.columns.tolist())

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    f"You are a Python data visualization assistant. "
                    f"The dataframe is called df and has the columns: {schema}. "
                    "When the user requests a visualization, generate a dashboard with multiple charts: "
                    "bar chart, pie chart, line chart, and boxplot if applicable. "
                    "Respect any requested colors. "
                    "Use matplotlib/seaborn subplots with smaller size (8x6). "
                    "⚠️ RULES: "
                    "- Always return valid Python code only (no explanations). "
                    "- Always enclose all strings in single quotes. "
                    "- Always close parentheses, brackets, and quotes. "
                    "- Never include markdown or backticks. "
                    "- Never output partial or truncated code. "
                    "- The code must run without syntax errors."
                )
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
    }

    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    content = data["choices"][0]["message"]["content"]

    content = content.replace("```python", "").replace("```", "").strip()
    content = sanitize_code(content)
    return content

# ===============================
# Fallback generator
# ===============================
def generate_plot_code(df: pd.DataFrame, color: str = "skyblue"):
    cols = df.columns.tolist()
    if len(cols) < 2:
        return "print('Not enough columns to plot')"

    col1, col2 = cols[0], cols[1]

    code = f"""
import matplotlib.pyplot as plt
import seaborn as sns

fig, axs = plt.subplots(2, 2, figsize=(6,4))  # reduced size per plot

sns.countplot(x='{col1}', data=df, ax=axs[0,0], color='{color}')
axs[0,0].set_title('Bar Chart of {col1}')

df['{col1}'].value_counts().plot.pie(
    autopct='%1.1f%%',
    ax=axs[0,1],
    colors=[plt.cm.Paired(i) for i in range(len(df['{col1}'].unique()))]
)
axs[0,1].set_ylabel('')
axs[0,1].set_title('Pie Chart of {col1}')

df.groupby('{col1}')['{col2}'].mean().plot(kind='line', marker='o', color='{color}', ax=axs[1,0])
axs[1,0].set_title('Line Chart of {col2} by {col1}')

sns.boxplot(x='{col1}', y='{col2}', data=df, ax=axs[1,1], color='{color}')
axs[1,1].set_title('Boxplot of {col2} by {col1}')

plt.tight_layout()
plt.show()
"""
    return textwrap.dedent(code)

# ===============================
# Execute code safely
# ===============================
def execute_plot_code(code: str, df: pd.DataFrame, fallback_color: str = "skyblue"):
    try:
        code = sanitize_code(code)
        ast.parse(code)  # validate syntax

        local_env = {"df": df, "plt": plt, "sns": sns, "pd": pd}
        exec(code, {}, local_env)
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.pyplot(plt.gcf())
        st.markdown("</div>", unsafe_allow_html=True)
        plt.clf()
    except SyntaxError as e:
        st.warning(f"⚠️ Groq returned invalid Python. Falling back. Error: {e}")
        safe_code = generate_plot_code(df, color=fallback_color)
        exec(safe_code, {"df": df, "plt": plt, "sns": sns, "pd": pd}, {})
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.pyplot(plt.gcf())
        st.markdown("</div>", unsafe_allow_html=True)
        plt.clf()
    except Exception as e:
        st.error(f"Plot generation failed: {e}")

# ===============================
# Streamlit UI
# ===============================
st.set_page_config(page_title="📊 AI Agent Dashboard", layout="wide")

# Custom UI Theme (ZeRaan style)
st.markdown("""
    <style>
        .stApp { background-color: #0d0f1a; color: white; }
        .stButton>button {
            background: linear-gradient(90deg, #6C5CE7, #a29bfe);
            color: white;
            border-radius: 8px;
            padding: 0.6em 1.2em;
            font-weight: 600;
        }
        .stTextInput>div>div>input {
            background-color: #1e2333;
            color: white;
        }
        .chart-container {
            background-color: #1e2333;
            border: 1px solid #6C5CE7;
            border-radius: 12px;
            padding: 15px;
            margin-bottom: 20px;
            box-shadow: 0 4px 12px rgba(108,92,231,0.3);
        }
    </style>
""", unsafe_allow_html=True)

st.title("💜 Ai Agent For Data Visualization ")

uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success(f"✅ File {uploaded_file.name} uploaded successfully!")
        st.dataframe(df.head(10))

        query = st.text_input("Ask me to visualize something (e.g., 'analyze cp and sex with purple color'):")

        picked_color = st.color_picker("Pick a default chart color", "#6C5CE7")

        if st.button("🚀 Generate Dashboard"):
            st.info("🔎 Generating dashboard visualizations...")
            try:
                code = call_groq(query, df)
            except Exception as e:
                st.warning(f"⚠️ Groq API failed, using fallback. Error: {e}")
                code = generate_plot_code(df, color=picked_color)

            execute_plot_code(code, df, fallback_color=picked_color)

    except Exception as e:
        st.error(f"❌ Failed to read file: {e}")

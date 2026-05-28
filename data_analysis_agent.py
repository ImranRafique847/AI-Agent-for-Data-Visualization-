import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import ast
import io
import boto3
from dotenv import load_dotenv

# ===============================
# Load AWS Config
# ===============================
load_dotenv()
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")

# ===============================
# Page Config (must be first Streamlit call)
# ===============================
st.set_page_config(page_title="📊 AI Data Visualization Agent", layout="wide")

# ===============================
# Custom UI Theme
# ===============================
st.markdown("""
    <style>
        .stApp { background-color: #0d0f1a; color: white; }
        .stButton>button {
            background: linear-gradient(90deg, #6C5CE7, #a29bfe);
            color: white;
            border-radius: 8px;
            padding: 0.6em 1.2em;
            font-weight: 600;
            border: none;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #a29bfe, #6C5CE7);
            transform: scale(1.02);
        }
        .stTextInput>div>div>input, .stTextArea>div>div>textarea {
            background-color: #1e2333;
            color: white;
            border: 1px solid #6C5CE7;
        }
        .chart-container {
            background-color: #1e2333;
            border: 1px solid #6C5CE7;
            border-radius: 12px;
            padding: 15px;
            margin-bottom: 20px;
            box-shadow: 0 4px 12px rgba(108,92,231,0.3);
        }
        .stat-card {
            background-color: #1e2333;
            border: 1px solid #6C5CE7;
            border-radius: 10px;
            padding: 15px;
            text-align: center;
        }
        .sidebar .sidebar-content { background-color: #1a1d2e; }
        h1, h2, h3 { color: #a29bfe; }
    </style>
""", unsafe_allow_html=True)

# ===============================
# Sidebar Settings
# ===============================
with st.sidebar:
    st.header("⚙️ Settings")

    st.subheader("AWS Configuration")

    # Show credentials status (loaded from .env)
    if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
        st.success("✅ AWS credentials loaded from .env")
        st.text(f"Region: {AWS_REGION}")
        st.text(f"Access Key: {AWS_ACCESS_KEY_ID[:8]}...")
    else:
        st.error("❌ AWS credentials not found. Add them to .env file:")
        st.code("AWS_ACCESS_KEY_ID=your_key\nAWS_SECRET_ACCESS_KEY=your_secret\nAWS_REGION=us-east-1", language="bash")

    st.divider()

    model_choice = st.selectbox(
        "Claude Model (Bedrock)",
        [
            "us.anthropic.claude-haiku-4-5-20251001-v1:0",
            "us.anthropic.claude-sonnet-4-6",
            "us.anthropic.claude-opus-4-7",
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        ],
        index=0,
        help="Haiku 4.5 = fast & cheap. Sonnet 4.6 = balanced. Opus 4.7 = best quality."
    )

    st.divider()

    chart_library = st.radio(
        "Chart Library",
        ["Matplotlib/Seaborn (Static)", "Plotly (Interactive)"],
        index=0
    )

    st.divider()

    picked_color = st.color_picker("Default Chart Color", "#6C5CE7")

    st.divider()

    chart_types = st.multiselect(
        "Preferred Chart Types",
        ["Bar Chart", "Pie Chart", "Line Chart", "Scatter Plot",
         "Box Plot", "Histogram", "Heatmap", "Area Chart"],
        default=["Bar Chart", "Line Chart", "Box Plot"]
    )

    st.divider()
    st.markdown("---")
    st.markdown("**Built with Claude on AWS Bedrock** 🤖")
    st.markdown("Upload data → Ask questions → Get visualizations")

# ===============================
# AWS Bedrock Client
# ===============================
def get_bedrock_client():
    """Create a Bedrock Runtime client using current credentials."""
    if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
        raise ValueError("AWS Access Key ID and Secret Access Key are required. Add them to your .env file.")
    session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )
    return session.client("bedrock-runtime")


# ===============================
# Call Claude via AWS Bedrock
# ===============================
def call_claude(prompt: str, df: pd.DataFrame, model: str, use_plotly: bool = False):
    """Call Claude via AWS Bedrock to generate visualization code."""
    client = get_bedrock_client()

    # Build a rich schema description
    schema_info = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        nunique = df[col].nunique()
        sample = df[col].dropna().head(3).tolist()
        schema_info.append(f"  - {col} ({dtype}, {nunique} unique values, sample: {sample})")
    schema_str = "\n".join(schema_info)

    stats_summary = df.describe(include='all').to_string()

    lib_instruction = ""
    if use_plotly:
        lib_instruction = (
            "Use Plotly (plotly.express as px, plotly.graph_objects as go) for interactive charts. "
            "Store each figure in a variable named fig1, fig2, fig3, etc. "
            "Do NOT call fig.show(). The figures will be rendered by Streamlit."
        )
    else:
        lib_instruction = (
            "Use matplotlib (plt) and seaborn (sns) for static charts. "
            "Use plt.subplots() for multiple charts. Use figsize=(12, 8) or appropriate size. "
            "Always call plt.tight_layout() at the end. Do NOT call plt.show()."
        )

    system_prompt = f"""You are an expert Python data visualization assistant.

The dataframe is called `df` and has these columns:
{schema_str}

Data statistics:
{stats_summary}

{lib_instruction}

RULES:
- Return ONLY valid, executable Python code. No explanations, no markdown, no backticks.
- The dataframe `df` is already loaded. Do NOT read any files.
- Available imports: pandas as pd, matplotlib.pyplot as plt, seaborn as sns, plotly.express as px, plotly.graph_objects as go
- Handle missing values with .dropna() or .fillna() where needed.
- Use proper labels, titles, and legends for all charts.
- If the user mentions a color, use that color. Otherwise use '{picked_color}'.
- Make charts visually appealing with proper formatting.
- For categorical data with many unique values (>15), show only top 10-15 categories.
- Always ensure the code is complete and syntactically valid.
- Do NOT use plt.show() or fig.show().

SMART CHART TYPE SELECTION:
- If the user explicitly asks for a chart type, use EXACTLY that type.
- If the user says "share", "proportion", "percentage", "distribution", "breakdown", "composition", "split", "ratio" → use a PIE CHART.
- If the user says "compare", "comparison", "versus", "vs", "difference", "rank", "top", "bottom", "highest", "lowest" → use a BAR CHART.
- If the user says "trend", "over time", "growth", "change", "timeline", "progress" → use a LINE CHART.
- If the user says "relationship", "correlation", "between X and Y" → use a SCATTER PLOT.
- If the user says "spread", "range", "outliers", "variability" → use a BOX PLOT.
- If the user says "frequency", "how many", "count distribution" → use a HISTOGRAM.
- If unclear, pick the most appropriate chart type based on the data and question.

LABELING AND PERCENTAGE RULES:
- ALWAYS label data points with their actual correct values (counts, percentages, amounts).
- For pie charts: pass raw counts to plt.pie() and use autopct='%1.1f%%' — matplotlib calculates correct percentages automatically.
- For bar charts: annotate each bar with its actual value using ax.bar_label() or ax.annotate().
- Percentages must ALWAYS sum to 100% across all displayed categories.
- Never manually calculate percentages incorrectly. Use value_counts() for category counts.
- When showing "top N" items in a pie chart, group the rest into an "Others" category so percentages still sum to 100%.
- Labels must be readable — rotate x-axis labels if needed, use legend for pie charts with many slices.

ACCURACY RULES:
- The numbers shown on charts MUST match the actual data in the dataframe.
- Always verify: if showing counts, use .value_counts(). If showing sums, use .groupby().sum(). If showing averages, use .groupby().mean().
- Never hardcode values. Always compute from the dataframe.
"""

    # Use the Bedrock Converse API
    response = client.converse(
        modelId=model,
        messages=[
            {
                "role": "user",
                "content": [{"text": prompt}]
            }
        ],
        system=[{"text": system_prompt}],
        inferenceConfig={
            "maxTokens": 4096,
            "temperature": 0.1,
        }
    )

    content = response["output"]["message"]["content"][0]["text"]

    # Clean up any accidental markdown formatting
    content = content.replace("```python", "").replace("```", "").strip()
    # Some models return \\n as literal characters instead of actual newlines
    if "\\n" in content and "\n" not in content:
        content = content.replace("\\n", "\n")

    return content


# ===============================
# Execute code safely
# ===============================
def execute_plot_code(code: str, df: pd.DataFrame, use_plotly: bool = False):
    """Execute generated visualization code safely."""
    try:
        # Validate syntax first
        ast.parse(code)

        local_env = {
            "df": df.copy(),
            "plt": plt,
            "sns": sns,
            "pd": pd,
            "px": px,
            "go": go,
        }
        exec(code, {}, local_env)

        if use_plotly:
            # Find all plotly figures in local_env
            figs_found = False
            for key, val in local_env.items():
                if isinstance(val, (go.Figure,)):
                    st.plotly_chart(val, use_container_width=True)
                    figs_found = True
            if not figs_found:
                st.warning("No Plotly figures were generated. Try rephrasing your query.")
        else:
            # Display matplotlib figure
            fig = plt.gcf()
            if fig.get_axes():
                st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
                st.pyplot(fig)
                st.markdown("</div>", unsafe_allow_html=True)

                # Offer download
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=150, bbox_inches='tight',
                           facecolor='#1e2333', edgecolor='none')
                buf.seek(0)
                st.download_button(
                    label="📥 Download Chart as PNG",
                    data=buf,
                    file_name="chart.png",
                    mime="image/png"
                )
            plt.clf()
            plt.close('all')

        return True

    except SyntaxError as e:
        st.error(f"⚠️ Generated code has a syntax error: {e}")
        with st.expander("View generated code"):
            st.code(code, language="python")
        return False

    except Exception as e:
        st.error(f"⚠️ Error executing visualization: {e}")
        with st.expander("View generated code"):
            st.code(code, language="python")
        return False


# ===============================
# Data Analysis Helper
# ===============================
def show_data_summary(df: pd.DataFrame):
    """Display a summary of the uploaded data."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Rows", f"{df.shape[0]:,}")
    with col2:
        st.metric("Columns", f"{df.shape[1]}")
    with col3:
        st.metric("Numeric Cols", f"{df.select_dtypes(include='number').shape[1]}")
    with col4:
        st.metric("Missing Values", f"{df.isnull().sum().sum():,}")

    with st.expander("📋 Column Details"):
        col_info = pd.DataFrame({
            "Type": df.dtypes.astype(str),
            "Non-Null": df.count(),
            "Null": df.isnull().sum(),
            "Unique": df.nunique(),
            "Sample": [str(df[col].dropna().iloc[0]) if len(df[col].dropna()) > 0 else "N/A" for col in df.columns]
        })
        st.dataframe(col_info, use_container_width=True)

    with st.expander("📊 Statistical Summary"):
        st.dataframe(df.describe(include='all'), use_container_width=True)


# ===============================
# Smart Query Suggestions
# ===============================
def get_query_suggestions(df: pd.DataFrame):
    """Generate smart query suggestions based on data."""
    suggestions = []
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if categorical_cols and numeric_cols:
        suggestions.append(f"Show a bar chart of average {numeric_cols[0]} by {categorical_cols[0]}")
        suggestions.append(f"Create a box plot of {numeric_cols[0]} grouped by {categorical_cols[0]}")

    if len(numeric_cols) >= 2:
        suggestions.append(f"Create a scatter plot of {numeric_cols[0]} vs {numeric_cols[1]}")
        suggestions.append(f"Show a correlation heatmap of all numeric columns")

    if categorical_cols:
        suggestions.append(f"Show the distribution of {categorical_cols[0]} as a pie chart")

    if len(numeric_cols) >= 1:
        suggestions.append(f"Show a histogram of {numeric_cols[0]}")

    return suggestions[:5]


# ===============================
# Main App
# ===============================
st.title("💜 AI Agent for Data Visualization")
st.caption("Powered by Claude AI — Upload data, ask questions, get beautiful visualizations")

uploaded_file = st.file_uploader(
    "Upload your data file",
    type=["csv", "xlsx", "xls", "json", "parquet"],
    help="Supported formats: CSV, Excel, JSON, Parquet"
)

if uploaded_file:
    try:
        # Load data based on file type
        file_ext = uploaded_file.name.split(".")[-1].lower()
        if file_ext == "csv":
            df = pd.read_csv(uploaded_file)
        elif file_ext in ["xlsx", "xls"]:
            df = pd.read_excel(uploaded_file)
        elif file_ext == "json":
            df = pd.read_json(uploaded_file)
        elif file_ext == "parquet":
            df = pd.read_parquet(uploaded_file)
        else:
            st.error("Unsupported file format")
            st.stop()

        st.success(f"✅ Loaded **{uploaded_file.name}** — {df.shape[0]:,} rows × {df.shape[1]} columns")

        # Show data summary
        show_data_summary(df)

        # Data preview
        with st.expander("👀 Data Preview (first 10 rows)", expanded=True):
            st.dataframe(df.head(10), use_container_width=True)

        st.divider()

        # Query suggestions
        suggestions = get_query_suggestions(df)
        if suggestions:
            st.markdown("**💡 Suggested queries:**")
            suggestion_cols = st.columns(len(suggestions))
            for i, suggestion in enumerate(suggestions):
                with suggestion_cols[i]:
                    if st.button(suggestion, key=f"sug_{i}", use_container_width=True):
                        st.session_state["query_input"] = suggestion

        # Query input
        query = st.text_area(
            "Ask me to visualize something:",
            value=st.session_state.get("query_input", ""),
            placeholder="e.g., 'Show a bar chart of sales by region with purple color' or 'Create a dashboard showing all key metrics'",
            height=80
        )

        col_btn1, col_btn2 = st.columns([1, 4])
        with col_btn1:
            generate_btn = st.button("🚀 Generate Visualization", type="primary")

        if generate_btn and query:
            use_plotly = "Plotly" in chart_library

            with st.spinner("🧠 Claude (via AWS Bedrock) is analyzing your data and generating visualizations..."):
                try:
                    # Add chart type preferences to the query
                    enhanced_query = query
                    if chart_types:
                        enhanced_query += f"\nPreferred chart types: {', '.join(chart_types)}"

                    code = call_claude(enhanced_query, df, model_choice, use_plotly=use_plotly)

                    st.markdown("### 📊 Generated Visualization")
                    success = execute_plot_code(code, df, use_plotly=use_plotly)

                    # If first attempt failed, retry with explicit instruction
                    if not success:
                        st.info("🔄 Retrying with clearer instructions...")
                        retry_query = (
                            enhanced_query +
                            "\n\nIMPORTANT: Your previous code had a syntax error. "
                            "Make sure to return properly formatted Python code with correct indentation "
                            "and newlines. Each statement must be on its own line."
                        )
                        code = call_claude(retry_query, df, model_choice, use_plotly=use_plotly)
                        success = execute_plot_code(code, df, use_plotly=use_plotly)

                    if success:
                        with st.expander("🔍 View Generated Code"):
                            st.code(code, language="python")

                except Exception as e:
                    error_msg = str(e)
                    if "AccessDeniedException" in error_msg:
                        st.error("❌ Access denied. Make sure you have enabled the Claude model in AWS Bedrock console (Model Access).")
                    elif "ResourceNotFoundException" in error_msg:
                        st.error("❌ Model not available. Go to AWS Console → Bedrock → Model Access and enable the selected model. Or try a different model from the sidebar dropdown.")
                    elif "ValidationException" in error_msg:
                        st.error(f"❌ Invalid request: {error_msg}")
                    elif "ThrottlingException" in error_msg:
                        st.warning("⏳ Rate limited by AWS. Please wait a moment and try again.")
                    elif "ExpiredTokenException" in error_msg or "credentials" in error_msg.lower():
                        st.error("❌ AWS credentials expired or invalid. Check your credentials in the sidebar.")
                    else:
                        st.error(f"❌ Error: {e}")

        elif generate_btn and not query:
            st.warning("Please enter a query first.")

    except Exception as e:
        st.error(f"❌ Failed to read file: {e}")

else:
    # Landing page when no file is uploaded
    st.markdown("""
    ### 🚀 How to use:
    1. **Configure AWS credentials** in the sidebar (or use default credentials from your environment)
    2. **Upload a data file** (CSV, Excel, JSON, or Parquet)
    3. **Ask a question** about your data in natural language
    4. **Get beautiful visualizations** generated by Claude AI on AWS Bedrock

    ### ✨ Features:
    - 🤖 Powered by Claude (Sonnet 4) via AWS Bedrock
    - 📊 Static charts (Matplotlib/Seaborn) or Interactive charts (Plotly)
    - 💡 Smart query suggestions based on your data
    - 📥 Download charts as PNG
    - 🎨 Customizable colors and chart types
    - 📁 Supports CSV, Excel, JSON, and Parquet files

    ### 🔑 AWS Setup:
    - Enable Claude model access in **AWS Console → Bedrock → Model Access**
    - Use IAM credentials with `bedrock:InvokeModel` and `bedrock:Converse` permissions
    - Or set `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` in `.env` file
    """)

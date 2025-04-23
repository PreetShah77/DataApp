import openai
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, send_file, session, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
import io
import json
import re
from charset_normalizer import detect
from flask_session import Session
from io import StringIO
import base64

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'Uploads'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SECRET_KEY'] = 'super-secret-key'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
Session(app)

# Initialize LLaMA API client
client = openai.OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY", "sk-XwJbTlOeKsDhf_V2XI18PA"),
    base_url=os.environ.get("OPENAI_BASE_URL", "https://chatapi.akash.network/api/v1")
)

# Function to detect file encoding for CSV
def get_file_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = detect(f.read(10000))
        return result['encoding'] if result['encoding'] else 'utf-8'

# Function to extract JSON from response
def extract_json_from_response(raw_response):
    try:
        print(f"Raw API response: {raw_response}")
        json_match = re.search(r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}', raw_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0).strip()
            print(f"Matched JSON: {json_str}")
            try:
                parsed = json.loads(json_str)
                print(f"Validated JSON string: {json_str}, type: {type(json_str)}")
                return json_str
            except json.JSONDecodeError as e:
                print(f"Validation failed for JSON: {json_str}, error: {str(e)}")
                return None
        print("No valid JSON object found in response")
        return None
    except Exception as e:
        print(f"Error extracting JSON: {e}")
        return None

# Function to normalize column names (lowercase, replace spaces with underscores)
def normalize_column_name(col_name):
    return col_name.lower().replace(" ", "_")

# Function to normalize an operation's column names
def normalize_operation_columns(operation):
    if "column" in operation:
        operation["column"] = normalize_column_name(operation["column"])
    if "x_axis" in operation:
        operation["x_axis"] = normalize_column_name(operation["x_axis"])
    if "y_axis" in operation:
        operation["y_axis"] = normalize_column_name(operation["y_axis"])
    if "aggregate_column" in operation:
        operation["aggregate_column"] = normalize_column_name(operation["aggregate_column"])
    if "group_by" in operation:
        operation["group_by"] = normalize_column_name(operation["group_by"])
    if "field" in operation:
        operation["field"] = normalize_column_name(operation["field"])
    if "columns" in operation:
        for col in operation["columns"]:
            col["original_name"] = normalize_column_name(col["original_name"])
            col["new_name"] = normalize_column_name(col["new_name"])
    if "x_column" in operation:
        operation["x_column"] = normalize_column_name(operation["x_column"])
    if "y_column" in operation:
        operation["y_column"] = normalize_column_name(operation["y_column"])
    return operation

# Helper function to get session-specific directory
def get_session_upload_dir():
    if 'session_id' not in session:
        session['session_id'] = str(os.urandom(16).hex())  # Unique ID for the session
    session_id = session['session_id']
    session_dir = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
    if not os.path.exists(session_dir):
        os.makedirs(session_dir)
    return session_dir

# Function to interpret NLP prompts using LLaMA
def interpret_prompt(prompt, df_columns):
    try:
        normalized_df_columns = [normalize_column_name(col) for col in df_columns]
        if not df_columns:
            return {"error": "No data uploaded. Please upload a CSV or Excel file first."}
        response = client.chat.completions.create(
            model="Meta-Llama-3-1-8B-Instruct-FP8",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a data analysis assistant. Given a user prompt and a list of DataFrame columns, "
                        "return a valid JSON object with the operation type (clean, transform, summarize, export, describe_columns, count_records, suggest_analysis, retrieve_row, visualize) "
                        "and parameters needed to execute it. The response MUST be pure JSON, with no additional text, "
                        "Markdown, or explanations. Supported operations:\n"
                        "- Clean: remove duplicates, drop nulls, standardize formats, remove columns, fill nulls, rename_columns (rename columns based on a list of original_name/new_name pairs).\n"
                        "- Transform: filter, group, calculate, pivot, find_max (group by column and find max aggregated value), calculate (use pandas string methods like str.replace, str[-n:] for string manipulations).\n"
                        "- Summarize: describe data numerically.\n"
                        "- Describe_columns: list column names and data types.\n"
                        "- Count_records: count total rows.\n"
                        "- Suggest_analysis: suggest possible analyses based on columns (e.g., for 'provide insights', suggest grouping, filtering, or summarizing).\n"
                        "- Retrieve_row: retrieve a specific row (e.g., first row) from the DataFrame.\n"
                        "- Visualize: create charts (e.g., donut, bar, pie, line) by grouping and aggregating data.\n"
                        "- Export: save as CSV.\n"
                        f"Columns: {df_columns}\n"
                        "Rules:\n"
                        "- All column names in the DataFrame are normalized to lowercase with spaces replaced by underscores (e.g., 'STATE' becomes 'state', 'PRODUCTLINE' becomes 'product_line'). Match user input case-insensitively to these normalized names.\n"
                        "- If 'product' is mentioned, assume 'productcode' unless 'productline' is explicitly stated.\n"
                        "- For 'group by X' or 'by grouping X', default to aggregating numeric columns (e.g., sales, quantityordered) with 'sum'.\n"
                        "- For 'which is the X with highest Y', use 'transform' with 'find_max', grouping by X and aggregating Y with 'sum', then sorting descending.\n"
                        "- For 'convert all columns names to snake case', use 'clean' with 'rename_columns' and provide a list of original_name/new_name pairs where new_name is the snake_case version.\n"
                        "- For 'rename X column to Y', use 'clean' with 'rename_columns' and match X case-insensitively against the current column names, replacing spaces with underscores.\n"
                        "- For phone number transformations (e.g., 'convert to 10 digits', 'remove X from Y'), use 'transform' with 'calculate' and pandas string methods: str.replace to remove characters (e.g., str.replace('-', '')), str[-n:] to extract substrings, and format with ' - ' as needed (e.g., XXX-XXX-XXXX). Infer user intent flexibly.\n"
                        "- Use ONLY pandas string methods (e.g., str.replace, str[-n:]) prefixed with the column name (e.g., 'phone.str.replace'), NO external functions.\n"
                        "- For 'provide insights' or similar, use 'suggest_analysis' to propose analyses based on available columns.\n"
                        "- For 'retrieve the first row' or similar, use 'retrieve_row' with 'row_index' set to 0.\n"
                        "- For chart requests like 'give me [chart_type] chart for X vs Y by grouping Z', use 'visualize' with 'chart_type' (e.g., 'donut', 'bar', 'pie', 'line'), 'x_column' (e.g., X), 'y_column' (e.g., Y), 'group_by' (e.g., Z), and 'aggregate' (default to 'sum'). Match X, Y, Z case-insensitively to normalized column names.\n"
                        "Examples:\n"
                        "- 'Fill null values in STATE with Guj': {\"operation\": \"clean\", \"type\": \"fill_null\", \"column\": \"state\", \"value\": \"Guj\"}\n"
                        "- 'Remove productline column': {\"operation\": \"clean\", \"type\": \"remove_column\", \"column\": \"product_line\"}\n"
                        "- 'Convert all columns names to snake case': {\"operation\": \"clean\", \"type\": \"rename_columns\", \"columns\": [{\"original_name\": \"ordernumber\", \"new_name\": \"order_number\"}, {\"original_name\": \"sales\", \"new_name\": \"sales\"}]}\n"
                        "- 'Rename productline column to products': {\"operation\": \"clean\", \"type\": \"rename_columns\", \"columns\": [{\"original_name\": \"product_line\", \"new_name\": \"products\"}]}\n"
                        "- 'Remove - from phone': {\"operation\": \"transform\", \"type\": \"calculate\", \"field\": \"phone\", \"expression\": \"phone.str.replace('-', '')\"}\n"
                        "- 'Transform the phone number and convert it to only 10 digits': {\"operation\": \"transform\", \"type\": \"calculate\", \"field\": \"phone\", \"expression\": \"phone.str.replace('[^0-9]', '', regex=True).str[-10:]\"}\n"
                        "- 'Provide me some important insights from this data': {\"operation\": \"suggest_analysis\"}\n"
                        "- 'Retrieve the first row of the dataframe': {\"operation\": \"retrieve_row\", \"row_index\": 0}\n"
                        "- 'Give me donut chart for sales vs state by grouping state column': {\"operation\": \"visualize\", \"chart_type\": \"donut\", \"x_column\": \"state\", \"y_column\": \"sales\", \"group_by\": \"state\", \"aggregate\": \"sum\"}\n"
                        "- 'Give me bar chart for sales vs state by grouping state column': {\"operation\": \"visualize\", \"chart_type\": \"bar\", \"x_column\": \"state\", \"y_column\": \"sales\", \"group_by\": \"state\", \"aggregate\": \"sum\"}\n"
                        "- 'Give me bar chart for product vs country': {\"operation\": \"visualize\", \"chart_type\": \"bar\", \"x_column\": \"country\", \"y_column\": \"productcode\", \"group_by\": \"country\", \"aggregate\": \"count\"}\n"
                        "- 'Give me pie chart for sales vs state by grouping state column': {\"operation\": \"visualize\", \"chart_type\": \"pie\", \"x_column\": \"state\", \"y_column\": \"sales\", \"group_by\": \"state\", \"aggregate\": \"sum\"}\n"
                        "- 'Give me line chart for sales vs year_id by grouping year_id column': {\"operation\": \"visualize\", \"chart_type\": \"line\", \"x_column\": \"year_id\", \"y_column\": \"sales\", \"group_by\": \"year_id\", \"aggregate\": \"sum\"}\n"
                        "- 'Which is the state with highest sale': {\"operation\": \"transform\", \"type\": \"find_max\", \"group_by\": \"state\", \"aggregate_column\": \"sales\", \"aggregate\": \"sum\", \"limit\": 1}\n"
                        "- 'Describe all columns and their datatypes': {\"operation\": \"describe_columns\"}\n"
                        "- 'How many records are there': {\"operation\": \"count_records\"}\n"
                        "- 'Group by COUNTRY': {\"operation\": \"transform\", \"type\": \"group\", \"column\": \"country\", \"aggregate_column\": \"sales\", \"aggregate\": \"sum\"}\n"
                        "If the prompt is unclear, columns are invalid, or an unsupported function is suggested, return: {\"error\": \"Unclear prompt, invalid columns, or unsupported function. Use pandas string methods prefixed with column name. Try: Remove X column, Describe columns.\"}"
                    )
                },
                {"role": "user", "content": prompt}
            ],
        )
        raw_response = response.choices[0].message.content
        print(f"Raw API response: {raw_response}")
        json_str = extract_json_from_response(raw_response)
        if json_str:
            try:
                print(f"Extracted JSON string: {json_str}, type: {type(json_str)}")
                parsed_json = json.loads(json_str)
                # Validate columns using normalized names
                normalized_df_columns = [normalize_column_name(col) for col in df_columns]
                if "column" in parsed_json:
                    normalized_col = normalize_column_name(parsed_json["column"])
                    if normalized_col not in normalized_df_columns:
                        return {"error": f"Column '{parsed_json['column']}' not found. Available columns: {df_columns}"}
                    parsed_json["column"] = normalized_col
                if "x_axis" in parsed_json:
                    normalized_col = normalize_column_name(parsed_json["x_axis"])
                    if normalized_col not in normalized_df_columns:
                        return {"error": f"Column '{parsed_json['x_axis']}' not found. Available columns: {df_columns}"}
                    parsed_json["x_axis"] = normalized_col
                if "y_axis" in parsed_json:
                    normalized_col = normalize_column_name(parsed_json["y_axis"])
                    if normalized_col not in normalized_df_columns:
                        return {"error": f"Column '{parsed_json['y_axis']}' not found. Available columns: {df_columns}"}
                    parsed_json["y_axis"] = normalized_col
                if "aggregate_column" in parsed_json:
                    normalized_col = normalize_column_name(parsed_json["aggregate_column"])
                    if normalized_col not in normalized_df_columns:
                        return {"error": f"Column '{parsed_json['aggregate_column']}' not found. Available columns: {df_columns}"}
                    parsed_json["aggregate_column"] = normalized_col
                if "group_by" in parsed_json:
                    normalized_col = normalize_column_name(parsed_json["group_by"])
                    if normalized_col not in normalized_df_columns:
                        return {"error": f"Column '{parsed_json['group_by']}' not found. Available columns: {df_columns}"}
                    parsed_json["group_by"] = normalized_col
                if "field" in parsed_json:
                    normalized_col = normalize_column_name(parsed_json["field"])
                    if normalized_col not in normalized_df_columns:
                        return {"error": f"Column '{parsed_json['field']}' not found. Available columns: {df_columns}"}
                    parsed_json["field"] = normalized_col
                if "columns" in parsed_json:
                    for col in parsed_json["columns"]:
                        normalized_col = normalize_column_name(col["original_name"])
                        if normalized_col not in normalized_df_columns:
                            return {"error": f"Column '{col['original_name']}' not found. Available columns: {df_columns}"}
                        col["original_name"] = normalized_col
                if "x_column" in parsed_json:
                    normalized_col = normalize_column_name(parsed_json["x_column"])
                    if normalized_col not in normalized_df_columns:
                        return {"error": f"Column '{parsed_json['x_column']}' not found. Available columns: {df_columns}"}
                    parsed_json["x_column"] = normalized_col
                if "y_column" in parsed_json:
                    normalized_col = normalize_column_name(parsed_json["y_column"])
                    if normalized_col not in normalized_df_columns:
                        return {"error": f"Column '{parsed_json['y_column']}' not found. Available columns: {df_columns}"}
                    parsed_json["y_column"] = normalized_col
                return parsed_json
            except json.JSONDecodeError as e:
                return {"error": f"Failed to parse JSON: {json_str}, error: {str(e)}"}
        return {"error": "API response contains no valid JSON. Please try again or upload data."}
    except Exception as e:
        print(f"API query exception: {str(e)}")
        return {"error": f"Failed to query API: {str(e)}"}

# Function to execute data operations
def execute_operation(operation, df):
    try:
        # Save updated CSV after modifications in session-specific directory
        def save_updated_csv(df):
            session_dir = get_session_upload_dir()
            output_file = os.path.join(session_dir, "output.csv")
            df.to_csv(output_file, index=False)
            return {"csv_file": "output.csv"}

        if operation["operation"] == "clean":
            if operation["type"] == "drop_nulls":
                df = df.dropna(subset=[operation["column"]])
                return df, f"Dropped nulls in {operation['column']}.", save_updated_csv(df)
            elif operation["type"] == "remove_duplicates":
                df = df.drop_duplicates()
                return df, "Removed duplicates.", save_updated_csv(df)
            elif operation["type"] == "standardize_format":
                if operation["format"] == "date":
                    df[operation["column"]] = pd.to_datetime(df[operation["column"]]).dt.strftime("%Y-%m-%d")
                    return df, f"Standardized date format in {operation['column']}.", save_updated_csv(df)
            elif operation["type"] == "remove_column":
                if operation["column"] not in df.columns:
                    return df, f"Column '{operation['column']}' not found in DataFrame.", None
                df = df.drop(columns=[operation["column"]])
                return df, f"Removed column {operation['column']}.", save_updated_csv(df)
            elif operation["type"] == "fill_null":
                if operation["column"] not in df.columns:
                    return df, f"Column '{operation['column']}' not found in DataFrame.", None
                df[operation["column"]] = df[operation["column"]].fillna(operation["value"])
                return df, f"Filled nulls in {operation['column']} with {operation['value']}.", save_updated_csv(df)
            elif operation["type"] == "rename_columns":
                rename_dict = {col["original_name"]: col["new_name"] for col in operation["columns"]}
                missing_cols = [col for col in rename_dict.keys() if col not in df.columns]
                if missing_cols:
                    return df, f"Columns {missing_cols} not found in DataFrame.", None
                df = df.rename(columns=rename_dict)
                return df, f"Renamed columns: {list(rename_dict.values())}.", save_updated_csv(df)

        elif operation["operation"] == "transform":
            if operation["type"] == "filter":
                condition = operation["condition"]
                df = df.query(condition)
                return df, f"Filtered data where {condition}.", save_updated_csv(df)
            elif operation["type"] == "group":
                if operation["column"] not in df.columns or operation["aggregate_column"] not in df.columns:
                    missing_cols = [col for col in [operation["column"], operation["aggregate_column"]] if col not in df.columns]
                    return df, f"Columns {missing_cols} not found in DataFrame.", None
                result = (df.groupby(operation["column"])[operation["aggregate_column"]]
                          .agg(operation["aggregate"]).reset_index())
                return result, f"Grouped by {operation['column']} and aggregated {operation['aggregate_column']} with {operation['aggregate']}.", save_updated_csv(result)
            elif operation["type"] == "calculate":
                print(f"Executing calculate with expression: {operation.get('expression', 'No expression')}")
                if "expression" in operation and operation["field"] in df.columns:
                    field = operation["field"]
                    expr = operation["expression"]
                    print(f"Evaluating expression: {expr}")
                    print(f"Condition check - 'str' in expr: {'str' in expr}, 'replace' in expr: {'replace' in expr}")
                    if "str" in expr and "replace" in expr:
                        print(f"Applying string transformation to {field}")
                        df[field] = df[field].astype(str)
                        try:
                            exec_expr = expr.replace(f"{field}.", f"df['{field}'].")
                            exec(exec_expr)
                            return df, f"Transformed {field} with expression: {expr}.", save_updated_csv(df)
                        except Exception as e:
                            return df, f"Error applying transformation: {str(e)}. Ensure valid pandas string methods.", None
                    else:
                        return df, f"Invalid expression '{expr}'. Use pandas string methods (e.g., str.replace, str[-n:]) prefixed with column name. Try a different prompt.", None
                return df, f"Column '{operation['field']}' not found in DataFrame.", None
            elif operation["type"] == "find_max":
                if operation["group_by"] not in df.columns or operation["aggregate_column"] not in df.columns:
                    missing_cols = [col for col in [operation["group_by"], operation["aggregate_column"]] if col not in df.columns]
                    return df, f"Columns {missing_cols} not found in DataFrame.", None
                result = (df.groupby(operation["group_by"])[operation["aggregate_column"]]
                          .agg(operation["aggregate"]).reset_index()
                          .sort_values(by=operation["aggregate_column"], ascending=False))
                if operation.get("limit"):
                    result = result.head(operation["limit"])
                top_value = result.iloc[0]
                return df, f"The {operation['group_by']} with the highest {operation['aggregate_column']} is {top_value[operation['group_by']]} with {top_value[operation['aggregate_column']]} total.", save_updated_csv(result)

        elif operation["operation"] == "summarize":
            summary = df.describe(include='all').to_string()
            return df, f"Summary:\n{summary}", None

        elif operation["operation"] == "describe_columns":
            dtypes = df.dtypes.astype(str).to_dict()
            description = "\n".join([f"{col}: {dtype}" for col, dtype in dtypes.items()])
            return df, f"Column Data Types:\n{description}", None

        elif operation["operation"] == "count_records":
            count = len(df)
            return df, f"Total records: {count}", None

        elif operation["operation"] == "suggest_analysis":
            suggestions = [
                "Group sales by country and compute the sum.",
                "Remove rows with null state.",
                "Fill nulls in state with 'Unknown'.",
                "Describe all columns and their data types.",
                "Count the total number of records.",
                "Which is the state with highest sale.",
                "Filter rows where sales > 5000."
            ]
            return df, f"Possible analyses:\n" + "\n".join(suggestions), None

        elif operation["operation"] == "retrieve_row":
            if "row_index" in operation:
                row_index = operation["row_index"]
                if len(df) > row_index:
                    row = df.iloc[row_index].to_dict()
                    row_str = "\n".join([f"{col}: {val}" for col, val in row.items()])
                    return df, f"First row:\n{row_str}", None
                else:
                    return df, "DataFrame is empty or row index out of range.", None
            return df, "Row index not specified.", None

        elif operation["operation"] == "visualize":
            if "chart_type" in operation:
                x_col = operation["x_column"]
                y_col = operation["y_column"]
                group_by = operation["group_by"]
                aggregate = operation.get("aggregate", "sum")
                # Check if required columns exist
                required_cols = [group_by, y_col]
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    return df, f"Columns {missing_cols} not found in DataFrame.", None
                # Filter out rows where group_by or y_col is null, making a copy to avoid SettingWithCopyWarning
                df_filtered = df.dropna(subset=[group_by, y_col]).copy()
                if df_filtered.empty:
                    return df, f"No valid data to plot after removing nulls in {group_by} and {y_col}.", None
                # Ensure y_col is numeric using .loc to avoid SettingWithCopyWarning
                df_filtered.loc[:, y_col] = pd.to_numeric(df_filtered[y_col], errors='coerce')
                df_filtered = df_filtered.dropna(subset=[y_col])
                if df_filtered.empty:
                    return df, f"No numeric data in {y_col} to plot.", None
                # Group and aggregate the data
                grouped_df = df_filtered.groupby(group_by)[y_col].agg(aggregate).reset_index()
                print(f"Grouped data for chart:\n{grouped_df}")
                if grouped_df.empty or len(grouped_df) == 0:
                    return df, f"No data to plot after grouping by {group_by}.", None

                # Create the chart based on chart_type
                plt.figure(figsize=(8, 6))
                chart_type = operation["chart_type"].lower()
                chart_filename = f"{chart_type}_chart.png"
                session_dir = get_session_upload_dir()
                chart_path = os.path.join(session_dir, chart_filename)

                try:
                    if chart_type == "donut":
                        plt.pie(grouped_df[y_col], labels=grouped_df[x_col], autopct='%1.1f%%')
                        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
                        fig = plt.gcf()
                        fig.gca().add_artist(centre_circle)
                    elif chart_type == "bar":
                        plt.bar(grouped_df[x_col], grouped_df[y_col])
                        plt.xticks(rotation=45, ha='right')
                    elif chart_type == "pie":
                        plt.pie(grouped_df[y_col], labels=grouped_df[x_col], autopct='%1.1f%%')
                    elif chart_type == "line":
                        plt.plot(grouped_df[x_col], grouped_df[y_col], marker='o')
                    else:
                        return df, f"Unsupported chart type: {chart_type}. Supported types: donut, bar, pie, line.", None

                    plt.title(f"{y_col} by {x_col}")
                    plt.tight_layout()
                    print(f"Attempting to save chart to: {chart_path}")
                    plt.savefig(chart_path, bbox_inches='tight')
                    plt.close()
                except Exception as e:
                    plt.close()
                    print(f"Chart generation error: {str(e)}")
                    return df, f"Failed to generate chart: {str(e)}.", None

                if not os.path.exists(chart_path):
                    print(f"Chart file not found at {chart_path} after saving.")
                    return df, "Failed to save chart image.", None
                print(f"Chart successfully saved to {chart_path}")

                # Also encode the image as base64 for fallback
                with open(chart_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                base64_string = f"data:image/png;base64,{base64_image}"

                return df, f"{chart_type.capitalize()} chart generated.", {
                    "chart_file": chart_filename,
                    "base64_image": base64_string
                }
            return df, "Chart type not specified.", None

        elif operation["operation"] == "export":
            if operation["type"] == "csv":
                return df, "Exported as CSV.", save_updated_csv(df)

        return df, "Operation not supported. Try: Remove X column, Describe columns.", None
    except Exception as e:
        print(f"Error in execute_operation: {str(e)}")
        return df, f"Error executing operation: {str(e)}. Try a different prompt.", None

# Flask routes
@app.route("/", methods=["GET", "POST"])
def index():
    if 'chat_history' not in session:
        session['chat_history'] = []
    if 'df' not in session:
        session['df'] = None
    if 'filename' not in session:
        session['filename'] = None
    if 'operations' not in session:
        session['operations'] = []
    message = ""
    csv_file = None
    chart_file = None
    base64_image = None

    if request.method == "POST":
        # Handle file upload
        if "file" in request.files and request.files["file"].filename:
            file = request.files["file"]
            if file and file.filename.endswith((".csv", ".xlsx")):
                filename = secure_filename(file.filename)
                session_dir = get_session_upload_dir()
                file_path = os.path.join(session_dir, filename)
                file.save(file_path)
                try:
                    if filename.endswith(".csv"):
                        encoding = get_file_encoding(file_path)
                        try:
                            df = pd.read_csv(file_path, encoding=encoding)
                        except UnicodeDecodeError:
                            for fallback in ['latin1', 'windows-1252', 'iso-8859-1']:
                                try:
                                    df = pd.read_csv(file_path, encoding=fallback)
                                    encoding = fallback
                                    break
                                except UnicodeDecodeError:
                                    continue
                            else:
                                raise ValueError("Unable to decode CSV with common encodings.")
                        message = f"File {filename} uploaded (encoding: {encoding})."
                    else:
                        df = pd.read_excel(file_path, engine='openpyxl')
                        message = f"File {filename} uploaded."
                    # Normalize column names in the DataFrame
                    df.columns = [normalize_column_name(col) for col in df.columns]
                    message += f" Columns: {list(df.columns)}"
                    session['df'] = df.to_json()
                    session['filename'] = filename
                    # Clear operations to start fresh with new data
                    session['operations'] = []
                    session['chat_history'].append({"role": "bot", "content": message})
                except Exception as e:
                    message = f"Error reading file: {str(e)}. Try UTF-8 encoding or checking file."
                    session['chat_history'].append({"role": "bot", "content": message})
            else:
                message = "Invalid file format. Upload a CSV or Excel file."
                session['chat_history'].append({"role": "bot", "content": message})

        # Handle prompt
        if "prompt" in request.form:
            if not session.get('df'):
                message = "No file uploaded. Please upload a CSV or Excel file."
                session['chat_history'].append({"role": "bot", "content": message})
            else:
                prompt = request.form["prompt"]
                session['chat_history'].append({"role": "user", "content": prompt})
                df = pd.read_json(StringIO(session['df']))
                operation = interpret_prompt(prompt, list(df.columns))
                if "error" in operation:
                    message = operation["error"]
                else:
                    # Normalize the current operation's columns
                    operation = normalize_operation_columns(operation)
                    # Apply all prior operations, ensuring they are normalized
                    for op in session['operations']:
                        op = normalize_operation_columns(op)
                        df, _, _ = execute_operation(op, df)
                    # Apply current operation
                    df, result, extra_data = execute_operation(operation, df)
                    session['df'] = df.to_json()
                    session['operations'].append(operation)
                    message = result
                    if extra_data:
                        if "csv_file" in extra_data:
                            csv_file = extra_data["csv_file"]
                        if "chart_file" in extra_data:
                            chart_file = extra_data["chart_file"]
                            print(f"Chart file passed to template: {chart_file}")
                        if "base64_image" in extra_data:
                            base64_image = extra_data["base64_image"]
                            print(f"Base64 image passed to template: {base64_image[:50]}...")
                session['chat_history'].append({"role": "bot", "content": message})

    # Always provide the latest CSV or chart if they exist
    session_dir = get_session_upload_dir()
    output_file_path = os.path.join(session_dir, "output.csv")
    if os.path.exists(output_file_path):
        csv_file = "output.csv"

    return render_template(
        "index.html",
        chat_history=session['chat_history'],
        csv_file=csv_file,
        chart_file=chart_file,
        base64_image=base64_image,
        filename=session.get('filename')
    )

# Route to serve files from the Uploads directory
@app.route('/Uploads/<filename>')
def serve_uploaded_file(filename):
    print(f"Attempting to serve file: {filename}")
    session_dir = get_session_upload_dir()
    return send_from_directory(session_dir, filename)

@app.route("/download/<filename>")
def download_file(filename):
    print(f"Downloading file: {filename}")
    session_dir = get_session_upload_dir()
    return send_file(os.path.join(session_dir, filename), as_attachment=True)

@app.route("/clear", methods=["POST"])
def clear_session():
    session.clear()
    for file in os.listdir(app.config['UPLOAD_FOLDER']):
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], file))
    return jsonify({"status": "Session cleared"})

@app.route("/test-matplotlib")
def test_matplotlib():
    try:
        plt.figure(figsize=(8, 6))
        plt.plot([1, 2, 3], [4, 5, 6])
        session_dir = get_session_upload_dir()
        chart_path = os.path.join(session_dir, "test.png")
        plt.savefig(chart_path)
        plt.close()
        if os.path.exists(chart_path):
            with open(chart_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            base64_string = f"data:image/png;base64,{base64_image}"
            return render_template("index.html", base64_image=base64_string, chat_history=[{"role": "bot", "content": "Test chart generated."}])
        else:
            return "Failed to save test chart.", 500
    except Exception as e:
        print(f"Matplotlib test error: {str(e)}")
        return f"Matplotlib test error: {str(e)}", 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

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
import shutil
from difflib import SequenceMatcher

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

# Function to convert a string to snake case
def to_snake_case(name):
    if not isinstance(name, str):
        return str(name)
    name = re.sub(r'([a-z])([A-Z])', r'\1_\2', name)  # Add underscore between lowercase and uppercase
    name = name.lower().replace(" ", "_")
    name = re.sub(r'(?<!^)(?=[A-Z])', '_', name.lower())
    name = re.sub(r'([a-z])([0-9])', r'\1_\2', name)
    name = re.sub(r'_+', '_', name)
    return name.strip('_')

# Function to normalize column names (converts to snake case)
def normalize_column_name(col_name):
    return to_snake_case(col_name)

# Function to calculate string similarity
def get_similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

# Function to find the closest matching column
def find_similar_column(target, columns, threshold=0.6):
    # First, try matching the target as-is or normalized
    normalized_target = normalize_column_name(target)
    if normalized_target in columns:
        return normalized_target
    
    # Check similarity with normalized columns
    best_match = None
    best_score = threshold
    for col in columns:
        score = get_similarity(normalized_target, col)
        if score > best_score:
            best_score = score
            best_match = col
    
    # Fallback: compare raw target (without normalization) against columns
    if not best_match:
        for col in columns:
            score = get_similarity(target, col)
            if score > best_score:
                best_score = score
                best_match = col
    
    return best_match if best_score >= threshold else None

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
        if isinstance(operation["group_by"], list):
            operation["group_by"] = [normalize_column_name(col) for col in operation["group_by"]]
        else:
            operation["group_by"] = normalize_column_name(operation["group_by"])
    if "field" in operation:
        operation["field"] = normalize_column_name(operation["field"])
    if "columns" in operation:
        for col in operation["columns"]:
            col["original_name"] = normalize_column_name(col["original_name"])
            col["new_name"] = normalize_column_name(col["new_name"])
    if "values" in operation:
        if isinstance(operation["values"], list):
            operation["values"] = [normalize_column_name(col) for col in operation["values"]]
        else:
            operation["values"] = normalize_column_name(operation["values"])
    if "x_column" in operation:
        operation["x_column"] = normalize_column_name(operation["x_column"])
    if "y_column" in operation:
        operation["y_column"] = normalize_column_name(operation["y_column"])
    return operation

# Helper function to get session-specific directory
def get_session_upload_dir():
    if 'session_id' not in session:
        session['session_id'] = str(os.urandom(16).hex())
    session_id = session['session_id']
    session_dir = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
    if not os.path.exists(session_dir):
        os.makedirs(session_dir)
    return session_dir

# Function to interpret NLP prompts using LLaMA
def interpret_prompt(prompt, df_columns):
    try:
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
                        "- Clean: remove duplicates, drop nulls, standardize_format, remove columns, fill nulls, rename_columns (rename columns based on a list of original_name/new_name pairs).\n"
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
                        "- All column names in the DataFrame are normalized to snake_case (e.g., 'STATE' becomes 'state', 'PRODUCTLINE' becomes 'product_line', 'QUANTITYORDERED' becomes 'quantity_ordered'). Match user input case-insensitively to these normalized names.\n"
                        "- If 'product' is mentioned, assume 'product_code' unless 'product_line' is explicitly stated.\n"
                        "- For 'group by X' or 'by grouping X', default to aggregating numeric columns (e.g., sales, quantity_ordered) with 'sum'.\n"
                        "- For 'which is the X with highest Y', use 'transform' with 'find_max', grouping by X and aggregating Y with 'sum', then sorting descending.\n"
                        "- For 'convert all columns names to snake case', use 'clean' with 'rename_columns' and provide a list of original_name/new_name pairs for ALL columns, ensuring they are already in snake_case (e.g., 'quantity_ordered' to 'quantity_ordered'). Since columns are already in snake_case, this may be a no-op.\n"
                        "- For 'rename X column to Y', use 'clean' with 'rename_columns' and match X case-insensitively against the current column names, which are in snake_case.\n"
                        "- For phone number transformations (e.g., 'convert to 10 digits', 'remove X from Y'), use 'transform' with 'calculate' and pandas string methods: str.replace to remove characters (e.g., str.replace('-', '')), str[-n:] to extract substrings, and format with ' - ' as needed (e.g., XXX-XXX-XXXX). Infer user intent flexibly.\n"
                        "- Use ONLY pandas string methods (e.g., str.replace, str[-n:]) prefixed with the column name (e.g., 'phone.str.replace'), NO external functions like np.where, np.select, or lambda.\n"
                        "- For conditional replacements like 'where X is Y replace with Z' in a column, use 'transform' with 'calculate' and pandas string methods, e.g., str.replace('Y', 'Z') for exact matches, ensuring the column is treated as a string.\n"
                        "- For 'provide insights' or similar, use 'suggest_analysis' to propose analyses based on available columns.\n"
                        "- For 'retrieve the first row' or similar, use 'retrieve_row' with 'row_index' set to 0.\n"
                        "- For chart requests like 'give me [chart_type] chart for X vs Y by grouping Z', use 'visualize' with 'chart_type' (e.g., 'donut', 'bar', 'pie', 'line'), 'x_column' (e.g., X), 'y_column' (e.g., Y), 'group_by' (e.g., Z), and 'aggregate' (default to 'sum'). Match X, Y, Z case-insensitively to normalized column names.\n"
                        "- For 'standardize the date format' or similar, use 'clean' with 'standardize_format', and include 'format': 'date' to specify the YYYY-MM-DD format.\n"
                        "Examples:\n"
                        "- 'Fill null values in STATE with Guj': {\"operation\": \"clean\", \"type\": \"fill_null\", \"column\": \"state\", \"value\": \"Guj\"}\n"
                        "- 'Remove productline column': {\"operation\": \"clean\", \"type\": \"remove_column\", \"column\": \"product_line\"}\n"
                        "- 'Remove dealsize column': {\"operation\": \"clean\", \"type\": \"remove_column\", \"column\": \"deal_size\"}\n"
                        "- 'Convert all columns names to snake case': {\"operation\": \"clean\", \"type\": \"rename_columns\", \"columns\": [{\"original_name\": \"order_number\", \"new_name\": \"order_number\"}, {\"original_name\": \"quantity_ordered\", \"new_name\": \"quantity_ordered\"}, {\"original_name\": \"price_each\", \"new_name\": \"price_each\"}, {\"original_name\": \"order_line_number\", \"new_name\": \"order_line_number\"}, {\"original_name\": \"sales\", \"new_name\": \"sales\"}, {\"original_name\": \"order_date\", \"new_name\": \"order_date\"}, {\"original_name\": \"status\", \"new_name\": \"status\"}, {\"original_name\": \"qtr_id\", \"new_name\": \"qtr_id\"}, {\"original_name\": \"month_id\", \"new_name\": \"month_id\"}, {\"original_name\": \"year_id\", \"new_name\": \"year_id\"}, {\"original_name\": \"product_line\", \"new_name\": \"product_line\"}, {\"original_name\": \"msrp\", \"new_name\": \"msrp\"}, {\"original_name\": \"product_code\", \"new_name\": \"product_code\"}, {\"original_name\": \"customer_name\", \"new_name\": \"customer_name\"}, {\"original_name\": \"phone\", \"new_name\": \"phone\"}, {\"original_name\": \"address_line1\", \"new_name\": \"address_line1\"}, {\"original_name\": \"address_line2\", \"new_name\": \"address_line2\"}, {\"original_name\": \"city\", \"new_name\": \"city\"}, {\"original_name\": \"state\", \"new_name\": \"state\"}, {\"original_name\": \"postal_code\", \"new_name\": \"postal_code\"}, {\"original_name\": \"country\", \"new_name\": \"country\"}, {\"original_name\": \"territory\", \"new_name\": \"territory\"}, {\"original_name\": \"contact_last_name\", \"new_name\": \"contact_last_name\"}, {\"original_name\": \"contact_first_name\", \"new_name\": \"contact_first_name\"}, {\"original_name\": \"deal_size\", \"new_name\": \"deal_size\"}]}\n"
                        "- 'Rename productline column to products': {\"operation\": \"clean\", \"type\": \"rename_columns\", \"columns\": [{\"original_name\": \"product_line\", \"new_name\": \"products\"}]}\n"
                        "- 'Remove - from phone': {\"operation\": \"transform\", \"type\": \"calculate\", \"field\": \"phone\", \"expression\": \"phone.str.replace('-', '')\"}\n"
                        "- 'Transform the phone number and convert it to only 10 digits': {\"operation\": \"transform\", \"type\": \"calculate\", \"field\": \"phone\", \"expression\": \"phone.str.replace('[^0-9]', '', regex=True).str[-10:]\"}\n"
                        "- 'In state column where there is Guj change with Ahm': {\"operation\": \"transform\", \"type\": \"calculate\", \"field\": \"state\", \"expression\": \"state.str.replace('Guj', 'Ahm')\"}\n"
                        "- 'Provide me some important insights from this data': {\"operation\": \"suggest_analysis\"}\n"
                        "- 'Retrieve the first row of the dataframe': {\"operation\": \"retrieve_row\", \"row_index\": 0}\n"
                        "- 'Retrieve the last row of the dataframe': {\"operation\": \"retrieve_row\", \"row_index\": -1}\n"
                        "- 'Give me donut chart for sales vs state by grouping state column': {\"operation\": \"visualize\", \"chart_type\": \"donut\", \"x_column\": \"state\", \"y_column\": \"sales\", \"group_by\": \"state\", \"aggregate\": \"sum\"}\n"
                        "- 'Give me bar chart for sales vs state by grouping state column': {\"operation\": \"visualize\", \"chart_type\": \"bar\", \"x_column\": \"state\", \"y_column\": \"sales\", \"group_by\": \"state\", \"aggregate\": \"sum\"}\n"
                        "- 'Give me bar chart for product vs country': {\"operation\": \"visualize\", \"chart_type\": \"bar\", \"x_column\": \"country\", \"y_column\": \"product_code\", \"group_by\": \"country\", \"aggregate\": \"count\"}\n"
                        "- 'Give me pie chart for sales vs state by grouping state column': {\"operation\": \"visualize\", \"chart_type\": \"pie\", \"x_column\": \"state\", \"y_column\": \"sales\", \"group_by\": \"state\", \"aggregate\": \"sum\"}\n"
                        "- 'Give me line chart for sales vs year_id by grouping year_id column': {\"operation\": \"visualize\", \"chart_type\": \"line\", \"x_column\": \"year_id\", \"y_column\": \"sales\", \"group_by\": \"year_id\", \"aggregate\": \"sum\"}\n"
                        "- 'Which is the state with highest sale': {\"operation\": \"transform\", \"type\": \"find_max\", \"group_by\": \"state\", \"aggregate_column\": \"sales\", \"aggregate\": \"sum\", \"limit\": 1}\n"
                        "- 'Describe all columns and their datatypes': {\"operation\": \"describe_columns\"}\n"
                        "- 'How many records are there': {\"operation\": \"count_records\"}\n"
                        "- 'Group by COUNTRY': {\"operation\": \"transform\", \"type\": \"group\", \"column\": \"country\", \"aggregate_column\": \"sales\", \"aggregate\": \"sum\"}\n"
                        "- 'Give me pivot table with state, city and aggregated sales': {\"operation\": \"transform\", \"type\": \"pivot\", \"group_by\": [\"state\", \"city\"], \"values\": [\"sales\"]}\n"
                        "- 'Standardize the date format': {\"operation\": \"clean\", \"type\": \"standardize_format\", \"column\": \"date\", \"format\": \"date\"}\n"
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
                # Post-process to correct 'standardize_formats' to 'standardize_format' and add 'format' if missing
                if parsed_json.get("type") == "standardize_formats":
                    parsed_json["type"] = "standardize_format"
                if parsed_json.get("type") == "standardize_format" and "format" not in parsed_json:
                    parsed_json["format"] = "date"  # Default to date format if not specified
                return parsed_json
            except json.JSONDecodeError as e:
                return {"error": f"Failed to parse JSON: {json_str}, error: {str(e)}"}
        return {"error": "API response contains no valid JSON. Please try again or upload data."}
    except Exception as e:
        print(f"API query exception: {str(e)}")
        return {"error": f"Failed to query API: {str(e)}"}
        
# Function to execute data operations
def execute_operation(operation, df, original_df):
    try:
        # Save updated CSV after modifications in session-specific directory
        def save_updated_csv(df, filename="output.csv"):
            session_dir = get_session_upload_dir()
            output_file = os.path.join(session_dir, filename)
            df.to_csv(output_file, index=False)
            return {"csv_file": filename}

        # Helper function to validate and adjust column names using similarity
        def validate_and_adjust_column(col_name, df_columns, operation_type):
            if col_name in df_columns:
                return col_name
            similar_col = find_similar_column(col_name, df_columns)
            if similar_col:
                print(f"Adjusted column '{col_name}' to similar column '{similar_col}' for operation '{operation_type}'")
                return similar_col
            return None

        # Validate and adjust columns before executing the operation
        if "column" in operation:
            adjusted_col = validate_and_adjust_column(operation["column"], df.columns, operation["operation"])
            if adjusted_col:
                operation["column"] = adjusted_col
            else:
                return df, f"Column '{operation['column']}' not found and no similar column identified. Available columns: {list(df.columns)}", None
        if "x_axis" in operation:
            adjusted_col = validate_and_adjust_column(operation["x_axis"], df.columns, operation["operation"])
            if adjusted_col:
                operation["x_axis"] = adjusted_col
            else:
                return df, f"Column '{operation['x_axis']}' not found and no similar column identified. Available columns: {list(df.columns)}", None
        if "y_axis" in operation:
            adjusted_col = validate_and_adjust_column(operation["y_axis"], df.columns, operation["operation"])
            if adjusted_col:
                operation["y_axis"] = adjusted_col
            else:
                return df, f"Column '{operation['y_axis']}' not found and no similar column identified. Available columns: {list(df.columns)}", None
        if "aggregate_column" in operation:
            adjusted_col = validate_and_adjust_column(operation["aggregate_column"], df.columns, operation["operation"])
            if adjusted_col:
                operation["aggregate_column"] = adjusted_col
            else:
                return df, f"Column '{operation['aggregate_column']}' not found and no similar column identified. Available columns: {list(df.columns)}", None
        if "group_by" in operation:
            if isinstance(operation["group_by"], list):
                adjusted_cols = []
                for col in operation["group_by"]:
                    adjusted_col = validate_and_adjust_column(col, df.columns, operation["operation"])
                    if adjusted_col:
                        adjusted_cols.append(adjusted_col)
                    else:
                        return df, f"Column '{col}' not found and no similar column identified. Available columns: {list(df.columns)}", None
                operation["group_by"] = adjusted_cols
            else:
                adjusted_col = validate_and_adjust_column(operation["group_by"], df.columns, operation["operation"])
                if adjusted_col:
                    operation["group_by"] = adjusted_col
                else:
                    return df, f"Column '{operation['group_by']}' not found and no similar column identified. Available columns: {list(df.columns)}", None
        if "field" in operation:
            adjusted_col = validate_and_adjust_column(operation["field"], df.columns, operation["operation"])
            if adjusted_col:
                operation["field"] = adjusted_col
            else:
                return df, f"Column '{operation['field']}' not found and no similar column identified. Available columns: {list(df.columns)}", None
        if "columns" in operation:
            for col in operation["columns"]:
                adjusted_col = validate_and_adjust_column(col["original_name"], df.columns, operation["operation"])
                if adjusted_col:
                    col["original_name"] = adjusted_col
                else:
                    return df, f"Column '{col['original_name']}' not found and no similar column identified. Available columns: {list(df.columns)}", None
        if "values" in operation:
            if isinstance(operation["values"], list):
                adjusted_cols = []
                for col in operation["values"]:
                    adjusted_col = validate_and_adjust_column(col, df.columns, operation["operation"])
                    if adjusted_col:
                        adjusted_cols.append(adjusted_col)
                    else:
                        return df, f"Column '{col}' not found and no similar column identified. Available columns: {list(df.columns)}", None
                operation["values"] = adjusted_cols
            else:
                adjusted_col = validate_and_adjust_column(operation["values"], df.columns, operation["operation"])
                if adjusted_col:
                    operation["values"] = adjusted_col
                else:
                    return df, f"Column '{operation['values']}' not found and no similar column identified. Available columns: {list(df.columns)}", None
        if "x_column" in operation:
            adjusted_col = validate_and_adjust_column(operation["x_column"], df.columns, operation["operation"])
            if adjusted_col:
                operation["x_column"] = adjusted_col
            else:
                return df, f"Column '{operation['x_column']}' not found and no similar column identified. Available columns: {list(df.columns)}", None
        if "y_column" in operation:
            adjusted_col = validate_and_adjust_column(operation["y_column"], df.columns, operation["operation"])
            if adjusted_col:
                operation["y_column"] = adjusted_col
            else:
                return df, f"Column '{operation['y_column']}' not found and no similar column identified. Available columns: {list(df.columns)}", None

        if operation["operation"] == "clean":
            if operation["type"] == "drop_nulls":
                df = df.dropna(subset=[operation["column"]])
                return df, f"Dropped nulls in {operation['column']}.", save_updated_csv(df)
            elif operation["type"] == "remove_duplicates":
                df = df.drop_duplicates()
                return df, "Removed duplicates.", save_updated_csv(df)
            elif operation["type"] == "standardize_format":
                if operation.get("format") == "date":
                    df[operation["column"]] = pd.to_datetime(df[operation["column"]], errors='coerce').dt.strftime("%Y-%m-%d")
                    return df, f"Standardized date format in {operation['column']} to YYYY-MM-DD.", save_updated_csv(df)
                else:
                    return df, f"Unsupported format '{operation.get('format', 'default')}'. Use 'date' for standardization.", None
            elif operation["type"] == "remove_column":
                target_column = operation["column"]
                df = df.drop(columns=[target_column])
                return df, f"Removed column {target_column}.", save_updated_csv(df)
            elif operation["type"] == "fill_null":
                df[operation["column"]] = df[operation["column"]].fillna(operation["value"])
                return df, f"Filled nulls in {operation['column']} with {operation['value']}.", save_updated_csv(df)
            elif operation["type"] == "rename_columns":
                rename_dict = {col["original_name"]: col["new_name"] for col in operation["columns"]}
                missing_cols = [col for col in rename_dict.keys() if col not in df.columns]
                if missing_cols:
                    return df, f"Columns {missing_cols} not found in DataFrame. Available columns: {list(df.columns)}", None
                df = df.rename(columns=rename_dict)
                return df, f"Renamed columns: {list(rename_dict.values())}.", save_updated_csv(df)

        elif operation["operation"] == "transform":
            if operation["type"] == "filter":
                condition = operation["condition"]
                df = df.query(condition)
                return df, f"Filtered data where {condition}.", save_updated_csv(df)
            elif operation["type"] == "group":
                result = (df.groupby(operation["column"])[operation["aggregate_column"]]
                          .agg(operation["aggregate"]).reset_index())
                return result, f"Grouped by {operation['column']} and aggregated {operation['aggregate_column']} with {operation['aggregate']}.", save_updated_csv(result)
            elif operation["type"] == "calculate":
                print(f"Executing calculate with expression: {operation.get('expression', 'No expression')}")
                field = operation["field"]
                expr = operation["expression"]
                print(f"Evaluating expression: {expr}")
                print(f"Condition check - 'str' in expr: {'str' in expr}, 'replace' in expr: {'replace' in expr}")
                if "str" in expr:
                    if "replace" in expr:
                        print(f"Applying string transformation to {field}")
                        # Log unique values before transformation
                        print(f"Unique values in {field} before transformation: {df[field].astype(str).unique()}")
                        # Convert to string, strip whitespace, and make replacement case-insensitive
                        df[field] = df[field].astype(str).str.strip()
                        try:
                            # Extract the old and new values from the expression for logging
                            import re
                            match = re.search(r"str\.replace\('([^']+)',\s*'([^']+)'\)", expr)
                            if match:
                                old_value, new_value = match.groups()
                                # Perform case-insensitive replacement
                                df[field] = df[field].str.replace(old_value, new_value, case=False)
                                # Log unique values after transformation
                                print(f"Unique values in {field} after transformation: {df[field].unique()}")
                                # Check if any replacements were made
                                if (df[field] == new_value).any():
                                    return df, f"Transformed {field} with expression: {expr}. Replaced '{old_value}' with '{new_value}'.", save_updated_csv(df)
                                else:
                                    return df, f"Transformed {field} with expression: {expr}. Replaced '{old_value}' with '{new_value}'.", save_updated_csv(df)
                            else:
                                return df, f"Could not parse replacement values from expression: {expr}.", None
                        except Exception as e:
                            return df, f"Error applying transformation: {str(e)}. Ensure valid pandas string methods.", None
                    else:
                        return df, f"Unsupported string method in expression '{expr}'. Use pandas string methods like str.replace, str[-n:].", None
                else:
                    return df, f"Invalid expression '{expr}'. Use pandas string methods (e.g., str.replace, str[-n:]) prefixed with column name. Try a different prompt.", None
            elif operation["type"] == "find_max":
                result = (df.groupby(operation["group_by"])[operation["aggregate_column"]]
                          .agg(operation["aggregate"]).reset_index()
                          .sort_values(by=operation["aggregate_column"], ascending=False))
                if operation.get("limit"):
                    result = result.head(operation["limit"])
                top_value = result.iloc[0]
                return df, f"The {operation['group_by']} with the highest {operation['aggregate_column']} is {top_value[operation['group_by']]} with {top_value[operation['aggregate_column']]} total.", save_updated_csv(result)
            elif operation["type"] == "pivot":
                group_by = operation["group_by"] if isinstance(operation["group_by"], list) else [operation["group_by"]]
                values = operation["values"] if isinstance(operation["values"], list) else [operation["values"]]
                pivot_df = df.pivot_table(index=group_by, values=values, aggfunc='sum').reset_index()
                # Save the pivot table as a separate file
                extra_data = save_updated_csv(pivot_df, "pivot_output.csv")
                # Preserve the original DataFrame by not modifying df
                return df, f"Pivot table created with index {group_by} and values {values}.", extra_data

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
                    return df, f"Row:\n{row_str}", None
                else:
                    return df, "DataFrame is empty or row index out of range.", None
            return df, "Row index not specified.", None

        elif operation["operation"] == "visualize":
            if "chart_type" in operation:
                x_col = operation["x_column"]
                y_col = operation["y_column"]
                group_by = operation["group_by"]
                aggregate = operation.get("aggregate", "sum")
                df_filtered = df.dropna(subset=[group_by, y_col]).copy()
                if df_filtered.empty:
                    return df, f"No valid data to plot after removing nulls in {group_by} and {y_col}.", None
                df_filtered.loc[:, y_col] = pd.to_numeric(df_filtered[y_col], errors='coerce')
                df_filtered = df_filtered.dropna(subset=[y_col])
                if df_filtered.empty:
                    return df, f"No numeric data in {y_col} to plot.", None
                grouped_df = df_filtered.groupby(group_by)[y_col].agg(aggregate).reset_index()
                print(f"Grouped data for chart:\n{grouped_df}")
                if grouped_df.empty or len(grouped_df) == 0:
                    return df, f"No data to plot after grouping by {group_by}.", None
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

@app.route("/", methods=["GET", "POST"])
def index():
    if 'chat_history' not in session:
        session['chat_history'] = []
    if 'df' not in session:
        session['df'] = None
    if 'original_df' not in session:
        session['original_df'] = None
    if 'filename' not in session:
        session['filename'] = None
    if 'operations' not in session:
        session['operations'] = []
    message = ""
    csv_file = None
    original_csv_file = None
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
                    # Normalize column names in the DataFrame immediately after loading
                    original_columns = df.columns.tolist()
                    df.columns = [normalize_column_name(col) for col in df.columns]
                    message += f" Columns (normalized to snake case): {list(df.columns)}"
                    session['df'] = df.to_json()
                    # Store the original DataFrame before any operations
                    session['original_df'] = df.to_json()
                    session['filename'] = filename
                    # Clear operations to start fresh with new data
                    session['operations'] = []
                    session['chat_history'].append({"role": "bot", "content": message})
                    # Save the original DataFrame as original.csv
                    session_dir = get_session_upload_dir()
                    original_file_path = os.path.join(session_dir, "original.csv")
                    df.to_csv(original_file_path, index=False)
                    original_csv_file = "original.csv"
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
                original_df = pd.read_json(StringIO(session['original_df'])) if session.get('original_df') else df.copy()
                operation = interpret_prompt(prompt, list(df.columns))
                if "error" in operation:
                    message = operation["error"]
                else:
                    # Normalize the current operation's columns
                    operation = normalize_operation_columns(operation)
                    # Apply all prior operations
                    for op in session['operations']:
                        op = normalize_operation_columns(op)
                        df, _, _ = execute_operation(op, df, original_df)
                    # Apply current operation
                    df, result, extra_data = execute_operation(operation, df, original_df)
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
    original_file_path = os.path.join(session_dir, "original.csv")
    if os.path.exists(original_file_path):
        original_csv_file = "original.csv"

    return render_template(
        "index.html",
        chat_history=session['chat_history'],
        csv_file=csv_file,
        original_csv_file=original_csv_file,
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
    session_dir = get_session_upload_dir()
    if os.path.exists(session_dir):
        try:
            shutil.rmtree(session_dir)
            print(f"Deleted session directory: {session_dir}")
        except Exception as e:
            print(f"Error deleting session directory {session_dir}: {str(e)}")
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
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

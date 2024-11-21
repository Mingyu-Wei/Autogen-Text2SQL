import pandas as pd

def preprocess_excel_schemas(excel_path):
    # Read all sheets from the Excel file
    data = pd.read_excel(excel_path, sheet_name=None)
    
    schema_texts = []
    for sheet_name, df in data.items():
        # Extract the relevant schema information from each sheet
        schema_text = f"Database: {sheet_name}\n"
        schema_text += "Columns:\n"
        for _, row in df.iterrows():
            column_name = row["MFDB Name"]
            column_label = row["Data Label"]
            column_type = row["Data Type"]
            schema_text += f"- {column_name} ({column_type}): {column_label}\n"
        schema_texts.append(schema_text)
    
    # Combine all schemas into a single string
    return "\n\n".join(schema_texts)

# Example usage
schema_text = preprocess_excel_schemas("./schema.xlsx")
with open("schema.txt", "w") as f:
    f.write(schema_text)

print("Preprocessed Schema:\n", schema_text)

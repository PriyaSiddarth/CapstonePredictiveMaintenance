# Load the raw dataset
df = pd.read_csv('CapstonePredictiveMaintenance/data/engine_data.csv')

# Validate that the expected columns are present before registering it
expected_columns = ["Engine rpm", "Lub oil pressure", "Fuel pressure",
                    "Coolant pressure","lub oil temp", "Coolant temp", "Engine Condition"]
# Identify missing columns
missing = [c for c in expected_columns if c not in df.columns ]

# Print missing columns
print(f"Missing columns: {missing}")

# Raise error only if columns are missing
if missing:
    raise ValueError(
        f"Dataset is missing expected columns: {missing}"
    )

print("Dataset registered successfully.")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("Columns:", list(df.columns))
print("Customer purchase distribution:")
print(df["ProdTaken"].value_counts())

import pandas as pd

# Read the Excel file
df = pd.read_excel('ground_truth_from_pdf.xlsx')

# Print information about the DataFrame
print("\nDataFrame Info:")
print("---------------")
print(df.info())

print("\nColumn Names:")
print("---------------")
print(df.columns.tolist())

print("\nFirst few rows:")
print("---------------")
print(df.head())

print("\nValue counts for each column:")
print("-----------------------------")
for column in df.columns:
    print(f"\n{column}:")
    print(df[column].value_counts().head())
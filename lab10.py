import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Example DataFrame
df = pd.DataFrame({
    'TypeName': ['Ultrabook', 'Notebook', 'Gaming', 'Notebook', 'Workstation'],
    'RAM': ['8GB', '16GB', '8GB', '4GB', '32GB']
})

ohe = OneHotEncoder(sparse_output=False)

enc_data = ohe.fit_transform(df[['TypeName']])

enc_df = pd.DataFrame(
    enc_data,
    columns=ohe.get_feature_names_out(['TypeName'])
)

df1 = df.join(enc_df)

print("After One Hot Encoding:")
print(df1)

le = LabelEncoder()

df1['RAM'] = le.fit_transform(df1['RAM'])

print("\nAfter Label Encoding RAM:")
print(df1)

print("\nRAM Classes:")
print(le.classes_)

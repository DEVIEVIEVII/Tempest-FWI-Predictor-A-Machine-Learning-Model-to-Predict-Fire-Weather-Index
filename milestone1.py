import pandas as pd
df = pd.read_csv("cleaned_dataset.csv")
print(df.head())
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("cleaned_dataset.csv")   


df.columns = df.columns.str.strip()


print("Missing Values:\n", df.isnull().sum())


df = df.dropna()


print("Duplicate Rows:", df.duplicated().sum())
df = df.drop_duplicates()


print("\nDataset Info:")
print(df.info())


if "area" in df.columns:
    df["area"] = np.log1p(df["area"])
    print("\nLog transformation applied to 'area' column")


print("\nFinal Dataset Shape:", df.shape)


df.to_csv("cleaned_forestfires.csv", index=False)

print("\n✅ Dataset is cleaned and ready for model building!")






df = pd.read_csv("cleaned_dataset.csv")  

df.columns = df.columns.str.strip()


df.hist(figsize=(12,8))
plt.tight_layout()
plt.show()

if "Classes" in df.columns:
    plt.figure(figsize=(6,4))
    sns.countplot(x="Classes", data=df)
    plt.title("Fire vs Not Fire Count")
    plt.show()


if "Temperature" in df.columns and "RH" in df.columns:
    plt.figure(figsize=(6,4))
    sns.scatterplot(x="Temperature", y="RH", data=df)
    plt.title("Temperature vs RH")
    plt.show()

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

if "Temperature" in df.columns:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df["Temperature"])
    plt.title("Boxplot - Temperature")
    plt.show()


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# veriyi oku
df = pd.read_csv("akilliev.csv", low_memory=False)

# time kolonunu düzelt
df["time"] = pd.to_numeric(df["time"], errors="coerce")
df = df.dropna(subset=["time"])
df["time"] = df["time"].astype("int64")

# tarih formatına çevir
df["time"] = pd.to_datetime(df["time"], unit="s")
df["saat"] = df["time"].dt.hour

# saatleri gruplara ayır
def gun_dilimi(saat):
    if 6 <= saat < 12:
        return "sabah"
    elif 12 <= saat < 18:
        return "ogle"
    else:
        return "aksam"

df["gun_dilimi"] = df["saat"].apply(gun_dilimi)

# cihazların kolonları
cols = ['Dishwasher [kW]', 'Furnace 1 [kW]', 'Furnace 2 [kW]',
        'Home office [kW]', 'Fridge [kW]', 'Wine cellar [kW]',
        'Garage door [kW]', 'Kitchen 12 [kW]', 'Kitchen 14 [kW]',
        'Kitchen 38 [kW]', 'Barn [kW]', 'Well [kW]',
        'Microwave [kW]', 'Living room [kW]', 'Solar [kW]']

# tahmin edilecek şey (sabah/ogle/aksam)
y = df["gun_dilimi"].map({"sabah": 0, "ogle": 1, "aksam": 2})

# x ve y'yi ayır
x = df[cols]

# train test ayır
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

# ortalama tüketimleri hesapla
pivot = df.pivot_table(
    index="gun_dilimi",
    values=cols,
    aggfunc="mean"
).T.reindex(columns=["sabah", "ogle", "aksam"])

print("\n========= ortalama degerler =========\n")
print(pivot.round(4))

# knn modeli
knn = Pipeline([
    ("scaler", StandardScaler()),
    ("model", KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1))
])

# random forest
rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1, max_depth=10)

models = {"knn": knn, "random forest": rf}

# modelleri çalıştır
print("\n===== basari oranlari =====\n")
for isim, model in models.items():
    model.fit(x_train, y_train)
    tahmin = model.predict(x_test)
    basari = accuracy_score(y_test, tahmin)
    print(f"{isim}: {basari:.4f}")

# grafik çiz
plt.figure(figsize=(8, 12))
plt.imshow(pivot.values, aspect="auto")
plt.colorbar(label="ortalama tuketim (kw)")

plt.yticks(range(len(pivot.index)), pivot.index)
plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45)

plt.title("cihaz tuketimleri - saat dilimleri")
plt.xlabel("zaman")
plt.ylabel("cihaz")

plt.tight_layout()
plt.show()
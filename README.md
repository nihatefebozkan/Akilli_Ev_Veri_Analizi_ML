# Akilli_Ev_Veri_Analizi_ML

Akıllı ev cihazlarının tüketim verilerine bakarak günün hangi saatinde olduğunu tahmin eden makine öğrenmesi projesi.

## Nasıl Çalışır

### 1 Veri okuma ve temizleme 

```
df = pd.read_csv("akilliev.csv", low_memory=False)
# time kolonunu düzelt
df["time"] = pd.to_numeric(df["time"], errors="coerce")
df = df.dropna(subset=["time"])
df["time"] = df["time"].astype("int64")

# tarih formatına çevir
df["time"] = pd.to_datetime(df["time"], unit="s")
df["saat"] = df["time"].dt.hour
```

### 2 Saatleri Sabah - Öğle - Akşam diye 3 gruba ayırma 
```
def gun_dilimi(saat):
    if 6 <= saat < 12:
        return "sabah"
    elif 12 <= saat < 18:
        return "ogle"
    else:
        return "aksam"

df["gun_dilimi"] = df["saat"].apply(gun_dilimi)
```

### Pivot Tablosu oluşturma
not: pivot tablosu oluştururken tarih bilgisinden çektiğim saati kullandım mean komutu ile baseline olusturdum.
```
pivot = df.pivot_table(
    index="gun_dilimi",
    values=cols,
    aggfunc="mean"
).T.reindex(columns=["sabah", "ogle", "aksam"])
```

### Model eğitimi
HEDEF(Tahmin) : Gün dilimleri (Sabah - Öğle - Akşam)
```
y = df["gun_dilimi"].map({"sabah": 0, "ogle": 1, "aksam": 2})
```



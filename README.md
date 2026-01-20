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
Datayı yükledim. Zaman kolununu numeric değere çevirdim ve eksik değerleri temizledim ve zamanı tam sayıya çevirdim.
Verideki time kısmını tarih saat formatına donusturdum. ve saat bilgisini çektim.

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
Projede pivot olarak saat dilimlerini kullanacağım için günü 3 e böldüm.

### Pivot Tablosu oluşturma
```
pivot = df.pivot_table(
    index="gun_dilimi",
    values=cols,
    aggfunc="mean"
).T.reindex(columns=["sabah", "ogle", "aksam"])
```

pivot tablosu oluştururken saat bilgilerinden çektiğim gun dilimlerini kullandımve ```aggfunc= "mean"``` komutu ile baseline(ortalama) olusturdum.


### Model eğitimi
HEDEF(Tahmin) : Gün dilimleri (Sabah - Öğle - Akşam)
```
y = df["gun_dilimi"].map({"sabah": 0, "ogle": 1, "aksam": 2})
```

### Terminal Çıktı 
<img src ="images/ss1.png">


### Heatmap Grafiği Çıktı





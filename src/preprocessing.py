"""Ham TEFAS ve BIST100 verisini analize hazır matrislere çevirir.

Çalıştırma (proje kökünden):
    python -m src.preprocessing

Akış:
    ham CSV'ler -> geniş fiyat matrisi -> BIST takvimine hizalama
    -> basit/log getiri -> kalite raporu -> data/processed/*.csv
"""

import numpy as np
import pandas as pd

from src import config

# Ham dosyalardaki olası sütun adları -> standart ad.
# tefas-crawler İngilizce (date, code, price, title) döndürür; elle
# kaydedilmiş dosyalarda Türkçe veya farklı adlar olabilir.
SUTUN_ESLEME = {
    "tarih": ["date", "tarih"],
    "kod": ["code", "kod", "fund_code", "fon_kodu"],
    "fiyat": ["price", "fiyat"],
    "fon_adi": ["title", "fon_adi", "fund_name"],
}


def _sutunlari_standartlastir(df: pd.DataFrame, gerekli: list[str]) -> pd.DataFrame:
    df = df.rename(columns=lambda c: str(c).strip().lstrip("﻿").lower())
    yeni_adlar = {}
    for standart, adaylar in SUTUN_ESLEME.items():
        bulunan = next((a for a in adaylar if a in df.columns), None)
        if bulunan:
            yeni_adlar[bulunan] = standart
    df = df.rename(columns=yeni_adlar)

    eksik = [s for s in gerekli if s not in df.columns]
    if eksik:
        raise ValueError(
            f"Gerekli sütun(lar) bulunamadı: {eksik}. "
            f"Dosyadaki sütunlar: {list(df.columns)}. "
            "SUTUN_ESLEME sözlüğüne ekleyin."
        )
    return df


def load_tefas_raw(path=config.TEFAS_RAW) -> tuple[pd.DataFrame, pd.Series | None]:
    """Uzun formatlı fon verisini (tarih, kod, fiyat) geniş matrise çevirir.

    Dönüş: (fiyatlar, fon_adlari)
        fiyatlar  : satır = tarih, sütun = fon kodu
        fon_adlari: kod -> fon adı (ham veride başlık sütunu yoksa None)
    """
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = _sutunlari_standartlastir(df, ["tarih", "kod", "fiyat"])

    df["tarih"] = pd.to_datetime(df["tarih"])
    df["kod"] = df["kod"].astype(str).str.strip().str.upper()
    df["fiyat"] = pd.to_numeric(df["fiyat"], errors="coerce")

    # Aynı (tarih, kod) iki kez gelmişse (ör. veri iki kez çekilip birleştirilmişse)
    # sonuncusu tutulur.
    onceki = len(df)
    df = df.drop_duplicates(subset=["tarih", "kod"], keep="last")
    if len(df) < onceki:
        print(f"  {onceki - len(df)} tekrar eden satır silindi.")

    fon_adlari = None
    if "fon_adi" in df.columns:
        fon_adlari = df.dropna(subset=["fon_adi"]).groupby("kod")["fon_adi"].last()

    fiyatlar = df.pivot(index="tarih", columns="kod", values="fiyat").sort_index()
    fiyatlar.columns.name = None
    return fiyatlar, fon_adlari


def load_benchmark_raw(path=config.BENCHMARK_RAW) -> pd.Series:
    """BIST100 kapanış serisi (tarih indeksli)."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.rename(columns=lambda c: str(c).strip().lstrip("﻿").lower())
    tarih_col = "tarih" if "tarih" in df.columns else "date"
    fiyat_col = next(c for c in ["kapanis", "close", "adj_close"] if c in df.columns)

    seri = pd.Series(
        pd.to_numeric(df[fiyat_col], errors="coerce").values,
        index=pd.to_datetime(df[tarih_col]),
        name="XU100",
    )
    seri = seri[~seri.index.duplicated(keep="last")].sort_index()
    return seri.where(seri > 0)


def align_to_benchmark(fiyatlar: pd.DataFrame, benchmark: pd.Series) -> pd.DataFrame:
    """Fon fiyatlarını BIST işlem günü takvimine hizalar.

    Neden BIST takvimi: beta/alfa gibi metrikler fon ve endeks getirisinin
    AYNI günlere ait olmasını gerektirir. TEFAS bazı günlerde (ör. yarım gün,
    resmî tatil farkları) fiyat yayımlamayabilir.

    Kurallar:
      - Sıfır/negatif fiyat geçersizdir -> NaN.
      - Boşluklar en fazla FFILL_LIMIT gün ileri doldurulur (son bilinen fiyat).
      - Fonun ilk geçerli fiyatından ÖNCESİ doldurulmaz: sonradan kurulan bir fona
        geçmiş fiyat uydurmak sahte sıfır getiri üretir.
    """
    fiyatlar = fiyatlar.where(fiyatlar > 0)
    takvim = benchmark.dropna().index
    # Takvim dışındaki fon günlerini de dahil edip sonra kesiyoruz; böylece
    # BIST'in açık olup TEFAS'ın fiyat vermediği gün, bir önceki TEFAS
    # fiyatıyla doldurulabiliyor.
    birlesik = fiyatlar.reindex(fiyatlar.index.union(takvim))
    doldurulmus = birlesik.ffill(limit=config.FFILL_LIMIT)
    # ffill fonun ilk verisinden önceyi zaten doldurmaz (öncesinde değer yok).
    return doldurulmus.reindex(takvim)


def compute_returns(fiyatlar: pd.DataFrame | pd.Series):
    """Basit ve log getiri.

    Basit getiri  r_t = P_t / P_{t-1} - 1  -> portföy getirisi (ağırlıklı toplam) için doğru olan.
    Log getiri    l_t = ln(P_t / P_{t-1}) -> zaman içinde toplanabilir; ln(1 + r).
    fill_method=None: pandas'ın eksik fiyatı sessizce doldurmasını engeller;
    eksik gün NaN getiri olarak kalır.
    """
    basit = fiyatlar.pct_change(fill_method=None)
    log = np.log(fiyatlar / fiyatlar.shift(1))
    return basit.iloc[1:], log.iloc[1:]


def quality_report(fiyatlar: pd.DataFrame, getiri: pd.DataFrame) -> pd.DataFrame:
    """Fon bazında veri kalitesi özeti."""
    satirlar = []
    for kod in config.fon_kodlari():
        if kod not in fiyatlar.columns:
            satirlar.append({"kod": kod, "durum": "VERİ YOK"})
            continue
        f = fiyatlar[kod]
        g = getiri[kod]
        ilk, son = f.first_valid_index(), f.last_valid_index()
        if ilk is None:
            satirlar.append({"kod": kod, "durum": "VERİ YOK"})
            continue
        aralik = f.loc[ilk:son]
        aykiri = g[g.abs() > config.OUTLIER_THRESHOLD]
        satirlar.append({
            "kod": kod,
            "durum": "OK",
            "ilk_tarih": ilk.date(),
            "son_tarih": son.date(),
            "gun_sayisi": int(aralik.notna().sum()),
            "eksik_gun": int(aralik.isna().sum()),
            "eksik_oran": round(aralik.isna().mean(), 4),
            "aykiri_gun": len(aykiri),
            "aykiri_tarihler": ", ".join(str(t.date()) for t in aykiri.index),
            "min_getiri": round(g.min(), 4),
            "max_getiri": round(g.max(), 4),
        })
    rapor = pd.DataFrame(satirlar).set_index("kod")

    # Takvimin başından sonra başlayan fon (ör. sonradan kurulmuş veya
    # veri eksik çekilmiş) analiz döneminin bir kısmını kaçırır.
    takvim_basi = fiyatlar.index.min().date()
    rapor["gec_baslayan"] = rapor["ilk_tarih"].apply(
        lambda t: bool(pd.notna(t) and t > takvim_basi)
    )
    return rapor


def main():
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Ham veriler okunuyor...")
    fiyatlar, fon_adlari = load_tefas_raw()
    benchmark = load_benchmark_raw()
    print(f"  TEFAS : {fiyatlar.shape[1]} fon, {fiyatlar.index.min().date()} -> {fiyatlar.index.max().date()}")
    print(f"  BIST  : {benchmark.notna().sum()} gün, {benchmark.index.min().date()} -> {benchmark.index.max().date()}")

    bilinmeyen = sorted(set(fiyatlar.columns) - set(config.fon_kodlari()))
    if bilinmeyen:
        print(f"  UYARI: config'te olmayan fonlar atlandı: {bilinmeyen}")
    fiyatlar = fiyatlar.reindex(columns=config.fon_kodlari())

    fiyatlar = align_to_benchmark(fiyatlar, benchmark)
    benchmark = benchmark.reindex(fiyatlar.index)

    basit, log = compute_returns(fiyatlar)
    bm_basit, bm_log = compute_returns(benchmark)
    rapor = quality_report(fiyatlar, basit)

    meta = config.metadata_df()
    if fon_adlari is not None:
        meta["fon_adi"] = fon_adlari.reindex(meta.index)

    out = config.PROCESSED_DIR
    fiyatlar.to_csv(out / "fiyatlar.csv", index_label="tarih")
    basit.to_csv(out / "getiri_basit.csv", index_label="tarih")
    log.to_csv(out / "getiri_log.csv", index_label="tarih")
    benchmark.to_frame("XU100").to_csv(out / "benchmark_fiyat.csv", index_label="tarih")
    pd.DataFrame({"basit": bm_basit, "log": bm_log}).to_csv(
        out / "benchmark_getiri.csv", index_label="tarih"
    )
    rapor.to_csv(out / "kalite_raporu.csv")
    meta.to_csv(out / "fon_metadata.csv")

    print(f"\nÇıktılar kaydedildi: {out}")
    print("\nKalite raporu:")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(rapor.drop(columns=["aykiri_tarihler"], errors="ignore"))

    sorunlu = rapor[(rapor["durum"] != "OK") | rapor["gec_baslayan"]
                    | (rapor["eksik_oran"].fillna(0) > 0.05) | (rapor["aykiri_gun"].fillna(0) > 0)]
    if len(sorunlu):
        print(f"\nDİKKAT - incelenmesi gereken fonlar: {list(sorunlu.index)}")


if __name__ == "__main__":
    main()

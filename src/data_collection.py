"""TEFAS'tan 22 fonun günlük fiyatlarını çeker -> data/raw/tefas_raw.csv

Çalıştırma (proje kökünden):
    python -m src.data_collection

TEFAS API'si uzun tarih aralıklarında eksik/boş yanıt dönebildiği için
sorgu 60 günlük parçalara bölünür ve sonuçlar birleştirilir.
"""

import time

import pandas as pd
from tefas import Crawler

from src import config

PARCA_GUN = 60
SUTUNLAR = ["date", "code", "title", "price"]


def tarih_parcalari(baslangic: str, bitis: str, gun: int = PARCA_GUN):
    bas, bit = pd.Timestamp(baslangic), pd.Timestamp(bitis)
    while bas <= bit:
        son = min(bas + pd.Timedelta(days=gun - 1), bit)
        yield bas.strftime("%Y-%m-%d"), son.strftime("%Y-%m-%d")
        bas = son + pd.Timedelta(days=1)


def fon_cek(crawler: Crawler, kod: str) -> pd.DataFrame:
    parcalar = []
    for bas, son in tarih_parcalari(config.START_DATE, config.END_DATE):
        for deneme in range(3):
            try:
                df = crawler.fetch(start=bas, end=son, name=kod, columns=SUTUNLAR)
                parcalar.append(df)
                break
            except Exception as e:  # ağ hatasında kısa bekleyip tekrar dene
                print(f"    {kod} {bas}->{son} hata ({e.__class__.__name__}), deneme {deneme + 1}/3")
                time.sleep(2 ** deneme)
        time.sleep(0.5)  # sunucuya nazik ol
    return pd.concat(parcalar, ignore_index=True) if parcalar else pd.DataFrame(columns=SUTUNLAR)


def main():
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    crawler = Crawler()
    tum = []
    for kod in config.fon_kodlari():
        df = fon_cek(crawler, kod)
        print(f"  {kod}: {len(df)} satır"
              + (f" ({df['date'].min()} -> {df['date'].max()})" if len(df) else " - VERİ GELMEDİ"))
        tum.append(df)

    veri = (pd.concat(tum, ignore_index=True)
            .drop_duplicates(subset=["date", "code"])
            .sort_values(["code", "date"]))
    veri.to_csv(config.TEFAS_RAW, index=False)
    print(f"\n{veri['code'].nunique()} fon, {len(veri)} satır -> {config.TEFAS_RAW}")

    eksik = sorted(set(config.fon_kodlari()) - set(veri["code"]))
    if eksik:
        print(f"UYARI: veri gelmeyen fonlar: {eksik}")


if __name__ == "__main__":
    main()

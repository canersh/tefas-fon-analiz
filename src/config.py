"""Proje genelinde kullanılan yollar, parametreler ve fon listesi.

Tüm modüller sabitleri buradan okur; böylece tarih aralığı veya fon listesi
değişince tek bir dosyayı düzenlemek yeterli olur.
"""

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Yollar
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

TEFAS_RAW = RAW_DIR / "tefas_raw.csv"
BENCHMARK_RAW = RAW_DIR / "benchmark_raw.csv"

# ---------------------------------------------------------------------------
# Parametreler
# ---------------------------------------------------------------------------
START_DATE = "2024-01-01"
END_DATE = "2026-09-27"
BENCHMARK_TICKER = "XU100.IS"

# Yıllıklandırma için yılda yaklaşık işlem günü sayısı (volatilite * sqrt(252)).
TRADING_DAYS = 252

# Bir fonun fiyatı en fazla kaç işlem günü ileri taşınabilir.
# Daha uzun boşluk gerçek bir veri sorunu demektir, doldurmak onu gizler.
FFILL_LIMIT = 5

# |günlük getiri| bu eşiği aşarsa kalite raporunda "aykırı" olarak işaretlenir.
# Silinmez; bölünme, birleşme veya veri hatası olup olmadığına elle bakılır.
OUTLIER_THRESHOLD = 0.25

# ---------------------------------------------------------------------------
# Fonlar
# ---------------------------------------------------------------------------
KATEGORILER = [
    "Teknoloji",
    "Para Piyasası",
    "BIST 30 Hisse",
    "Altın/Kıymetli Madenler",
    "Değişken/Çoklu Varlık",
]

SIRKETLER = ["İş", "Garanti", "Ak", "QNB", "Tacirler"]

FONLAR = {
    # Teknoloji
    "ITP": {"sirket": "İş", "kategori": "Teknoloji"},
    "ICZ": {"sirket": "İş", "kategori": "Teknoloji"},
    "GRT": {"sirket": "Garanti", "kategori": "Teknoloji"},
    "AFT": {"sirket": "Ak", "kategori": "Teknoloji"},
    "OJT": {"sirket": "QNB", "kategori": "Teknoloji"},
    # Para Piyasası
    "TI1": {"sirket": "İş", "kategori": "Para Piyasası"},
    "GTL": {"sirket": "Garanti", "kategori": "Para Piyasası"},
    "ALE": {"sirket": "Ak", "kategori": "Para Piyasası"},
    "OPJ": {"sirket": "QNB", "kategori": "Para Piyasası"},
    # BIST 30 Hisse
    "TIE": {"sirket": "İş", "kategori": "BIST 30 Hisse"},
    "GOH": {"sirket": "Garanti", "kategori": "BIST 30 Hisse"},
    "AKU": {"sirket": "Ak", "kategori": "BIST 30 Hisse"},
    "BID": {"sirket": "QNB", "kategori": "BIST 30 Hisse"},
    # Altın / Kıymetli Madenler
    "TTA": {"sirket": "İş", "kategori": "Altın/Kıymetli Madenler"},
    "GTA": {"sirket": "Garanti", "kategori": "Altın/Kıymetli Madenler"},
    "AFO": {"sirket": "Ak", "kategori": "Altın/Kıymetli Madenler"},
    "OJK": {"sirket": "QNB", "kategori": "Altın/Kıymetli Madenler"},
    # Değişken / Çoklu Varlık
    "ALR": {"sirket": "Ak", "kategori": "Değişken/Çoklu Varlık"},
    "IKP": {"sirket": "İş", "kategori": "Değişken/Çoklu Varlık"},
    "GPG": {"sirket": "Garanti", "kategori": "Değişken/Çoklu Varlık"},
    "OPB": {"sirket": "QNB", "kategori": "Değişken/Çoklu Varlık"},
    "TCD": {"sirket": "Tacirler", "kategori": "Değişken/Çoklu Varlık"},
}


def fon_kodlari() -> list[str]:
    return list(FONLAR)


def kategoriye_gore(kategori: str) -> list[str]:
    if kategori not in KATEGORILER:
        raise ValueError(f"Bilinmeyen kategori: {kategori!r}. Geçerli: {KATEGORILER}")
    return [kod for kod, bilgi in FONLAR.items() if bilgi["kategori"] == kategori]


def sirkete_gore(sirket: str) -> list[str]:
    if sirket not in SIRKETLER:
        raise ValueError(f"Bilinmeyen şirket: {sirket!r}. Geçerli: {SIRKETLER}")
    return [kod for kod, bilgi in FONLAR.items() if bilgi["sirket"] == sirket]


def metadata_df() -> pd.DataFrame:
    """Fon kodu indeksli (sirket, kategori) tablosu."""
    df = pd.DataFrame.from_dict(FONLAR, orient="index")
    df.index.name = "kod"
    return df


# Import anında tutarlılık kontrolü: listeye elle fon eklerken yapılan
# hatalar analiz aşamasına kadar fark edilmeden kalmasın.
assert len(FONLAR) == 22, f"22 fon bekleniyordu, {len(FONLAR)} var"
for _kod, _bilgi in FONLAR.items():
    assert _bilgi["kategori"] in KATEGORILER, f"{_kod}: geçersiz kategori"
    assert _bilgi["sirket"] in SIRKETLER, f"{_kod}: geçersiz şirket"

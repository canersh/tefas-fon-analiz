import os
import pandas as pd
import yfinance as yf

TICKER = "XU100.IS"
START_DATE = "2024-01-01"
END_DATE = "2026-09-28"
OUTPUT_PATH = "data/raw/benchmark_raw.csv"

def fetch_benchmark_data():
    print(f"[{TICKER}] BIST 100 verisi çekiliyor... ({START_DATE} -> 2026-09-27)")
    
    df = yf.download(TICKER, start=START_DATE, end=END_DATE, progress=False)
    
    if df.empty:
        print("Hata: Benchmark verisi çekilemedi.")
        return None
    
    # Çoklu indeks (MultiIndex) varsa düzelt
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    
    df.reset_index(inplace=True)
    
    # Sütun isimlerini küçük harfe çevir
    df.columns = [str(c).lower().replace(' ', '_') for c in df.columns]
    
    # Tarih sütununu belirle
    date_col = 'date' if 'date' in df.columns else df.columns[0]
    df['tarih'] = pd.to_datetime(df[date_col]).dt.strftime('%Y-%m-%d')
    
    # Kapanış fiyatını belirle (adj_close yoksa close al)
    if 'adj_close' in df.columns:
        df['kapanis'] = df['adj_close']
    elif 'close' in df.columns:
        df['kapanis'] = df['close']
    else:
        df['kapanis'] = df.iloc[:, 1]
        
    df['hacim'] = df['volume'] if 'volume' in df.columns else 0
    
    clean_df = df[['tarih', 'kapanis', 'hacim']].dropna().drop_duplicates(subset=['tarih'])
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    clean_df.to_csv(OUTPUT_PATH, index=False)
    
    print("\n==========================================")
    print("Benchmark verisi başarıyla kaydedildi!")
    print(f"Dosya yolu: {OUTPUT_PATH}")
    print(f"Toplam işlem günü: {len(clean_df)}")
    print(f"İlk gün: {clean_df['tarih'].iloc[0]} | Son gün: {clean_df['tarih'].iloc[-1]}")
    print("==========================================")
    return clean_df

if __name__ == "__main__":
    fetch_benchmark_data()

# v2 ilerleme notları

Her görev sonunda Implementer buraya kısa bir Türkçe not ekler: ne değişti, neyin ölçüldüğü
(komut veya run dizini ile), ne açık kaldı. Ölçülmemiş sayı yazılmaz.

## Durum

- G0: T1 incelemede; kapı henüz geçilmedi
- Proposer: G0 geçene kadar kapalı

## G0-T1 — istatistik ve sizing portu

### Ek doğrulama ve ilk inceleme

- Sahip iki port sapmasını onayladı. `v2-test` grubu statsmodels içeriyor;
  beş sentetik seride HAC karşılaştırması `rtol=1e-8`, Bartlett, maxlags=5 ve
  `use_correction=True` ile geçti. `python -m pytest -q -s
  tests/v2/test_hac_statsmodels.py tests/v2/test_spa_acceptance.py`: 12 geçti.
- SPA güç testi: T=1000, günlük Sharpe 2/√252; 173/200 (%86,5) edge saptandı.
  Önceki %3 yanlış alarm testi değiştirilmedi ve yukarıdaki komutta geçti.
- İlk incelemede sıfıra yakın HAC standart hatasının yanlış anlamlılık üretmesi
  düzeltildi; rapor `docs/v2/reviews/G0-T1.md` içinde.
- İkinci incelemenin çalıştırma bulgusu giderildi: güncel ağaçta
  `python -m pytest -q tests` 321 geçti, 32 uyarı.
- Ek açık MINOR: dejenerasyon durumundaki NaN değerleri ledger JSON biçimine
  doğrudan yazılamaz; ileride INVALID sınıflandırması yapan çağıran bunları
  dönüştürmelidir; bu görev ledger entegrasyonu içermediğinden ertelendi.
  `1e-12` RMS eşiği yalnızca sayısal olarak tam dejenerasyonu
  yakalamak için kullanılır; yakın dejenerasyon için kalibre edilmiş değildir.
- Açık MINOR: pandas indeks hizalaması henüz doğrulanmıyor; girdi dizileri
  çağıran tarafından hizalanmalı. Sıfır uygun günlü sütun için bağımsız test
  henüz eklenmedi. Bazı formül tekrarına dayanan kontroller ek kapsama değil
  tutarlılık kontrolüdür. Tam test için `.[test,v2,v2-test]` kurulmalıdır;
  eksik opsiyonel bağımlılıklarla test toplama desteklenmiyor. SPA güç testinde
  252 günlük yıllıklaştırma sahibi tarafından istendi, kaynak bootstrap ise
  365 kullanır; sonuçlar bu nedenle doğrudan aynı yıllık ölçek değildir.
  Kullanılmayan `BTC` sabiti kaynak portuna sadakat için korundu.

**Ne değişti.** `ddawnlll/tycho @ 0a309c4` (D:\tycho-src-readonly, HEAD doğrulandı, salt-okunur)
kopyalandı. Yeni dosyalar: `tycho/v2/{__init__,stats,weights,nulls}.py`,
`tests/v2/{test_stats_port,test_weights_port,test_nulls_port,test_stats_acceptance,test_weights_acceptance,test_spa_acceptance}.py`;
`pyproject.toml`'a yeni `v2` opsiyonel grubu (`arch>=8,<9`, `scipy>=1.11,<2`) eklendi. Her portlanan
fonksiyonun başında kaynak yol+commit yorumu var; `stats.py`'ye ayrıca `hansen_spa` ve
`romano_wolf_stepm` ince sarmalayıcıları eklendi (arch SPA / StepM, işaret dönüşümü tek yerde).

**Mekanik doğrulama.** AST karşılaştırması ile kaynak gövdelere karşı kontrol edildi: `weights.py`
üç fonksiyonda birebir aynı; `nulls.py` portlanan sekiz fonksiyonda birebir aynı; `stats.py`'de
`_sharpe_variance`, `mean_holding_period`, `_annualized_sharpe` birebir aynı, kalan üç fonksiyonda
fark yalnızca aşağıda yazılı bilinçli sapmalar.

**Ölçülenler.**
- Önceki ağaçta `python -m pytest -q tests` → 315 passed; bu sayı güncel değildir.
- `python -m pyflakes tycho scripts tests/v2` → temiz; `python scripts/check_frozen.py` → geçti.
- SPA saf-gürültü red oranı: **%3.0** (6/200 seed; 20 strateji, T=1000, reps=500, blok=10,
  eşik p ≤ 0.05, Hansen "consistent" p-değeri). Kabul sınırı ~%5.
- DSR: N=1 ve sıfır trial varyansında PSR(0)'a eşit; N=1,10,100,1000 için kesin azalan.
- `beta_adjusted_alpha`: 5 tohumda plantlanmış alpha=0.0008 ve beta=0.6, 3 standart hata içinde.
- `target_weights`: 200 rastgele durumda gross cap ve varlık başına vol katkısı (|w_i|·vol_i ≤ b)
  korunuyor; s ≤ 1.
- `arch` çalışma ortamına kuruldu (`uv pip install arch` → arch 8.0.0, statsmodels 0.15.0 yan
  bağımlılık olarak geldi). `.venv/` gitignore'da, repoya girmedi.

**Bilinçli sapmalar (sahibin onayı gerekiyor).**
1. `beta_adjusted_alpha`, kaynaktaki `statsmodels` HAC OLS yerine numpy'de açık Newey-West
   (Bartlett, `n/(n-k)` düzeltmesi) kullanıyor; görev yalnızca `arch` + `scipy` bağımlılığına izin
   veriyor. Sonuç sözlüğüne `alpha_se` / `beta_se` eklendi (mevcut anahtarlar korundu).
2. `nulls.py` yalnızca saf üreticileri içeriyor (`scramble`, `random_entry`, `blocks`,
   `apply_gross_cap`, `eligible_days`). Kaynaktaki `no_trade`, `equal_risk_buy_and_hold`,
   `trend_following_btc`, `run_baselines`, `run_nulls` E-001 defterine bağlı; E-001 v2'de yok.
   Bu yüzden `run_baselines`/`run_nulls` ve E-001 defterine/sentetik tape'e ihtiyaç duyan 5 kaynak
   testi portlanmadı. `tests/v2/test_weights_port.py` ayrıca kaynaktaki `test_config_wiring`'i
   içermiyor (kaynak reponun `config/tycho.toml`'unu okuyor; bu repoda öyle bir dosya yok).

**Açık kalan.**
- İki inceleme çalıştırıldı; kayıt ve açık MINOR bulgular yukarıdadır.

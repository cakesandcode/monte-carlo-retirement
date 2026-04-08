# Data Sources

All historical data used by the retirement financial modeling system.
Data is downloaded on first run and cached locally under `data/cache/`.

---

## 1. Equity Returns

### U.S. Large Cap (S&P 500)
- **Source:** Damodaran Online — NYU Stern School of Business
- **URL:** https://pages.stern.nyu.edu/~adamodar/pc/datasets/histretSP.xls
- **Coverage:** 1928–present (annual total returns)
- **Format:** Excel (.xls) — Sheet "Returns by year"
- **Notes:** Includes dividends. Longest-running authoritative U.S. equity series.
- **Direct data page:** https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/histretSP.html

### U.S. Large Cap (live/recent)
- **Source:** Yahoo Finance via yfinance — ticker `^GSPC`
- **URL:** https://finance.yahoo.com/quote/%5EGSPC/history/
- **Coverage:** 1950–present (daily, resampled to annual)

### U.S. Small Cap
- **Source:** Yahoo Finance via yfinance — ticker `^RUT` (Russell 2000)
- **URL:** https://finance.yahoo.com/quote/%5ERUT/history/
- **Coverage:** 1978–present
- **Pre-1978 fallback:** Fama-French small cap premium embedded in fallback parameters

### International Developed Markets
- **Source:** Yahoo Finance via yfinance — ticker `EFA` (iShares MSCI EAFE)
- **URL:** https://finance.yahoo.com/quote/EFA/history/
- **Coverage:** 2001–present (ETF inception)
- **Proxy for earlier data:** MSCI EAFE Index (embedded fallback, 1970+)

### Emerging Markets
- **Source:** Yahoo Finance via yfinance — ticker `EEM` (iShares MSCI EM)
- **URL:** https://finance.yahoo.com/quote/EEM/history/
- **Coverage:** 2003–present (ETF inception)
- **Proxy for earlier data:** MSCI EM Index (embedded fallback, 1987+)

### REITs
- **Source:** Yahoo Finance via yfinance — ticker `VNQ` (Vanguard Real Estate ETF)
- **URL:** https://finance.yahoo.com/quote/VNQ/history/
- **Coverage:** 2004–present (ETF inception)
- **Historical proxy:** NAREIT All Equity REIT Index (embedded, 1972+)
- **NAREIT data:** https://www.reit.com/data-research/reit-indexes/annual-index-values-returns

---

## 2. Fixed Income Returns

### U.S. Aggregate Bonds
- **Source:** Yahoo Finance via yfinance — ticker `AGG` (iShares Core U.S. Aggregate Bond ETF)
- **URL:** https://finance.yahoo.com/quote/AGG/history/
- **Coverage:** 2003–present (ETF inception)
- **Proxy for earlier data:** 10-Year Treasury total return (FRED GS10 + duration adjustment)

### TIPS (Inflation-Protected Bonds)
- **Source:** Yahoo Finance via yfinance — ticker `TIP` (iShares TIPS Bond ETF)
- **URL:** https://finance.yahoo.com/quote/TIP/history/
- **Coverage:** 2003–present (first TIPS ETF)
- **TIPS program began:** 1997

### Cash / T-Bills
- **Source:** Yahoo Finance via yfinance — ticker `BIL` (SPDR Bloomberg 1-3 Month T-Bill ETF)
- **URL:** https://finance.yahoo.com/quote/BIL/history/
- **Coverage:** 2007–present
- **Longer history via FRED:** TB3MS series (see below)

---

## 3. Inflation Data (CPI)

### U.S. CPI-U All Items
- **Source:** Federal Reserve Economic Data (FRED) — St. Louis Fed
- **Series ID:** CPIAUCSL
- **Direct CSV download URL:** https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL
- **Series page:** https://fred.stlouisfed.org/series/CPIAUCSL
- **Coverage:** January 1947–present (monthly)
- **Converted to:** Annual average, year-over-year percent change
- **Notes:** Bureau of Labor Statistics data. No API key required for direct CSV download.

---

## 4. Treasury Yields

### 10-Year Treasury Constant Maturity Rate
- **Source:** FRED — GS10
- **Direct CSV URL:** https://fred.stlouisfed.org/graph/fredgraph.csv?id=GS10
- **Series page:** https://fred.stlouisfed.org/series/GS10
- **Coverage:** April 1953–present (monthly)

### 3-Month Treasury Bill Secondary Market Rate
- **Source:** FRED — TB3MS
- **Direct CSV URL:** https://fred.stlouisfed.org/graph/fredgraph.csv?id=TB3MS
- **Series page:** https://fred.stlouisfed.org/series/TB3MS
- **Coverage:** January 1934–present (monthly)

---

## 5. Social Security Parameters

### Bend Points (PIA Formula)
- **Source:** Social Security Administration
- **URL:** https://www.ssa.gov/oact/cola/bendpoints.html
- **2025 bend points:** $1,226 / $7,391 (monthly AIME)
- **Embedded in code:** `config/defaults.py` → `SS_BEND_POINTS_2025`

### Full Retirement Age by Birth Year
- **Source:** SSA
- **URL:** https://www.ssa.gov/planners/retire/agereduction.html
- **Embedded in code:** `models/social_security.py` → `FRA_TABLE`

### COLA History
- **Source:** SSA
- **URL:** https://www.ssa.gov/oact/cola/colaseries.html

---

## 6. Tax Parameters

### Federal Income Tax Brackets (2025)
- **Source:** IRS Revenue Procedure 2024-40
- **URL:** https://www.irs.gov/newsroom/irs-provides-tax-inflation-adjustments-for-tax-year-2025
- **Embedded in code:** `config/defaults.py` → `TAX_BRACKETS_2025`

### Standard Deduction (2025)
- **Source:** IRS Publication 501 (2025)
- **URL:** https://www.irs.gov/publications/p501
- **Single:** $15,000 | **MFJ:** $30,000 | **Age 65+ bonus:** $1,550/person

### IRMAA Medicare Surcharges (2025)
- **Source:** CMS (Centers for Medicare & Medicaid Services)
- **URL:** https://www.medicare.gov/basics/costs/medicare-costs/costs-at-a-glance
- **Embedded in code:** `models/tax.py` → `IRMAA_TIERS_2025`

### RMD Uniform Lifetime Table
- **Source:** IRS Publication 590-B, Appendix B
- **URL:** https://www.irs.gov/publications/p590b
- **SECURE 2.0 RMD start age:** 73 (2023+)
- **Embedded in code:** `models/portfolio.py` → `IRS_UNIFORM_LIFETIME_TABLE`

---

## 7. Schwab ETF Equivalents

The following Schwab ETFs have expense ratios under 15 bps and serve as
direct proxies for the 8 modeled asset classes:

| Asset Class | Schwab ETF | Ticker | Expense Ratio |
|---|---|---|---|
| US Large Cap | Schwab US Broad Market ETF | SCHB | 0.03% |
| US Small Cap | Schwab US Small-Cap ETF | SCHA | 0.04% |
| International Developed | Schwab International Equity ETF | SCHF | 0.06% |
| Emerging Markets | Schwab Emerging Markets Equity ETF | SCHE | 0.11% |
| US Bonds | Schwab US Aggregate Bond ETF | SCHZ | 0.03% |
| TIPS | Schwab US TIPS ETF | SCHP | 0.03% |
| Cash | Schwab Value Advantage Money Fund | SWVXX | variable |
| REITs | Schwab US REIT ETF | SCHH | 0.07% |

**Schwab ETF screener:** https://www.schwab.com/etfs/schwab-etfs
**Schwab account types:** https://www.schwab.com/retirement-ira

---

## 8. Download Instructions

### Automatic (via app or loader)
Run from project root:
```python
from data.loader import DataLoader
loader = DataLoader()
status = loader.download_all(force_refresh=False)
for src, result in status.items():
    print(f"{src}: {result}")
```
Or click **"Download Historical Data"** in the Streamlit app before running simulation.

### Manual Fallback
If automatic download fails (network restriction, firewall):
1. Visit each URL above in a browser
2. Save files to `data/cache/` with these names:
   - `historical_returns.csv` — annual asset class returns (columns = asset classes, index = year)
   - `cpi_annual.csv` — single column `inflation_rate`, index = year
3. The system will detect and use these cached files automatically

### No-Download Mode
If no download is possible, the simulation runs on **embedded fallback statistics** (1970–2024 means, standard deviations, and correlation matrix). Results will be statistically representative but will not reflect post-2024 market conditions. All fallback parameters are documented in `data/loader.py`.

---

*All data sources are publicly available at no cost. No API keys are required.*
*Last reviewed: 2026-03-08*

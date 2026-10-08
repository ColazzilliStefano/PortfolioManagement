# Markowitz Portfolio Management

Progetto di analisi e ottimizzazione di portafogli con il modello media-varianza di Markowitz.

Il progetto scarica dati storici, costruisce rendimenti mensili, stima rendimento atteso e covarianza, calcola portafogli ottimali e verifica le strategie con un backtest walk-forward out-of-sample.

## Obiettivo

Confrontare tre strategie di allocazione:

- **Max Sharpe**: massimizza il rapporto tra rendimento atteso in eccesso e volatilita.
- **Min Variance**: minimizza la volatilita del portafoglio.
- **Equal Weight**: benchmark 1/N, con peso uguale su tutti gli asset rischiosi.

Il risk-free e rappresentato da `BIL` e viene escluso dall'ottimizzazione degli asset rischiosi, ma utilizzato nei calcoli dello Sharpe e del Sortino.

## Universo investibile

Gli asset sono definiti in [config/assets.yaml](config/assets.yaml):

| Ticker | Ruolo |
|---|---|
| ACWI | Azionario globale |
| AGG | Obbligazionario aggregate USA |
| TLT | Treasury USA lunga scadenza |
| LQD | Credito investment grade |
| HYG | High yield |
| VNQ | Real estate USA |
| DBC | Commodities |
| GLD | Oro |
| BIL | Proxy risk-free, escluso dall'ottimizzazione |

L'universo viene validato prima dell'analisi. Se nei dati processati compaiono ticker non configurati, lo script Markowitz si interrompe invece di produrre risultati ambigui.

## Struttura

```text
markowitz-stage/
├── config/
│   ├── config.yaml
│   └── assets.yaml
├── data/
│   ├── raw/prices/
│   ├── interim/
│   └── processed/
├── outputs/
│   ├── figures/
│   ├── tables/
│   └── data_quality/
├── scripts/
│   ├── run_01_download.py
│   ├── run_02_clean.py
│   ├── run_03_validate.py
│   └── run_04_markowitz.py
├── src/
│   ├── data/
│   ├── markowitz/
│   │   └── shrinkage.py
│   ├── analytics/
│   ├── utils/
│   └── viz/
├── tests/
├── requirements.txt
└── README.md
```

## Pipeline di esecuzione

Eseguire sempre i comandi dalla root del progetto:

```powershell
cd C:\Users\user\PortfolioManagement\markowitz-stage
.\.venv\Scripts\python.exe scripts\run_01_download.py
.\.venv\Scripts\python.exe scripts\run_02_clean.py
.\.venv\Scripts\python.exe scripts\run_03_validate.py
.\.venv\Scripts\python.exe scripts\run_04_markowitz.py
```

### 1. Download

[run_01_download.py](scripts/run_01_download.py) legge l'universo da `assets.yaml`, aggiunge BIL come risk-free e scarica i prezzi da Yahoo Finance in:

```text
data/raw/prices/prices_raw.parquet
```

### 2. Pulizia e trasformazione

[run_02_clean.py](scripts/run_02_clean.py):

- ordina e pulisce i prezzi;
- applica forward fill limitato;
- elimina asset con storia insufficiente;
- calcola rendimenti logaritmici per lo storage;
- aggrega i rendimenti a frequenza mensile;
- identifica outlier;
- salva dati, metadata e report di qualita.

Output principali:

```text
data/processed/prices.parquet
data/processed/returns.parquet
data/processed/returns_monthly.parquet
data/processed/metadata.parquet
data/processed/outliers.parquet
data/processed/universe.yaml
```

### 3. Validazione

[run_03_validate.py](scripts/run_03_validate.py) controlla che i file necessari esistano e che i rendimenti mensili:

- siano leggibili;
- non contengano valori mancanti;
- abbiano indice ordinato;
- non contengano date duplicate.

### 4. Analisi Markowitz

[run_04_markowitz.py](scripts/run_04_markowitz.py):

1. separa BIL dagli asset rischiosi;
2. stima rendimento medio annualizzato e matrice di covarianza annualizzata;
3. calcola portafoglio Min Variance e Max Sharpe;
4. costruisce la frontiera efficiente;
5. esegue il backtest walk-forward;
6. confronta Max Sharpe, Min Variance ed Equal Weight;
7. misura errore di stima, instabilita dei pesi e turnover;
8. salva tabelle e figure.

## Teoria utilizzata

Per un vettore di pesi $w$, rendimento atteso $\mu$ e matrice di covarianza $\Sigma$:

$$
E[R_p] = w^T\mu
$$

$$
\sigma_p = \sqrt{w^T\Sigma w}
$$

$$
Sharpe_p = \frac{E[R_p] - r_f}{\sigma_p}
$$

I rendimenti sono archiviati come log-return, ma vengono convertiti in
simple-return prima della stima, aggregazione e calcolo delle metriche:

$$R_{simple,t}=\exp(r_{log,t})-1$$

$$V_t=\prod_{i=1}^{t}(1+R_{simple,i})$$

L'ottimizzazione corrente e long-only, con somma dei pesi uguale a 1 e limite
massimo per asset pari a 0.30, coerente con il confronto GMP. Non sono
consentite posizioni short.

## Risultati correnti

I risultati riportati qui sotto sono quelli rigenerati con lo stesso universo
rischioso del GMP: ACWI/AGG/TLT/LQD/HYG/VNQ/DBC/GLD e BIL come risk-free.

### Performance out-of-sample

| Strategia | CAGR | Volatilita | Sharpe | Sortino | Max Drawdown | Calmar |
|---|---:|---:|---:|---:|---:|---:|
| Max Sharpe OOS | 6,38% | 7,79% | 0,63 | 0,84 | -19,10% | 0,33 |
| Min Variance OOS | 3,72% | 6,09% | 0,36 | 0,47 | -16,69% | 0,22 |
| Equal Weight | 5,06% | 7,90% | 0,46 | 0,65 | -16,81% | 0,30 |

Interpretazione:

- Max Sharpe ottiene il rendimento e lo Sharpe piu alti, ma anche il rischio e il drawdown piu elevati.
- Min Variance riduce significativamente volatilita e drawdown, pagando un costo in termini di rendimento.
- Equal Weight e un benchmark intermedio e offre un confronto utile contro l'ottimizzazione parametrica.

### Pesi in-sample

#### Max Sharpe

| Asset | Peso |
|---|---:|
| ACWI | 30,00% |
| AGG | 29,89% |
| HYG | 10,11% |
| GLD | 30,00% |

Gli altri asset ricevono peso nullo o numericamente prossimo a zero.

#### Min Variance

| Asset | Peso |
|---|---:|
| AGG | 30,00% |
| LQD | 30,00% |
| HYG | 14,30% |
| DBC | 12,74% |
| TLT | 10,81% |
| GLD | 2,14% |
| Altri asset | 0,00% |


### Errore di stima

L'esperimento in [critique.py](src/markowitz/critique.py) mostra la differenza tra Sharpe stimato sul campione di training e Sharpe sul periodo successivo:

- Sharpe in-sample tipico: circa `0,64-0,76`;
- Sharpe out-of-sample spesso vicino a zero o negativo;
- in alcuni split l'out-of-sample e positivo, ma molto variabile.

Questo risultato e coerente con la teoria: il portafoglio Max Sharpe dipende fortemente dalle stime di rendimento medio, che sono rumorose e instabili.

### Turnover

Il turnover annuale al ribilanciamento e spesso elevato, con valori che arrivano a circa `1,83`. Questo significa che l'allocazione ottimale cambia in modo sostanziale tra una finestra e la successiva.

I risultati correnti **non includono costi di transazione**. Il rendimento netto reale sarebbe quindi probabilmente piu basso, soprattutto per Max Sharpe.

## Prima estensione: Ledoit-Wolf shrinkage

E stata aggiunta una sezione sperimentale di shrinkage in [shrinkage.py](src/markowitz/shrinkage.py).
La baseline continua a usare la covarianza campionaria; in parallelo viene stimata una covarianza Ledoit-Wolf:

$$
\hat{\Sigma}_{shrunk} = (1-\lambda)\hat{\Sigma} + \lambda F
$$

Il parametro $\lambda$ viene stimato automaticamente da Ledoit-Wolf. Nell'ultima esecuzione:

```text
shrinkage_alpha = 0,0733
```

Il confronto in-sample e disponibile in [shrinkage_comparison.csv](outputs/tables/shrinkage_comparison.csv):

| Modello | Rendimento atteso | Volatilita | Sharpe |
|---|---:|---:|---:|
| Covarianza campionaria - Max Sharpe | 6,38% | 9,81% | 0,522 |
| Covarianza campionaria - Min Variance | 2,44% | 4,39% | 0,269 |
| Ledoit-Wolf - Max Sharpe | 6,72% | 10,33% | 0,529 |
| Ledoit-Wolf - Min Variance | 2,60% | 5,70% | 0,234 |

Questi numeri sono **in-sample** e non devono essere interpretati come miglioramento definitivo. Il prossimo controllo necessario e un walk-forward OOS con la stessa covarianza shrinked ricalcolata a ogni ribilanciamento.

I pesi sperimentali sono salvati in:

- [shrinkage_tangency_weights.csv](outputs/tables/shrinkage_tangency_weights.csv)
- [shrinkage_minvar_weights.csv](outputs/tables/shrinkage_minvar_weights.csv)

La baseline originale resta nei file `tangency_weights.csv` e `minvar_weights.csv`, così il confronto rimane riproducibile.

## Figure principali

### Spazio dei portafogli e frontiera

La figura unificata mostra portafogli casuali, frontiera efficiente, Max Sharpe e Min Variance nello stesso grafico.

![Spazio dei portafogli e frontiera efficiente](outputs/figures/frontier_markowitz.png)

### Pesi Max Sharpe

![Pesi Max Sharpe](outputs/figures/weights_tangency.png)

### Pesi Min Variance

![Pesi Min Variance](outputs/figures/weights_minvar.png)

### Equity curve out-of-sample

![Equity curve](outputs/figures/equity_curves.png)

### Drawdown

![Drawdown](outputs/figures/drawdowns.png)

### Instabilita dei pesi

![Instabilita dei pesi](outputs/figures/weights_instability.png)

### Turnover

![Turnover](outputs/figures/turnover.png)

### Errore di stima

![Errore di stima](outputs/figures/estimation_error.png)

Altre figure disponibili:

- [Capital Allocation Line](outputs/figures/capital_allocation_line.png)
- [Rolling Sharpe](outputs/figures/rolling_sharpe.png)
- [Correlation heatmap](outputs/figures/correlation_heatmap.png)
- [In-sample vs out-of-sample](outputs/figures/insample_vs_oos.png)

## Tabelle generate

Le tabelle ufficiali sono in [outputs/tables](outputs/tables):

- `tangency_weights.csv`: pesi del portafoglio Max Sharpe;
- `minvar_weights.csv`: pesi del portafoglio Min Variance;
- `performance_summary.csv`: metriche OOS;
- `walkforward_summary.csv`: rendimenti annuali OOS;
- `rolling_weights.csv`: pesi nel tempo;
- `turnover.csv`: turnover al ribilanciamento;
- `estimation_error.csv`: confronto in-sample/out-of-sample;
- `sensitivity_mu.csv`: sensibilita dei pesi alle perturbazioni di mu.
- `shrinkage_comparison.csv`: confronto in-sample tra covarianza campionaria e Ledoit-Wolf.
- `shrinkage_tangency_weights.csv`: pesi Max Sharpe con covarianza shrinked.
- `shrinkage_minvar_weights.csv`: pesi Min Variance con covarianza shrinked.

La cartella `scripts/outputs` non e una cartella valida di risultati e non deve essere utilizzata.

## Criticita attuali

### 1. Sensibilita alle stime

Il Max Sharpe e molto sensibile a rendimento medio, covarianza e periodo storico. I pesi nulli di alcuni asset non dimostrano che siano inutili in assoluto: indicano solo che non sono risultati ottimali con queste stime e questi vincoli.

### 2. Concentrazione

Il portafoglio Max Sharpe e concentrato in quattro asset, soprattutto GLD e ACWI. Questo aumenta il rischio di errore di stima e la dipendenza da specifici regimi di mercato.

### 3. Costi di transazione assenti

Il turnover elevato puo ridurre sensibilmente la performance dopo costi, spread e slippage. I rendimenti attuali sono lordi.

### 4. Risk-free

BIL viene usato come serie mensile time-varying nelle metriche OOS e nella stima del Max Sharpe; la media annualizzata viene riportata solo come descrizione sintetica del campione.

### 5. Dati e date

Il dataset grezzo arriva fino al 16 settembre 2026, ma il runner esclude l'ultimo mese parziale: il campione usato nelle metriche termina a `2026-08-31`.

### 6. Assenza di vincoli realistici

Al momento non sono presenti vincoli di:

- peso minimo o massimo personalizzato per asset;
- turnover massimo;
- costi di transazione;
- tracking error;
- esposizione massima per asset class;
- volatilita target.

## Prossimi passi consigliati

1. **Validare lo shrinkage OOS** nel walk-forward, ricalcolando Ledoit-Wolf a ogni finestra.
2. **Aggiungere costi di transazione** e confrontare rendimento lordo e netto.
3. **Applicare una stima robusta di mu**, oppure usare rendimenti attesi piu conservativi.
4. **Aggiungere vincoli di concentrazione**, ad esempio un limite del 25-35% per singolo asset.
5. **Testare ribilanciamenti diversi**, per esempio mensile, trimestrale e annuale.
6. **Confrontare piu finestre di training**, ad esempio 36, 60 e 120 mesi.
7. **Aggiungere benchmark esterni**, come Equal Weight, 60/40 e un portafoglio risk parity.
8. **Separare chiaramente anni completi e anno parziale 2026**.
9. **Aggiungere test automatici** per shrinkage, pesi, metriche e assenza di look-ahead bias.
10. **Produrre una relazione finale** distinguendo sempre risultati in-sample, out-of-sample e analisi di sensibilita.

## Verifiche eseguite

Lo stato corrente e verificato con:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q src scripts
.\.venv\Scripts\python.exe scripts\run_03_validate.py
```

Esito corrente:

```text
2 passed
Contratto dati rispettato
Universo coerente con assets.yaml
13 figure generate
8 tabelle generate
```

## Nota interpretativa finale

Il progetto e tecnicamente funzionante e i risultati sono coerenti con il modello di Markowitz. La conclusione piu importante non e che Max Sharpe sia sempre superiore, ma che l'ottimizzazione dei rendimenti medi produce portafogli concentrati e instabili. Il confronto OOS e quindi essenziale: Min Variance e Equal Weight rappresentano benchmark importanti per capire se il beneficio dell'ottimizzazione sopravvive fuori campione e dopo l'introduzione dei costi di transazione.

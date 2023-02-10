__version__ = 'dev'
import pandas as pd
import numpy as np
from datetime import timedelta
import random
import yfinance as yf
#!pip install git+https://github.com/fedefliguer/chartlevels@main#egg=chartlevels
from chartlevels import *

def descarga(tickers, empieza, termina):
    '''
    Función que desde un array de tickers, un string de fecha de comienzo y de fin descarga y genera la base de dataframes y la de retornos.
    '''
    i = 0
    df_global = pd.DataFrame()
    for ticker in tickers:
        df = yf.download(ticker, start=empieza, end=termina).reset_index()
        i = i + 1
        print(i)
        df['Ticker'] = ticker
        df_global = df_global.append(df, ignore_index=True)

    daily_returns = df_global[['Date', 'Ticker', 'Adj Close']].copy()
    daily_returns.loc[:, 'Return'] = np.where(
        daily_returns['Ticker'] == daily_returns['Ticker'].shift(1), 
        daily_returns['Adj Close']/daily_returns['Adj Close'].shift(1) - 1,
        np.nan)

    return df_global, daily_returns

def build_individual_strategy(dataset, estrategia):
    '''
    Agrega indicadores al dataset, precedidos del string 'i_', necesarios para el cálculo de las estrategias.
    Luego agrega estrategias al dataset, precedidas del string 'e_'.
    Una estrategia es una columna compuesta por strings comenzados por C, M o V. El string comenzado por C significa compra, con un máximo de días según continúe el string. El string comenzado por M significa mantener. El string comenzado por V significa venta a partir de los días según se continúe.
    Por ejemplo, 'C90' implica que ese día se comprará el activo, con un máximo de 90 días conservandolo. 'V7' implica que ese día se venderá el activo, si es que pasaron al menos 7 días desde la compra.
    '''
    if estrategia == 'e_RSI':
        # RSI
        returns = dataset["Close"].pct_change()
        up = returns.clip(
            lower=0
        )  # levanta los valores que fueron superiores al valor indicado = 0
        down = returns.clip(upper=0)
        ema_up = up.ewm(com=14, adjust=False).mean()
        ema_down = down.ewm(com=14, adjust=False).mean()
        rs = -(ema_up / ema_down)
        datarsi = 100 - (100 / (1 + rs))
        rsi = round(datarsi, 2)
        dataset["i_RSI"] = datarsi
        
        # ESTRATEGIAS
        dataset["e_RSI"] = np.where(
            dataset["i_RSI"] < 30, "C30", np.where(dataset["i_RSI"] > 70, "V0", "M")
        )
        
        # LIMPIO PRIMEROS DÍAS POR TICKER
        dias_minimos = 14

    if estrategia == 'e_STOCH/RSI_(20+1l/30)_/(70/30d+50)/90d':

        # RSI
        returns = dataset["Close"].pct_change()
        up = returns.clip(
            lower=0
        )  # levanta los valores que fueron superiores al valor indicado = 0
        down = returns.clip(upper=0)
        ema_up = up.ewm(com=14, adjust=False).mean()
        ema_down = down.ewm(com=14, adjust=False).mean()
        rs = -(ema_up / ema_down)
        datarsi = 100 - (100 / (1 + rs))
        rsi = round(datarsi, 2)
        dataset["i_RSI"] = datarsi

        # ESTOCASTICO
        stok = (
            100
            * (dataset["Close"] - dataset["Low"].rolling(14).min())
            / (dataset["High"].rolling(14).max() - dataset["Low"].rolling(14).min())
        )
        dataset["i_STOK"] = stok
        stod = stok.rolling(3).mean()
        dataset["i_STOD"] = stod

        dataset["e_RSI_30_70/30d+50/90d"] = np.where(
                    dataset["i_RSI"].isna(),
                    np.nan,
                    np.where(
                        dataset["i_RSI"] < 30,
                        "C90",
                        np.where(
                            dataset["i_RSI"] > 70, "V0", np.where(dataset["i_RSI"] > 50, "V30", "M")
                        ),
                    ),
                )

        dataset["e_STOCH_20+1l_80/30d"] = np.where(
        (
        (dataset["i_STOK"] < 20)
        & (dataset["i_STOK"] > dataset["i_STOD"])
        & (dataset["i_STOK"].shift(1) > dataset["i_STOD"].shift(1))
        & (dataset["i_STOK"].shift(2) < dataset["i_STOD"].shift(2))
        ),
        "C30",
         np.where(
                (
                ((dataset["i_STOK"] > 80) | (dataset["i_STOK"].shift(1) > 80))
                & (dataset["i_STOK"] < dataset["i_STOD"])
                & (dataset["i_STOK"].shift(1) > dataset["i_STOD"].shift(1))
                )
                , "V0"
                , "M"
                )
        )

        dataset["e_STOCH/RSI_(20+1l/30)_/(70/30d+50)/90d"] = np.where(
        (dataset["e_STOCH_20+1l_80/30d"] == "C30")
        | (dataset["e_RSI_30_70/30d+50/90d"] == "C90"),
        "C90",
        np.where(
            (dataset["e_RSI_30_70/30d+50/90d"] == "V0"),
            "V0",
            np.where((dataset["e_RSI_30_70/30d+50/90d"] == "V30"), "V30", "M"),
        ),
        )

        dias_minimos = 14

    if estrategia == 'e_distancia_soporte_resistencia_3x':
        dataset_aux = pd.DataFrame()
        for t in dataset.Ticker.unique():
            dataset_ticker = dataset[dataset.Ticker == t].copy()
            dataset_ticker = calculador_soportes_resistencias(dataset_ticker)
            dataset_ticker["e_distancia_soporte_resistencia_3x"] = np.where(
                ((dataset_ticker["resistencia_10_valor"]/dataset_ticker["Close"]) - 1) > ((dataset_ticker["Close"]/dataset_ticker["soporte_10_valor"]) - 1) * 3,
                "C30",
                np.where(
                ((dataset_ticker["resistencia_10_valor"]/dataset_ticker["Close"]) - 1) * 3 < ((dataset_ticker["Close"]/dataset_ticker["soporte_10_valor"]) - 1),
                "V0",
                "M")
            )
            dataset_aux = pd.concat([dataset_aux, dataset_ticker])
        dataset = dataset_aux.copy()
        
        # LIMPIO PRIMEROS DÍAS POR TICKER
        dias_minimos = 60
    
    dataset = dataset[dataset.Ticker == dataset.Ticker.shift(dias_minimos)]
    
    return dataset

def build_trades(dataset, estrategia):
    '''
    Construye un dataset completo de posibles trades que podría considerar la estrategia, con todos los días posibles de compras y ventas.
    '''
    posibles_compras = dataset[dataset[estrategia].str.match("C")][['Ticker', 'Date', estrategia]]
    posibles_compras['dias_compras'] = posibles_compras[estrategia].str.split('C', expand=True)[1].astype(int)
    
    # Construyo los posibles cierres por vencimiento
    candidatos_trades_vencimiento = pd.DataFrame()
    for i in range(0, 12):
        posibles_cierres_vencimiento = posibles_compras.copy()
        posibles_cierres_vencimiento.columns = ['Ticker', 'Date_compra', estrategia, 'dias_compras']
        posibles_cierres_vencimiento['Date_venta'] = posibles_cierres_vencimiento['Date_compra'] + pd.to_timedelta(posibles_cierres_vencimiento['dias_compras'] + i, unit='D')
        candidatos_trades_vencimiento = candidatos_trades_vencimiento.append(posibles_cierres_vencimiento)
    
    if len(dataset[dataset[estrategia].str.match("V")][['Ticker', 'Date', estrategia]]) > 0:
        posibles_ventas = dataset[dataset[estrategia].str.match("V")][['Ticker', 'Date', estrategia]]
        posibles_ventas['dias_ventas'] = posibles_ventas[estrategia].str.split('V', expand=True)[1].astype(int)
        candidatos_trades_cierre = posibles_compras.merge(posibles_ventas, how='inner', left_on=['Ticker'], right_on = ['Ticker'], suffixes=('_compra', '_venta'))

        # Solo fechas de venta posteriores a la fecha de compra
        candidatos_trades_cierre = candidatos_trades_cierre[candidatos_trades_cierre.Date_venta > candidatos_trades_cierre.Date_compra]

        # Solo fechas de venta no posteriores a la fecha límite planteada para la compra
        candidatos_trades_cierre = candidatos_trades_cierre[candidatos_trades_cierre['Date_compra'] + pd.to_timedelta(candidatos_trades_cierre['dias_compras'], unit='D') > candidatos_trades_cierre['Date_venta']]

        # Solo fechas de venta no anteriores a la primer fecha con la que ese requisito aplica para la venta
        candidatos_trades_cierre = candidatos_trades_cierre[(candidatos_trades_cierre['Date_venta'] - candidatos_trades_cierre['Date_compra']).dt.days > candidatos_trades_cierre['dias_ventas']]

        # Unifico
        candidatos_trades = pd.concat([candidatos_trades_cierre[['Ticker', 'Date_compra', 'Date_venta']], candidatos_trades_vencimiento[['Ticker', 'Date_compra', 'Date_venta']]]).sort_values(by=['Ticker', 'Date_compra'])
        
        # Excluyo fechas de venta que no son días hábiles
        candidatos_trades = candidatos_trades.merge(dataset[['Ticker', 'Date', 'Close']], how='left', left_on=['Ticker','Date_venta'], right_on = ['Ticker', 'Date']).dropna()[['Ticker','Date_compra','Date_venta']]

    else:
        candidatos_trades = candidatos_trades_vencimiento
        candidatos_trades = candidatos_trades.merge(dataset[['Ticker', 'Date', 'Close']], how='left', left_on=['Ticker','Date_venta'], right_on = ['Ticker', 'Date']).dropna()[['Ticker','Date_compra','Date_venta']]
        
    # Me quedo con la primer venta por compra
    dataset_trades = candidatos_trades.groupby(['Ticker', 'Date_compra']).agg({'Date_venta': 'min'}).reset_index()
        
    return dataset_trades    

def filtro_fundamental(df_trades):
    url = 'https://raw.githubusercontent.com/fedefliguer/trading_ideas/main/202302%20-%20analisis_fundamental.csv'
    df_fundamental = pd.read_csv(url, index_col=0)
    df_fundamental['Ticker'] = df_fundamental['ticker']
    df_fundamental['Date'] = pd.to_datetime(df_fundamental['Date'])
    df_fundamental = df_fundamental[['Ticker', 'Date', 'Posible_compra']]
    dataset_trades = pd.merge_asof(left=df_trades.sort_values(by=['Date_compra']), 
                  right=df_fundamental.sort_values(by=['Date']), 
                  left_on='Date_compra',
                  right_on='Date',
                  by='Ticker')
    
    dataset_trades = dataset_trades[dataset_trades.Posible_compra == 'Sí'][['Ticker','Date_compra','Date_venta']]
    return dataset_trades

def build_strategy_returns(df_trades, daily_returns, benchmark_returns, duplicados = 'random', fr = 0):
    '''
    En base a los trades posibles, a la decisión en caso de tener más de un activo disponible el mismo día y a la tasa libre de riesgo, construye el set de retornos de la estrategia.
    '''
    strategy_returns = []
    for day in benchmark_returns.index:
        posible_tickers = df_trades[(day > df_trades.Date_compra)&(day <= df_trades.Date_venta)].Ticker.unique()
        if len(posible_tickers) == 0:
            ticker = np.nan
            if type(fr) == 'float':
                daily_return_strategy = fr / 365
            if fr == 'benchmark':
                daily_return_strategy = benchmark_returns[day]

        elif len(posible_tickers) == 1:
            ticker = posible_tickers[0]
            daily_return_strategy = daily_returns[(daily_returns.Date == day)&(daily_returns.Ticker == ticker)].Return.values[0]
            daily_return_strategy = daily_return_strategy - 0.0005
            if pd.isna(daily_return_strategy):
                daily_return_strategy = 0

        elif len(posible_tickers) > 1:
            if duplicados == 'random': # En caso de más de uno, me quedo con el RSI más bajo
                ticker = random.sample(list(posible_tickers), 1)[0]
            daily_return_strategy = daily_returns[(daily_returns.Date == day)&(daily_returns.Ticker == ticker)].Return.values[0]
            daily_return_strategy = daily_return_strategy - 0.0005
            if pd.isna(daily_return_strategy):
                daily_return_strategy = 0

        strategy_returns = np.append(strategy_returns, daily_return_strategy)
        #print('El día', day, 'se elige el activo', ticker, 'cuyo retorno fue', daily_return_strategy)

    return strategy_returns

def getKelly(returns):
    perdedores = returns[returns < 0]
    ganadores = returns[returns > 0]

    p_win = len(ganadores) / len(returns)
    p_loss = 1 - p_win

    win_m = ganadores.mean()
    loss_m = -perdedores.mean()

    kelly = p_win/loss_m  - p_loss/win_m

    return kelly

def resumen(strategy, benchmark):
    '''
    De la iteración, construye un dataframe resumen del resultado.
    '''
    resumen_iteration = pd.DataFrame(columns = ['metrica','grupo','valor'])
    strategy_lin = strategy.pct_change()
    benchmark_lin = benchmark.pct_change() 
    retorno_acum_st = ((strategy_lin+1).prod()-1)
    retorno_acum_bench = ((benchmark_lin+1).prod()-1)
    resumen_iteration.loc[len(resumen_iteration)] = ['retorno_acum', 'strategy', retorno_acum_st]
    resumen_iteration.loc[len(resumen_iteration)] = ['retorno_acum', 'benchmark', retorno_acum_bench]
    years = (strategy_lin.index[-1]-strategy_lin.index[0]).days/365
    cagr_st = (1+retorno_acum_st)**(1/years)-1
    years = (benchmark_lin.index[-1]-benchmark_lin.index[0]).days/365
    cagr_bench = (1+retorno_acum_bench)**(1/years)-1
    resumen_iteration.loc[len(resumen_iteration)] = ['cagr', 'strategy', cagr_st]
    resumen_iteration.loc[len(resumen_iteration)] = ['cagr', 'benchmark', cagr_bench]
    volatilidad_st = strategy_lin.std() * 252**0.5
    volatilidad_bench = benchmark_lin.std() * 252**0.5
    resumen_iteration.loc[len(resumen_iteration)] = ['volatilidad', 'strategy', volatilidad_st]
    resumen_iteration.loc[len(resumen_iteration)] = ['volatilidad', 'benchmark', volatilidad_bench]
    #sharpe_st = (cagr_st - free_rate_risk) / volatilidad_st
    #sharpe_benchmark = (cagr_bench - free_rate_risk) / volatilidad_bench
    #resumen_iteration.loc[len(resumen_iteration)] = ['sharpe', 'strategy', sharpe_st]
    #resumen_iteration.loc[len(resumen_iteration)] = ['sharpe', 'benchmark', sharpe_benchmark]
    #kelly_st = getKelly(strategy_log)
    #kelly_benchmark = getKelly(benchmark_log)
    #resumen_iteration.loc[len(resumen_iteration)] = ['kelly', 'strategy', kelly_st]
    #resumen_iteration.loc[len(resumen_iteration)] = ['kelly', 'benchmark', kelly_benchmark]

    return resumen_iteration

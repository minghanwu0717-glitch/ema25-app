# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from io import BytesIO

st.set_page_config(page_title="EMA23 回測 APP (Week1 Starter+)", layout="wide")
st.title("EMA23 回測 APP · Week1 Starter+")
st.caption("教學用途，非投資建議。")

# ========== 側邊欄參數 ==========
with st.sidebar:
    st.subheader("參數設定")
    ticker = st.text_input("股票代碼（台股請加 .TW，如 2330.TW）", "2330.TW")
    period = st.selectbox("區間", ["1周", "1個月", "1年", "2年"], index=2)
    today = pd.to_datetime("today").normalize()
    if period == "1周":
        start = today - pd.Timedelta(days=7)
    elif period == "1個月":
        start = today - pd.DateOffset(months=1)
    elif period == "1年":
        start = today - pd.DateOffset(years=1)
    elif period == "2年":
        start = today - pd.DateOffset(years=2)
    end = today

    # 新增策略選擇
    strategy = st.selectbox(
        "快速選擇策略",
        [
            "長線保守",
            "長線穩健",
            "當沖建議（5m 版）",
            "隔日沖（日線）"
        ],
        index=0
    )
    # 根據選擇自動填入參數
    if strategy == "長線保守":
        ema_span, buy_x, sell_y, consec_n = 100, 0.0, -1.0, 5
        st.markdown(
            """
            **A（長線－保守跟趨勢）**  
            週期：日線｜參數：EMA=100，X=0%，Y=−1%，N=5  
            重點：連續5天站上長均線才上車，跌破均線1%才下車，交易少、抱比較久。
            """
        )
    elif strategy == "長線穩健":
        ema_span, buy_x, sell_y, consec_n = 67, 0.3, -0.5, 3
        st.markdown(
            """
            **B（長線－標準穩健）**  
            週期：日線｜參數：EMA=67，X=0.3%，Y=−0.5%，N=3  
            重點：稍微嚴格的趨勢跟隨，三連漲且離均線>0.3%才進，回落0.5%才出。
            """
        )
    elif strategy == "當沖建議（5m 版）":
        ema_span, buy_x, sell_y, consec_n = 23, 0.10, -0.10, 2
        st.markdown(
            """
            **C（當沖）**  
            週期：5 分鐘（短線）｜參數：EMA=23，X=0.10%，Y=−0.10%，N=2  
            重點：用小帶寬+連續2根K確認，快進快出，當天收盤前強制平倉。
            """
        )
    elif strategy == "隔日沖（日線）":
        ema_span, buy_x, sell_y, consec_n = 23, 0.5, -0.3, 2
        st.markdown(
            """
            **D（隔日沖）**  
            週期：日線｜參數：EMA=23，X=0.5%，Y=−0.3%，N=2  
            重點：連續2天強勢才進，隔日跌回均線−0.3%就出；可用「明日預掛觸發價」掛單。
            """
        )

    run = st.button("執行回測")

# ========== 資料讀取 ==========
@st.cache_data(show_spinner=False)
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    print(f"下載資料：{ticker}, {start} ~ {end}, 筆數={len(df)}")
    return df

def _pick_close_series(df: pd.DataFrame) -> pd.Series:
    """從 yfinance DataFrame 可靠取出單一 Close（處理單層/多層欄位與多欄情況）"""
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.get_level_values(0)
        if "Close" in lvl0:
            obj = df["Close"]
        elif "Adj Close" in lvl0:
            obj = df["Adj Close"]
        else:
            raise ValueError("找不到 Close 或 Adj Close 欄位")
        s = obj.iloc[:, 0] if isinstance(obj, pd.DataFrame) else obj
    else:
        if "Close" in df.columns:
            obj = df["Close"]
        elif "Adj Close" in df.columns:
            obj = df["Adj Close"]
        else:
            raise ValueError("找不到 Close 或 Adj Close 欄位")
        s = obj.iloc[:, 0] if isinstance(obj, pd.DataFrame) else obj
    s = pd.to_numeric(s, errors="coerce")
    s.name = "Close"
    return s

# ========== 指標計算 ==========
def add_indicators(df, span=23):
    close = _pick_close_series(df)
    ema = close.ewm(span=span, adjust=False).mean()

    out = pd.DataFrame(index=close.index)
    out["Close"] = close.astype(float)
    out["EMA"] = ema.astype(float)
    out["DiffPct"] = (out["Close"] - out["EMA"]) / out["EMA"] * 100.0
    return out

# ========== 回測邏輯（含交易明細） ==========
def build_trades(d: pd.DataFrame) -> pd.DataFrame:
    """依 Position 0→1/1→0 產生交易表；若最後仍持有，最後一天視為出場（教學簡化）。"""
    pos = d["Position"].astype(float)
    change = pos.diff().fillna(pos)  # 0->1 = +1, 1->0 = -1
    entry_idx = change[change > 0].index.tolist()
    exit_idx  = change[change < 0].index.tolist()

    # 若最後仍持有，補一筆出場在最後一天
    if len(entry_idx) > len(exit_idx):
        exit_idx.append(d.index[-1])

    rows = []
    for en, ex in zip(entry_idx, exit_idx):
        if ex <= en:
            continue
        en_px = float(d.loc[en, "Close"])
        ex_px = float(d.loc[ex, "Close"])
        ret = ex_px / en_px - 1.0
        rows.append({
            "EntryDate": en,
            "ExitDate":  ex,
            "EntryPx":   en_px,
            "ExitPx":    ex_px,
            "Ret":       ret,
            "HoldDays":  int((pd.to_datetime(ex) - pd.to_datetime(en)).days)
        })
    trades = pd.DataFrame(rows)
    return trades

def backtest(df, buy_x=0.0, sell_y=0.0, consec_n=1):
    d = df.copy()
    d["BuySig"] = (d["DiffPct"] >= float(buy_x))
    d["Consec"] = d["BuySig"].rolling(int(consec_n)).sum() >= int(consec_n)
    sig = np.select([d["Consec"], (d["DiffPct"] <= float(sell_y))], [1, 0], default=np.nan)
    d["Position"] = pd.Series(sig, index=d.index).ffill().fillna(0.0).astype(float)

    close = d["Close"].astype(float)
    ret = close.pct_change().fillna(0.0)
    d["StrategyRet"] = (ret * d["Position"].shift(1).fillna(0.0)).astype(float)
    d["CumRet"] = (1.0 + d["StrategyRet"]).cumprod() - 1.0
    d["BuyHold"] = (1.0 + ret).cumprod() - 1.0

    trades = build_trades(d)
    return d, trades

# ========== Excel 匯出 ==========
def to_excel(bt_df: pd.DataFrame, trades_df: pd.DataFrame):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        bt_df.to_excel(writer, index=True, sheet_name="backtest")
        trades_df.to_excel(writer, index=False, sheet_name="trades")
    return output.getvalue()

# ========== 主流程 ==========
if run:
    with st.spinner("下載資料中…"):
        raw = load_data(ticker, start, end)

    if raw is None or raw.empty:
        st.warning("下載不到資料，請確認代碼或日期區間。")
    else:
        data = add_indicators(raw, ema_span)
        bt, trades = backtest(data, buy_x, sell_y, consec_n)

        # ---- 績效摘要（以「交易」計算勝率）----
        total_ret = float(bt["CumRet"].iloc[-1])
        bh_ret = float(bt["BuyHold"].iloc[-1])
        trade_win_rate = float((trades["Ret"] > 0).mean()) if len(trades) else 0.0

        st.subheader("績效摘要")
        st.write(f"✔️ 策略總報酬：{total_ret:.2%} 　|　買進抱牢：{bh_ret:.2%} 　|　勝率（交易筆數）：{trade_win_rate:.2%}")

        # ---- 明日預掛觸發價（用「最後一天 EMA」計算）----
        last_ema = float(bt["EMA"].iloc[-1])
        buy_trig  = last_ema * (1.0 + float(buy_x) / 100.0)
        sell_trig = last_ema * (1.0 + float(sell_y) / 100.0)
        st.info(f"📌 依當日 EMA：買進觸發 ≥ **{buy_trig:.2f}**；賣出觸發 ≤ **{sell_trig:.2f}**")

        # ---- 畫圖（收盤＋EMA＋買賣點）----
        bt_plot = bt.copy().assign(Date=bt.index)

        fig = px.line(
            bt_plot, x="Date", y=["Close", "EMA"],
            title=f"{ticker} 收盤價 vs EMA{ema_span}",
        )

        # 買賣點標記
        pos = bt["Position"]
        change = pos.diff().fillna(pos)
        entry_idx = change[change > 0].index
        exit_idx  = change[change < 0].index

        fig.add_trace(go.Scatter(
            x=entry_idx, y=bt.loc[entry_idx, "Close"],
            mode="markers", name="Buy",
            marker_symbol="triangle-up", marker_size=10
        ))
        fig.add_trace(go.Scatter(
            x=exit_idx, y=bt.loc[exit_idx, "Close"],
            mode="markers", name="Sell",
            marker_symbol="triangle-down", marker_size=10
        ))
        st.plotly_chart(fig, use_container_width=True)

        # ---- 交易明細（中文欄位說明）----
        st.subheader(f"交易明細（{len(trades)} 筆）")
        trades_zh = trades.rename(columns={
            "EntryDate": "進場日期",
            "ExitDate": "出場日期",
            "EntryPx": "進場價格",
            "ExitPx": "出場價格",
            "Ret": "投報率%",
            "HoldDays": "持有天數"
        }).copy()
        # 投報率% 欄位格式化為百分比字串
        trades_zh["投報率%"] = (trades_zh["投報率%"] * 100).map(lambda x: f"{x:.2f}%")
        st.dataframe(trades_zh.tail(20), use_container_width=True)

        # ---- 匯出 Excel（兩張表）----
        st.download_button(
            "匯出結果（Excel：backtest + trades）",
            data=to_excel(bt[["Close","EMA","DiffPct","Position","StrategyRet","CumRet","BuyHold"]], trades_zh),
            file_name=f"{ticker}_ema23_backtest.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        # ---- 底部策略參數說明 ----
        st.markdown("""
---
#### 策略參數說明

**EMA（Exponential Moving Average）**  
指數移動平均線。期間＝你設定的天數/根數（如 23、67、100）。越短越靈敏、越長越穩。  
平滑係數 α = 2 / (期間 + 1)。

**X（%）＝買進門檻**  
定義：DiffPct = (收盤 − EMA) / EMA × 100%。  
當 DiffPct ≥ X，且連續滿 N 天/根，就判定進場。X 通常為 ≥0 的小百分比。

**Y（%）＝賣出/出場門檻**  
當 DiffPct ≤ Y 就出場。為避免來回洗價，Y 常設為 0 或負值（如 −0.3%、−1%）。

**N＝連續天數/根數濾波**  
需連續 N 天/根都達到「DiffPct ≥ X」才進場。N 越大訊號越少、越穩健。

**掛單參考價：**  
買進觸發價 = EMA_today × (1 + X/100)  
賣出觸發價 = EMA_today × (1 + Y/100)

**常見範圍：**  
長線：EMA 67–100，X 0～0.5%，Y −1%～0%，N 3～5  
隔日/短波：EMA 20～30，X 0.3～0.8%，Y −0.3%～0%，N 2  
當沖（5分K）：EMA 約 20～30，X ≈ 0.1%，Y ≈ −0.1%，N 2

**一句話記：**  
EMA 定方向、X 設進門檻、Y 設出門檻、N 防雜訊。
""")
else:
    st.info("在左側輸入代碼與日期後，按「執行回測」。")

# app.py
import os
import subprocess
import tempfile
import yaml
import pandas as pd
import numpy as np
import streamlit as st
import mysql.connector
from dotenv import load_dotenv
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Nifty 50 Stock Performance Dashboard")

# --- Load MySQL credentials from .env ---
load_dotenv()
mysql_host = os.getenv("MYSQL_HOST", "localhost")
mysql_user = os.getenv("MYSQL_USER", "root")
mysql_password = os.getenv("MYSQL_PASSWORD", "aira")
mysql_db = "stock_analysis"

# --- Sidebar: Folder with RAR (RAR file path)+ sector CSV ---
st.sidebar.subheader("Data Paths")
folder_path = st.sidebar.text_input(
    "Folder path with RAR files", r"C:\Users\abanu\Desktop"
)
sector_csv_path = st.sidebar.text_input(
    "Sector CSV path", r"C:\Users\abanu\Desktop\sector_mapping.csv"
)

# --- WinRAR path (Since we need to extract RAR file, we use WinRAR) Already installed it ---
winrar_path = r"C:\Program Files\WinRAR\WinRAR.exe"
if not os.path.exists(winrar_path):
    st.error("WinRAR.exe not found. Install WinRAR.")
    st.stop()


# --- Detect first RAR part ---
def detect_first_rar(folder):
    for file in os.listdir(folder):
        if file.lower().endswith(".rar") and (
            "part1" in file.lower()
            or "part01" in file.lower()
            or "part001" in file.lower()
        ):
            return os.path.join(folder, file)
        elif file.lower().endswith(".rar") and "part" not in file.lower():
            return os.path.join(folder, file)
    return None


rar_path = detect_first_rar(folder_path)
if not rar_path:
    st.error("No RAR file found in folder")
    st.stop()
st.write(f"Found RAR: {rar_path}")  # Displays the RAR file path

# --- Temporary folder for extraction ---
extract_folder = tempfile.mkdtemp()
st.write(f"Extracting to: {extract_folder}")

# --- Extract RAR ---
try:
    subprocess.run([winrar_path, "x", "-y", rar_path, extract_folder], check=True)
    st.success("Extraction complete")
except subprocess.CalledProcessError as e:
    st.error(f"Extraction failed: {e}")
    st.stop()

# --- Connect to MySQL ---
try:
    conn = mysql.connector.connect(
        host=mysql_host, user=mysql_user, password=mysql_password, database=mysql_db
    )
    cursor = conn.cursor()
    st.sidebar.success("Connected to MySQL")
except Exception as e:
    st.error(f"MySQL Connection Failed\n{e}")
    st.stop()

# --- Create table ---
cursor.execute(
    """
CREATE TABLE IF NOT EXISTS stock_data (
    Ticker VARCHAR(50),
    Date DATETIME,
    Open FLOAT,
    High FLOAT,
    Low FLOAT,
    Close FLOAT,
    Volume BIGINT,
    Month VARCHAR(20),
    PRIMARY KEY (Ticker, Date)
)
"""
)
conn.commit()


# --- Load YAML ---
def load_yaml_files(folder):
    all_rows = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith((".yaml", ".yml")):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = yaml.safe_load(f)
                        if isinstance(data, list):
                            all_rows.extend(data)
                        elif isinstance(data, dict):
                            if "Ticker" in data:
                                all_rows.append(data)
                            else:
                                for value in data.values():
                                    if isinstance(value, list):
                                        all_rows.extend(value)
                except Exception as e:
                    st.warning(f"Failed to read {file_path}: {e}")
    return pd.DataFrame(all_rows)


stock_df = load_yaml_files(extract_folder)
if stock_df.empty:
    st.error("No data loaded from YAML files.")
    st.stop()
st.write("Sample loaded data:")
st.dataframe(stock_df.head(5))

# --- Normalize columns ---
stock_df.columns = stock_df.columns.str.strip()
stock_df.rename(
    columns={
        "ticker": "Ticker",
        "date": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
        "month": "Month",
    },
    inplace=True,
)

# --- Detect and convert Date ---
if "Date" not in stock_df.columns:
    for col in stock_df.columns:
        if "date" in col.lower():
            stock_df.rename(columns={col: "Date"}, inplace=True)
            break
stock_df["Date"] = pd.to_datetime(stock_df["Date"], errors="coerce")
stock_df = stock_df.dropna(subset=["Date"])
stock_df["Date"] = stock_df["Date"].dt.strftime("%Y-%m-%d %H:%M:%S")

# --- Insert into MySQL ---
inserted = 0
for _, row in stock_df.iterrows():
    sql = """
    REPLACE INTO stock_data
    (Ticker, Date, Open, High, Low, Close, Volume, Month)
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
    """
    cursor.execute(
        sql,
        (
            row.get("Ticker"),
            row.get("Date"),
            row.get("Open"),
            row.get("High"),
            row.get("Low"),
            row.get("Close"),
            row.get("Volume"),
            row.get("Month"),
        ),
    )
    inserted += 1
conn.commit()
st.success(f"Inserted {inserted} rows into MySQL")

# --- Load from MySQL ---
df = pd.read_sql("SELECT * FROM stock_data", conn)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# --- Key Metrics ---
metrics = []
for symbol, group in df.groupby("Ticker"):
    group = group.sort_values("Date")
    if len(group) < 2:
        continue
    first_close = group["Close"].iloc[0]
    last_close = group["Close"].iloc[-1]
    yearly_return = ((last_close - first_close) / first_close) * 100
    metrics.append(
        {
            "Symbol": symbol,
            "Yearly Return": yearly_return,
            "Average Price": group["Close"].mean(),
            "Average Volume": group["Volume"].mean(),
        }
    )
metrics_df = pd.DataFrame(metrics)

# --- Market Summary ---
green_count = (metrics_df["Yearly Return"] > 0).sum()
red_count = (metrics_df["Yearly Return"] < 0).sum()
st.subheader("Market Summary")
st.write(f"Green Stocks: {green_count}, Red Stocks: {red_count}")
st.write(f"Average Price: {metrics_df['Average Price'].mean():.2f}")
st.write(f"Average Volume: {metrics_df['Average Volume'].mean():.0f}")

# --- Top 10 Gainers / Losers ---
# --- Top 10 Best & Worst Performing Stocks (Highlight + Colored Charts) ---

# Top 10 Green (Best Performers)
top10_green = metrics_df.sort_values("Yearly Return", ascending=False).head(10)
st.markdown("### Top 10 Best Performing Stocks (Green)")
st.dataframe(
    top10_green.style.background_gradient(
        subset=["Yearly Return"], cmap="Greens"
    ).format(
        {
            "Yearly Return": "{:.2f}%",
            "Average Price": "{:.2f}",
            "Average Volume": "{:.0f}",
        }
    )
)

# Green Bar Chart
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(top10_green["Symbol"], top10_green["Yearly Return"], color="green")
ax.set_title("Top 10 Best Performing Stocks")
ax.set_ylabel("Yearly Return (%)")
ax.set_xticklabels(top10_green["Symbol"], rotation=45)
st.pyplot(fig)

# Top 10 Red (Worst Performers)
top10_red = metrics_df.sort_values("Yearly Return").head(10)
st.markdown("### Top 10 Worst Performing Stocks (Red)")
st.dataframe(
    top10_red.style.background_gradient(subset=["Yearly Return"], cmap="Reds").format(
        {
            "Yearly Return": "{:.2f}%",
            "Average Price": "{:.2f}",
            "Average Volume": "{:.0f}",
        }
    )
)

# Red Bar Chart
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(top10_red["Symbol"], top10_red["Yearly Return"], color="red")
ax.set_title("Top 10 Worst Performing Stocks")
ax.set_ylabel("Yearly Return (%)")
ax.set_xticklabels(top10_red["Symbol"], rotation=45)
st.pyplot(fig)


# --- Volatility Analysis ---
st.subheader("Volatility Analysis (Top 10 Most Volatile Stocks)")

volatility = []
for symbol, group in df.groupby("Ticker"):
    group = group.sort_values("Date")
    if len(group) > 1:
        # Explicit daily return calculation
        group["Daily Return"] = (group["Close"] - group["Close"].shift(1)) / group[
            "Close"
        ].shift(1)
        vol = group["Daily Return"].std() * 100  # std dev in %
        volatility.append({"Symbol": symbol, "Volatility": vol})

vol_df = pd.DataFrame(volatility).dropna()

# Pick Top 10
top_vol = vol_df.sort_values("Volatility", ascending=False).head(10)

# Styled Table
st.dataframe(
    top_vol.style.background_gradient(subset=["Volatility"], cmap="Reds").format(
        {"Volatility": "{:.2f}%"}
    )
)

# Bar Chart
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(top_vol["Symbol"], top_vol["Volatility"], color="orange")
ax.set_title("Top 10 Most Volatile Stocks")
ax.set_ylabel("Volatility (Std Dev of Daily Returns, %)")
ax.set_xticklabels(top_vol["Symbol"], rotation=45)
st.pyplot(fig)


# --- Cumulative Return ---
st.subheader("Cumulative Return (Top 5 Stocks)")
cumulative = pd.DataFrame()
for symbol, group in df.groupby("Ticker"):
    group = group.sort_values("Date")
    group["Cumulative Return"] = (group["Close"] / group["Close"].iloc[0] - 1) * 100
    cumulative[symbol] = group["Cumulative Return"].values
top5_symbols = (
    metrics_df.sort_values("Yearly Return", ascending=False).head(5)["Symbol"].tolist()
)
st.line_chart(cumulative[top5_symbols])

# --- Sector-wise Performance ---
if os.path.exists(sector_csv_path):
    sector_df = pd.read_csv(sector_csv_path)
    merged = metrics_df.merge(
        sector_df, left_on="Symbol", right_on="Ticker", how="left"
    )
    sector_avg = (
        merged.groupby("Sector")["Yearly Return"].mean().sort_values(ascending=False)
    )
    st.subheader("Sector-wise Average Yearly Return")
    st.bar_chart(sector_avg)
else:
    st.warning("Sector CSV not found. Skipping sector analysis.")

# --- Correlation Heatmap ---
# --- Stock Price Correlation ---
st.subheader("Stock Price Correlation Heatmap")

# Pivot to get closing prices with Date as rows and Ticker as columns
pivot = df.pivot(index="Date", columns="Ticker", values="Close")

# Calculate daily returns explicitly
daily_returns = pivot.pct_change()

# Compute correlation matrix using pandas.DataFrame.corr()
corr_matrix = daily_returns.corr()

# Show correlation matrix in a table (optional)
st.dataframe(corr_matrix.style.background_gradient(cmap="coolwarm").format("{:.2f}"))

# Plot heatmap
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(corr_matrix, cmap="coolwarm", annot=False, linewidths=0.5, ax=ax)
ax.set_title("Stock Price Correlation Heatmap", fontsize=14)
st.pyplot(fig)


# --- Monthly Top 5 Gainers / Losers ---
st.subheader("Monthly Top 5 Gainers and Losers")
df["MonthNum"] = df["Date"].dt.to_period("M")
monthly_results = []
for month, group in df.groupby("MonthNum"):
    temp = group.groupby("Ticker").apply(
        lambda x: (x["Close"].iloc[-1] - x["Close"].iloc[0]) / x["Close"].iloc[0] * 100
    )
    temp = temp.sort_values()
    losers = temp
    losers = temp.head(5).reset_index()
    losers.columns = ["Symbol", "Monthly Return"]

    gainers = temp.tail(5).reset_index()
    gainers.columns = ["Symbol", "Monthly Return"]

    losers["Month"] = str(month)
    gainers["Month"] = str(month)

    losers["Type"] = "Loser"
    gainers["Type"] = "Gainer"

    monthly_results.append(losers)
    monthly_results.append(gainers)

if monthly_results:
    monthly_df = pd.concat(monthly_results)
    st.dataframe(monthly_df)

    # Plot Monthly Performance
    fig, ax = plt.subplots(figsize=(12, 6))
    for key, grp in monthly_df.groupby("Type"):
        ax.plot(
            grp["Month"], grp["Monthly Return"], marker="o", linestyle="-", label=key
        )
    ax.legend()
    ax.set_title("Monthly Top 5 Gainers and Losers")
    ax.set_ylabel("Return (%)")
    ax.set_xlabel("Month")
    plt.xticks(rotation=45)
    st.pyplot(fig)
else:
    st.warning("No monthly data available for analysis")

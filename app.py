import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import plotly.graph_objects as go
import io
import sqlite3
import hashlib
import json
from datetime import datetime

st.set_page_config(page_title="AI Portfolio Optimizer", page_icon="💰", layout="wide")

ASSETS = {
    "Large Cap ETF": "NIFTYBEES.NS",
    "Mid Cap ETF": "JUNIORBEES.NS",
    "Gold ETF": "GOLDBEES.NS",
    "Liquid Fund (Cash-like)": "LIQUIDBEES.NS",
}

GLOBAL_ASSETS = {
    "US Nasdaq 100 (Global)": "MON100.NS",
    "US S&P 500 Top 50 (Global)": "MASPTOP50.NS",
    "Hang Seng / China (Global)": "HNGSNGBEES.NS",
}

DEFENSIVE_ASSETS = {"Gold ETF", "Liquid Fund (Cash-like)"}


# ---------------- DATABASE FUNCTIONS ----------------

def init_db():
    conn = sqlite3.connect("portfolio.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS portfolios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            created_at TEXT,
            profile TEXT,
            score INTEGER,
            amount REAL,
            weights TEXT,
            port_return REAL,
            port_vol REAL,
            sharpe REAL
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS expenses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            date TEXT,
            category TEXT,
            amount REAL,
            note TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS budget_settings (
            username TEXT PRIMARY KEY,
            monthly_income REAL
        )
    """)
    conn.commit()
    conn.close()


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def create_user(username, password):
    conn = sqlite3.connect("portfolio.db")
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)",
                   (username, hash_password(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def check_login(username, password):
    conn = sqlite3.connect("portfolio.db")
    c = conn.cursor()
    c.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
    row = c.fetchone()
    conn.close()
    if row is None:
        return False
    return row[0] == hash_password(password)


def save_portfolio(username, profile, score, amount, weights, asset_names, port_return, port_vol, sharpe):
    conn = sqlite3.connect("portfolio.db")
    c = conn.cursor()
    weights_dict = {name: float(w) for name, w in zip(asset_names, weights)}
    c.execute("""
        INSERT INTO portfolios (username, created_at, profile, score, amount, weights, port_return, port_vol, sharpe)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (username, datetime.now().strftime("%Y-%m-%d %H:%M"), profile, score, amount,
          json.dumps(weights_dict), float(port_return), float(port_vol), float(sharpe)))
    conn.commit()
    conn.close()


def get_saved_portfolios(username):
    conn = sqlite3.connect("portfolio.db")
    c = conn.cursor()
    c.execute("""SELECT created_at, profile, score, amount, weights, port_return, port_vol, sharpe
                 FROM portfolios WHERE username = ? ORDER BY id DESC""", (username,))
    rows = c.fetchall()
    conn.close()
    return rows


def add_expense(username, date, category, amount, note):
    conn = sqlite3.connect("portfolio.db")
    c = conn.cursor()
    c.execute("INSERT INTO expenses (username, date, category, amount, note) VALUES (?, ?, ?, ?, ?)",
               (username, date, category, amount, note))
    conn.commit()
    conn.close()


def get_expenses(username):
    conn = sqlite3.connect("portfolio.db")
    c = conn.cursor()
    c.execute("SELECT id, date, category, amount, note FROM expenses WHERE username = ? ORDER BY date DESC",
               (username,))
    rows = c.fetchall()
    conn.close()
    return rows


def delete_expense(expense_id):
    conn = sqlite3.connect("portfolio.db")
    c = conn.cursor()
    c.execute("DELETE FROM expenses WHERE id = ?", (expense_id,))
    conn.commit()
    conn.close()


def set_monthly_income(username, income):
    conn = sqlite3.connect("portfolio.db")
    c = conn.cursor()
    c.execute("""
        INSERT INTO budget_settings (username, monthly_income) VALUES (?, ?)
        ON CONFLICT(username) DO UPDATE SET monthly_income = excluded.monthly_income
    """, (username, income))
    conn.commit()
    conn.close()


def get_monthly_income(username):
    conn = sqlite3.connect("portfolio.db")
    c = conn.cursor()
    c.execute("SELECT monthly_income FROM budget_settings WHERE username = ?", (username,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else 0.0


init_db()


# ---------------- CORE LOGIC FUNCTIONS ----------------

@st.cache_data(ttl=3600)
def get_price_data(tickers, period="5y"):
    data = yf.download(tickers, period=period)["Close"]
    return data.dropna()


def calculate_risk_score(age, experience, horizon, reaction):
    score = 0
    score += {"1 year": 5, "5 years": 15, "10+ years": 25}[horizon]
    score += {"None": 5, "Beginner": 10, "Intermediate": 20, "Expert": 25}[experience]
    score += {"Sell everything immediately": 0, "Sell some": 10, "Do nothing": 20, "Buy more": 25}[reaction]
    score += max(0, 25 - (age - 18) // 2)
    return score


def optimize_portfolio(returns, risk_free_rate=0.06, equity_cap=1.0):
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    n = len(mean_returns)

    def neg_sharpe(weights):
        port_return = np.dot(weights, mean_returns)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -(port_return - risk_free_rate) / port_vol

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "ineq", "fun": lambda w: equity_cap - (w[0] + w[1])},
    ]
    bounds = tuple((0, 1) for _ in range(n))
    init_guess = np.array([1 / n] * n)

    result = minimize(neg_sharpe, init_guess, method="SLSQP", bounds=bounds, constraints=constraints)
    return result.x, mean_returns, cov_matrix


def optimize_portfolio_general(returns, asset_names, defensive_names, risk_free_rate=0.06, equity_cap=1.0):
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    n = len(mean_returns)

    equity_idx = [i for i, name in enumerate(asset_names) if name not in defensive_names]

    def neg_sharpe(weights):
        port_return = np.dot(weights, mean_returns)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -(port_return - risk_free_rate) / port_vol

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "ineq", "fun": lambda w: equity_cap - sum(w[i] for i in equity_idx)},
    ]
    bounds = tuple((0, 1) for _ in range(n))
    init_guess = np.array([1 / n] * n)

    result = minimize(neg_sharpe, init_guess, method="SLSQP", bounds=bounds, constraints=constraints)
    return result.x, mean_returns, cov_matrix


def generate_explanation(profile, score, weights, asset_names, port_return, port_vol, sharpe, equity_cap):
    equity_weight = weights[0] + weights[1]
    gold_weight = weights[2]
    cash_weight = weights[3]

    top_idx = int(np.argmax(weights))
    top_asset = asset_names[top_idx]
    top_weight = weights[top_idx]

    lines = []
    lines.append(
        f"Based on your answers, you scored **{score}/100**, placing you in the **{profile}** risk category. "
        f"This means your portfolio was built with a maximum of **{equity_cap*100:.0f}%** allowed in equity-type assets (Large Cap + Mid Cap)."
    )
    lines.append(
        f"The optimizer picked **{top_asset}** as your largest holding at **{top_weight*100:.1f}%**, "
        f"because — across the last 5 years of data — it offered the best risk-adjusted return within your allowed limits."
    )
    if equity_weight > 0.5:
        lines.append(f"Your total equity exposure is **{equity_weight*100:.1f}%**, which is relatively high. This typically means higher potential growth, but also bigger swings in value year to year.")
    elif equity_weight > 0.2:
        lines.append(f"Your total equity exposure is **{equity_weight*100:.1f}%** — a balanced mix of growth potential and stability.")
    else:
        lines.append(f"Your total equity exposure is only **{equity_weight*100:.1f}%**, prioritizing capital protection over growth.")

    if gold_weight > 0.1:
        lines.append(f"**{gold_weight*100:.1f}%** was allocated to Gold — this acts as a hedge, since gold often moves opposite to stocks during market stress.")
    if cash_weight > 0.1:
        lines.append(f"**{cash_weight*100:.1f}%** is in a Liquid/Cash-like fund, giving you stability and quick access to funds if needed.")

    if sharpe > 1:
        lines.append(f"Your portfolio's Sharpe Ratio of **{sharpe:.2f}** is strong — you're getting good return for each unit of risk taken.")
    elif sharpe > 0.5:
        lines.append(f"Your Sharpe Ratio of **{sharpe:.2f}** is decent — a reasonable trade-off between risk and return.")
    else:
        lines.append(f"Your Sharpe Ratio of **{sharpe:.2f}** is on the lower side — this can happen in choppy market periods, but the mix is still the best available given your risk limits.")

    return "\n\n".join(lines)


def generate_generic_explanation(weights, asset_names, defensive_names, port_return, port_vol, sharpe, equity_cap):
    equity_weight = sum(w for w, name in zip(weights, asset_names) if name not in defensive_names)
    top_idx = int(np.argmax(weights))
    top_asset = asset_names[top_idx]
    top_weight = weights[top_idx]

    lines = []
    lines.append(
        f"This mix was built with a maximum of **{equity_cap*100:.0f}%** allowed in growth/equity-type assets, "
        f"and **{top_asset}** came out as the largest holding at **{top_weight*100:.1f}%** based on 5 years of historical data."
    )
    lines.append(f"Total growth-asset exposure (domestic + global equity) is **{equity_weight*100:.1f}%**.")
    if sharpe > 1:
        lines.append(f"Sharpe Ratio of **{sharpe:.2f}** is strong — good return for the risk taken.")
    elif sharpe > 0.5:
        lines.append(f"Sharpe Ratio of **{sharpe:.2f}** is decent.")
    else:
        lines.append(f"Sharpe Ratio of **{sharpe:.2f}** is on the lower side for this data window.")
    return "\n\n".join(lines)


def generate_pdf_report(profile, score, weights, asset_names, amount, port_return, port_vol, sharpe, explanation):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import cm

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - 2 * cm

    c.setFont("Helvetica-Bold", 16)
    c.drawString(2 * cm, y, "AI Portfolio Optimizer - Report")
    y -= 1 * cm

    c.setFont("Helvetica", 10)
    c.drawString(2 * cm, y, f"Generated: {datetime.now().strftime('%d %b %Y, %H:%M')}")
    y -= 1 * cm

    c.setFont("Helvetica-Bold", 12)
    c.drawString(2 * cm, y, f"Risk Profile: {profile} (Score: {score}/100)")
    y -= 0.8 * cm
    c.drawString(2 * cm, y, f"Investment Amount: Rs. {amount:,.0f}")
    y -= 1 * cm

    c.setFont("Helvetica-Bold", 12)
    c.drawString(2 * cm, y, "Recommended Allocation:")
    y -= 0.7 * cm
    c.setFont("Helvetica", 10)
    for name, w in zip(asset_names, weights):
        c.drawString(2.5 * cm, y, f"- {name}: {w*100:.1f}%  (Rs. {amount*w:,.0f})")
        y -= 0.6 * cm

    y -= 0.5 * cm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2 * cm, y, "Portfolio Stats:")
    y -= 0.7 * cm
    c.setFont("Helvetica", 10)
    c.drawString(2.5 * cm, y, f"Expected Annual Return: {port_return*100:.1f}%")
    y -= 0.6 * cm
    c.drawString(2.5 * cm, y, f"Expected Volatility: {port_vol*100:.1f}%")
    y -= 0.6 * cm
    c.drawString(2.5 * cm, y, f"Sharpe Ratio: {sharpe:.2f}")
    y -= 1 * cm

    c.setFont("Helvetica-Bold", 12)
    c.drawString(2 * cm, y, "Why this portfolio:")
    y -= 0.7 * cm
    c.setFont("Helvetica", 9)
    plain_text = explanation.replace("**", "")
    for paragraph in plain_text.split("\n\n"):
        words = paragraph.split(" ")
        line = ""
        for word in words:
            if len(line + word) > 95:
                c.drawString(2.5 * cm, y, line)
                y -= 0.5 * cm
                line = ""
            line += word + " "
        if line:
            c.drawString(2.5 * cm, y, line)
            y -= 0.5 * cm
        y -= 0.3 * cm

    c.setFont("Helvetica-Oblique", 8)
    c.drawString(2 * cm, 1.5 * cm, "This is an educational tool, not financial advice.")

    c.save()
    buffer.seek(0)
    return buffer


def generate_rebalancing_advice(target_weights, asset_names, drift_pct=5):
    np.random.seed(42)
    drift = np.random.uniform(-drift_pct / 100, drift_pct / 100, len(target_weights))
    current_weights = np.array(target_weights) + drift
    current_weights = np.clip(current_weights, 0, 1)
    current_weights = current_weights / current_weights.sum()

    advice = []
    for name, target, current in zip(asset_names, target_weights, current_weights):
        diff = current - target
        if diff > 0.02:
            advice.append(f"🔴 **Sell {diff*100:.1f}%** of {name} (currently {current*100:.1f}%, target {target*100:.1f}%)")
        elif diff < -0.02:
            advice.append(f"🟢 **Buy {abs(diff)*100:.1f}%** more of {name} (currently {current*100:.1f}%, target {target*100:.1f}%)")
        else:
            advice.append(f"⚪ {name} is close to target ({current*100:.1f}% vs {target*100:.1f}%) — no action needed")

    return current_weights, advice


# ---------------- GOAL CALCULATOR FUNCTIONS ----------------

def sip_future_value(monthly_amount, annual_rate_pct, years):
    r = annual_rate_pct / 100 / 12
    n = years * 12
    if r == 0:
        return monthly_amount * n
    fv = monthly_amount * (((1 + r) ** n - 1) / r) * (1 + r)
    return fv


def required_sip_for_goal(target_amount, annual_rate_pct, years):
    r = annual_rate_pct / 100 / 12
    n = years * 12
    if r == 0:
        return target_amount / n
    monthly = target_amount / ((((1 + r) ** n - 1) / r) * (1 + r))
    return monthly


def calculate_emi(principal, annual_rate_pct, tenure_months):
    r = annual_rate_pct / 100 / 12
    if r == 0:
        return principal / tenure_months
    emi = principal * r * (1 + r) ** tenure_months / (((1 + r) ** tenure_months) - 1)
    return emi


# ---------------- RETIREMENT PLANNER FUNCTIONS ----------------

def retirement_corpus_required(current_monthly_expense, current_age, retirement_age, life_expectancy,
                                inflation_pct, post_retirement_return_pct):
    years_to_retirement = max(0, retirement_age - current_age)
    years_in_retirement = max(1, life_expectancy - retirement_age)

    expense_at_retirement = current_monthly_expense * (1 + inflation_pct / 100) ** years_to_retirement
    annual_expense_at_retirement = expense_at_retirement * 12

    real_rate = (1 + post_retirement_return_pct / 100) / (1 + inflation_pct / 100) - 1
    n = years_in_retirement

    if abs(real_rate) < 1e-6:
        corpus = annual_expense_at_retirement * n
    else:
        corpus = annual_expense_at_retirement * (1 - (1 + real_rate) ** (-n)) / real_rate

    return corpus, expense_at_retirement, years_to_retirement, years_in_retirement


def required_monthly_investment_for_retirement(corpus_needed, current_savings, years_to_retirement, expected_return_pct):
    future_value_of_current_savings = current_savings * (1 + expected_return_pct / 100) ** years_to_retirement
    remaining_needed = max(0, corpus_needed - future_value_of_current_savings)
    if years_to_retirement > 0:
        monthly_sip = required_sip_for_goal(remaining_needed, expected_return_pct, years_to_retirement)
    else:
        monthly_sip = 0
    return monthly_sip, future_value_of_current_savings, remaining_needed


def simulate_retirement_drawdown(corpus_needed, years_in_retirement, expense_at_retirement, inflation_pct, post_retirement_return_pct):
    balance = corpus_needed
    monthly_expense = expense_at_retirement
    yearly_balances = [balance]
    for _ in range(int(years_in_retirement)):
        annual_expense = monthly_expense * 12
        balance = balance * (1 + post_retirement_return_pct / 100) - annual_expense
        monthly_expense *= (1 + inflation_pct / 100)
        yearly_balances.append(max(0, balance))
    return yearly_balances


# ---------------- MONTE CARLO + CORRELATION ----------------

def run_monte_carlo(mean_returns, cov_matrix, weights, amount, years=10, simulations=500):
    port_return = np.dot(weights, mean_returns)
    port_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))

    np.random.seed(1)
    days = years * 252
    daily_return = port_return / 252
    daily_vol = port_vol / np.sqrt(252)

    all_paths = np.zeros((simulations, years + 1))
    all_paths[:, 0] = amount

    for sim in range(simulations):
        value = amount
        yearly_checkpoints = np.linspace(0, days, years + 1).astype(int)
        daily_returns_sim = np.random.normal(daily_return, daily_vol, days)
        cumulative = np.cumprod(1 + daily_returns_sim)
        path_values = amount * cumulative
        for yi, day_idx in enumerate(yearly_checkpoints):
            if day_idx == 0:
                all_paths[sim, yi] = amount
            else:
                all_paths[sim, yi] = path_values[day_idx - 1]

    return all_paths


def generate_random_portfolios(mean_returns, cov_matrix, n_portfolios=2000):
    n_assets = len(mean_returns)
    results = np.zeros((3, n_portfolios))
    for i in range(n_portfolios):
        w = np.random.random(n_assets)
        w /= np.sum(w)
        p_return = np.dot(w, mean_returns)
        p_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        results[0, i] = p_vol
        results[1, i] = p_return
        results[2, i] = (p_return - 0.06) / p_vol
    return results


# ---------------- LOGIN GATE ----------------

if "username" not in st.session_state:
    st.session_state.username = None

if st.session_state.username is None:
    st.markdown("## 💰 AI Portfolio Optimizer — Login")
    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:
        login_user = st.text_input("Username", key="login_user")
        login_pass = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            if check_login(login_user, login_pass):
                st.session_state.username = login_user
                st.rerun()
            else:
                st.error("Invalid username or password.")

    with tab2:
        new_user = st.text_input("Choose a username", key="new_user")
        new_pass = st.text_input("Choose a password", type="password", key="new_pass")
        if st.button("Sign Up"):
            if new_user.strip() == "" or new_pass.strip() == "":
                st.error("Username and password can't be empty.")
            elif create_user(new_user, new_pass):
                st.success("Account created! Please log in using the Login tab.")
            else:
                st.error("That username is already taken.")

    st.stop()

else:
    st.sidebar.write(f"👤 Logged in as **{st.session_state.username}**")
    if st.sidebar.button("Log out"):
        st.session_state.username = None
        st.rerun()


# ---------------- MAIN APP HEADER ----------------

st.markdown(
    """
    <div style='text-align: center; padding: 10px 0 20px 0;'>
        <h1 style='margin-bottom: 0;'>💰 AI Portfolio Optimizer</h1>
        <p style='color: gray; font-size: 16px;'>Data-driven portfolio allocation based on your risk profile</p>
    </div>
    """,
    unsafe_allow_html=True
)

main_tabs = st.tabs([
    "🧮 Portfolio Optimizer",
    "🌍 Global Markets",
    "🎯 Goal Calculators",
    "🧓 Retirement Planner",
    "💵 Budget & Expenses",
    "📈 Analytics",
    "🏠 Dashboard",
    "⚡ Scenario Analysis",
])

# =====================================================================
# TAB 1: PORTFOLIO OPTIMIZER (original flow)
# =====================================================================
with main_tabs[0]:
    st.subheader("📋 Step 1: Tell us about yourself")

    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Your age", 18, 75, 30)
        experience = st.selectbox("Investment experience", ["None", "Beginner", "Intermediate", "Expert"])
        horizon = st.selectbox("Investment horizon", ["1 year", "5 years", "10+ years"])
    with col2:
        reaction = st.selectbox(
            "If your investment fell 20% in a month, you would:",
            ["Sell everything immediately", "Sell some", "Do nothing", "Buy more"]
        )
        amount = st.number_input("Investment amount (₹)", min_value=1000, value=100000, step=1000)

    st.divider()
    st.subheader("🚀 Step 2: Get your recommendation")

    if st.button("Calculate my risk profile & portfolio", type="primary", use_container_width=True):
        score = calculate_risk_score(age, experience, horizon, reaction)

        if score <= 30:
            profile, equity_cap = "Conservative", 0.3
        elif score <= 70:
            profile, equity_cap = "Moderate", 0.6
        else:
            profile, equity_cap = "Aggressive", 0.9

        with st.spinner("Downloading market data and optimizing your portfolio..."):
            try:
                tickers = list(ASSETS.values())
                prices = get_price_data(tickers)
                returns = prices.pct_change().dropna()
                weights, mean_returns, cov_matrix = optimize_portfolio(returns, equity_cap=equity_cap)
            except Exception as e:
                st.error(f"Couldn't fetch market data right now: {e}")
                st.stop()

        port_return = np.dot(weights, mean_returns)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (port_return - 0.06) / port_vol

        st.session_state.result = {
            "score": score,
            "profile": profile,
            "equity_cap": equity_cap,
            "weights": weights,
            "mean_returns": mean_returns,
            "cov_matrix": cov_matrix,
            "port_return": port_return,
            "port_vol": port_vol,
            "sharpe": sharpe,
            "amount": amount,
            "returns_df": returns,
        }

    if "result" in st.session_state:
        r = st.session_state.result
        score = r["score"]
        profile = r["profile"]
        equity_cap = r["equity_cap"]
        weights = r["weights"]
        mean_returns = r["mean_returns"]
        cov_matrix = r["cov_matrix"]
        port_return = r["port_return"]
        port_vol = r["port_vol"]
        sharpe = r["sharpe"]
        result_amount = r["amount"]

        st.divider()
        m1, m2 = st.columns(2)
        m1.metric("Risk Score", f"{score}/100")
        m2.metric("Risk Profile", profile)

        st.subheader("Recommended Allocation")
        for name, w in zip(ASSETS.keys(), weights):
            st.write(f"**{name}**: {w*100:.1f}%  →  ₹{result_amount*w:,.0f}")

        pie_fig = go.Figure(data=[go.Pie(labels=list(ASSETS.keys()), values=weights, hole=0.4)])
        pie_fig.update_layout(title="Asset Allocation")
        st.plotly_chart(pie_fig, use_container_width=True)

        st.subheader("📊 Portfolio Stats")
        s1, s2, s3 = st.columns(3)
        s1.metric("Expected Annual Return", f"{port_return*100:.1f}%")
        s2.metric("Expected Volatility", f"{port_vol*100:.1f}%")
        s3.metric("Sharpe Ratio", f"{sharpe:.2f}")

        years_arr = np.arange(0, 11)
        projected_values = result_amount * (1 + port_return) ** years_arr
        growth_fig = go.Figure()
        growth_fig.add_trace(go.Scatter(x=years_arr, y=projected_values, mode="lines+markers", name="Projected Value"))
        growth_fig.update_layout(title="Projected Portfolio Growth (10 Years)", xaxis_title="Years", yaxis_title="Value (₹)")
        st.plotly_chart(growth_fig, use_container_width=True)

        asset_returns = mean_returns.values
        asset_vols = np.sqrt(np.diag(cov_matrix.values))

        scatter_fig = go.Figure()
        scatter_fig.add_trace(go.Scatter(
            x=asset_vols * 100, y=asset_returns * 100,
            mode="markers+text", text=list(ASSETS.keys()), textposition="top center",
            marker=dict(size=10, color="gray"), name="Individual Assets"
        ))
        scatter_fig.add_trace(go.Scatter(
            x=[port_vol * 100], y=[port_return * 100],
            mode="markers+text", text=["Your Portfolio"], textposition="top center",
            marker=dict(size=16, color="red", symbol="star"), name="Your Portfolio"
        ))
        scatter_fig.update_layout(title="Risk vs Return", xaxis_title="Volatility / Risk (%)", yaxis_title="Expected Annual Return (%)")
        st.plotly_chart(scatter_fig, use_container_width=True)

        st.subheader("🤖 Why this portfolio?")
        explanation = generate_explanation(profile, score, weights, list(ASSETS.keys()), port_return, port_vol, sharpe, equity_cap)
        st.markdown(explanation)

        st.subheader("📑 Export")
        df_export = pd.DataFrame({
            "Asset": list(ASSETS.keys()),
            "Allocation %": [w * 100 for w in weights],
            "Amount (₹)": [result_amount * w for w in weights]
        })
        csv_data = df_export.to_csv(index=False).encode("utf-8")
        st.download_button(label="⬇️ Download Allocation (CSV)", data=csv_data, file_name="portfolio_allocation.csv", mime="text/csv")

        pdf_buffer = generate_pdf_report(profile, score, weights, list(ASSETS.keys()), result_amount, port_return, port_vol, sharpe, explanation)
        st.download_button(label="⬇️ Download Full Report (PDF)", data=pdf_buffer, file_name="portfolio_report.pdf", mime="application/pdf")

        st.subheader("💾 Save")
        if st.button("Save this portfolio to my account"):
            save_portfolio(st.session_state.username, profile, score, result_amount, weights, list(ASSETS.keys()), port_return, port_vol, sharpe)
            st.success("Portfolio saved! Check the Dashboard tab to see it.")

        st.subheader("🔄 Rebalancing Check")
        st.write("Simulating how your portfolio might drift after some time, and what to do about it:")

        current_weights, rebalance_advice = generate_rebalancing_advice(weights, list(ASSETS.keys()))
        for line in rebalance_advice:
            st.write(line)

        compare_fig = go.Figure()
        compare_fig.add_trace(go.Bar(name="Target", x=list(ASSETS.keys()), y=[w * 100 for w in weights]))
        compare_fig.add_trace(go.Bar(name="Current (drifted)", x=list(ASSETS.keys()), y=[w * 100 for w in current_weights]))
        compare_fig.update_layout(title="Target vs Current Allocation", yaxis_title="Allocation (%)", barmode="group")
        st.plotly_chart(compare_fig, use_container_width=True)

        st.caption("⚠️ Educational tool only, not financial advice. Based on 5 years of historical price data.")

    st.divider()
    st.subheader("📂 My Saved Portfolios")

    saved = get_saved_portfolios(st.session_state.username)
    if not saved:
        st.write("No saved portfolios yet — calculate one above and click 'Save'.")
    else:
        for row in saved:
            created_at, profile_s, score_s, amount_s, weights_json, p_return, p_vol, p_sharpe = row
            weights_dict = json.loads(weights_json)
            with st.expander(f"{created_at} — {profile_s} — ₹{amount_s:,.0f}"):
                st.write(f"Risk Score: {score_s}/100")
                st.write(f"Expected Return: {p_return*100:.1f}% | Volatility: {p_vol*100:.1f}% | Sharpe: {p_sharpe:.2f}")
                for name, w in weights_dict.items():
                    st.write(f"- {name}: {w*100:.1f}%")


# =====================================================================
# TAB 2: GLOBAL MARKETS
# =====================================================================
with main_tabs[1]:
    st.subheader("🌍 Add Global Market Exposure")
    st.write(
        "These are India-listed ETFs/FoFs that give exposure to international markets "
        "(US tech, US large-cap, and Hong Kong/China), so you can diversify beyond domestic equities "
        "without needing a separate overseas brokerage account."
    )

    global_choices = st.multiselect(
        "Choose global assets to include alongside your domestic portfolio",
        options=list(GLOBAL_ASSETS.keys()),
        default=list(GLOBAL_ASSETS.keys())[:2],
    )

    g1, g2, g3 = st.columns(3)
    with g1:
        g_amount = st.number_input("Investment amount (₹)", min_value=1000, value=100000, step=1000, key="global_amount")
    with g2:
        g_equity_cap = st.slider("Max growth-asset allocation (%)", 10, 100, 60, key="global_equity_cap") / 100
    with g3:
        st.write("")
        st.write("")
        run_global = st.button("Build Global-Diversified Portfolio", type="primary", use_container_width=True)

    if not global_choices:
        st.info("Pick at least one global asset above to build a combined portfolio.")

    if run_global and global_choices:
        combined_assets = dict(ASSETS)
        for name in global_choices:
            combined_assets[name] = GLOBAL_ASSETS[name]

        with st.spinner("Downloading market data (domestic + global) and optimizing..."):
            try:
                tickers = list(combined_assets.values())
                prices = get_price_data(tickers)
                returns = prices.pct_change().dropna()
                g_weights, g_mean_returns, g_cov_matrix = optimize_portfolio_general(
                    returns, list(combined_assets.keys()), DEFENSIVE_ASSETS, equity_cap=g_equity_cap
                )
            except Exception as e:
                st.error(f"Couldn't fetch market data right now: {e}")
                st.stop()

        g_port_return = np.dot(g_weights, g_mean_returns)
        g_port_vol = np.sqrt(np.dot(g_weights.T, np.dot(g_cov_matrix, g_weights)))
        g_sharpe = (g_port_return - 0.06) / g_port_vol

        st.session_state.global_result = {
            "weights": g_weights,
            "asset_names": list(combined_assets.keys()),
            "mean_returns": g_mean_returns,
            "cov_matrix": g_cov_matrix,
            "port_return": g_port_return,
            "port_vol": g_port_vol,
            "sharpe": g_sharpe,
            "amount": g_amount,
            "equity_cap": g_equity_cap,
        }

    if "global_result" in st.session_state:
        gr = st.session_state.global_result
        g_weights = gr["weights"]
        g_asset_names = gr["asset_names"]
        g_mean_returns = gr["mean_returns"]
        g_cov_matrix = gr["cov_matrix"]
        g_port_return = gr["port_return"]
        g_port_vol = gr["port_vol"]
        g_sharpe = gr["sharpe"]
        g_result_amount = gr["amount"]

        st.divider()
        st.subheader("Recommended Global-Diversified Allocation")
        for name, w in zip(g_asset_names, g_weights):
            tag = " 🌍" if name in GLOBAL_ASSETS else ""
            st.write(f"**{name}**{tag}: {w*100:.1f}%  →  ₹{g_result_amount*w:,.0f}")

        g_pie = go.Figure(data=[go.Pie(labels=g_asset_names, values=g_weights, hole=0.4)])
        g_pie.update_layout(title="Global-Diversified Asset Allocation")
        st.plotly_chart(g_pie, use_container_width=True)

        s1, s2, s3 = st.columns(3)
        s1.metric("Expected Annual Return", f"{g_port_return*100:.1f}%")
        s2.metric("Expected Volatility", f"{g_port_vol*100:.1f}%")
        s3.metric("Sharpe Ratio", f"{g_sharpe:.2f}")

        g_asset_returns = g_mean_returns.values
        g_asset_vols = np.sqrt(np.diag(g_cov_matrix.values))
        g_scatter = go.Figure()
        g_scatter.add_trace(go.Scatter(
            x=g_asset_vols * 100, y=g_asset_returns * 100,
            mode="markers+text", text=g_asset_names, textposition="top center",
            marker=dict(size=10, color="gray"), name="Individual Assets"
        ))
        g_scatter.add_trace(go.Scatter(
            x=[g_port_vol * 100], y=[g_port_return * 100],
            mode="markers+text", text=["Your Global Portfolio"], textposition="top center",
            marker=dict(size=16, color="blue", symbol="star"), name="Your Portfolio"
        ))
        g_scatter.update_layout(title="Risk vs Return (Domestic + Global)", xaxis_title="Volatility / Risk (%)", yaxis_title="Expected Annual Return (%)")
        st.plotly_chart(g_scatter, use_container_width=True)

        st.subheader("🤖 Why this mix?")
        g_explanation = generate_generic_explanation(g_weights, g_asset_names, DEFENSIVE_ASSETS, g_port_return, g_port_vol, g_sharpe, gr["equity_cap"])
        st.markdown(g_explanation)

        if st.button("Save this global portfolio to my account"):
            save_portfolio(st.session_state.username, "Global-Diversified", 0, g_result_amount, g_weights, g_asset_names, g_port_return, g_port_vol, g_sharpe)
            st.success("Global portfolio saved! Check the Dashboard tab to see it.")

        st.caption("⚠️ Educational tool only, not financial advice. Global exposure here is via India-listed ETFs/FoFs tracking overseas indices, so currency and tracking effects apply.")


# =====================================================================
# TAB 3: GOAL CALCULATORS
# =====================================================================
with main_tabs[2]:
    st.subheader("🎯 Goal-Based Calculators")

    calc_choice = st.radio(
        "Choose a calculator",
        ["SIP Future Value", "Goal-Based SIP (how much to invest monthly)", "EMI Calculator"],
        horizontal=True
    )

    st.divider()

    if calc_choice == "SIP Future Value":
        st.write("Estimate what a monthly investment (SIP) could grow to.")
        c1, c2, c3 = st.columns(3)
        with c1:
            monthly_amt = st.number_input("Monthly investment (₹)", min_value=500, value=5000, step=500)
        with c2:
            sip_rate = st.number_input("Expected annual return (%)", min_value=1.0, max_value=30.0, value=12.0, step=0.5)
        with c3:
            sip_years = st.number_input("Investment duration (years)", min_value=1, max_value=40, value=10, step=1)

        if st.button("Calculate SIP Future Value", type="primary"):
            fv = sip_future_value(monthly_amt, sip_rate, sip_years)
            invested = monthly_amt * sip_years * 12
            gains = fv - invested

            m1, m2, m3 = st.columns(3)
            m1.metric("Total Invested", f"₹{invested:,.0f}")
            m2.metric("Estimated Gains", f"₹{gains:,.0f}")
            m3.metric("Future Value", f"₹{fv:,.0f}")

            years_range = np.arange(0, sip_years + 1)
            values = [sip_future_value(monthly_amt, sip_rate, y) if y > 0 else 0 for y in years_range]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=years_range, y=values, mode="lines+markers", fill="tozeroy"))
            fig.update_layout(title="SIP Growth Over Time", xaxis_title="Years", yaxis_title="Value (₹)")
            st.plotly_chart(fig, use_container_width=True)

    elif calc_choice == "Goal-Based SIP (how much to invest monthly)":
        st.write("Find out how much you need to invest monthly to reach a target amount.")
        c1, c2, c3 = st.columns(3)
        with c1:
            goal_amt = st.number_input("Target amount (₹)", min_value=10000, value=2000000, step=10000)
        with c2:
            goal_rate = st.number_input("Expected annual return (%)", min_value=1.0, max_value=30.0, value=12.0, step=0.5, key="goal_rate")
        with c3:
            goal_years = st.number_input("Time horizon (years)", min_value=1, max_value=40, value=10, step=1, key="goal_years")

        if st.button("Calculate Required Monthly SIP", type="primary"):
            required = required_sip_for_goal(goal_amt, goal_rate, goal_years)
            total_invested = required * goal_years * 12

            m1, m2 = st.columns(2)
            m1.metric("Required Monthly SIP", f"₹{required:,.0f}")
            m2.metric("Total You'll Invest", f"₹{total_invested:,.0f}")
            st.info(f"Investing ₹{required:,.0f}/month for {goal_years} years at {goal_rate}% expected annual return should get you to your ₹{goal_amt:,.0f} goal.")

    else:  # EMI Calculator
        st.write("Calculate your monthly loan repayment (EMI).")
        c1, c2, c3 = st.columns(3)
        with c1:
            principal = st.number_input("Loan amount (₹)", min_value=10000, value=500000, step=10000)
        with c2:
            emi_rate = st.number_input("Annual interest rate (%)", min_value=1.0, max_value=30.0, value=9.0, step=0.1)
        with c3:
            tenure_years = st.number_input("Loan tenure (years)", min_value=1, max_value=30, value=5, step=1)

        if st.button("Calculate EMI", type="primary"):
            tenure_months = tenure_years * 12
            emi = calculate_emi(principal, emi_rate, tenure_months)
            total_payment = emi * tenure_months
            total_interest = total_payment - principal

            m1, m2, m3 = st.columns(3)
            m1.metric("Monthly EMI", f"₹{emi:,.0f}")
            m2.metric("Total Interest", f"₹{total_interest:,.0f}")
            m3.metric("Total Payment", f"₹{total_payment:,.0f}")

            pie = go.Figure(data=[go.Pie(labels=["Principal", "Interest"], values=[principal, total_interest], hole=0.4)])
            pie.update_layout(title="Principal vs Interest")
            st.plotly_chart(pie, use_container_width=True)


# =====================================================================
# TAB 4: RETIREMENT PLANNER
# =====================================================================
with main_tabs[3]:
    st.subheader("🧓 Retirement Planner")
    st.write("Estimate the retirement corpus you'll need, and how much to invest monthly to get there.")

    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        ret_current_age = st.number_input("Current age", min_value=18, max_value=70, value=30, key="ret_age")
        ret_retirement_age = st.number_input("Retirement age", min_value=ret_current_age + 1, max_value=80, value=60, key="ret_retire_age")
    with rc2:
        ret_life_expectancy = st.number_input("Life expectancy", min_value=ret_retirement_age + 1, max_value=100, value=85, key="ret_life_exp")
        ret_monthly_expense = st.number_input("Current monthly expenses (₹)", min_value=1000, value=50000, step=1000, key="ret_expense")
    with rc3:
        ret_current_savings = st.number_input("Current retirement savings (₹)", min_value=0, value=500000, step=10000, key="ret_savings")

    rc4, rc5, rc6 = st.columns(3)
    with rc4:
        ret_inflation = st.number_input("Expected inflation (%)", min_value=0.0, max_value=15.0, value=6.0, step=0.5, key="ret_inflation")
    with rc5:
        ret_pre_return = st.number_input("Expected return before retirement (%)", min_value=1.0, max_value=25.0, value=11.0, step=0.5, key="ret_pre_return")
    with rc6:
        ret_post_return = st.number_input("Expected return during retirement (%)", min_value=1.0, max_value=20.0, value=7.0, step=0.5, key="ret_post_return")

    if st.button("Calculate Retirement Plan", type="primary", use_container_width=True):
        corpus, expense_at_retirement, years_to_retirement, years_in_retirement = retirement_corpus_required(
            ret_monthly_expense, ret_current_age, ret_retirement_age, ret_life_expectancy,
            ret_inflation, ret_post_return
        )
        monthly_sip, fv_current_savings, remaining_needed = required_monthly_investment_for_retirement(
            corpus, ret_current_savings, years_to_retirement, ret_pre_return
        )

        st.session_state.retirement_result = {
            "corpus": corpus,
            "expense_at_retirement": expense_at_retirement,
            "years_to_retirement": years_to_retirement,
            "years_in_retirement": years_in_retirement,
            "monthly_sip": monthly_sip,
            "fv_current_savings": fv_current_savings,
        }

    if "retirement_result" in st.session_state:
        rr = st.session_state.retirement_result

        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("Retirement Corpus Needed", f"₹{rr['corpus']:,.0f}")
        m2.metric("Monthly Expense at Retirement", f"₹{rr['expense_at_retirement']:,.0f}")
        m3.metric("Required Monthly SIP", f"₹{rr['monthly_sip']:,.0f}")

        st.info(
            f"With {rr['years_to_retirement']} years to retirement and {rr['years_in_retirement']} years in retirement, "
            f"your current savings are projected to grow to ₹{rr['fv_current_savings']:,.0f}. "
            f"Investing ₹{rr['monthly_sip']:,.0f}/month should cover the remaining gap."
        )

        years_axis = np.arange(0, rr["years_to_retirement"] + 1)
        accumulation = [
            rr["fv_current_savings"] * (y / rr["years_to_retirement"]) ** 1 if rr["years_to_retirement"] > 0 else rr["fv_current_savings"]
            for y in years_axis
        ]
        # simple SIP + savings growth curve for illustration
        proj = []
        for y in years_axis:
            fv_savings_y = ret_current_savings * (1 + ret_pre_return / 100) ** y
            fv_sip_y = sip_future_value(rr["monthly_sip"], ret_pre_return, y) if y > 0 else 0
            proj.append(fv_savings_y + fv_sip_y)

        accum_fig = go.Figure()
        accum_fig.add_trace(go.Scatter(x=years_axis, y=proj, mode="lines+markers", name="Projected Corpus"))
        accum_fig.add_trace(go.Scatter(x=years_axis, y=[rr["corpus"]] * len(years_axis), mode="lines", line=dict(dash="dash", color="red"), name="Target Corpus"))
        accum_fig.update_layout(title="Corpus Accumulation to Retirement", xaxis_title="Years from now", yaxis_title="Value (₹)")
        st.plotly_chart(accum_fig, use_container_width=True)

        drawdown = simulate_retirement_drawdown(rr["corpus"], rr["years_in_retirement"], rr["expense_at_retirement"], ret_inflation, ret_post_return)
        draw_years = np.arange(0, len(drawdown))
        draw_fig = go.Figure()
        draw_fig.add_trace(go.Scatter(x=draw_years, y=drawdown, mode="lines+markers", fill="tozeroy", name="Remaining Corpus"))
        draw_fig.update_layout(title="Corpus Drawdown During Retirement", xaxis_title="Years into retirement", yaxis_title="Remaining Value (₹)")
        st.plotly_chart(draw_fig, use_container_width=True)

        st.caption("⚠️ Educational estimate only. Actual returns, inflation, and expenses will vary — revisit this plan periodically.")


# =====================================================================
# TAB 5: BUDGET & EXPENSE TRACKING
# =====================================================================
with main_tabs[4]:
    st.subheader("💵 Budget & Expense Tracking")

    income_col1, income_col2 = st.columns([2, 1])
    with income_col1:
        current_income = get_monthly_income(st.session_state.username)
        new_income = st.number_input("Monthly income (₹)", min_value=0.0, value=float(current_income), step=1000.0, key="income_input")
    with income_col2:
        st.write("")
        st.write("")
        if st.button("Save Income"):
            set_monthly_income(st.session_state.username, new_income)
            st.success("Monthly income saved.")

    st.divider()
    st.markdown("#### ➕ Add an Expense")

    EXPENSE_CATEGORIES = ["Housing", "Food & Groceries", "Transport", "Utilities", "Entertainment",
                           "Healthcare", "Investments/Savings", "Shopping", "Education", "Other"]

    e1, e2, e3, e4 = st.columns([1, 1, 1, 2])
    with e1:
        exp_date = st.date_input("Date", value=datetime.now())
    with e2:
        exp_category = st.selectbox("Category", EXPENSE_CATEGORIES)
    with e3:
        exp_amount = st.number_input("Amount (₹)", min_value=0.0, value=0.0, step=100.0, key="exp_amount")
    with e4:
        exp_note = st.text_input("Note (optional)", key="exp_note")

    if st.button("Add Expense", type="primary"):
        if exp_amount <= 0:
            st.error("Enter an amount greater than zero.")
        else:
            add_expense(st.session_state.username, exp_date.strftime("%Y-%m-%d"), exp_category, exp_amount, exp_note)
            st.success("Expense added.")
            st.rerun()

    st.divider()
    st.markdown("#### 📋 Your Expenses")

    expenses = get_expenses(st.session_state.username)
    if not expenses:
        st.info("No expenses logged yet. Add one above to get started.")
    else:
        exp_df = pd.DataFrame(expenses, columns=["id", "Date", "Category", "Amount", "Note"])
        exp_df["Date"] = pd.to_datetime(exp_df["Date"])
        exp_df["Month"] = exp_df["Date"].dt.strftime("%Y-%m")

        current_month = datetime.now().strftime("%Y-%m")
        month_options = sorted(exp_df["Month"].unique(), reverse=True)
        selected_month = st.selectbox("View month", month_options, index=month_options.index(current_month) if current_month in month_options else 0)

        month_df = exp_df[exp_df["Month"] == selected_month]
        total_expenses = month_df["Amount"].sum()
        savings = new_income - total_expenses
        savings_rate = (savings / new_income * 100) if new_income > 0 else 0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Monthly Income", f"₹{new_income:,.0f}")
        m2.metric("Total Expenses", f"₹{total_expenses:,.0f}")
        m3.metric("Savings", f"₹{savings:,.0f}")
        m4.metric("Savings Rate", f"{savings_rate:.1f}%")

        cat_summary = month_df.groupby("Category")["Amount"].sum().reset_index()
        if not cat_summary.empty:
            exp_pie = go.Figure(data=[go.Pie(labels=cat_summary["Category"], values=cat_summary["Amount"], hole=0.4)])
            exp_pie.update_layout(title=f"Spending by Category — {selected_month}")
            st.plotly_chart(exp_pie, use_container_width=True)

        monthly_trend = exp_df.groupby("Month")["Amount"].sum().reset_index().sort_values("Month")
        trend_fig = go.Figure()
        trend_fig.add_trace(go.Bar(x=monthly_trend["Month"], y=monthly_trend["Amount"], name="Total Expenses"))
        trend_fig.update_layout(title="Monthly Spending Trend", xaxis_title="Month", yaxis_title="Amount (₹)")
        st.plotly_chart(trend_fig, use_container_width=True)

        st.markdown(f"#### Transactions — {selected_month}")
        display_df = month_df[["Date", "Category", "Amount", "Note"]].sort_values("Date", ascending=False)
        st.dataframe(display_df, use_container_width=True)

        with st.expander("Delete a transaction"):
            del_options = {f"{row['Date'].strftime('%Y-%m-%d')} | {row['Category']} | ₹{row['Amount']:,.0f}": row_id
                           for row_id, row in zip(month_df["id"], month_df.to_dict("records"))}
            if del_options:
                to_delete_label = st.selectbox("Select transaction to delete", list(del_options.keys()))
                if st.button("Delete Selected Transaction"):
                    delete_expense(del_options[to_delete_label])
                    st.success("Transaction deleted.")
                    st.rerun()

    st.caption("⚠️ Educational tool only. Track consistently for the most useful savings-rate picture.")


# =====================================================================
# TAB 6: ANALYTICS (Correlation + Monte Carlo)
# =====================================================================
with main_tabs[5]:
    st.subheader("📈 Deeper Analytics")
    st.write("Uses the same 5-year historical data as the optimizer.")

    if st.button("Load / Refresh Analytics", type="primary"):
        with st.spinner("Fetching data and running analysis..."):
            tickers = list(ASSETS.values())
            prices = get_price_data(tickers)
            returns = prices.pct_change().dropna()
            mean_returns = returns.mean() * 252
            cov_matrix = returns.cov() * 252
            corr_matrix = returns.corr()

            st.session_state.analytics = {
                "returns": returns,
                "mean_returns": mean_returns,
                "cov_matrix": cov_matrix,
                "corr_matrix": corr_matrix,
            }

    if "analytics" in st.session_state:
        a = st.session_state.analytics
        asset_names = list(ASSETS.keys())

        st.markdown("#### Correlation Heatmap")
        st.write("Shows how closely each asset's returns move together. Values near 0 or negative are good for diversification.")
        corr_vals = a["corr_matrix"].values
        heatmap_fig = go.Figure(data=go.Heatmap(
            z=corr_vals,
            x=asset_names,
            y=asset_names,
            colorscale="RdBu",
            zmid=0,
            text=np.round(corr_vals, 2),
            texttemplate="%{text}"
        ))
        heatmap_fig.update_layout(title="Asset Correlation Matrix")
        st.plotly_chart(heatmap_fig, use_container_width=True)

        st.divider()
        st.markdown("#### Efficient Frontier (Random Portfolios)")
        st.write("Each dot is a randomly generated portfolio mix. This shows the range of possible risk/return combinations across your 4 assets.")

        sims = generate_random_portfolios(a["mean_returns"].values, a["cov_matrix"].values, n_portfolios=1500)
        frontier_fig = go.Figure()
        frontier_fig.add_trace(go.Scatter(
            x=sims[0] * 100, y=sims[1] * 100,
            mode="markers",
            marker=dict(size=5, color=sims[2], colorscale="Viridis", showscale=True, colorbar=dict(title="Sharpe")),
            name="Random Portfolios"
        ))
        frontier_fig.update_layout(title="Efficient Frontier", xaxis_title="Volatility / Risk (%)", yaxis_title="Expected Annual Return (%)")
        st.plotly_chart(frontier_fig, use_container_width=True)

        st.divider()
        st.markdown("#### Monte Carlo Simulation")
        st.write("Simulates many possible future paths for a portfolio, showing the range of outcomes rather than a single prediction.")

        if "result" in st.session_state:
            mc_weights = st.session_state.result["weights"]
            mc_amount = st.session_state.result["amount"]
            st.caption("Using your last calculated portfolio from the Portfolio Optimizer tab.")
        else:
            mc_weights = np.array([0.25, 0.25, 0.25, 0.25])
            mc_amount = 100000
            st.caption("No portfolio calculated yet — using an equal-weighted example portfolio. Go to the Portfolio Optimizer tab first for a personalized simulation.")

        mc_years = st.slider("Simulation years", 1, 20, 10)
        mc_sims = st.slider("Number of simulated paths", 100, 1000, 300, step=100)

        if st.button("Run Monte Carlo Simulation", type="primary"):
            with st.spinner("Running simulations..."):
                paths = run_monte_carlo(a["mean_returns"].values, a["cov_matrix"].values, mc_weights, mc_amount, years=mc_years, simulations=mc_sims)

            mc_fig = go.Figure()
            years_axis = np.arange(0, mc_years + 1)
            for i in range(min(mc_sims, 150)):  # cap lines drawn for performance
                mc_fig.add_trace(go.Scatter(x=years_axis, y=paths[i], mode="lines", line=dict(width=0.5, color="rgba(100,150,250,0.15)"), showlegend=False))

            median_path = np.median(paths, axis=0)
            p10 = np.percentile(paths, 10, axis=0)
            p90 = np.percentile(paths, 90, axis=0)
            mc_fig.add_trace(go.Scatter(x=years_axis, y=median_path, mode="lines", line=dict(width=3, color="white"), name="Median outcome"))
            mc_fig.add_trace(go.Scatter(x=years_axis, y=p90, mode="lines", line=dict(width=2, color="green", dash="dash"), name="90th percentile"))
            mc_fig.add_trace(go.Scatter(x=years_axis, y=p10, mode="lines", line=dict(width=2, color="red", dash="dash"), name="10th percentile"))
            mc_fig.update_layout(title=f"Monte Carlo: {mc_sims} Simulated Paths Over {mc_years} Years", xaxis_title="Years", yaxis_title="Portfolio Value (₹)")
            st.plotly_chart(mc_fig, use_container_width=True)

            m1, m2, m3 = st.columns(3)
            m1.metric("Median Outcome", f"₹{median_path[-1]:,.0f}")
            m2.metric("10th Percentile (pessimistic)", f"₹{p10[-1]:,.0f}")
            m3.metric("90th Percentile (optimistic)", f"₹{p90[-1]:,.0f}")
    else:
        st.info("Click 'Load / Refresh Analytics' above to begin.")


# =====================================================================
# TAB 7: DASHBOARD
# =====================================================================
with main_tabs[6]:
    st.subheader("🏠 Your Dashboard")

    saved_d = get_saved_portfolios(st.session_state.username)

    if not saved_d:
        st.info("No saved portfolios yet. Calculate and save one in the Portfolio Optimizer tab to populate your dashboard.")
    else:
        total_invested = sum(row[3] for row in saved_d)
        avg_sharpe = np.mean([row[7] for row in saved_d])
        avg_return = np.mean([row[5] for row in saved_d])
        num_portfolios = len(saved_d)

        st.markdown("#### Summary")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Total Saved Portfolios", num_portfolios)
        d2.metric("Total ₹ Across Portfolios", f"₹{total_invested:,.0f}")
        d3.metric("Avg. Expected Return", f"{avg_return*100:.1f}%")
        d4.metric("Avg. Sharpe Ratio", f"{avg_sharpe:.2f}")

        health_score = min(100, max(0, int(avg_sharpe * 40 + 30)))
        st.markdown("#### Portfolio Health Score")
        st.progress(health_score / 100)
        st.write(f"**{health_score}/100** — based on average Sharpe Ratio across your saved portfolios. Higher means better risk-adjusted returns on average.")

        st.divider()
        st.markdown("#### Portfolio History")

        hist_df = pd.DataFrame([
            {
                "Date": row[0],
                "Risk Profile": row[1],
                "Amount (₹)": row[3],
                "Expected Return (%)": round(row[5] * 100, 1),
                "Volatility (%)": round(row[6] * 100, 1),
                "Sharpe": round(row[7], 2),
            }
            for row in saved_d
        ])
        st.dataframe(hist_df, use_container_width=True)

        trend_fig = go.Figure()
        trend_fig.add_trace(go.Scatter(
            x=list(range(len(saved_d))),
            y=[row[7] for row in reversed(saved_d)],
            mode="lines+markers",
            name="Sharpe Ratio over saved portfolios"
        ))
        trend_fig.update_layout(title="Sharpe Ratio Trend (oldest → newest)", xaxis_title="Portfolio #", yaxis_title="Sharpe Ratio")
        st.plotly_chart(trend_fig, use_container_width=True)

    st.divider()
    st.markdown("#### Budget Snapshot")
    dash_income = get_monthly_income(st.session_state.username)
    dash_expenses = get_expenses(st.session_state.username)
    if dash_income or dash_expenses:
        if dash_expenses:
            dash_exp_df = pd.DataFrame(dash_expenses, columns=["id", "Date", "Category", "Amount", "Note"])
            dash_exp_df["Date"] = pd.to_datetime(dash_exp_df["Date"])
            this_month = datetime.now().strftime("%Y-%m")
            month_total = dash_exp_df[dash_exp_df["Date"].dt.strftime("%Y-%m") == this_month]["Amount"].sum()
        else:
            month_total = 0
        b1, b2, b3 = st.columns(3)
        b1.metric("Monthly Income", f"₹{dash_income:,.0f}")
        b2.metric("This Month's Expenses", f"₹{month_total:,.0f}")
        b3.metric("This Month's Savings", f"₹{dash_income - month_total:,.0f}")
    else:
        st.info("Set your income and log expenses in the Budget & Expenses tab to see a snapshot here.")


# =====================================================================
# TAB 8: SCENARIO ANALYSIS
# =====================================================================
with main_tabs[7]:
    st.subheader("⚡ Scenario / Stress Testing")
    st.write("See how your portfolio might react to a sudden market move.")

    if "result" not in st.session_state:
        st.info("Calculate a portfolio in the Portfolio Optimizer tab first, then come back here to stress-test it.")
    else:
        r = st.session_state.result
        weights = r["weights"]
        result_amount = r["amount"]
        asset_names = list(ASSETS.keys())

        st.caption(f"Stress-testing your last calculated {r['profile']} portfolio (₹{result_amount:,.0f}).")

        scenario = st.selectbox(
            "Choose a scenario",
            [
                "Custom market shock",
                "Market falls 15% (equities), Gold +5%",
                "Market crash 30% (equities), Gold +10%",
                "Equities flat, Gold falls 10%",
            ]
        )

        # weights order: Large Cap, Mid Cap, Gold, Liquid Fund
        if scenario == "Custom market shock":
            c1, c2 = st.columns(2)
            with c1:
                equity_shock = st.slider("Equity shock (%)", -50, 50, -15)
            with c2:
                gold_shock = st.slider("Gold shock (%)", -50, 50, 5)
            shocks = np.array([equity_shock, equity_shock, gold_shock, 0]) / 100
        elif scenario == "Market falls 15% (equities), Gold +5%":
            shocks = np.array([-0.15, -0.15, 0.05, 0.0])
        elif scenario == "Market crash 30% (equities), Gold +10%":
            shocks = np.array([-0.30, -0.30, 0.10, 0.0])
        else:
            shocks = np.array([0.0, 0.0, -0.10, 0.0])

        if st.button("Run Scenario", type="primary"):
            asset_values_before = weights * result_amount
            asset_values_after = asset_values_before * (1 + shocks)

            total_before = asset_values_before.sum()
            total_after = asset_values_after.sum()
            pct_change = (total_after - total_before) / total_before * 100

            m1, m2, m3 = st.columns(3)
            m1.metric("Portfolio Value Before", f"₹{total_before:,.0f}")
            m2.metric("Portfolio Value After", f"₹{total_after:,.0f}", delta=f"{pct_change:.1f}%")
            m3.metric("₹ Impact", f"₹{total_after - total_before:,.0f}")

            comp_fig = go.Figure()
            comp_fig.add_trace(go.Bar(name="Before", x=asset_names, y=asset_values_before))
            comp_fig.add_trace(go.Bar(name="After Scenario", x=asset_names, y=asset_values_after))
            comp_fig.update_layout(title="Asset Values: Before vs After Scenario", yaxis_title="Value (₹)", barmode="group")
            st.plotly_chart(comp_fig, use_container_width=True)

            if pct_change < -10:
                st.warning(f"This scenario would reduce your portfolio by {abs(pct_change):.1f}%. Your diversification (Gold/Liquid Fund allocation) is cushioning some of the equity shock.")
            elif pct_change < 0:
                st.info(f"This scenario has a moderate impact — a {abs(pct_change):.1f}% reduction, softened by your non-equity holdings.")
            else:
                st.success(f"This scenario would actually increase your portfolio value by {pct_change:.1f}%.")

    st.caption("⚠️ Educational tool only. Real market shocks affect assets in more complex, correlated ways than this simplified simulation shows.")

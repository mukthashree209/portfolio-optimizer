import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import plotly.graph_objects as go
import io
from datetime import datetime

st.markdown(
    """
    <div style='text-align: center; padding: 10px 0 20px 0;'>
        <h1 style='margin-bottom: 0;'>💰 AI Portfolio Optimizer</h1>
        <p style='color: gray; font-size: 16px;'>Data-driven portfolio allocation based on your risk profile</p>
    </div>
    """,
    unsafe_allow_html=True
)
st.divider()
st.write("Answer a few questions and get a data-driven portfolio allocation.")

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

ASSETS = {
    "Large Cap ETF": "NIFTYBEES.NS",
    "Mid Cap ETF": "JUNIORBEES.NS",
    "Gold ETF": "GOLDBEES.NS",
    "Liquid Fund (Cash-like)": "LIQUIDBEES.NS",
}


@st.cache_data(ttl=3600)
def get_price_data(tickers, period="5y"):
    data = yf.download(tickers, period=period)["Close"]
    return data.dropna()


def calculate_risk_score():
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
        lines.append(
            f"Your total equity exposure is **{equity_weight*100:.1f}%**, which is relatively high. "
            f"This typically means higher potential growth, but also bigger swings in value year to year."
        )
    elif equity_weight > 0.2:
        lines.append(
            f"Your total equity exposure is **{equity_weight*100:.1f}%** — a balanced mix of growth potential and stability."
        )
    else:
        lines.append(
            f"Your total equity exposure is only **{equity_weight*100:.1f}%**, prioritizing capital protection over growth."
        )

    if gold_weight > 0.1:
        lines.append(
            f"**{gold_weight*100:.1f}%** was allocated to Gold — this acts as a hedge, since gold often moves opposite to stocks during market stress."
        )

    if cash_weight > 0.1:
        lines.append(
            f"**{cash_weight*100:.1f}%** is in a Liquid/Cash-like fund, giving you stability and quick access to funds if needed."
        )

    if sharpe > 1:
        lines.append(f"Your portfolio's Sharpe Ratio of **{sharpe:.2f}** is strong — you're getting good return for each unit of risk taken.")
    elif sharpe > 0.5:
        lines.append(f"Your Sharpe Ratio of **{sharpe:.2f}** is decent — a reasonable trade-off between risk and return.")
    else:
        lines.append(f"Your Sharpe Ratio of **{sharpe:.2f}** is on the lower side — this can happen in choppy market periods, but the mix is still the best available given your risk limits.")

    return "\n\n".join(lines)


def generate_pdf_report(profile, score, weights, asset_names, amount, port_return, port_vol, sharpe, explanation):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import cm

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - 2*cm

    c.setFont("Helvetica-Bold", 16)
    c.drawString(2*cm, y, "AI Portfolio Optimizer - Report")
    y -= 1*cm

    c.setFont("Helvetica", 10)
    c.drawString(2*cm, y, f"Generated: {datetime.now().strftime('%d %b %Y, %H:%M')}")
    y -= 1*cm

    c.setFont("Helvetica-Bold", 12)
    c.drawString(2*cm, y, f"Risk Profile: {profile} (Score: {score}/100)")
    y -= 0.8*cm
    c.drawString(2*cm, y, f"Investment Amount: Rs. {amount:,.0f}")
    y -= 1*cm

    c.setFont("Helvetica-Bold", 12)
    c.drawString(2*cm, y, "Recommended Allocation:")
    y -= 0.7*cm
    c.setFont("Helvetica", 10)
    for name, w in zip(asset_names, weights):
        c.drawString(2.5*cm, y, f"- {name}: {w*100:.1f}%  (Rs. {amount*w:,.0f})")
        y -= 0.6*cm

    y -= 0.5*cm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2*cm, y, "Portfolio Stats:")
    y -= 0.7*cm
    c.setFont("Helvetica", 10)
    c.drawString(2.5*cm, y, f"Expected Annual Return: {port_return*100:.1f}%")
    y -= 0.6*cm
    c.drawString(2.5*cm, y, f"Expected Volatility: {port_vol*100:.1f}%")
    y -= 0.6*cm
    c.drawString(2.5*cm, y, f"Sharpe Ratio: {sharpe:.2f}")
    y -= 1*cm

    c.setFont("Helvetica-Bold", 12)
    c.drawString(2*cm, y, "Why this portfolio:")
    y -= 0.7*cm
    c.setFont("Helvetica", 9)
    plain_text = explanation.replace("**", "")
    for paragraph in plain_text.split("\n\n"):
        words = paragraph.split(" ")
        line = ""
        for word in words:
            if len(line + word) > 95:
                c.drawString(2.5*cm, y, line)
                y -= 0.5*cm
                line = ""
            line += word + " "
        if line:
            c.drawString(2.5*cm, y, line)
            y -= 0.5*cm
        y -= 0.3*cm

    c.setFont("Helvetica-Oblique", 8)
    c.drawString(2*cm, 1.5*cm, "This is an educational tool, not financial advice.")

    c.save()
    buffer.seek(0)
    return buffer


def generate_rebalancing_advice(target_weights, asset_names, drift_pct=5):
    np.random.seed(42)
    drift = np.random.uniform(-drift_pct/100, drift_pct/100, len(target_weights))
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


st.subheader("🚀 Step 2: Get your recommendation")

if st.button("Calculate my risk profile & portfolio", type="primary", use_container_width=True):
    score = calculate_risk_score()

    if score <= 30:
        profile, equity_cap = "Conservative", 0.3
    elif score <= 70:
        profile, equity_cap = "Moderate", 0.6
    else:
        profile, equity_cap = "Aggressive", 0.9

    st.divider()
    m1, m2 = st.columns(2)
    m1.metric("Risk Score", f"{score}/100")
    m2.metric("Risk Profile", profile)

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

    st.subheader("Recommended Allocation")
    for name, w in zip(ASSETS.keys(), weights):
        st.write(f"**{name}**: {w*100:.1f}%  →  ₹{amount*w:,.0f}")

    pie_fig = go.Figure(data=[go.Pie(labels=list(ASSETS.keys()), values=weights, hole=0.4)])
    pie_fig.update_layout(title="Asset Allocation")
    st.plotly_chart(pie_fig, use_container_width=True)

    st.subheader("📊 Portfolio Stats")
    s1, s2, s3 = st.columns(3)
    s1.metric("Expected Annual Return", f"{port_return*100:.1f}%")
    s2.metric("Expected Volatility", f"{port_vol*100:.1f}%")
    s3.metric("Sharpe Ratio", f"{sharpe:.2f}")

    years = np.arange(0, 11)
    projected_values = amount * (1 + port_return) ** years
    growth_fig = go.Figure()
    growth_fig.add_trace(go.Scatter(x=years, y=projected_values, mode="lines+markers", name="Projected Value"))
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
        "Allocation %": [w*100 for w in weights],
        "Amount (₹)": [amount*w for w in weights]
    })
    csv_data = df_export.to_csv(index=False).encode("utf-8")
    st.download_button(label="⬇️ Download Allocation (CSV)", data=csv_data, file_name="portfolio_allocation.csv", mime="text/csv")

    pdf_buffer = generate_pdf_report(profile, score, weights, list(ASSETS.keys()), amount, port_return, port_vol, sharpe, explanation)
    st.download_button(label="⬇️ Download Full Report (PDF)", data=pdf_buffer, file_name="portfolio_report.pdf", mime="application/pdf")

    st.subheader("🔄 Rebalancing Check")
    st.write("Simulating how your portfolio might drift after some time, and what to do about it:")

    current_weights, rebalance_advice = generate_rebalancing_advice(weights, list(ASSETS.keys()))
    for line in rebalance_advice:
        st.write(line)

    compare_fig = go.Figure()
    compare_fig.add_trace(go.Bar(name="Target", x=list(ASSETS.keys()), y=[w*100 for w in weights]))
    compare_fig.add_trace(go.Bar(name="Current (drifted)", x=list(ASSETS.keys()), y=[w*100 for w in current_weights]))
    compare_fig.update_layout(title="Target vs Current Allocation", yaxis_title="Allocation (%)", barmode="group")
    st.plotly_chart(compare_fig, use_container_width=True)

    st.caption("⚠️ Educational tool only, not financial advice. Based on 5 years of historical price data.")
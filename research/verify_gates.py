import pandas as pd
df = pd.read_csv('logs/closed_positions.csv', engine='python', on_bad_lines='skip')
for c in ['net_realized_pnl', 'confidence', 'entry_price', 'size_usdc']:
    if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
closed = df[df['status'].astype(str).str.upper() == 'CLOSED'].copy()
closed['win'] = closed['net_realized_pnl'] > 0

gate1 = closed[closed['market_family'].astype(str) != 'btc_other']
gate2 = gate1[gate1['confidence'] >= 0.50]
gate3 = gate2[(gate2['entry_price'] >= 0.15) & (gate2['entry_price'] <= 0.86)]
gate4 = gate3[gate3['size_usdc'] >= 0.15]

print(f'Final Gate: {len(gate4)} trades, {gate4["win"].mean()*100:.1f}% win rate, {gate4["net_realized_pnl"].sum():.2f} PnL')

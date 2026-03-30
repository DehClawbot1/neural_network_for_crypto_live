import logging
from pathlib import Path

import pandas as pd

from shadow_purgatory import ResilientCLOBClient, ShadowPurgatory

logging.basicConfig(level=logging.INFO, format="%(asctime)s [DOA-Resurrector] %(message)s")


class DOAResurrector:
    def __init__(self, log_path="logs/shadow_results.csv", model_bundle_path=None, clob_client=None):
        self.log_path = Path(log_path)
        self.clob = clob_client or ResilientCLOBClient()
        self.purgatory = ShadowPurgatory(
            model_bundle_path=model_bundle_path,
            clob_client=self.clob,
            log_path=self.log_path,
        )

    def run_audit(self, limit=50):
        if not self.log_path.exists():
            logging.error("Shadow results log not found.")
            return pd.DataFrame()

        df = pd.read_csv(self.log_path, engine="python", on_bad_lines="skip")
        if "outcome" not in df.columns:
            logging.error("No outcome column found in results.")
            return pd.DataFrame()

        doa_trades = df[df["outcome"] == "DOA"].tail(limit).copy()
        if doa_trades.empty:
            logging.info("No DOA trades found to resurrect.")
            return pd.DataFrame()

        logging.info("Resurrecting %s DOA trades...", len(doa_trades))

        ghost_outcomes = []
        for _, row in doa_trades.iterrows():
            outcome, ghost_ret, trade_count = self.purgatory._check_path(row)
            ghost_outcomes.append(
                {
                    "market": row.get("market_title"),
                    "prob": row.get("meta_prob"),
                    "expected_slip": row.get("expected_slip_bps"),
                    "actual_slippage": row.get("entry_slippage_bps"),
                    "outcome": outcome,
                    "pnl": ghost_ret,
                    "trade_count": trade_count,
                }
            )

        results_df = pd.DataFrame(ghost_outcomes)
        self._report(results_df)
        return results_df

    def _report(self, results_df):
        total = len(results_df)
        tp_count = int((results_df["outcome"] == "TP").sum()) if total else 0
        sl_count = int((results_df["outcome"] == "SL").sum()) if total else 0
        exp_count = int((results_df["outcome"] == "EXPIRED").sum()) if total else 0
        pending_count = int((results_df["outcome"] == "PENDING").sum()) if total else 0

        ghost_win_rate = (tp_count / total) if total > 0 else 0.0
        avg_ghost_pnl = results_df["pnl"].mean() if total else 0.0

        print("\n" + "=" * 50)
        print("👻 DOA RESURRECTION REPORT: ALPHA LEAK AUDIT")
        print("=" * 50)
        print(f"Total Vetoed Trades Analyzed: {total}")
        print(f"Ghost Win Rate (TP hit): {ghost_win_rate:.2%}")
        print(f"Avg Ghost PnL: {avg_ghost_pnl:+.2%}")
        print("-" * 50)
        print(f"🛡 Bullets Dodged (SL): {sl_count}")
        print(f"💸 Alpha Leaked (TP): {tp_count}")
        print(f"⏳ Neutral/Expired: {exp_count}")
        if pending_count:
            print(f"🕒 Pending/API-limited: {pending_count}")
        print("-" * 50)

        if ghost_win_rate > 0.65:
            print("VERDICT: 🔴 OVER-VETOING. You are killing high-quality alpha.")
            print("Action: Consider relaxing the EV_adj threshold in shadow_purgatory.py.")
        elif ghost_win_rate < 0.40:
            print("VERDICT: 🟢 VETO IS EFFECTIVE. DOA trades are largely toxic.")
        else:
            print("VERDICT: 🟡 NEUTRAL. Veto is catching noise, but not significant alpha.")
        print("=" * 50 + "\n")


def resurrect_doa_trades(log_path="logs/shadow_results.csv", limit=50, model_bundle_path=None, clob_client=None):
    return DOAResurrector(
        log_path=log_path,
        model_bundle_path=model_bundle_path,
        clob_client=clob_client,
    ).run_audit(limit=limit)


if __name__ == "__main__":
    auditor = DOAResurrector()
    auditor.run_audit()

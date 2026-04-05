"""
BTC Sentiment Feature Fetcher

Pulls alpha-generating sentiment signals that technical indicators miss:
  1. Fear & Greed Index (alternative.me) — strong contrarian signal
  2. Google Trends "bitcoin" searches — retail interest proxy
  3. Twitter/X search sentiment — fast crowd mood via public RSS mirrors + VADER NLP
  4. Reddit r/bitcoin + r/cryptocurrency sentiment — crowd mood via VADER NLP

These features measure CROWD PSYCHOLOGY — not price action.
When fear is extreme, smart money buys. When greed peaks, corrections follow.

Architecture matches btc_onchain_features.py:
  - Individual fetch_*() methods return timestamped DataFrames
  - fetch_all_and_merge(candle_df) enriches candles via merge_asof
  - Derived features: z-scores, momentum, contrarian extremes
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)


def _safe_merge_asof(left, right, on, **kwargs):
    """merge_asof that tolerates NaT/null in the left merge key."""
    if left.empty or right.empty or on not in left.columns or on not in right.columns:
        return left
    mask = left[on].notna()
    if not mask.any():
        return left
    valid = left[mask].copy().sort_values(on)
    work_right = right.copy().sort_values(on)
    merged = pd.merge_asof(valid, work_right, on=on, **kwargs)
    if mask.all():
        return merged
    return pd.concat([merged, left[~mask]], ignore_index=True)


# ---------------------------------------------------------------------------
# VADER sentiment (lazy-loaded to avoid import overhead)
# ---------------------------------------------------------------------------
_vader = None


def _get_vader():
    """Lazy-load VADER sentiment analyzer (avoids full nltk import → scipy chain)."""
    global _vader
    if _vader is None:
        try:
            # Direct import avoids nltk.__init__ pulling in scipy via collocations
            import importlib
            vader_mod = importlib.import_module("nltk.sentiment.vader")
            _vader = vader_mod.SentimentIntensityAnalyzer()
        except (ImportError, LookupError):
            try:
                import nltk
                nltk.download("vader_lexicon", quiet=True)
                import importlib
                vader_mod = importlib.import_module("nltk.sentiment.vader")
                _vader = vader_mod.SentimentIntensityAnalyzer()
            except Exception as exc:
                logger.warning("VADER unavailable: %s", exc)
                return None
        except Exception as exc:
            logger.warning("VADER unavailable: %s", exc)
            return None
    return _vader


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class BTCSentimentFeatures:
    """
    Fetch sentiment features for BTC from free public APIs.
    Designed to enrich OHLCV candle DataFrames for ML training.
    """

    FEAR_GREED_URL = "https://api.alternative.me/fng/"
    REDDIT_BASE = "https://www.reddit.com"
    REDDIT_SUBREDDITS = ["bitcoin", "cryptocurrency"]
    TWITTER_SEARCH_TERMS = ["bitcoin", "btc", "#bitcoin", "#btc"]
    NITTER_INSTANCES = [
        "https://nitter.poast.org",
        "https://nitter.net",
        "https://nitter.privacydev.net",
    ]

    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "BTC-Sentiment-Bot/1.0 (research; non-commercial)",
        })

    # ------------------------------------------------------------------
    # 1. Fear & Greed Index
    # ------------------------------------------------------------------

    def fetch_fear_greed_history(self, limit: int = 0) -> pd.DataFrame:
        """
        Fetch historical Fear & Greed Index (daily granularity).

        limit=0 returns ALL history (since Feb 2018).
        Returns DataFrame with columns: [timestamp, fgi_value, fgi_class]
        """
        try:
            resp = self._session.get(
                self.FEAR_GREED_URL,
                params={"limit": limit, "format": "json"},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json().get("data", [])

            if not data:
                logger.warning("Fear & Greed API returned empty data")
                return self._empty_fgi()

            rows = []
            for entry in data:
                rows.append({
                    "timestamp": pd.to_datetime(int(entry["timestamp"]), unit="s", utc=True),
                    "fgi_value": int(entry["value"]),
                    "fgi_class": entry.get("value_classification", ""),
                })

            df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
            logger.info("Fear & Greed: fetched %d daily values (%.0f days back)",
                        len(df), (df["timestamp"].max() - df["timestamp"].min()).days)
            return df

        except Exception as exc:
            logger.warning("Failed to fetch Fear & Greed Index: %s", exc)
            return self._empty_fgi()

    @staticmethod
    def _empty_fgi() -> pd.DataFrame:
        return pd.DataFrame(columns=["timestamp", "fgi_value", "fgi_class"])

    # ------------------------------------------------------------------
    # 2. Google Trends
    # ------------------------------------------------------------------

    def fetch_google_trends(
        self,
        keyword: str = "bitcoin",
        timeframe: str = "today 5-y",
    ) -> pd.DataFrame:
        """
        Fetch Google Trends interest-over-time for a keyword.

        Returns DataFrame with columns: [timestamp, gtrends_bitcoin]
        Values are 0-100 (relative search interest).

        Uses pytrends if available and compatible, otherwise falls back
        to a direct Google Trends embed widget scrape.
        """
        # Try pytrends first
        df = self._fetch_gtrends_pytrends(keyword, timeframe)
        if not df.empty:
            return df

        # Fallback: try direct Google Trends multiline CSV endpoint
        df = self._fetch_gtrends_direct(keyword)
        if not df.empty:
            return df

        logger.warning("Google Trends: all methods failed for '%s'", keyword)
        return self._empty_gtrends()

    def _fetch_gtrends_pytrends(self, keyword: str, timeframe: str) -> pd.DataFrame:
        """Try fetching via pytrends library."""
        try:
            from pytrends.request import TrendReq

            pytrend = TrendReq(hl="en-US", tz=0, retries=2, backoff_factor=1.0)
            pytrend.build_payload(kw_list=[keyword], timeframe=timeframe, geo="")
            df = pytrend.interest_over_time()

            if df.empty:
                return self._empty_gtrends()

            df = df.reset_index()
            df = df.rename(columns={"date": "timestamp", keyword: "gtrends_bitcoin"})
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

            if "isPartial" in df.columns:
                df = df.drop(columns=["isPartial"])

            df = df[["timestamp", "gtrends_bitcoin"]].sort_values("timestamp").reset_index(drop=True)
            logger.info("Google Trends (pytrends): fetched %d data points for '%s'", len(df), keyword)
            return df

        except ImportError:
            logger.debug("pytrends not installed")
            return self._empty_gtrends()
        except Exception as exc:
            logger.debug("pytrends failed: %s", exc)
            return self._empty_gtrends()

    def _fetch_gtrends_direct(self, keyword: str) -> pd.DataFrame:
        """
        Fallback: fetch Google Trends CSV via the public multiline endpoint.
        Returns weekly data for last 12 months.
        """
        try:
            url = "https://trends.google.com/trends/api/widgetdata/multiline/csv"
            # This endpoint is fragile and may break, so we catch all errors
            resp = self._session.get(
                url,
                params={
                    "req": f'{{"time":"today 12-m","resolution":"WEEK","locale":"en-US","comparisonItem":[{{"keyword":"{keyword}","geo":"","time":"today 12-m"}}],"requestOptions":{{"property":"","backend":"IZG","category":0}}}}',
                    "token": "",  # may need valid token
                    "tz": "0",
                },
                timeout=15,
            )
            if resp.status_code != 200:
                return self._empty_gtrends()

            # Parse CSV response (skip first 3 header lines)
            lines = resp.text.strip().split("\n")
            if len(lines) < 4:
                return self._empty_gtrends()

            from io import StringIO
            csv_data = "\n".join(lines[3:])  # skip metadata lines
            df = pd.read_csv(StringIO(csv_data))

            if df.empty or len(df.columns) < 2:
                return self._empty_gtrends()

            df.columns = ["timestamp", "gtrends_bitcoin"]
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df["gtrends_bitcoin"] = pd.to_numeric(df["gtrends_bitcoin"], errors="coerce")
            df = df.dropna().sort_values("timestamp").reset_index(drop=True)

            logger.info("Google Trends (direct): fetched %d data points for '%s'", len(df), keyword)
            return df

        except Exception as exc:
            logger.debug("Google Trends direct fetch failed: %s", exc)
            return self._empty_gtrends()

    @staticmethod
    def _empty_gtrends() -> pd.DataFrame:
        return pd.DataFrame(columns=["timestamp", "gtrends_bitcoin"])

    # ------------------------------------------------------------------
    # 3. Twitter/X Sentiment
    # ------------------------------------------------------------------

    def fetch_twitter_sentiment(
        self,
        search_terms: list[str] | None = None,
        per_term_limit: int = 40,
    ) -> pd.DataFrame:
        """
        Fetch recent Twitter/X posts via public RSS mirrors and compute VADER sentiment.

        Returns daily aggregates with columns:
          [timestamp, twitter_sentiment, twitter_post_count, twitter_sentiment_pos,
           twitter_sentiment_neg, twitter_engagement_proxy]
        """
        if search_terms is None:
            search_terms = self.TWITTER_SEARCH_TERMS

        vader = _get_vader()
        if vader is None:
            logger.warning("VADER not available — skipping Twitter sentiment")
            return self._empty_twitter()

        all_scores = []
        for term in search_terms:
            try:
                posts = self._fetch_nitter_posts(term, limit=per_term_limit)
            except Exception as exc:
                logger.warning("Twitter sentiment fetch failed for '%s': %s", term, exc)
                continue

            for post in posts:
                text = str(post.get("text", "") or "").strip()
                if len(text) < 5:
                    continue

                scores = vader.polarity_scores(text)
                all_scores.append({
                    "timestamp": pd.to_datetime(post.get("timestamp"), utc=True),
                    "compound": scores["compound"],
                    "pos": scores["pos"],
                    "neg": scores["neg"],
                    "engagement_proxy": float(post.get("engagement_proxy", 1.0) or 1.0),
                    "query": term,
                })

        if not all_scores:
            return self._empty_twitter()

        df = pd.DataFrame(all_scores).dropna(subset=["timestamp"]).sort_values("timestamp")
        df["date"] = df["timestamp"].dt.floor("D")
        df["weight"] = np.log1p(df["engagement_proxy"].clip(lower=0)) + 1.0

        daily = df.groupby("date").agg(
            twitter_sentiment=("compound", lambda x: np.average(x, weights=df.loc[x.index, "weight"])),
            twitter_post_count=("compound", "count"),
            twitter_sentiment_pos=("pos", "mean"),
            twitter_sentiment_neg=("neg", "mean"),
            twitter_engagement_proxy=("engagement_proxy", "mean"),
        ).reset_index()

        daily = daily.rename(columns={"date": "timestamp"})
        daily["timestamp"] = pd.to_datetime(daily["timestamp"], utc=True)
        daily = daily.sort_values("timestamp").reset_index(drop=True)
        logger.info(
            "Twitter sentiment: %d posts → %d daily aggregates across %s",
            len(df),
            len(daily),
            search_terms,
        )
        return daily

    def _fetch_nitter_posts(self, query: str, limit: int = 40) -> list[dict]:
        """
        Fetch recent posts for a query from public Nitter RSS mirrors.
        """
        from urllib.parse import quote_plus
        import xml.etree.ElementTree as ET

        encoded_query = quote_plus(query)
        for base_url in self.NITTER_INSTANCES:
            rss_url = f"{base_url}/search/rss?f=tweets&q={encoded_query}"
            try:
                resp = self._session.get(rss_url, timeout=15)
                if resp.status_code != 200 or not resp.text.strip():
                    continue

                root = ET.fromstring(resp.text)
                items = root.findall(".//item")
                posts = []
                for item in items[:limit]:
                    title = (item.findtext("title") or "").strip()
                    description = (item.findtext("description") or "").strip()
                    pub_date = (item.findtext("pubDate") or "").strip()
                    link = (item.findtext("link") or "").strip()
                    guid = (item.findtext("guid") or "").strip()

                    text = description or title
                    if not text:
                        continue

                    timestamp = pd.to_datetime(pub_date, utc=True, errors="coerce")
                    if pd.isna(timestamp):
                        continue

                    engagement_proxy = max(float(len(text.split())), 1.0)
                    if "#" in text:
                        engagement_proxy += 1.0
                    if "http" in text.lower():
                        engagement_proxy += 0.5

                    posts.append({
                        "timestamp": timestamp,
                        "text": text,
                        "link": link,
                        "guid": guid,
                        "engagement_proxy": engagement_proxy,
                    })

                if posts:
                    return posts
            except Exception as exc:
                logger.debug("Nitter RSS fetch failed for '%s' via %s: %s", query, base_url, exc)
                continue

        return []

    @staticmethod
    def _empty_twitter() -> pd.DataFrame:
        return pd.DataFrame(columns=[
            "timestamp",
            "twitter_sentiment",
            "twitter_post_count",
            "twitter_sentiment_pos",
            "twitter_sentiment_neg",
            "twitter_engagement_proxy",
        ])

    # ------------------------------------------------------------------
    # 4. Reddit Sentiment
    # ------------------------------------------------------------------

    def fetch_reddit_sentiment(
        self,
        subreddits: list[str] | None = None,
        post_limit: int = 100,
        sort: str = "hot",
    ) -> pd.DataFrame:
        """
        Fetch recent Reddit posts and compute VADER sentiment scores.

        Uses Reddit's public JSON API (no auth required).
        Returns DataFrame with columns: [timestamp, reddit_sentiment, reddit_post_count,
                                          reddit_sentiment_pos, reddit_sentiment_neg]
        """
        if subreddits is None:
            subreddits = self.REDDIT_SUBREDDITS

        vader = _get_vader()
        if vader is None:
            logger.warning("VADER not available — skipping Reddit sentiment")
            return self._empty_reddit()

        all_scores = []

        for sub in subreddits:
            try:
                posts = self._fetch_reddit_posts(sub, limit=post_limit, sort=sort)
                for post in posts:
                    title = post.get("title", "")
                    selftext = post.get("selftext", "")[:500]  # cap text length
                    text = f"{title}. {selftext}".strip()

                    if not text or len(text) < 5:
                        continue

                    scores = vader.polarity_scores(text)
                    created_utc = post.get("created_utc", 0)

                    all_scores.append({
                        "timestamp": pd.to_datetime(int(created_utc), unit="s", utc=True),
                        "compound": scores["compound"],
                        "pos": scores["pos"],
                        "neg": scores["neg"],
                        "score": post.get("score", 0),
                        "num_comments": post.get("num_comments", 0),
                        "subreddit": sub,
                    })

            except Exception as exc:
                logger.warning("Reddit fetch failed for r/%s: %s", sub, exc)
                continue

        if not all_scores:
            return self._empty_reddit()

        df = pd.DataFrame(all_scores)

        # Aggregate to daily sentiment (weighted by post score/engagement)
        df["weight"] = np.log1p(df["score"].clip(lower=0) + df["num_comments"].clip(lower=0))
        df["weight"] = df["weight"].replace(0, 1.0)

        # Floor to day for aggregation
        df["date"] = df["timestamp"].dt.floor("D")

        daily = df.groupby("date").agg(
            reddit_sentiment=("compound", lambda x: np.average(x, weights=df.loc[x.index, "weight"])),
            reddit_post_count=("compound", "count"),
            reddit_sentiment_pos=("pos", "mean"),
            reddit_sentiment_neg=("neg", "mean"),
            reddit_avg_score=("score", "mean"),
            reddit_avg_comments=("num_comments", "mean"),
        ).reset_index()

        daily = daily.rename(columns={"date": "timestamp"})
        daily["timestamp"] = pd.to_datetime(daily["timestamp"], utc=True)
        daily = daily.sort_values("timestamp").reset_index(drop=True)

        logger.info("Reddit sentiment: %d posts → %d daily aggregates across %s",
                     len(df), len(daily), subreddits)
        return daily

    def _fetch_reddit_posts(self, subreddit: str, limit: int = 100, sort: str = "hot") -> list[dict]:
        """Fetch posts from a subreddit using the public JSON API."""
        posts = []
        after = None

        while len(posts) < limit:
            batch_limit = min(100, limit - len(posts))
            url = f"{self.REDDIT_BASE}/r/{subreddit}/{sort}.json"
            params = {"limit": batch_limit, "raw_json": 1}
            if after:
                params["after"] = after

            try:
                resp = self._session.get(url, params=params, timeout=15)
                if resp.status_code == 429:
                    logger.warning("Reddit rate limited, waiting 5s...")
                    time.sleep(5)
                    continue
                resp.raise_for_status()

                data = resp.json().get("data", {})
                children = data.get("children", [])
                if not children:
                    break

                for child in children:
                    post_data = child.get("data", {})
                    posts.append(post_data)

                after = data.get("after")
                if not after:
                    break

                time.sleep(1.0)  # be respectful to Reddit

            except Exception as exc:
                logger.warning("Reddit API error for r/%s: %s", subreddit, exc)
                break

        return posts

    @staticmethod
    def _empty_reddit() -> pd.DataFrame:
        return pd.DataFrame(columns=[
            "timestamp", "reddit_sentiment", "reddit_post_count",
            "reddit_sentiment_pos", "reddit_sentiment_neg",
            "reddit_avg_score", "reddit_avg_comments",
        ])

    # ------------------------------------------------------------------
    # 4. Fetch All & Merge
    # ------------------------------------------------------------------

    def fetch_all_and_merge(
        self,
        candle_df: pd.DataFrame,
        fetch_trends: bool = True,
        fetch_twitter: bool = True,
        fetch_reddit: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch all sentiment features and merge into candle DataFrame.
        Uses forward-fill to align daily sentiment to sub-daily candles.

        Args:
            candle_df: OHLCV DataFrame with 'timestamp' column
            fetch_trends: whether to fetch Google Trends (can be slow/rate-limited)
            fetch_twitter: whether to fetch Twitter/X sentiment via public RSS mirrors
            fetch_reddit: whether to fetch Reddit sentiment (real-time only, not historical)

        Returns:
            Enriched DataFrame with sentiment columns.
        """
        df = candle_df.copy()
        if "timestamp" not in df.columns:
            logger.warning("BTCSentimentFeatures: no timestamp column, skipping")
            return df

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.sort_values("timestamp")

        # --- Fear & Greed Index (daily, back to 2018) ---
        fgi_df = self.fetch_fear_greed_history(limit=0)
        if not fgi_df.empty:
            df = _safe_merge_asof(df, fgi_df[["timestamp", "fgi_value"]], on="timestamp", direction="backward")
        else:
            df["fgi_value"] = np.nan

        # --- Google Trends (weekly, back to 5 years) ---
        if fetch_trends:
            gtrends_df = self.fetch_google_trends(keyword="bitcoin", timeframe="today 5-y")
            if not gtrends_df.empty:
                df = _safe_merge_asof(df, gtrends_df, on="timestamp", direction="backward")
            else:
                df["gtrends_bitcoin"] = np.nan
        else:
            df["gtrends_bitcoin"] = np.nan

        # --- Twitter/X Sentiment (recent daily aggregate, not full history) ---
        if fetch_twitter:
            twitter_df = self.fetch_twitter_sentiment(per_term_limit=40)
            if not twitter_df.empty:
                df = pd.merge_asof(
                    df,
                    twitter_df[[
                        "timestamp",
                        "twitter_sentiment",
                        "twitter_post_count",
                        "twitter_sentiment_pos",
                        "twitter_sentiment_neg",
                        "twitter_engagement_proxy",
                    ]],
                    on="timestamp",
                    direction="backward",
                )
            else:
                self._fill_twitter_nans(df)
        else:
            self._fill_twitter_nans(df)

        # --- Reddit Sentiment (current snapshot, not historical) ---
        if fetch_reddit:
            reddit_df = self.fetch_reddit_sentiment(post_limit=100)
            if not reddit_df.empty:
                df = _safe_merge_asof(
                    df, reddit_df[["timestamp", "reddit_sentiment", "reddit_post_count",
                                    "reddit_sentiment_pos", "reddit_sentiment_neg"]],
                    on="timestamp", direction="backward",
                )
            else:
                self._fill_reddit_nans(df)
        else:
            self._fill_reddit_nans(df)

        # --- Derived Features ---
        df = self._compute_derived_features(df)

        n_fgi = fgi_df.shape[0] if not fgi_df.empty else 0
        logger.info(
            "BTCSentimentFeatures: merged %d FGI, trends=%s, twitter=%s, reddit=%s into %d candles",
            n_fgi,
            "yes" if fetch_trends and "gtrends_bitcoin" in df.columns else "no",
            "yes" if fetch_twitter and "twitter_sentiment" in df.columns else "no",
            "yes" if fetch_reddit and "reddit_sentiment" in df.columns else "no",
            len(df),
        )
        return df

    @staticmethod
    def _fill_twitter_nans(df: pd.DataFrame) -> None:
        """Fill Twitter columns with NaN when not fetched."""
        for col in [
            "twitter_sentiment",
            "twitter_post_count",
            "twitter_sentiment_pos",
            "twitter_sentiment_neg",
            "twitter_engagement_proxy",
        ]:
            df[col] = np.nan

    @staticmethod
    def _fill_reddit_nans(df: pd.DataFrame) -> None:
        """Fill Reddit columns with NaN when not fetched."""
        for col in ["reddit_sentiment", "reddit_post_count",
                     "reddit_sentiment_pos", "reddit_sentiment_neg"]:
            df[col] = np.nan

    @staticmethod
    def _compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute derived sentiment features from raw values.

        Fear & Greed derived:
          - fgi_zscore: how extreme is current FGI vs recent history
          - fgi_extreme_fear: binary flag when FGI < 20 (strong buy signal historically)
          - fgi_extreme_greed: binary flag when FGI > 80 (warning/sell signal)
          - fgi_momentum: rate of change of FGI (rising/falling fear)
          - fgi_contrarian: |FGI - 50| / 50 — distance from neutral (0=neutral, 1=extreme)

        Google Trends derived:
          - gtrends_zscore: how extreme is search interest vs recent history
          - gtrends_spike: binary flag for sudden search interest surge
          - gtrends_momentum: rate of change (rising retail interest)

        Twitter/X derived:
          - twitter_sentiment_zscore: how extreme is X sentiment vs recent
          - twitter_bullish: binary flag for strongly positive X mood
          - twitter_bearish: binary flag for strongly negative X mood

        Reddit derived:
          - reddit_sentiment_zscore: how extreme is sentiment vs recent
          - reddit_bullish: binary flag for strongly positive sentiment
        """

        # --- Fear & Greed Index features ---
        if "fgi_value" in df.columns:
            fgi = df["fgi_value"]

            # Z-score vs rolling 30-day mean
            fgi_mean = fgi.rolling(30, min_periods=5).mean()
            fgi_std = fgi.rolling(30, min_periods=5).std().replace(0, np.nan)
            df["fgi_zscore"] = (fgi - fgi_mean) / fgi_std

            # Extreme zones (contrarian signals)
            df["fgi_extreme_fear"] = (fgi < 20).astype(float)
            df["fgi_extreme_greed"] = (fgi > 80).astype(float)

            # Momentum: daily change in FGI
            df["fgi_momentum"] = fgi.diff()
            df["fgi_momentum_3d"] = fgi.diff(3)

            # Contrarian distance from neutral (0-1 scale)
            df["fgi_contrarian"] = (fgi - 50).abs() / 50.0

            # Normalized to 0-1
            df["fgi_normalized"] = fgi / 100.0

        # --- Google Trends features ---
        if "gtrends_bitcoin" in df.columns:
            gt = df["gtrends_bitcoin"]

            # Z-score vs rolling mean
            gt_mean = gt.rolling(30, min_periods=5).mean()
            gt_std = gt.rolling(30, min_periods=5).std().replace(0, np.nan)
            df["gtrends_zscore"] = (gt - gt_mean) / gt_std

            # Spike detection: >2 std above mean
            df["gtrends_spike"] = ((gt - gt_mean) > 2 * gt_std.fillna(1)).astype(float)

            # Momentum (weekly change)
            df["gtrends_momentum"] = gt.diff()
            df["gtrends_momentum_4w"] = gt.diff(4)

            # Normalized to 0-1
            df["gtrends_normalized"] = gt / 100.0

        # --- Twitter/X Sentiment features ---
        if "twitter_sentiment" in df.columns:
            ts = df["twitter_sentiment"]
            ts_mean = ts.rolling(14, min_periods=3).mean()
            ts_std = ts.rolling(14, min_periods=3).std().replace(0, np.nan)
            df["twitter_sentiment_zscore"] = (ts - ts_mean) / ts_std
            df["twitter_bullish"] = (ts > 0.2).astype(float)
            df["twitter_bearish"] = (ts < -0.2).astype(float)
            df["twitter_sentiment_momentum"] = ts.diff()

        # --- Reddit Sentiment features ---
        if "reddit_sentiment" in df.columns:
            rs = df["reddit_sentiment"]

            # Z-score
            rs_mean = rs.rolling(14, min_periods=3).mean()
            rs_std = rs.rolling(14, min_periods=3).std().replace(0, np.nan)
            df["reddit_sentiment_zscore"] = (rs - rs_mean) / rs_std

            # Binary bullish/bearish flags
            df["reddit_bullish"] = (rs > 0.2).astype(float)
            df["reddit_bearish"] = (rs < -0.2).astype(float)

            # Momentum
            df["reddit_sentiment_momentum"] = rs.diff()

        return df

    # ------------------------------------------------------------------
    # Standalone fetch (for live prediction, no merge needed)
    # ------------------------------------------------------------------

    def fetch_current_snapshot(self) -> dict:
        """
        Fetch current sentiment values for live prediction.
        Returns a flat dict suitable for model.predict().
        """
        result = {}

        # Fear & Greed (latest)
        try:
            resp = self._session.get(
                self.FEAR_GREED_URL,
                params={"limit": 7, "format": "json"},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json().get("data", [])
            if data:
                latest = data[0]
                fgi_val = int(latest["value"])
                result["fgi_value"] = fgi_val
                result["fgi_normalized"] = fgi_val / 100.0
                result["fgi_extreme_fear"] = float(fgi_val < 20)
                result["fgi_extreme_greed"] = float(fgi_val > 80)
                result["fgi_contrarian"] = abs(fgi_val - 50) / 50.0

                # Compute momentum from last 7 days
                if len(data) >= 3:
                    vals = [int(d["value"]) for d in data[:3]]
                    result["fgi_momentum"] = float(vals[0] - vals[1])
                    result["fgi_momentum_3d"] = float(vals[0] - vals[2])
        except Exception as exc:
            logger.debug("FGI snapshot failed: %s", exc)

        # Twitter/X sentiment (current)
        try:
            twitter_df = self.fetch_twitter_sentiment(per_term_limit=20)
            if not twitter_df.empty:
                latest = twitter_df.iloc[-1]
                result["twitter_sentiment"] = float(latest.get("twitter_sentiment", 0))
                result["twitter_post_count"] = float(latest.get("twitter_post_count", 0))
                result["twitter_sentiment_pos"] = float(latest.get("twitter_sentiment_pos", 0))
                result["twitter_sentiment_neg"] = float(latest.get("twitter_sentiment_neg", 0))
                result["twitter_engagement_proxy"] = float(latest.get("twitter_engagement_proxy", 0))
                result["twitter_bullish"] = float(result.get("twitter_sentiment", 0) > 0.2)
                result["twitter_bearish"] = float(result.get("twitter_sentiment", 0) < -0.2)
        except Exception as exc:
            logger.debug("Twitter snapshot failed: %s", exc)

        # Reddit sentiment (current)
        try:
            reddit_df = self.fetch_reddit_sentiment(post_limit=50)
            if not reddit_df.empty:
                latest = reddit_df.iloc[-1]
                result["reddit_sentiment"] = float(latest.get("reddit_sentiment", 0))
                result["reddit_post_count"] = float(latest.get("reddit_post_count", 0))
                result["reddit_sentiment_pos"] = float(latest.get("reddit_sentiment_pos", 0))
                result["reddit_sentiment_neg"] = float(latest.get("reddit_sentiment_neg", 0))
                result["reddit_bullish"] = float(result.get("reddit_sentiment", 0) > 0.2)
                result["reddit_bearish"] = float(result.get("reddit_sentiment", 0) < -0.2)
        except Exception as exc:
            logger.debug("Reddit snapshot failed: %s", exc)

        return result

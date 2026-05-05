"""
Post-collection: combine, process, and run full analysis on newly collected data.

Usage after background collections finish:
    python -m src.data.post_collect
"""
import pandas as pd
import logging
import os
import time

logger = logging.getLogger(__name__)


def combine_and_process(
    btc_file: str = "data/raw/kraken_long.csv",
    eth_file: str = "data/raw/kraken_eth_long.csv",
    existing_btc: str = "data/raw/kraken_combined.csv",
    output_btc: str = "data/raw/kraken_final.csv",
    output_eth: str = "data/raw/kraken_eth.csv",
):
    """Combine newly collected data with existing and process."""
    from src.data.process.pipeline import OrderBookDataProcessor

    # BTC
    dfs = []
    if os.path.exists(btc_file):
        new_btc = pd.read_csv(btc_file)
        logger.info(f"New BTC data: {len(new_btc)} rows")
        dfs.append(new_btc)
    if os.path.exists(existing_btc):
        existing = pd.read_csv(existing_btc)
        logger.info(f"Existing BTC data: {len(existing)} rows")
        dfs.append(existing)

    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
        combined.to_csv(output_btc, index=False)
        logger.info(f"Combined BTC: {len(combined)} rows → {output_btc}")

        proc = OrderBookDataProcessor()
        processed = proc.process(input_file=output_btc, output_file="data/processed/kraken_final.csv")
        logger.info(f"Processed BTC: {len(processed)} rows")

        from src.analysis.run.run_all import run_all
        logger.info("Running full analysis on BTC...")
        run_all(
            input_file="data/processed/kraken_final.csv",
            output_dir="plots",
            results_file="results/rq_analysis_final.json",
        )
        logger.info("BTC analysis complete.")

    # ETH
    if os.path.exists(eth_file):
        eth = pd.read_csv(eth_file)
        eth = eth.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
        eth.to_csv(output_eth, index=False)
        logger.info(f"ETH data: {len(eth)} rows → {output_eth}")

        proc = OrderBookDataProcessor(symbol="ETHUSD")
        processed = proc.process(input_file=output_eth, output_file="data/processed/kraken_eth.csv")
        logger.info(f"Processed ETH: {len(processed)} rows")

        from src.analysis.run.run_all import run_all
        logger.info("Running full analysis on ETH...")
        run_all(
            input_file="data/processed/kraken_eth.csv",
            output_dir="plots/eth",
            results_file="results/rq_analysis_eth.json",
        )
        logger.info("ETH analysis complete.")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Wait for background collections to finish
    for fname, label in [("data/raw/kraken_long.csv", "BTC"), ("data/raw/kraken_eth_long.csv", "ETH")]:
        waited = 0
        while not os.path.exists(fname):
            if waited > 1800:
                logger.warning(f"{label} file not found after 30 min, skipping")
                break
            time.sleep(30)
            waited += 30
            if waited % 120 == 0:
                logger.info(f"Waiting for {label} collection... ({waited}s)")

    combine_and_process()


if __name__ == "__main__":
    main()

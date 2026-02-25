# Makefile for Motor TPL Pricing Engine
# Usage:
#   make ingest
#   make validate_raw
#   make stage
#   make validate_staged
#   make join
#   make all
#   make clean

PYTHON := python
FREQ_RAW := data/raw/freMTPL2freq.csv
SEV_RAW  := data/raw/freMTPL2sev.csv

RAW_SNAP_DIR := data/raw_snapshots
STAGING_DIR := data/staging
REPORT_DIR := artifacts/reports

# -----------------------------
# Ingest raw CSV -> raw snapshots
# -----------------------------
ingest:
	$(PYTHON) -m src.data.ingest \
		--freq "$(FREQ_RAW)" \
		--sev  "$(SEV_RAW)" \
		--out  "$(RAW_SNAP_DIR)" \
		--manifest "$(REPORT_DIR)/ingest_manifest.json"

# -----------------------------
# Validate raw snapshots (optional gate)
# -----------------------------
validate_raw:
	$(PYTHON) -m src.data.validate \
		--freq "$(RAW_SNAP_DIR)/freMTPL2freq__*.parquet" \
		--sev  "$(RAW_SNAP_DIR)/freMTPL2sev__*.parquet" \
		--out  "$(REPORT_DIR)"

# -----------------------------
# Stage data (canonicalize + cap exposure)
# -----------------------------
stage:
	$(PYTHON) -m src.data.staging \
		--freq-snapshot "$(RAW_SNAP_DIR)/freMTPL2freq__*.parquet" \
		--sev-snapshot  "$(RAW_SNAP_DIR)/freMTPL2sev__*.parquet" \
		--out           "$(STAGING_DIR)" \
		--report        "$(REPORT_DIR)/staging_report.json" \
		--exposure-cap  1.0

# -----------------------------
# Validate staged data (recommended CI gate)
# -----------------------------
validate_staged:
	$(PYTHON) -m src.data.validate \
		--freq "$(STAGING_DIR)/freq_staged.parquet" \
		--sev  "$(STAGING_DIR)/sev_staged.parquet" \
		--out  "$(REPORT_DIR)"

# -----------------------------
# Build severity training dataset
# -----------------------------
join:
	$(PYTHON) -m src.data.joins \
		--freq   "$(STAGING_DIR)/freq_staged.parquet" \
		--sev    "$(STAGING_DIR)/sev_staged.parquet" \
		--out    "$(STAGING_DIR)/sev_train.parquet" \
		--report "$(REPORT_DIR)/sev_join_report.json"

# -----------------------------
# Full pipeline
# -----------------------------
all: ingest stage validate_staged join

# -----------------------------
# Cleanup artifacts
# -----------------------------
clean:
	rm -rf $(RAW_SNAP_DIR)/*
	rm -rf $(STAGING_DIR)/*
	rm -rf $(REPORT_DIR)/*
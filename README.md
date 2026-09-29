# XAI Climate-Vegetation Case Study

Small repository for downloading climate/vegetation datasets and storing analysis outputs for explainable AI workflows.

This repository accompanies the work "Insights from Explainable Machine Learning: A Case Study in Climate-Vegetation Modeling", currently under review.

## Repository Layout

- `download_zenodo.sh`: downloads the three biome archives (temperate, boreal, tropical).
- `utils/download_zenodo_file.py`: helper used by the download script.
- `datasets/`: target folder for downloaded and extracted data from Zenodo.
- `correlation_graphs/`, `feature_importance/`, `explained_gpp_variance/`: analysis outputs.

## Quick Start

1. Move to the repository root.
2. Ensure Python and `requests` are available.
3. Export your Zenodo token.
4. Run the download script.

```bash
cd ...
python -c "import requests" || pip install requests
export ZENODO_TOKEN="<your_zenodo_token>"
bash download_zenodo.sh
```

## Zenodo Token Note

The current downloader uses a draft-file API endpoint and expects `ZENODO_TOKEN`.
Access to draft files is permission-based, so only the record owner or authorized collaborators can download with their own token.
At the moment, a Zenodo token must be requested from the authors. The Zenodo draft will be published after publication.

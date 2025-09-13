# VITRO_VIVO_Correlation

# ChEMBL-PubChem-ProteinAtlas Integration Tool

an attempt to generate examples of invitro vs invivo data correlation for drugs

## Setup Instructions

```
pip install -r requirements.txt
```

### 2. Prepare Data Files

Make sure you have the `protein_atlas.tsv` file in the same directory as your Streamlit app. This file should contain the Protein Atlas data with the following key columns:

### 3. Run the Application

```bash
streamlit run streamlined_app.py
```

## Features

### 1. Compound Search
- **Input Types Supported:**
  - SMILES strings (e.g., `CCO` for ethanol)
  - PubChem CIDs (e.g., `2244` for aspirin)
  - ChEMBL IDs (e.g., `CHEMBL25` for aspirin)

## API Rate Limits
The app makes calls to:
- **ChEMBL REST API**: Generally no strict limits, but be respectful
- **PubChem REST API**: Limit to ~5 requests per second

## Data Sources
1. **ChEMBL Database**: Bioactivity data for drug discovery
2. **PubChem Database**: Chemical information and biological activities  
3. **Human Protein Atlas**: Protein expression and localization data

## Example Compounds to Test

- **Aspirin**: CID=2244, ChEMBL=CHEMBL25, SMILES=CC(=O)OC1=CC=CC=C1C(=O)O
- **Caffeine**: CID=2519, ChEMBL=CHEMBL113, SMILES=CN1C=NC2=C1C(=O)N(C(=O)N2C)C
- **Ibuprofen**: CID=3672, ChEMBL=CHEMBL521, SMILES=CC(C)CC1=CC=C(C=C1)C(C)C(=O)O



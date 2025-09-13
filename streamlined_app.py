import streamlit as st
import pandas as pd
import requests
import time
from rdkit import Chem
import numpy as np
from io import BytesIO
import json
from typing import Dict, List, Optional
import plotly.express as px
import plotly.graph_objects as go

# Configure Streamlit page
st.set_page_config(
    page_title="ChEMBL-PubChem-ProteinAtlas Tool",
    page_icon="🧬",
    layout="wide"
)

# Add custom CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class EnhancedCompoundAnalyzer:
    def __init__(self):
        self.chembl_base_url = "https://www.ebi.ac.uk/chembl/api/data"
        self.pubchem_base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Streamlit-ChEMBL-App/1.0'
        })
        
    def search_compound_by_identifier(self, identifier: str, id_type: str) -> Dict:
        """Unified compound search method"""
        results = {"chembl": None, "pubchem": None, "smiles": None, "name": None}
        
        try:
            if id_type == "SMILES":
                results.update(self._search_by_smiles(identifier))
            elif id_type == "PubChem CID":
                results.update(self._search_by_cid(identifier))
            elif id_type == "ChEMBL ID":
                results.update(self._search_by_chembl_id(identifier))
        except Exception as e:
            st.error(f"Search error: {e}")
            
        return results
    
    def _search_by_smiles(self, smiles: str) -> Dict:
        """Search by SMILES string"""
        results = {"smiles": smiles}
        
        # Validate SMILES first
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            st.error("Invalid SMILES string")
            return results
            
        # Search ChEMBL - try multiple approaches
        try:
            # First try exact flexmatch
            url = f"{self.chembl_base_url}/molecule.json"
            params = {"molecule_structures__canonical_smiles__flexmatch": smiles, "limit": 1}
            response = self.session.get(url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                if data.get('molecules'):
                    results["chembl"] = data['molecules'][0]
                    results["name"] = data['molecules'][0].get('pref_name')
            
            # If no results, try canonical SMILES from RDKit
            if not results.get("chembl"):
                canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
                if canonical_smiles != smiles:
                    params = {"molecule_structures__canonical_smiles__flexmatch": canonical_smiles, "limit": 1}
                    response = self.session.get(url, params=params, timeout=30)
                    if response.status_code == 200:
                        data = response.json()
                        if data.get('molecules'):
                            results["chembl"] = data['molecules'][0]
                            results["name"] = data['molecules'][0].get('pref_name')
            
            # If still no results, try connectivity search (more flexible)
            if not results.get("chembl"):
                params = {"molecule_structures__canonical_smiles__connectivity": smiles, "limit": 5}
                response = self.session.get(url, params=params, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('molecules'):
                        results["chembl"] = data['molecules'][0]
                        results["name"] = data['molecules'][0].get('pref_name')
            
            # If still no results, try InChI Key approach
            if not results.get("chembl"):
                try:
                    from rdkit.Chem import inchi
                    inchi_key = inchi.MolToInchiKey(mol)
                    if inchi_key:
                        params = {"molecule_structures__standard_inchi_key": inchi_key, "limit": 1}
                        response = self.session.get(url, params=params, timeout=30)
                        if response.status_code == 200:
                            data = response.json()
                            if data.get('molecules'):
                                results["chembl"] = data['molecules'][0]
                                results["name"] = data['molecules'][0].get('pref_name')
                except:
                    pass  # InChI generation failed, continue
                    
        except Exception as e:
            st.warning(f"ChEMBL search failed: {e}")
        
        # Search PubChem
        try:
            url = f"{self.pubchem_base_url}/compound/smiles/{smiles}/JSON"
            response = self.session.get(url, timeout=30)
            if response.status_code == 200:
                data = response.json()
                if data.get('PC_Compounds'):
                    results["pubchem"] = data['PC_Compounds'][0]
        except Exception as e:
            st.warning(f"PubChem search failed: {e}")
        
        return results
    
    def _search_by_cid(self, cid: str) -> Dict:
        """Search by PubChem CID"""
        results = {}
        
        try:
            # Get compound from PubChem
            url = f"{self.pubchem_base_url}/compound/cid/{cid}/JSON"
            response = self.session.get(url, timeout=30)
            if response.status_code == 200:
                data = response.json()
                if data.get('PC_Compounds'):
                    compound = data['PC_Compounds'][0]
                    results["pubchem"] = compound
                    
                    # Extract SMILES
                    smiles = self._extract_smiles_from_pubchem(compound)
                    if smiles:
                        results["smiles"] = smiles
                        # Search ChEMBL with SMILES
                        chembl_results = self._search_by_smiles(smiles)
                        results["chembl"] = chembl_results.get("chembl")
                        results["name"] = chembl_results.get("name")
                    
                    # If SMILES search didn't find ChEMBL entry, try synonym search
                    if not results.get("chembl"):
                        synonyms = self._extract_synonyms_from_pubchem(compound)
                        if synonyms:
                            chembl_results = self._search_chembl_by_synonyms(synonyms)
                            if chembl_results:
                                results["chembl"] = chembl_results.get("chembl")
                                results["name"] = chembl_results.get("name")
                                # Get SMILES from ChEMBL if we didn't have it
                                if not results.get("smiles") and results.get("chembl"):
                                    chembl_compound = results.get("chembl")
                                    if chembl_compound:
                                        structures = chembl_compound.get('molecule_structures', {})
                                        if structures and structures.get('canonical_smiles'):
                                            results["smiles"] = structures['canonical_smiles']
        except Exception as e:
            st.warning(f"CID search failed: {e}")
        
        return results
    
    def _search_by_chembl_id(self, chembl_id: str) -> Dict:
        """Search by ChEMBL ID"""
        results = {}
        
        try:
            # Clean the ChEMBL ID if needed
            if not chembl_id.startswith('CHEMBL'):
                chembl_id = f"CHEMBL{chembl_id}"
            
            # Get compound from ChEMBL
            url = f"{self.chembl_base_url}/molecule/{chembl_id}.json"
            response = self.session.get(url, timeout=30)
            if response.status_code == 200:
                data = response.json()
                results["chembl"] = data
                results["name"] = data.get('pref_name')
                
                # Extract SMILES
                structures = data.get('molecule_structures', {})
                if structures.get('canonical_smiles'):
                    results["smiles"] = structures['canonical_smiles']
                    
                    # Search PubChem with SMILES
                    if results["smiles"]:
                        pubchem_results = self._search_by_smiles(results["smiles"])
                        results["pubchem"] = pubchem_results.get("pubchem")
            else:
                st.warning(f"ChEMBL ID {chembl_id} not found")
                
        except Exception as e:
            st.warning(f"ChEMBL ID search failed: {e}")
        
        return results
    
    def _extract_smiles_from_pubchem(self, compound_data: Dict) -> Optional[str]:
        """Extract SMILES from PubChem compound data"""
        try:
            props = compound_data.get('props', [])
            for prop in props:
                urn = prop.get('urn', {})
                if (urn.get('label') == 'SMILES' and 
                    urn.get('name') == 'Canonical'):
                    return prop.get('value', {}).get('sval')
        except:
            pass
        return None
    
    def _extract_synonyms_from_pubchem(self, compound_data: Dict) -> List[str]:
        """Extract synonyms from PubChem compound data"""
        synonyms = []
        try:
            props = compound_data.get('props', [])
            for prop in props:
                urn = prop.get('urn', {})
                if urn.get('label') == 'IUPAC Name':
                    name_type = urn.get('name', '')
                    if name_type in ['Allowed', 'CAS-like Style', 'Preferred', 'Systematic', 'Traditional']:
                        synonyms.append(prop.get('value', {}).get('sval', ''))
        except:
            pass
        
        # Also try to get additional synonyms from PubChem synonyms endpoint
        try:
            cid = compound_data.get('id', {}).get('id', {}).get('cid')
            if cid:
                url = f"{self.pubchem_base_url}/compound/cid/{cid}/synonyms/JSON"
                response = self.session.get(url, timeout=30)
                if response.status_code == 200:
                    syn_data = response.json()
                    if syn_data.get('InformationList', {}).get('Information'):
                        info = syn_data['InformationList']['Information'][0]
                        if info.get('Synonym'):
                            # Take first 5 synonyms to avoid too many API calls
                            synonyms.extend(info['Synonym'][:5])
        except:
            pass
        
        return list(set(synonyms))  # Remove duplicates
    
    def _search_chembl_by_synonyms(self, synonyms: List[str]) -> Dict:
        """Search ChEMBL using compound synonyms"""
        for synonym in synonyms:
            if not synonym or len(synonym) < 3:  # Skip very short names
                continue
                
            try:
                # Search for molecules by synonym
                url = f"{self.chembl_base_url}/molecule.json"
                params = {
                    "molecule_synonyms__molecule_synonym__icontains": synonym,
                    "limit": 5
                }
                response = self.session.get(url, params=params, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('molecules'):
                        # Return the first match
                        molecule = data['molecules'][0]
                        return {
                            "chembl": molecule,
                            "name": molecule.get('pref_name')
                        }
                
                # Also try searching by preferred name
                params = {
                    "pref_name__icontains": synonym,
                    "limit": 5
                }
                response = self.session.get(url, params=params, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('molecules'):
                        molecule = data['molecules'][0]
                        return {
                            "chembl": molecule,
                            "name": molecule.get('pref_name')
                        }
                        
                time.sleep(0.2)  # Rate limiting between synonym searches
                
            except Exception as e:
                st.warning(f"Synonym search failed for '{synonym}': {e}")
                continue
        
        return {}
    
    def get_activities_for_compound(self, chembl_id: str) -> pd.DataFrame:
        """Get comprehensive activity data for a compound"""
        all_activities = []
        
        try:
            url = f"{self.chembl_base_url}/activity.json"
            params = {
                "molecule_chembl_id": chembl_id,
                "limit": 1000
            }
            
            with st.spinner("Fetching activity data..."):
                response = self.session.get(url, params=params, timeout=60)
                
                if response.status_code == 200:
                    data = response.json()
                    all_activities = data.get('activities', [])
                    
                    # Handle pagination
                    total_count = data.get('page_meta', {}).get('total_count', 0)
                    
                    if total_count > 1000:
                        st.info(f"Found {total_count} activities. Retrieving additional pages...")
                        
                        while data.get('page_meta', {}).get('next'):
                            time.sleep(0.5)  # Rate limiting
                            next_url = data['page_meta']['next']
                            
                            # Fix URL if it's missing the scheme
                            if next_url.startswith('/'):
                                next_url = f"https://www.ebi.ac.uk{next_url}"
                            elif not next_url.startswith('http'):
                                next_url = f"https://www.ebi.ac.uk/chembl/api/data{next_url}"
                            
                            response = self.session.get(next_url, timeout=60)
                            if response.status_code == 200:
                                data = response.json()
                                all_activities.extend(data.get('activities', []))
                                st.progress(len(all_activities) / total_count)
                            else:
                                break
                
        except Exception as e:
            st.error(f"Error fetching activities: {e}")
        
        return self._process_activities_to_dataframe(all_activities)
    
    def _process_activities_to_dataframe(self, activities: List[Dict]) -> pd.DataFrame:
        """Convert raw activities to structured DataFrame"""
        processed = []
        
        for activity in activities:
            # Basic activity info
            record = {
                'assay_chembl_id': activity.get('assay_chembl_id'),
                'target_chembl_id': activity.get('target_chembl_id'),
                'activity_type': activity.get('standard_type'),
                'activity_value': activity.get('standard_value'),
                'activity_unit': activity.get('standard_units'),
                'activity_relation': activity.get('standard_relation'),
                'pchembl_value': activity.get('pchembl_value'),  # Add pchembl value
                'assay_type': activity.get('assay_type'),
                'assay_description': activity.get('assay_description', ''),
                'cell_line': activity.get('cell_line_name'),  # Use correct ChEMBL field
                'cell_line_id': activity.get('cell_line_id'),  # ChEMBL cell line ID
                'cell_type': activity.get('assay_cell_type'),  # Alternative cell line field
                'organism': activity.get('assay_organism'),
                'tissue': activity.get('assay_tissue'),
                'data_validity_comment': activity.get('data_validity_comment'),
                'bao_label': activity.get('bao_label'),  # BAO label for assay classification
                'bao_format': activity.get('bao_format')  # Alternative BAO field
            }
            
            # Classify as Biochemical or Cellular based on BAO labels
            bao_label = str(record.get('bao_label', '')).lower()
            bao_format = str(record.get('bao_format', '')).lower()
            
            # Debug: Add specific debugging for this classification
            if 'single protein format' in bao_label or 'single protein format' in bao_format:
                record['assay_classification'] = 'Biochemical'
            elif ('cell-based format' in bao_label or 'cell-based format' in bao_format or
                  'organism-based format' in bao_label or 'organism-based format' in bao_format or
                  'cell membrane format' in bao_label or 'cell membrane format' in bao_format):
                record['assay_classification'] = 'Cellular'
            else:
                record['assay_classification'] = 'Other'
                # Debug: Let's see what's not matching
                if record.get('bao_label'):
                    print(f"DEBUG: BAO label not matched: '{record.get('bao_label')}'")
            
            # Keep the original B/F classification for compatibility
            assay_desc = str(record['assay_description']).lower()
            if any(term in assay_desc for term in ['binding', 'affinity', 'displacement', 'ki', 'kd']):
                record['bf_type'] = 'B'  # Binding
            elif any(term in assay_desc for term in ['functional', 'inhibition', 'ic50', 'ec50', 'activity']):
                record['bf_type'] = 'F'  # Functional
            else:
                record['bf_type'] = 'Unknown'
            
            processed.append(record)
        
        df = pd.DataFrame(processed)
        
        # Add target information
        if not df.empty:
            df = self._enrich_with_target_info(df)
        
        return df
    
    def _enrich_with_target_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add target information to activities DataFrame"""
        unique_targets = df['target_chembl_id'].dropna().unique()
        target_info_cache = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, target_id in enumerate(unique_targets):
            status_text.text(f"Fetching target info {i+1}/{len(unique_targets)}: {target_id}")
            
            if target_id not in target_info_cache:
                try:
                    url = f"{self.chembl_base_url}/target/{target_id}.json"
                    response = self.session.get(url, timeout=30)
                    if response.status_code == 200:
                        target_data = response.json()
                        
                        # Extract UniProt IDs and names
                        components = target_data.get('target_components', [])
                        uniprot_ids = [comp.get('accession') for comp in components if comp.get('accession')]
                        target_names = [comp.get('component_synonym') for comp in components if comp.get('component_synonym')]
                        
                        target_info_cache[target_id] = {
                            'uniprot_ids': '; '.join(uniprot_ids) if uniprot_ids else 'Unknown',
                            'target_name': '; '.join(target_names) if target_names else target_data.get('pref_name', 'Unknown'),
                            'target_type': target_data.get('target_type', 'Unknown')
                        }
                    else:
                        target_info_cache[target_id] = {
                            'uniprot_ids': 'Unknown',
                            'target_name': 'Unknown',
                            'target_type': 'Unknown'
                        }
                except Exception as e:
                    target_info_cache[target_id] = {
                        'uniprot_ids': 'Error',
                        'target_name': 'Error',
                        'target_type': 'Error'
                    }
                    
                time.sleep(0.2)  # Rate limiting
            
            progress_bar.progress((i + 1) / len(unique_targets))
        
        progress_bar.empty()
        status_text.empty()
        
        # Add target info to DataFrame
        df['uniprot_ids'] = df['target_chembl_id'].map(lambda x: target_info_cache.get(x, {}).get('uniprot_ids', 'Unknown'))
        df['target_name'] = df['target_chembl_id'].map(lambda x: target_info_cache.get(x, {}).get('target_name', 'Unknown'))
        df['target_type'] = df['target_chembl_id'].map(lambda x: target_info_cache.get(x, {}).get('target_type', 'Unknown'))
        
        return df

@st.cache_data
def load_protein_atlas_data():
    """Load Protein Atlas data with caching"""
    try:
        df = pd.read_csv('proteinatlas.tsv', sep='\t')
        return df
    except FileNotFoundError:
        st.error("⚠️ Protein Atlas demo file (proteinatlas.tsv) not found!")
        return pd.DataFrame()

def analyze_protein_expression(uniprot_ids: List[str], protein_atlas_df: pd.DataFrame) -> Dict:
    """Analyze protein expression patterns"""
    if protein_atlas_df.empty or not uniprot_ids:
        return {}
    
    # Clean and split UniProt IDs
    clean_uniprot_ids = []
    for uid_string in uniprot_ids:
        if pd.notna(uid_string) and uid_string not in ['Unknown', 'Error']:
            clean_uniprot_ids.extend([uid.strip() for uid in str(uid_string).split(';')])
    
    if not clean_uniprot_ids:
        return {}
    
    # Find matching proteins
    expression_data = protein_atlas_df[protein_atlas_df['Uniprot'].isin(clean_uniprot_ids)]
    
    if expression_data.empty:
        return {}
    
    # Extract cell line information
    cell_lines = set()
    tissue_info = []
    
    for _, row in expression_data.iterrows():
        # Extract tissue specificity
        if pd.notna(row.get('RNA tissue specificity')):
            tissue_info.append(row['RNA tissue specificity'])
        
        # Extract cell line information from various columns
        cell_columns = ['RNA tissue cell type enrichment', 'Subcellular location']
        for col in cell_columns:
            if pd.notna(row.get(col)):
                cell_info = str(row[col])
                if any(keyword in cell_info.lower() for keyword in ['cell', 'line', 'culture']):
                    cell_lines.add(cell_info)
    
    return {
        'expression_data': expression_data,
        'cell_lines': list(cell_lines),
        'tissue_info': tissue_info,
        'num_proteins_found': len(expression_data)
    }

def create_activity_visualizations(activities_df: pd.DataFrame):
    """Create comprehensive activity visualizations"""
    
    if activities_df.empty:
        st.warning("No activity data to visualize")
        return
    
    # Activity type distribution (only show allowed types)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Activity Type Distribution")
        activity_counts = activities_df['activity_type'].value_counts()
        
        fig = px.bar(
            x=activity_counts.values,
            y=activity_counts.index,
            orientation='h',
            title="Activity Types (AC50/EC50/DC50/IC50/Ki/Kd/MIC)",
            labels={'x': 'Count', 'y': 'Activity Type'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Assay Classification Distribution")
        if 'assay_classification' in activities_df.columns:
            classification_counts = activities_df['assay_classification'].value_counts()
            
            fig = px.pie(
                values=classification_counts.values,
                names=classification_counts.index,
                title="Biochemical vs Cellular Distribution"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No assay classification data available")
    
    # Binding vs Functional classification (only B and F)
    st.subheader("⚖️ Binding vs Functional Assays")
    
    bf_counts = activities_df[activities_df['bf_type'].isin(['B', 'F'])]['bf_type'].value_counts()
    
    if not bf_counts.empty:
        fig = go.Figure(data=[
            go.Bar(
                x=bf_counts.index,
                y=bf_counts.values,
                marker_color=['#FF6B6B', '#4ECDC4'],
                text=bf_counts.values,
                textposition='auto'
            )
        ])
        fig.update_layout(
            title="Assay Classification: Binding (B) vs Functional (F)",
            xaxis_title="Assay Type",
            yaxis_title="Count",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No binding or functional assays found")
    
    # Activity value distribution using pchembl values (already filtered to allowed types)
    activity_data = activities_df[activities_df['pchembl_value'].notna()].copy()
    
    if not activity_data.empty:
        # Convert pchembl_value to numeric, handling any conversion errors
        activity_data['pchembl_value_numeric'] = pd.to_numeric(
            activity_data['pchembl_value'], errors='coerce'
        )
        
        # Filter out any values that couldn't be converted
        activity_data = activity_data[activity_data['pchembl_value_numeric'].notna()]
        
        if not activity_data.empty:
            st.subheader("📈 pChEMBL Value Distribution")
            
            fig = px.histogram(
                activity_data,
                x='pchembl_value_numeric',
                nbins=30,
                title="Distribution of pChEMBL Values (AC50/EC50/DC50/IC50/Ki/Kd/MIC)",
                labels={'pchembl_value_numeric': 'pChEMBL Value', 'count': 'Frequency'}
            )
            fig.add_vline(
                x=activity_data['pchembl_value_numeric'].median(),
                line_dash="dash",
                line_color="red",
                annotation_text=f"Median: {activity_data['pchembl_value_numeric'].median():.2f}"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No valid pChEMBL values found for visualization")
    else:
        st.warning("No pChEMBL data available for the selected activity types")

def main():
    st.title("🧬 Integrated ChEMBL-PubChem-ProteinAtlas Analysis Tool")
    
    # Initialize components
    analyzer = EnhancedCompoundAnalyzer()
    
    # Load Protein Atlas data
    with st.spinner("🔄 Loading Protein Atlas data..."):
        protein_atlas_df = load_protein_atlas_data()
    
    if not protein_atlas_df.empty:
        st.success(f"✅ Loaded Protein Atlas data: {len(protein_atlas_df)} proteins")
    else:
        st.error("❌ Cannot load Protein Atlas data. Some features will be unavailable.")
    
    # Sidebar for input
    st.sidebar.header("🔍 Compound Search")
    
    input_type = st.sidebar.selectbox(
        "Select input type:",
        ["SMILES", "PubChem CID", "ChEMBL ID"],
        help="Choose the type of identifier you want to use"
    )
    
    # Input examples
    examples = {
        "SMILES": "CC(=O)OC1=CC=CC=C1C(=O)O (Aspirin)",
        "PubChem CID": "2244 (Aspirin)",
        "ChEMBL ID": "CHEMBL25 (Aspirin)"
    }
    
    compound_input = st.sidebar.text_input(
        f"Enter {input_type}:",
        placeholder=examples[input_type],
        help=f"Example: {examples[input_type]}"
    )
    
    # Analysis options
    st.sidebar.header("⚙️ Analysis Options")
    
    include_functional = st.sidebar.checkbox(
        "Include functional assays", value=True,
        help="Include functional (F-type) assays in analysis"
    )
    
    include_binding = st.sidebar.checkbox(
        "Include binding assays", value=True,
        help="Include binding (B-type) assays in analysis"
    )
    
    # Main analysis
    if st.sidebar.button("🚀 Start Analysis", type="primary"):
        if not compound_input.strip():
            st.error("⚠️ Please enter a compound identifier")
        else:
            # Search compound
            with st.spinner("🔍 Searching compound databases..."):
                compound_data = analyzer.search_compound_by_identifier(
                    compound_input.strip(), input_type
                )
            
            # Display search results
            st.header("🔍 Compound Search Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if compound_data.get("chembl"):
                    chembl_id = compound_data["chembl"].get("molecule_chembl_id", "Unknown")
                    st.success(f"**ChEMBL:** {chembl_id}")
                    if compound_data.get("name"):
                        st.info(f"**Name:** {compound_data['name']}")
                else:
                    st.warning("**ChEMBL:** Not found")
            
            with col2:
                if compound_data.get("pubchem"):
                    st.success("**PubChem:** Found ✅")
                else:
                    st.warning("**PubChem:** Not found")
            
            with col3:
                # Add molecular structure visualization
                if compound_data.get("smiles"):
                    try:
                        from rdkit.Chem import Draw
                        from rdkit.Chem.Draw import rdMolDraw2D
                        import base64
                        
                        mol = Chem.MolFromSmiles(compound_data['smiles'])
                        if mol:
                            # Generate molecule image
                            drawer = rdMolDraw2D.MolDraw2DCairo(300, 300)
                            drawer.DrawMolecule(mol)
                            drawer.FinishDrawing()
                            img_data = drawer.GetDrawingText()
                            
                            # Convert to base64 for display
                            img_base64 = base64.b64encode(img_data).decode()
                            st.markdown(f'<img src="data:image/png;base64,{img_base64}" width="250">', unsafe_allow_html=True)
                    except Exception as e:
                        st.warning("Could not display molecular structure")
                
            # Proceed with ChEMBL analysis if available
            if compound_data.get("chembl") and compound_data["chembl"].get("molecule_chembl_id"):
                chembl_id = compound_data["chembl"]["molecule_chembl_id"]
                
                # Get activity data
                activities_df = analyzer.get_activities_for_compound(chembl_id)
                
                if not activities_df.empty:
                    st.header("📊 Activity Data Analysis")
                    
                    # Debug: Show raw data statistics
                    st.write("**Debug Information:**")
                    st.write(f"- Total activities fetched: {len(activities_df)}")
                    
                    # Show assay_type distribution
                    assay_type_counts = activities_df['assay_type'].value_counts()
                    st.write(f"- Assay types found: {assay_type_counts.to_dict()}")
                    
                    # Show BAO label distribution
                    if 'bao_label' in activities_df.columns:
                        bao_counts = activities_df['bao_label'].value_counts()
                        st.write(f"- BAO labels found: {bao_counts.to_dict()}")
                    
                    # Show assay classification distribution
                    if 'assay_classification' in activities_df.columns:
                        classification_counts = activities_df['assay_classification'].value_counts()
                        st.write(f"- Assay classifications: {classification_counts.to_dict()}")
                    
                    # Apply filters step by step with debugging
                    # Step 1: Filter by assay type (B or F)
                    bf_filtered = activities_df[activities_df['assay_type'].isin(['B', 'F'])]
                    st.write(f"- After B/F filter: {len(bf_filtered)} activities")
                    
                    # Step 2: Filter by activity type (only keep specified types)
                    allowed_activity_types = ['AC50', 'EC50', 'DC50', 'IC50', 'Ki', 'Kd', 'MIC']
                    activity_type_filtered = bf_filtered[bf_filtered['activity_type'].isin(allowed_activity_types)]
                    st.write(f"- After activity type filter (AC50/EC50/DC50/IC50/Ki/Kd/MIC only): {len(activity_type_filtered)} activities")
                    
                    # Step 3: Filter by BAO classification (keep only Biochemical and Cellular)
                    bao_filtered = activity_type_filtered[activity_type_filtered['assay_classification'].isin(['Biochemical', 'Cellular'])]
                    st.write(f"- After BAO filter (Biochemical + Cellular only): {len(bao_filtered)} activities")
                    
                    # Show distribution after BAO filtering
                    if not bao_filtered.empty:
                        bao_dist = bao_filtered['assay_classification'].value_counts()
                        st.write(f"  - Biochemical: {bao_dist.get('Biochemical', 0)}")
                        st.write(f"  - Cellular: {bao_dist.get('Cellular', 0)}")
                    
                    filtered_df = bao_filtered.copy()
                    
                    # Step 4: Apply binding/functional filters
                    if not include_functional:
                        before_count = len(filtered_df)
                        filtered_df = filtered_df[filtered_df['bf_type'] != 'F']
                        st.write(f"- After excluding functional: {len(filtered_df)} (removed {before_count - len(filtered_df)})")
                    
                    if not include_binding:
                        before_count = len(filtered_df)
                        filtered_df = filtered_df[filtered_df['bf_type'] != 'B']
                        st.write(f"- After excluding binding: {len(filtered_df)} (removed {before_count - len(filtered_df)})")
                    
                    st.info(f"**Final result:** {len(filtered_df)} activities after all filtering")
                    
                    # Show sample of filtered data if available (only with activity values)
                    if not filtered_df.empty:
                        # Filter out activities without values and those with "%" units for display
                        display_df = filtered_df[
                            (filtered_df['activity_value'].notna()) & 
                            (filtered_df['activity_unit'] != '%')
                        ]
                        if not display_df.empty:
                            st.write("**Sample of filtered data:**")
                            display_cols = ['target_name', 'assay_type', 'assay_classification', 'activity_type', 'activity_value', 'activity_unit']
                            st.dataframe(display_df[display_cols].head(), use_container_width=True)
                        else:
                            st.warning("No activities with valid values found in filtered data")
                    
                    # Only create visualizations if we have data
                    if not filtered_df.empty:
                        # Create visualizations
                        create_activity_visualizations(filtered_df)
                    else:
                        st.warning("⚠️ No data remaining after filtering. Try adjusting your filter settings.")
                        
                        # Show what was filtered out
                        st.write("**Possible issues:**")
                        if len(bf_filtered) == 0:
                            st.write("- No assays with type 'B' or 'F' found")
                        if len(bao_filtered) == 0:
                            st.write("- No assays with BAO labels 'Single_protein' or 'cell-base-format' found")
                        if not include_functional and not include_binding:
                            st.write("- Both functional and binding assays are excluded")
                        
                    # Detailed analysis
                    st.header("🔬 Detailed Activity Analysis")
                    
                    # Separate biochemical and cellular assays based on BAO classification
                    # Filter out activities without values
                    biochemical_df = filtered_df[
                        (filtered_df['assay_classification'] == 'Biochemical') & 
                        (filtered_df['activity_value'].notna())
                    ]
                    
                    # For cellular, check multiple cell line fields and improve detection
                    cellular_df = filtered_df[
                        (filtered_df['assay_classification'] == 'Cellular') & 
                        (filtered_df['activity_value'].notna()) &
                        (
                            (filtered_df['cell_line'].notna()) | 
                            (filtered_df['cell_line_id'].notna()) |
                            (filtered_df['cell_type'].notna()) |
                            (filtered_df['assay_description'].str.contains('cell', case=False, na=False)) |
                            (filtered_df['assay_description'].str.contains('CELL_LINE', case=False, na=False)) |
                            (filtered_df['assay_description'].str.contains('TARGET_TYPE: CELL_LINE', case=False, na=False))
                        )
                    ]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("🧪 Biochemical Assays")
                        st.metric("BAO: Single Protein Format", len(biochemical_df))
                        
                        if not biochemical_df.empty:
                            st.dataframe(
                                biochemical_df[['target_name', 'activity_type', 'activity_value', 'activity_unit', 'assay_type']].head(10),
                                use_container_width=True
                            )
                    
                    with col2:
                        st.subheader("🔬 Cellular Assays")
                        st.metric("BAO: Cell-Based Format", len(cellular_df))
                        
                        if not cellular_df.empty:
                            # Display cell line info from multiple possible fields with improved extraction
                            display_cellular = cellular_df.copy()
                            
                            # Create comprehensive cell info by combining available fields
                            def extract_cell_info(row):
                                cell_info_parts = []
                                
                                # ChEMBL cell line name (primary)
                                if pd.notna(row.get('cell_line')):
                                    cell_info_parts.append(f"Cell: {row['cell_line']}")
                                
                                # ChEMBL cell line ID
                                if pd.notna(row.get('cell_line_id')):
                                    cell_info_parts.append(f"ID: {row['cell_line_id']}")
                                
                                # Alternative cell type field
                                if pd.notna(row.get('cell_type')) and row.get('cell_type') != row.get('cell_line'):
                                    cell_info_parts.append(f"Type: {row['cell_type']}")
                                
                                # Extract from assay description if available
                                if pd.notna(row.get('assay_description')):
                                    desc = str(row['assay_description'])
                                    # Look for CELL_LINE patterns in description
                                    import re
                                    cell_patterns = [
                                        r'CELL_LINE[:\s]+([A-Za-z0-9\-]+)',
                                        r'TARGET_TYPE:\s*CELL_LINE[:\s]+([A-Za-z0-9\-]+)',
                                        r'cell\s+line[:\s]+([A-Za-z0-9\-]+)'
                                    ]
                                    for pattern in cell_patterns:
                                        matches = re.findall(pattern, desc, re.IGNORECASE)
                                        if matches:
                                            cell_info_parts.append(f"Desc: {matches[0]}")
                                            break
                                
                                if cell_info_parts:
                                    return '; '.join(cell_info_parts)
                                else:
                                    return 'Cell-based (type not specified)'
                            
                            display_cellular['cell_info'] = display_cellular.apply(extract_cell_info, axis=1)
                            
                            st.dataframe(
                                display_cellular[['target_name', 'activity_type', 'activity_value', 'cell_info', 'assay_type']].head(10),
                                use_container_width=True
                            )
                    
                    # Protein Atlas integration
                    if not protein_atlas_df.empty and not filtered_df.empty:
                        st.header("🎯 Target Protein Expression Analysis")
                        
                        unique_uniprot_ids = filtered_df['uniprot_ids'].dropna().unique().tolist()
                        
                        if unique_uniprot_ids:
                            with st.spinner("🔍 Analyzing protein expression..."):
                                expression_analysis = analyze_protein_expression(unique_uniprot_ids, protein_atlas_df)
                            
                            if expression_analysis:
                                st.success(f"Found expression data for {expression_analysis['num_proteins_found']} target proteins")
                                
                                # Display expression data
                                if not expression_analysis['expression_data'].empty:
                                    st.dataframe(
                                        expression_analysis['expression_data'][
                                            ['Gene', 'Uniprot', 'Gene description', 'RNA tissue specificity']
                                        ],
                                        use_container_width=True
                                    )
                                
                                # Display identified cell lines
                                if expression_analysis['cell_lines']:
                                    st.subheader("📱 Identified Cell Lines with Target Expression")
                                    for i, cell_line in enumerate(expression_analysis['cell_lines'][:10]):
                                        st.write(f"{i+1}. {cell_line}")
                            else:
                                st.warning("No matching protein expression data found")
                        else:
                            st.warning("No UniProt IDs found for target matching")
                    
                    # Generate final report with target-based comparison
                    st.header("📋 Target-Based Biochemical vs Cellular Comparison")
                    
                    # Group data by target for comparison
                    target_comparison = []
                    
                    # Get unique targets that have both biochemical and cellular data
                    targets_with_biochemical = set(biochemical_df['target_chembl_id'].dropna())
                    targets_with_cellular = set(cellular_df['target_chembl_id'].dropna())
                    common_targets = targets_with_biochemical.intersection(targets_with_cellular)
                    
                    st.info(f"Found {len(common_targets)} targets with both biochemical and cellular assay data")
                    
                    if common_targets:
                        for target_id in common_targets:
                            # Get biochemical data for this target
                            target_biochemical = biochemical_df[biochemical_df['target_chembl_id'] == target_id]
                            target_cellular = cellular_df[cellular_df['target_chembl_id'] == target_id]
                            
                            # Calculate median activity values for allowed activity types only
                            allowed_activity_types = ['AC50', 'EC50', 'DC50', 'IC50', 'Ki', 'Kd', 'MIC']
                            
                            for activity_type in allowed_activity_types:
                                biochem_activities = target_biochemical[
                                    (target_biochemical['activity_type'] == activity_type) &
                                    (target_biochemical['activity_value'].notna())
                                ]
                                
                                cellular_activities = target_cellular[
                                    (target_cellular['activity_type'] == activity_type) &
                                    (target_cellular['activity_value'].notna())
                                ]
                                
                                if not biochem_activities.empty and not cellular_activities.empty:
                                    # Convert to numeric
                                    biochem_values = pd.to_numeric(biochem_activities['activity_value'], errors='coerce').dropna()
                                    cellular_values = pd.to_numeric(cellular_activities['activity_value'], errors='coerce').dropna()
                                    
                                    if len(biochem_values) > 0 and len(cellular_values) > 0:
                                        # Get target info
                                        target_info = biochem_activities.iloc[0]
                                        cellular_info = cellular_activities.iloc[0]
                                        
                                        # Get protein expression data for this target
                                        uniprot_id = target_info['uniprot_ids']
                                        expression_level = "Unknown"
                                        
                                        # Improved cell line extraction for comparison
                                        cell_line_used = None
                                        if pd.notna(cellular_info.get('cell_line')):
                                            cell_line_used = cellular_info['cell_line']
                                        elif pd.notna(cellular_info.get('cell_line_id')):
                                            cell_line_used = f"ID:{cellular_info['cell_line_id']}"
                                        elif pd.notna(cellular_info.get('cell_type')):
                                            cell_line_used = cellular_info['cell_type']
                                        else:
                                            cell_line_used = "Unknown cell line"
                                        
                                        target_comparison.append({
                                            'Target_ChEMBL_ID': target_id,
                                            'Target_Name': target_info['target_name'],
                                            'UniProt_ID': uniprot_id,
                                            'Activity_Type': activity_type,
                                            'Biochemical_Median': biochem_values.median(),
                                            'Biochemical_Unit': target_info['activity_unit'],
                                            'Cellular_Median': cellular_values.median(),
                                            'Cellular_Unit': cellular_info['activity_unit'],
                                            'Fold_Difference': cellular_values.median() / biochem_values.median() if biochem_values.median() > 0 else None,
                                            'Cell_Line': cell_line_used,
                                            'Protein_Expression_Level': expression_level,
                                            'Biochemical_Count': len(biochem_values),
                                            'Cellular_Count': len(cellular_values)
                                        })
                    
                    if target_comparison:
                        comparison_df = pd.DataFrame(target_comparison)
                        
                        # Display summary metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Targets Compared", len(comparison_df))
                        
                        with col2:
                            avg_fold_diff = comparison_df['Fold_Difference'].dropna().median()
                            st.metric("Median Fold Difference", f"{avg_fold_diff:.1f}x" if pd.notna(avg_fold_diff) else "N/A")
                        
                        with col3:
                            targets_with_expression = len(comparison_df[comparison_df['Protein_Expression_Level'] != 'Unknown'])
                            st.metric("Targets with Expression Data", targets_with_expression)
                        
                        with col4:
                            unique_cell_lines = comparison_df['Cell_Line'].nunique()
                            st.metric("Unique Cell Lines", unique_cell_lines)
                        
                        # Display comparison table
                        st.subheader("🎯 Target Activity Comparison")
                        st.dataframe(comparison_df, use_container_width=True)
                        
                        # Create visualization
                        if not comparison_df.empty:
                            st.subheader("📊 Biochemical vs Cellular Activity Correlation")
                            
                            # Add explanation about the plot
                            st.info("""
                            **Plot Interpretation Guide:**
                            - **Both axes are logarithmic** - each tick represents a 10-fold change
                            - **Same units** - both axes use the same activity units (nM, µM, etc.) for the same activity type
                            - **Diagonal line** = perfect correlation (biochemical = cellular)
                            - **Points above line** = cellular activity is higher than biochemical
                            - **Points below line** = biochemical activity is higher than cellular
                            """)
                            
                            valid_data = comparison_df.dropna(subset=['Biochemical_Median', 'Cellular_Median'])
                            if not valid_data.empty:
                                # Get the most common activity type and unit for labeling
                                common_activity_type = valid_data['Activity_Type'].mode().iloc[0] if not valid_data['Activity_Type'].empty else 'Activity'
                                
                                # Create enhanced scatter plot with better axis labels
                                fig = px.scatter(
                                    valid_data,
                                    x='Biochemical_Median',
                                    y='Cellular_Median',
                                    color='Activity_Type',
                                    hover_data=['Target_Name', 'Activity_Type', 'Cell_Line', 'Biochemical_Unit', 'Cellular_Unit'],
                                    title=f"Biochemical vs Cellular Activity Correlation (Log Scale)<br><sub>Both axes in same units - comparing {common_activity_type} values</sub>",
                                    labels={
                                        'Biochemical_Median': f'Biochemical {common_activity_type} (Log Scale)',
                                        'Cellular_Median': f'Cellular {common_activity_type} (Log Scale)'
                                    },
                                    log_x=True,
                                    log_y=True
                                )
                                
                                # Add annotation about log scale and units
                                fig.add_annotation(
                                    x=0.02, y=0.98,
                                    xref='paper', yref='paper',
                                    text="📊 Both axes: Log scale, same units<br>🎯 Perfect correlation = diagonal line<br>📈 Points above line: cellular > biochemical",
                                    showarrow=False,
                                    bgcolor="rgba(255,255,255,0.8)",
                                    bordercolor="gray",
                                    borderwidth=1,
                                    font=dict(size=10)
                                )
                                
                                # Get the range for the lines (extend slightly for better visualization)
                                min_val = min(valid_data['Biochemical_Median'].min(), valid_data['Cellular_Median'].min()) * 0.8
                                max_val = max(valid_data['Biochemical_Median'].max(), valid_data['Cellular_Median'].max()) * 1.2
                                
                                # Create x values for the reference lines
                                x_vals = [min_val, max_val]
                                
                                # Perfect correlation (1:1) - Main reference line
                                fig.add_scatter(
                                    x=x_vals,
                                    y=x_vals,
                                    mode='lines',
                                    name='Perfect correlation (1:1)',
                                    line=dict(color='black', width=3),
                                    showlegend=True
                                )
                                
                                # 10-fold difference lines (more relevant for log scale)
                                fig.add_scatter(
                                    x=x_vals,
                                    y=[val * 10 for val in x_vals],
                                    mode='lines',
                                    name='Cellular 10x higher',
                                    line=dict(color='red', dash='dash', width=2),
                                    showlegend=True
                                )
                                
                                fig.add_scatter(
                                    x=x_vals,
                                    y=[val / 10 for val in x_vals],
                                    mode='lines',
                                    name='Biochemical 10x higher',
                                    line=dict(color='blue', dash='dash', width=2),
                                    showlegend=True
                                )
                                
                                # Update layout with better axis formatting
                                fig.update_layout(
                                    height=600,
                                    xaxis_title=f"Biochemical {common_activity_type} (Log Scale)",
                                    yaxis_title=f"Cellular {common_activity_type} (Log Scale)",
                                    legend=dict(
                                        yanchor="top",
                                        y=0.99,
                                        xanchor="right",
                                        x=0.99,
                                        bgcolor="rgba(255,255,255,0.8)"
                                    )
                                )
                                
                                st.plotly_chart(fig, use_container_width=True, key="correlation_plot")
                                
                                # Add summary statistics about correlation
                                import numpy as np
                                
                                # Calculate correlation coefficient (on log scale)
                                correlation = np.corrcoef(
                                    np.log10(valid_data['Biochemical_Median']), 
                                    np.log10(valid_data['Cellular_Median'])
                                )[0, 1]
                                
                                # Count points within different fold-change ranges
                                within_2fold = 0
                                within_10fold = 0
                                total_points = len(valid_data)
                                
                                for _, row in valid_data.iterrows():
                                    x_val = row['Biochemical_Median']
                                    y_val = row['Cellular_Median']
                                    ratio = max(x_val/y_val, y_val/x_val)  # Fold difference
                                    
                                    if ratio <= 2:
                                        within_2fold += 1
                                    if ratio <= 10:
                                        within_10fold += 1
                                
                                # Display correlation statistics
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Correlation (log scale)", f"{correlation:.3f}")
                                with col2:
                                    st.metric("Within 2-fold", f"{within_2fold}/{total_points} ({within_2fold/total_points*100:.1f}%)")
                                with col3:
                                    st.metric("Within 10-fold", f"{within_10fold}/{total_points} ({within_10fold/total_points*100:.1f}%)")
                                
                                # Remove duplicate plotly_chart call
                            else:
                                st.warning("No valid data points for correlation plot")
                    else:
                        st.warning("No target comparison data available")
                        
                        # Create alternative comparison using highest cellular pChEMBL values
                        st.subheader("📋 Alternative Analysis: Top Cellular Activities per Target")
                        
                        if not cellular_df.empty:
                            # Group by target and find best activity for each target
                            cellular_comparison = []
                            
                            for target_id in cellular_df['target_chembl_id'].dropna().unique():
                                target_cellular = cellular_df[cellular_df['target_chembl_id'] == target_id]
                                
                                # For each allowed activity type, find the best value
                                allowed_activity_types = ['AC50', 'EC50', 'DC50', 'IC50', 'Ki', 'Kd', 'MIC']
                                
                                best_activity = None
                                best_pchembl = -1
                                
                                for activity_type in allowed_activity_types:
                                    type_activities = target_cellular[
                                        (target_cellular['activity_type'] == activity_type) &
                                        (target_cellular['activity_value'].notna()) &
                                        (target_cellular['pchembl_value'].notna())
                                    ]
                                    
                                    if not type_activities.empty:
                                        # Convert pchembl to numeric and find highest
                                        pchembl_values = pd.to_numeric(type_activities['pchembl_value'], errors='coerce').dropna()
                                        
                                        if len(pchembl_values) > 0:
                                            max_pchembl = pchembl_values.max()
                                            
                                            if max_pchembl > best_pchembl:
                                                best_pchembl = max_pchembl
                                                best_idx = type_activities.loc[pchembl_values.idxmax()]
                                                best_activity = best_idx
                                
                                if best_activity is not None:
                                    # Get protein expression data for this target
                                    uniprot_id = best_activity['uniprot_ids']
                                    expression_level = "Unknown"
                                    
                                    if not protein_atlas_df.empty and pd.notna(uniprot_id) and uniprot_id != 'Unknown':
                                        clean_uniprot = uniprot_id.split(';')[0].strip()
                                        protein_data = protein_atlas_df[protein_atlas_df['Uniprot'] == clean_uniprot]
                                        
                                        if not protein_data.empty:
                                            if 'RNA tissue cell type enrichment' in protein_data.columns:
                                                expression_level = protein_data.iloc[0]['RNA tissue cell type enrichment']
                                            elif 'RNA tissue specificity' in protein_data.columns:
                                                expression_level = protein_data.iloc[0]['RNA tissue specificity']
                                    
                                    # Improved cell line information extraction
                                    cell_line_info = None
                                    if pd.notna(best_activity.get('cell_line')):
                                        cell_line_info = best_activity['cell_line']
                                    elif pd.notna(best_activity.get('cell_line_id')):
                                        cell_line_info = f"ID:{best_activity['cell_line_id']}"
                                    elif pd.notna(best_activity.get('cell_type')):
                                        cell_line_info = best_activity['cell_type']
                                    else:
                                        # Try to extract from assay description
                                        desc = str(best_activity.get('assay_description', ''))
                                        import re
                                        cell_patterns = [
                                            r'CELL_LINE[:\s]+([A-Za-z0-9\-]+)',
                                            r'TARGET_TYPE:\s*CELL_LINE[:\s]+([A-Za-z0-9\-]+)',
                                        ]
                                        for pattern in cell_patterns:
                                            matches = re.findall(pattern, desc, re.IGNORECASE)
                                            if matches:
                                                cell_line_info = matches[0]
                                                break
                                        if not cell_line_info:
                                            cell_line_info = "Unknown cell line"
                                    
                                    cellular_comparison.append({
                                        'Target_ChEMBL_ID': target_id,
                                        'Target_Name': best_activity['target_name'],
                                        'UniProt_ID': uniprot_id,
                                        'Best_Activity_Type': best_activity['activity_type'],
                                        'Best_Activity_Value': best_activity['activity_value'],
                                        'Activity_Unit': best_activity['activity_unit'],
                                        'Best_pChEMBL_Value': best_pchembl,
                                        'Cell_Line': cell_line_info,
                                        'Protein_Expression_Level': expression_level,
                                        'Assay_ChEMBL_ID': best_activity['assay_chembl_id']
                                    })
                            
                            if cellular_comparison:
                                cellular_comparison_df = pd.DataFrame(cellular_comparison)
                                
                                # Sort by best pChEMBL value (highest first)
                                cellular_comparison_df = cellular_comparison_df.sort_values('Best_pChEMBL_Value', ascending=False)
                                
                                # Display summary metrics
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric("Targets Analyzed", len(cellular_comparison_df))
                                
                                with col2:
                                    avg_pchembl = cellular_comparison_df['Best_pChEMBL_Value'].median()
                                    st.metric("Median pChEMBL", f"{avg_pchembl:.2f}" if pd.notna(avg_pchembl) else "N/A")
                                
                                with col3:
                                    targets_with_expression = len(cellular_comparison_df[cellular_comparison_df['Protein_Expression_Level'] != 'Unknown'])
                                    st.metric("Targets with Expression Data", targets_with_expression)
                                
                                with col4:
                                    unique_cell_lines = cellular_comparison_df['Cell_Line'].nunique()
                                    st.metric("Unique Cell Lines", unique_cell_lines)
                                
                                st.info("📊 **Top Cellular Activities per Target** (sorted by highest pChEMBL value)")
                                st.dataframe(cellular_comparison_df, use_container_width=True)
                                
                                # Create visualization of pChEMBL values
                                if len(cellular_comparison_df) > 1:
                                    st.subheader("📈 Best pChEMBL Values by Target")
                                    
                                    fig = px.bar(
                                        cellular_comparison_df.head(20),  # Show top 20 targets
                                        x='Target_Name',
                                        y='Best_pChEMBL_Value',
                                        color='Best_Activity_Type',
                                        title="Top 20 Targets by Best Cellular pChEMBL Value",
                                        labels={'Best_pChEMBL_Value': 'pChEMBL Value', 'Target_Name': 'Target'}
                                    )
                                    fig.update_layout(xaxis_tickangle=45, height=500)
                                    st.plotly_chart(fig, use_container_width=True, key="pchembl_bar_chart")
                                
                                # Download button for cellular comparison data
                                csv_buffer = BytesIO()
                                cellular_comparison_df.to_csv(csv_buffer, index=False)
                                csv_buffer.seek(0)
                                
                                st.download_button(
                                    label="📥 Download Cellular Activities Report (CSV)",
                                    data=csv_buffer,
                                    file_name=f"cellular_activities_{compound_input.replace('/', '_').replace(' ', '_')}.csv",
                                    mime="text/csv",
                                    help="Download the best cellular activities per target as CSV",
                                    key=f"download_cellular_{chembl_id}"
                                )
                                
                                # Show protein expression insights for cellular data
                                if targets_with_expression > 0:
                                    st.subheader("🧬 Protein Expression Insights (Cellular Targets)")
                                    
                                    cellular_expression_summary = cellular_comparison_df[cellular_comparison_df['Protein_Expression_Level'] != 'Unknown'].groupby('Protein_Expression_Level').agg({
                                        'Target_Name': 'count',
                                        'Best_pChEMBL_Value': 'median'
                                    }).round(2)
                                    
                                    if not cellular_expression_summary.empty:
                                        st.dataframe(cellular_expression_summary.rename(columns={
                                            'Target_Name': 'Number of Targets',
                                            'Best_pChEMBL_Value': 'Median pChEMBL Value'
                                        }), use_container_width=True)
                            else:
                                st.warning("No cellular activities with pChEMBL values found for analysis.")
                        
                        # Still show all filtered data (only with activity values)
                        st.subheader("📋 All Filtered Assay Data")
                        
                        all_data = []
                        for _, row in filtered_df.iterrows():
                            # Only include rows with activity values and exclude "%" units
                            if pd.notna(row['activity_value']) and row['activity_unit'] != '%':
                                all_data.append({
                                    'Assay_ChEMBL_ID': row['assay_chembl_id'],
                                    'Target_ChEMBL_ID': row['target_chembl_id'],
                                    'Target_Name': row['target_name'],
                                    'UniProt_ID': row['uniprot_ids'],
                                    'Assay_Classification': row['assay_classification'],
                                    'ChEMBL_Assay_Type': row['assay_type'],
                                    'BF_Type': row['bf_type'],
                                    'Activity_Type': row['activity_type'],
                                    'Activity_Value': row['activity_value'],
                                    'Activity_Unit': row['activity_unit'],
                                    'pChEMBL_Value': row.get('pchembl_value', ''),
                                    'BAO_Label': row.get('bao_label', ''),
                                    'Cell_Line': row.get('cell_line', row.get('cell_type', '')),
                                    'Organism': row['organism']
                                })
                        
                        all_data_df = pd.DataFrame(all_data)
                        st.dataframe(all_data_df, use_container_width=True)
                        
                        # Download button for all data (without auto-refresh)
                        csv_buffer = BytesIO()
                        all_data_df.to_csv(csv_buffer, index=False)
                        csv_buffer.seek(0)
                        
                        st.download_button(
                            label="📥 Download All Assay Data (CSV)",
                            data=csv_buffer,
                            file_name=f"all_assays_{compound_input.replace('/', '_').replace(' ', '_')}.csv",
                            mime="text/csv",
                            help="Download all filtered assay data as CSV",
                            key=f"download_all_data_{chembl_id}"  # Unique key to prevent refresh
                        )
                else:
                    st.warning("No activity data found for this compound")
            else:
                st.error("Compound not found in ChEMBL. Cannot proceed with activity analysis.")

if __name__ == "__main__":
    main()




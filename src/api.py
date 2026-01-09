import pandas as pd
import requests
import time
import json
import os
import sys

from typing import Optional
from tqdm import tqdm
from urllib.parse import quote

def get_epa_cpdat_categories(
    chemical_name: str, 
    max_retries: int = 3,
    delay: float = 0.2,
    name_type: str = "complete"
) -> Optional[dict]:
    """
    Query PubChem using PUG REST API to get EPA CPDat chemical and product categories
    for a given chemical name.
    
    Args:
        chemical_name: The name of the chemical to query
        max_retries: Maximum number of retry attempts (default: 3)
        delay: Delay between requests in seconds to respect rate limits (default: 0.2)
        name_type: Type of name matching to use. Options:
            - "complete": Exact match only (default, recommended - gets the primary compound)
            - "word": Word-based matching (more flexible but may return wrong compound)
            
    Returns:
        A dictionary containing:
            - 'cid': PubChem Compound ID
            - 'name': Chemical name used
            - 'functional_use': List of EPA CPDat reported functional use categories
            - 'product_categories': List of EPA CPDat Product Use Categories (PUC)
            - 'raw_data': List of all raw category entries with full details
        Returns None if the chemical is not found or an error occurs.
    """
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    
    # URL-encode the chemical name properly
    encoded_name = quote(chemical_name, safe='')
    
    # Step 1: Get CID from chemical name
    cid = None
    
    for attempt in range(max_retries):
        try:
            cid_url = f"{base_url}/compound/name/{encoded_name}/cids/JSON?name_type={name_type}"
            time.sleep(delay)  # Respect rate limits (max 5 requests/second)
            response = requests.get(cid_url, timeout=30)
            
            if response.status_code == 404:
                # Try the other matching type as fallback
                fallback_type = "word" if name_type == "complete" else "complete"
                cid_url_fallback = f"{base_url}/compound/name/{encoded_name}/cids/JSON?name_type={fallback_type}"
                time.sleep(delay)
                response = requests.get(cid_url_fallback, timeout=30)
                
                if response.status_code == 404:
                    print(f"Chemical '{chemical_name}' not found in PubChem")
                    return None
                    
            elif response.status_code == 503:
                print(f"Server busy, retrying... (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay * (attempt + 1) * 2)  # Exponential backoff
                continue
            
            response.raise_for_status()
            cid_data = response.json()
            cids = cid_data.get("IdentifierList", {}).get("CID", [])
            
            if not cids:
                print(f"No CID found for '{chemical_name}'")
                return None
            
            # Take the first CID (for complete match, this is the primary compound)
            cid = cids[0]
            break
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching CID (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return None
            time.sleep(delay * (attempt + 1) * 2)
    
    if cid is None:
        return None
    
    # Step 2: Query the CPDat table directly via SDQ (Service Data Query)
    return _query_cpdat_by_cid(cid, chemical_name, max_retries, delay)


def get_epa_cpdat_categories_by_inchikey(
    inchikey: str, 
    max_retries: int = 3,
    delay: float = 0.2
) -> Optional[dict]:
    """
    Query PubChem using PUG REST API to get EPA CPDat chemical and product categories
    for a given InChIKey.
    
    Args:
        inchikey: The InChIKey of the chemical to query
        max_retries: Maximum number of retry attempts (default: 3)
        delay: Delay between requests in seconds to respect rate limits (default: 0.2)
            
    Returns:
        A dictionary containing:
            - 'cid': PubChem Compound ID
            - 'inchikey': The InChIKey used for the query
            - 'functional_use': List of EPA CPDat reported functional use categories
            - 'product_categories': List of EPA CPDat Product Use Categories (PUC)
            - 'raw_data': List of all raw category entries with full details
        Returns None if the chemical is not found or an error occurs.
    """
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    
    # URL-encode the InChIKey properly
    encoded_inchikey = quote(inchikey, safe='')
    
    # Step 1: Get CID from InChIKey
    cid = None
    
    for attempt in range(max_retries):
        try:
            cid_url = f"{base_url}/compound/inchikey/{encoded_inchikey}/cids/JSON"
            time.sleep(delay)  # Respect rate limits (max 5 requests/second)
            response = requests.get(cid_url, timeout=30)
            
            if response.status_code == 404:
                print(f"InChIKey '{inchikey}' not found in PubChem")
                return None
                    
            elif response.status_code == 503:
                print(f"Server busy, retrying... (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay * (attempt + 1) * 2)  # Exponential backoff
                continue
            
            response.raise_for_status()
            cid_data = response.json()
            cids = cid_data.get("IdentifierList", {}).get("CID", [])
            
            if not cids:
                print(f"No CID found for InChIKey '{inchikey}'")
                return None
            
            # Take the first CID
            cid = cids[0]
            break
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching CID (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return None
            time.sleep(delay * (attempt + 1) * 2)
    
    if cid is None:
        return None
    
    # Step 2: Query the CPDat table directly via SDQ (Service Data Query)
    result = _query_cpdat_by_cid(cid, inchikey, max_retries, delay)
    if result:
        # Rename 'name' to 'inchikey' in the result
        result['inchikey'] = result.pop('name')
    return result


def _query_cpdat_by_cid(
    cid: int,
    identifier: str,
    max_retries: int = 3,
    delay: float = 0.2
) -> Optional[dict]:
    """
    Internal function to query the CPDat table by CID.
    
    Args:
        cid: PubChem Compound ID
        identifier: The original identifier (name or inchikey) for the result
        max_retries: Maximum number of retry attempts
        delay: Delay between requests in seconds
        
    Returns:
        A dictionary containing EPA CPDat categories, or None on error.
    """
    sdq_query = {
        "select": "*",
        "collection": "cpdat",
        "where": {
            "ands": [
                {"cid": str(cid)}
            ]
        }
    }
    
    sdq_url = f"https://pubchem.ncbi.nlm.nih.gov/sdq/sdqagent.cgi?infmt=json&outfmt=json&query={quote(json.dumps(sdq_query))}"
    
    for attempt in range(max_retries):
        try:
            time.sleep(delay)
            response = requests.get(sdq_url, timeout=60)
            
            if response.status_code == 503:
                print(f"Server busy, retrying... (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay * (attempt + 1) * 2)
                continue
            
            response.raise_for_status()
            sdq_data = response.json()
            
            # Parse the SDQ response
            functional_use = []
            product_categories = []
            raw_data = []
            
            output_set = sdq_data.get("SDQOutputSet", [])
            if output_set and len(output_set) > 0:
                rows = output_set[0].get("rows", [])
                
                for row in rows:
                    source = row.get("source", "")
                    category = row.get("category", "")
                    category_desc = row.get("catogorydesc", "")  # Note: typo in API response
                    
                    raw_data.append({
                        "source": source,
                        "category": category,
                        "description": category_desc
                    })
                    
                    if source == "Reported Functional Use":
                        if category and category not in functional_use:
                            functional_use.append(category)
                    elif source == "Product Use Category (PUC)":
                        if category and category not in product_categories:
                            product_categories.append(category)
            
            return {
                "cid": cid,
                "name": identifier,
                "functional_use": functional_use,
                "product_categories": product_categories,
                "raw_data": raw_data
            }
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching CPDat data (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return {
                    "cid": cid, 
                    "name": identifier,
                    "functional_use": [], 
                    "product_categories": [],
                    "raw_data": []
                }
            time.sleep(delay * (attempt + 1) * 2)
    
    return None


def get_epa_cpdat_by_cid(
    cid: int,
    max_retries: int = 3,
    delay: float = 0.2
) -> Optional[dict]:
    """
    Query PubChem EPA CPDat data directly by CID.
    
    Args:
        cid: PubChem Compound ID
        max_retries: Maximum number of retry attempts (default: 3)
        delay: Delay between requests in seconds (default: 0.2)
        
    Returns:
        A dictionary containing EPA CPDat categories, or None on error.
    """
    sdq_query = {
        "select": "*",
        "collection": "cpdat",
        "where": {
            "ands": [
                {"cid": str(cid)}
            ]
        }
    }
    
    sdq_url = f"https://pubchem.ncbi.nlm.nih.gov/sdq/sdqagent.cgi?infmt=json&outfmt=json&query={quote(json.dumps(sdq_query))}"
    
    for attempt in range(max_retries):
        try:
            time.sleep(delay)
            response = requests.get(sdq_url, timeout=60)
            
            if response.status_code == 503:
                print(f"Server busy, retrying... (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay * (attempt + 1) * 2)
                continue
            
            response.raise_for_status()
            sdq_data = response.json()
            
            functional_use = []
            product_categories = []
            raw_data = []
            
            output_set = sdq_data.get("SDQOutputSet", [])
            if output_set and len(output_set) > 0:
                rows = output_set[0].get("rows", [])
                
                for row in rows:
                    source = row.get("source", "")
                    category = row.get("category", "")
                    category_desc = row.get("catogorydesc", "")
                    
                    raw_data.append({
                        "source": source,
                        "category": category,
                        "description": category_desc
                    })
                    
                    if source == "Reported Functional Use":
                        if category and category not in functional_use:
                            functional_use.append(category)
                    elif source == "Product Use Category (PUC)":
                        if category and category not in product_categories:
                            product_categories.append(category)
            
            return {
                "cid": cid,
                "functional_use": functional_use,
                "product_categories": product_categories,
                "raw_data": raw_data
            }
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching CPDat data (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return {
                    "cid": cid,
                    "functional_use": [],
                    "product_categories": [],
                    "raw_data": []
                }
            time.sleep(delay * (attempt + 1) * 2)
    
    return None


cpdat_classes_to_chemsource = {'Arts and crafts/office supplies':"INDUSTRIAL",
 'Batteries':"INDUSTRIAL",
 'Cleaning and safety':"INDUSTRIAL",
 'Cleaning products and household care':"INDUSTRIAL",
 'Cons. electronics, mech. appliances, and machinery':"INDUSTRIAL",
 'Construction and building materials':"INDUSTRIAL",
 'Electronics/small appliances':"INDUSTRIAL",
 'Furniture and furnishings':"INDUSTRIAL",
 'Home maintenance':"INDUSTRIAL",
 'Laboratory supplies':"INDUSTRIAL",
 'Landscape/yard':"INDUSTRIAL",
 'Manufactured formulations':"INDUSTRIAL",
 'Other vehicles/mass transit':"INDUSTRIAL",
 'Personal care':"PERSONAL CARE",
 'Pesticides':"INDUSTRIAL",
 'Pet care':"INDUSTRIAL",
 'Raw materials':"INDUSTRIAL",
 'Specialty occupational products':"INDUSTRIAL",
 'Sports equipment':"INDUSTRIAL",
 'Vehicle':"INDUSTRIAL"}

def get_cpdat_puc_superclasses(chemical_name):
    cpdat_data = get_epa_cpdat_categories(chemical_name)
    if not cpdat_data:
        return None
    else:
        product_categories = cpdat_data.get("product_categories", [])
        superclasses = set()
        for category in product_categories:
            superclass = category.split("->")[0].strip()
            superclasses.add(superclass)
        return list(superclasses)

def get_cpdat_puc_superclasses_inchikey(inchikey):
    # Handle None, NaN, empty string, and non-string values
    if inchikey is None or (isinstance(inchikey, float) and pd.isna(inchikey)) or not isinstance(inchikey, str) or not inchikey.strip():
        return None
    cpdat_data = get_epa_cpdat_categories_by_inchikey(inchikey)
    if not cpdat_data:
        return None
    else:
        product_categories = cpdat_data.get("product_categories", [])
        superclasses = set()
        for category in product_categories:
            superclass = category.split("->")[0].strip()
            superclasses.add(superclass)
        return list(superclasses)

def map_cpdat_to_chemsource(cpdat_superclasses):
    if not cpdat_superclasses:
        return None
    chemsource_set = set()
    for superclass in cpdat_superclasses:
        chemsource = cpdat_classes_to_chemsource.get(superclass)
        if chemsource:
            chemsource_set.add(chemsource)
    return list(chemsource_set)

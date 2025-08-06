from urllib import response
import requests
import json
from urllib.parse import quote
from tqdm import tqdm
import argparse
import os

def call_differential_diagnosis_sams(hpo_dict, base_url="https://genecascade.org/sams-cgi/DifferentialDiagnosis.cgi"):
    """
    Call the Differential Diagnosis CGI with HPO IDs.
    Thought to run for the Phenopacket data in JSONL format, under /home/lasa14/scratch-llm/data/phenopackets/phenopackets_json.jsonl
    Output in JSONL format
    
    Args:
        hpo_dict (dict): Dictionary with HPO IDs as keys and their values (e.g., 2 for present)
        base_url (str): Base URL for the CGI endpoint
    
    Returns:
        requests.Response: The response object from the API call
    """
    try:
        # Convert the dictionary to JSON string
        hpo_json = json.dumps(hpo_dict)
        encoded_hpo = quote(hpo_json)  # URL encode the JSON string        
        full_url = f"{base_url}?hpo={encoded_hpo}" # Construct the full URL      
        response = requests.get(full_url) # HTTP request
        response.raise_for_status() # Check if the request was successful
        
        return response
        
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

def main(data_file, output_file=None):
    # Check if the data file exists
    if not os.path.exists(data_file):
        print(f"Error: Data file '{data_file}' not found.")
        return
    
    # Set default output file if not provided
    if output_file is None:
        output_file = '/home/lasa14/scratch-llm/results/symptoms_mode/SAMS/symptoms_sams_results.jsonl'
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    
    phenopackets = [] # load phenopacket JSON data 
    with open(data_file, 'r') as f:
        for line in f:
            try:
                phenopackets.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")

    sams_results = []
    for pheno in tqdm(phenopackets, desc="Processing phenopackets"):
        hpo_data = {} # SAMS API expects HPO IDs as input instad of the full HPO term
        for symptom in pheno.get("symptoms_hpo", []):
            hpo_id = symptom.split(":")[-1]
            hpo_id = str(int(hpo_id))
            hpo_data[hpo_id] = 2  # 2 indicates present in SAMS

        response = call_differential_diagnosis_sams(hpo_data) # call SAMS API
    
        if response:
            try:
                json_response = response.json()
                predictions = json_response['prediction']

                # Format all predictions as a numbered list
                response_lines = []
                for i, pred in enumerate(predictions, 1):
                    disease_name = pred[1]  # Extract disease name (second element)
                    response_lines.append(f"{i}. {disease_name}")
                
                # Join all lines with newlines
                formatted_response = "\n".join(response_lines)
                
                sams_result = {
                    "id": pheno["id"],
                    "gold": {
                        "disease_name": pheno["gold"]["disease_name"],
                        "disease_id": pheno["gold"]["disease_id"]
                    },
                    "seed": 1234,
                    "symptoms": pheno["symptoms"],
                    "response": formatted_response
                }
                print(f"Processed {pheno['id']}")

                sams_results.append(sams_result)

            except json.JSONDecodeError:
                print(f"Response is not valid JSON for {pheno['id']}")
            except KeyError as e:
                print(f"Missing key in response for {pheno['id']}: {e}")
        else:
            print(f"Failed to get response from the API for {pheno['id']}")
    
    # Save results to a JSON file
    with open(output_file, 'w') as f:
        for result in sams_results:
            f.write(json.dumps(result) + "\n")
    print(f"Results saved to {output_file}")
    print(f"Processed {len(sams_results)} successful predictions out of {len(phenopackets)} total")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Call SAMS API for differential diagnosis")
    parser.add_argument('--data', required=True, help='Path to the phenopackets JSONL file')
    parser.add_argument('--output', help='Path to the output JSONL file (optional)')
    
    args = parser.parse_args()
    
    main(args.data, args.output)
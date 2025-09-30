# Knowledge graph-based retrieval augmented generation (RAG) for rare disease diagnosis and symptom identification

Msc Bioinformatics - Master Thesis project by Laura Santa Cruz

The idea of this project was to implement a Retrieval Augmented Generation (RAG) system to improve the responses given by LLM when it comes to rare diseases. RAG allows to use up-to-date information reducing LLM hallucinations.

For this purpose, [PrimeKG](https://www.nature.com/articles/s41597-023-01960-3) was used as an external database. 

Two case scenarios were implemented:
- Disease symptom description from a disease of interest (DiseaseMode)
- Generating a differential diagnosis from a list of symptoms (SymptomsMode)

## Symptoms description (DiseaseMode)

The `DiseaseMode/disease_mode.py` can be run with:

```
python DiseaseMode/disease_mode.py \
    --dataset [path to the dataset in JSON format] \
    --outdir  [directory to save output to] \
    --runs [number of runs]
```

The `--dataset` file should be in JSON format and contain the "ground truth symptoms" associated with each disease. The structure should be the following:
``` 
 "Spermatogenic failure 96": [
        "Male infertility",
        "Non-obstructive azoospermia",
        "Spermatocyte maturation arrest"
    ],
```
Each disease in in the dataset will be run the number of times specified in the `--runs` command. It will generate two files, `[dataset]_rag_results.json` and `[dataset]_no_rag_results.json`. Each one containing the final list of symptoms generated with the RAG and the LLM, respectively.

### Evaluation

In order to evaluate the results from the RAG framework and the base LLM, the the two files generated as output in the previous step should be given as input. Also the same file as in `--dataset`.

```
python DiseaseMode/disease_mode_evaluation.py \
    --data [path to the dataset in JSON format] \
    --rag-results [dataset]_rag_results.json \
    --no-rag-results [dataset]_no_rag_results.json \
    --outdir [directory to save output to]
```
This will output one file `[dataset]_results.csv` containing different evaluation metrics.


## Differential diagnosis (SymptomsMode)

The `SymptomsMode/symptoms_mode.py` can be run with:

```
python3 SymptomsMode/symptoms_mode.py \
    --dataset [path to the dataset in JSONL format] \
    --runs [number of runs]
``` 

The `--dataset` should have the following structure:

``` 
{"id": "PMID_38054405_Family_2_individual_P3", "gold": {"disease_id": "OMIM:620746", "disease_name": "Neurodevelopmental disorder with hypotonia and characteristic brain abnormalities"}, "symptoms": ["Gray matter heterotopia", "Short corpus callosum", "Seizure", "Microcephaly", "Macrotia", "Global developmental delay", "Axial hypotonia", "Appendicular hypotonia", "Failure to thrive", "Small for gestational age", "Absent speech", "Developmental regression", "Anteverted nares", "Long eyelashes", "Wide nasal bridge"]}

```

Where `gold` referrs to the diagnosed OMIM disease ID and disease name associated with that symptomatology.<br><br>
The output will consist on two files `symptoms_rag_results.jsonl` and `symptoms_no_rag_results.jsonl` for the RAG and LLM responses, respectively.


### Evaluation

The evaluation was done with [PhEval](https://github.com/monarch-initiative/pheval.llm) so you can refer to their documentation.
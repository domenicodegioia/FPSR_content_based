# Do Recommender Systems Really Leverage Multimodal Content? A Comprehensive Analysis on Multimodal Representations for Recommendation

This repository contains the code to reproduce the experiments from the paper "_Do Recommender Systems Really Leverage Multimodal Content? A Comprehensive Analysis on Multimodal Representations for Recommendation_", accepted at ACM CIKM ‘25.


## Setting Up the Virtual Environment

After cloning the repository, it is recommended to create a virtual environment for installing the required dependencies. The codebase was developed using **Python 3.11.6** with **CUDA 11.8**.  

To set up the virtual environment using `venv`, follow these steps:  

```sh
python3 -m venv venv
source venv/bin/activate
sh install.sh
```

## Prompting
To generate structured metadata for images, we employ predefined prompts tailored to each dataset: **Baby, Pets, and Clothing**. These prompts guide the selection of five key attributes that best describe the image content. As mentioned in the paper, we use the LVLMs Qwen2VL and Phi3.5VL.


### Baby Dataset  
```
Imagine you’re creating metadata for an image database of baby products. Your task is to select five keywords that best represent the image content. Fill in the blanks: [Category], [Age Group], [Purpose], [Material], [Usage Context]. For example, your answer will be: [Category] {Feeding}, [Age Group] {Infant}, [Purpose] {Hygiene}, [Material] {Silicone}, [Usage Context] {Home}.
```

### Pets Dataset
```
Imagine you’re creating metadata for an image database of pet-related items. Your task is to select five keywords that best represent the image content. Fill in the blanks: [Category], [Pet Type], [Purpose], [Material], [Usage Context]. For example, your answer will be: [Category] {Toys}, [Pet Type] {Dog}, [Purpose] {Entertainment}, [Material] {Rubber}, [Usage Context] {Outdoor}.
```

### Clothing Dataset  

```
Imagine you're creating metadata for an image database of clothing items. Your task is to select five keywords that best represent the image content. Fill in the blanks: [Type], [Color], [Wear Location], [Material], [Style]. For example, your answer will be: [Type] {dress}, [Color] {red}, [Wear Location] {torso}, [Material] {cotton}, [Style] {casual}. 
```
### LVLMs Output Format

The final output from LVLMs is structured as a CSV file with the following format:
```
Image Name,Keywords
B0BHTBS5RM,"[Category] {Pet Accessories}, [Pet Type] {Dog}, [Purpose] {Support}, [Material] {Plastic}, [Usage Context] {Home}"
B0BX76YVP9,"[Category] {Leashes}, [Pet Type] {Dog}, [Purpose] {Training}, [Material] {Leather}, [Usage Context] {Outdoor}."
B0BM6V2SH8,"[Category] {Dog Treats}, [Pet Type] {Dog}, [Purpose] {Chewing}, [Material] {Meat}, [Usage Context] {Training Aid}"
```
The csv outputs extracted using both LVLMs are available in the folder `extracted_keywords/`.

## Data processing (optional)

To generate the one-hot encoding for the keywords in each dataset, follow these steps:
First, install the necessary dependencies using the following command:  

```sh
pip3 install nltk rich
```

To create the one-hot encoding file, run the following command:

```sh
python3 utils/preprocessing_csv.py --input_file <my-input-file.csv>
```
This script processes the input CSV files and prepares the data for keyword extraction. The preprocessing steps include lemmatization, clustering, categorization, and one-hot encoding.

Before running the script, explicitly define the five categories to match the attributes in your dataset. For example, in the script, you could define the labels as:

```python
labels = ['category', 'pet type', 'purpose', 'material', 'usage context']
```
Make sure to adjust these labels to reflect the attributes present in your CSV file.

Command-Line Arguments:
The script uses the `parse_arguments` function to handle CLI input:

- `--input_file`: Path to the input CSV file.  
- `--save_lemmatized`: Whether to save a file of lemmatized keywords.  
- `--model_name`: Which SentenceTransformer model to use.  
- `--device`: Select "cuda" or "cpu" for processing.  
- `--frequency_threshold`: Minimum frequency of a keyword to consider.  
- `--num_clusters`: Number of clusters.  
- `--distance_threshold`: Threshold for clustering.  
- `--category_limit`: How many top keywords to keep per category. e.g. 50 + the extra'Other' category. The final file will have 250 unique keywords + 5 'Other' for each one of the five categories.
- `--remove_duplicates`: Remove duplicates from the final keywords list.  

 Core Steps:
1. **Lemmatization**  
   Converts keywords to a standardized base form.

2. **Clustering**  
   Groups similar keywords using hierarchical clustering with distance thresholds or a fixed number of clusters.

3. **Categorization**  
   Assigns keywords to the five pre-defined labels and marks any that do not match as "Other."

4. **One-Hot Encoding**  
   Produces a TSV file of one-hot-encoded features, suitable for subsequent Attribute item-KNN.

After running the script, a directory named processed_<timestamp> will be created in the same folder as your input CSV. This directory will contain the following:

- Logs of the entire process.
- Processed files, including clustered results and one-hot-encoded outputs.
- A log file documenting the processing steps.


The **already processed files** containing the one-hot encoding for each dataset are available in `data/<dataset-name>`.

## Running the Experiments  

### Downloading and Extracting Features  

To run the experiments, first download the required feature files from the following [link](https://drive.google.com/drive/folders/13KI-o9ghbGCF_DDuMzWuetiLy6q54T9z?usp=sharing).

Once downloaded, place the corresponding ZIP folder into the appropriate dataset directory `data/<dataset-name>`.

Next, extract the features by executing the following commands:  

```sh
cd data/<dataset-name>
unzip <features_dir.zip>
```

After extracting the features, navigate back to the root directory of the project.

To start the experiments, use the following command:
```sh
CUBLAS_WORKSPACE_CONFIG=:16:8 python3 start_experiments.py --config <config_name>
```
All necessary configuration files for reproducing our experiments are available in the `config_files` directory. This directory also contains configuration files for running experiments with various baselines. To evaluate both LVLMs answers with the `Attribute Item-kNN` model, please update the configuration file by specifying the appropriate side information. Replace or add the following block in the relevant configuration:
```sh
    side_information:
      - dataloader: ItemAttributesOH
        attribute_file: ../data/{0}/baby_phi35vl_onehot_mapped.tsv
```
Make sure to apply the same modification for each dataset.

## The Team

Currently, this repository is maintained by:
- Matteo Attimonelli (matteo.attimonelli@poliba.it)
- Danilo Danese (danilo.danese@poliba.it)
- Claudio Pomo (claudio.pomo@poliba.it)

The scientific supervision is driven by:

- Fedelucio Narducci (fedelucio.narducci@poliba.it)
- Tommaso Di Noia (tommaso.dinoia@poliba.it)
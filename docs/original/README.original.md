# Birmingham-Council-Purchase-Analysis

## Overview
This repository contains a historical dataset of purchase card transactions for the Birmingham City Council. The dataset provides a comprehensive collection of transactional data, enabling various analytical tasks.

## Analytical Tasks
With this dataset, you can perform the following tasks:

### Clustering
- Discovering profiles and identifying patterns in the transactions.
- Detecting unusual transactions (anomaly detection).

### Forecasting
- Predicting future transactional behaviors.
- Forecasting expenditures and determining the next likely purchases.



## Getting Started

### Prerequisites
Ensure you have the following installed:
- [Anaconda](https://www.anaconda.com/products/distribution)
- Python 3.9

### Setting Up the Environment

1. **Create a Conda Environment**:
    ```bash
    conda create -n pct python=3.9
    conda activate pct
    ```

2. **Install Jupyter and Dependencies**:
    ```bash
    conda install jupyter
    conda install ipykernel
    pip install pandas
    ```

3. **Create a Jupyter Kernel**:
    ```bash
    python -m ipykernel install --user --name pct --display-name "Purchasing Card Transactions"
    ```

4. **Installing the Requirements**:
Install the required packages listed in `requirements.txt`:

### Execution

1. **Run the main.py file:**
In the repo folder and terminal after installing prerequisites run:
    ```bash
    python main.py
    ```


## Contributions
Contributions are welcome! Please submit a pull request or open an issue to discuss your ideas or improvements.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
Data provided by Birmingham City Council under the Open Government Licence v2.  
[Dataset Source](https://www.cityobservatory.birmingham.gov.uk/@birmingham-city-council/purchase-card-transactions)
# clifford-group-equivariant-simplicial-transformers



## Description
This repository contains code and resources for the "Clifford Group Equivariant Simplicial Transformers" (CGESTs) project. The project aims to develop a new class of equivariant transformers by leveraging simplicial complexes and Clifford algebra to achieve equivariance to the full symmetry group. The codebase includes implementations of the proposed model, experiments, and evaluation scripts.

We introduce an extension to the domain of geometric deep learning with CGESTs, enhancing the expressivity of Transformers to match that of Simplicial Message Passing Neural Networks (MPNNs). This facilitates efficient and scalable solutions for geometric graph data, including triangles and higher-level graph structures. Our model has been successfully implemented on the three-dimensional N-body problem, aiming to establish a foundation for broader applications in physics, chemistry, robotics, and computer vision. 

## Report
Find a detailed report on the project [here](/blogpost.md).

## Requirements
- Python 3.8+
- Anaconda or Miniconda
   
## Code Organization

├── media/                         # Contains media files, such as images and diagrams
│   └── diagram_1.png              # Example image used for documentation or visualization
├── results/                       # Stores output files from model runs
│   ├── hyperparameter_search/     # Subdirectory for hyperparameter tuning results
│   ├── fin_model_6388474.out      # Output file from a model run
│   ├── model_edges_zeros_6388476.out  # Output file detailing model edges
│   └── model_nodes_6388473.out    # Output file detailing model nodes
├── src/                           # Main source code directory
│   ├── algebra/                   # Placeholder for algebra-related scripts
│   ├── data/                      # Contains datasets and data processing scripts
│   │   ├── nbody_dataset/         # Directory for N-body simulation dataset
│   │   └── nbody.py               # Script for handling N-body dataset
│   ├── lib/                       # Contains libraries and modules used across the project
│   │   ├── nbody_model/           # Directory for N-body simulation models
│   │   │   ├── modules/           # Contains submodules related to the N-body model
│   │   │   ├── __init__.py        # Initialization script for the N-body model
│   │   │   ├── transformer.py     # Transformer module script
│   │   │   └── hyperparameter_testing.py  # Script for hyperparameter testing
│   │   └── env.yaml               # Environment configuration file
│   │   ├── main.py                # Main script to run the project
│   │   ├── unit_test_model.py     # Unit tests for the model
│   ├── models/                    # Placeholder for model scripts
│   ├── scripts/                   # Contains additional scripts used in the project
│   │   └── __init__.py            # Initialization script for the scripts directory
│   └── __init__.py                # Initialization script for the src directory
├── .gitignore                     # Specifies files and directories to be ignored by Git
├── LICENSE                        # Licensing information for the project
├── README.md                      # Provides an overview and documentation for the project
├── blogpost.md                    # Markdown file for the associated blog post
└── demo.ipynb                     # Jupyter notebook demonstrating the usage and functionality of the project


## Installation
1. Clone the repository:
   ```bash
   git clone git@github.com:raoulritter/clifford-group-equivariant-simplicial-transformers.git 
   ```
2. Install the conda environment:
   ```bash
   conda env create -f src/lib/env.yaml
   conda activate cgest_env
   ```
3. Run the main script:
   ```bash
   python nbody_main.py
   ```
   or use other scripts provided in the repository under *scripts*.


## Usage
To use the scripts, ensure you have installed the required dependencies and activated the conda environment. You can then run the main script or any other scripts provided in the repository.

For example, to run the main script:
```bash
python nbody_main.py
```
## Demo and Reproducibility
% use pretrained models
% run scripts 

## Data
Data for the n-body problem and other experiments can be found in the `nbody_dataset` directory. 
Ensure that you have the necessary data files before running the scripts.
% Add info on where data comes from and what it is

## References
% Fill in
## Acknowledgements
% Fill in with TA and DavidRuhe

## Citation
% fill in
If you use this code or find it helpful in your research, please cite it as follows:
```
@misc{,
  author = {Your Name},
  title = {},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/bellavg/dl_on_the_dl}}
}
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```


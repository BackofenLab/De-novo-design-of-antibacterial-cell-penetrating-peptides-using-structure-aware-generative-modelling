# De novo design of antibacterial cell-penetrating peptides using structure-aware generative modelling   
**Abstract**
Cell-penetrating peptides (CPPs) offer a powerful route for delivering therapeutic agents  into cells, with potential applications ranging from antimicrobial and anticancer therapies to intracellular drug delivery. However, their rational design remains challenging due to the lack of understanding of how sequence, structure and function interact. Furthermore, traditional drug discovery approaches typically explore only a narrow region of peptide chemical space. In this study, we present a structure-informed generative framework that integrates sequence embeddings with structure-related latent representations enabling de novo design of CPPs under defined sequential and structural constraints, including both natural and non-natural variants. The model can generate hundreds of candidates and rank them using a multi-metric evaluation approach that considers predicted uptake, CPP resemblance, structural compatibility, docking affinity, and protein-peptide interaction scores. Benchmarking shows that the framework successfully recovers ~70% of top-ranked generated peptides falling within the physicochemical patterns characteristic of known CPPs while exploring novel regions of chemical space with high predicted activity. To evaluate the biological relevance of our designs, we designed and synthesized  top four leading peptides for antibacterial function and tested their antibacterial potency, as well as their cytotoxic profiles in mammalian cells. All the peptides showed minimal cytotoxicity in human cell lines, while at least one of them displayed micromolar antibacterial potency. These results demonstrate a scalable peptide-engineering tool that is capable of generating functional CPP-like molecules.

## Features  
Allows for training 3 different models, explained in the paper "De novo design of antibacterial cell-penetrating peptides using structure-aware generative modelling"
  
## Prerequisites
Before you run this script, ensure you have the following installed:  


Make sure to install the cuda version if you want to make use of your gpu. 

Other dependencies for creating and using the models are listed in **requirements.txt**

  
  
  
## Installation  
To set up your environment for this tool, follow these steps:  
  
**Clone the repository:**    

```
git clone https://github.com/BackofenLab/De-novo-design-of-antibacterial-cell-penetrating-peptides-using-structure-aware-generative-modelling.git
```

  
**Install required Python using conda:**    
To install required python packages we recommend the use of miniconda


**Creating a Miniconda environment:**


First we install Miniconda for python 3. Miniconda can be downloaded from here:

https://docs.conda.io/en/latest/miniconda.html

Then Miniconda should be installed. On a linux machine the command is similar to this one:
```
bash Miniconda3-latest-Linux-x86_64.sh
```
Then we create an environment. The necessary setup is provided in the "environment.yml" file inside the "for_environment" directory

In order to install the corresponding environment one can execute the following command from the "HVSeeker-DNA" directory

```
conda env create -f requirements.txt --name cpp
```

### Activation of the environment

Before running any cpp model one need to activate the corresponding environment.

```
conda activate cpp
```

  
## Usage  
This script can be run from the command line with various options. 


**Basic Usage**  

To use HVSeeker you can either train models yourself or download our pretrained models from https://drive.google.com/drive/folders/1wHWgxH3Y9YSNJXugtZZrLI4PWJ6SDkaK?usp=sharing
We recommend using the the corresponding padding model. 

All shared global variables in the script have to be set upfront in the **globals.py** file.


```
python FT_VAE_main.py 
```

exectues the main training logic of the main 2 submodels used in the paper (TransformerVAE and CoordVAE). 

```
python main.py 
```
trains the TransformerVAE

```
python ./Model/TSC_vae.py 
```
trains the CoordVAE

```
python FT_PepEmbedNet.py
```
trains the PeptideEmbedNet

```
python FT_VAE_testing.py
```
combines all 3 models to create new samples according to the embedding of a sample input. 

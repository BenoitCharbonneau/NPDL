# NPDL
## Installation
Install dependencies on a python environment (3.x)
```
$ pip3 install -r requirements.txt
```
Execute the following commands:
```
$ cd datasets/
$ ./get_datasets.sh
$ ./get_sentiment_analysis.sh
$ cd ../utils/cython/
$ ./build_cython.sh
```
For the seq2seq dataset, download at the following link (no download script since kaggle asks for authentication):
https://www.kaggle.com/thoughtvector/customer-support-on-twitter
```
This project was made for academic purposes (Université de Sherbrooke) and aims to help graduate students understand the implementation details of various neural network architectures. The library runs on CPU, and as a result many of the models tested in the notebooks are not meant to reach great levels of performance, especially the more complicated models (e.g. seq2seq Transformers).
```
The code is commented in french (requested by our supervising professor). Some function signatures were modified with "_npdl" to make it harder for students to find this code with a copy-paste. The commit history on this repository is empty, since the projet was originally made on the university's GitLab and copied to GitHub afterwards.
```
Contributors: 
Benoit Charbonneau - Benoit.Charbonneau2@USherbrooke.ca
Pierre-Luc Parent - Pierre-Luc.Parent@USherbrooke.ca
Pierre-Marc Jodoin (supervisor) - Pierre-Marc.Jodoin@USherbrooke.ca

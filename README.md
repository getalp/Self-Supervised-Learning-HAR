# Leave-One-Dataset-Out (LODO)
Tensorflow implementation of **Self-Supervised Learning with LODO**:

This work has been published in **SmartComp2023**.

**Comparing Self-Supervised Learning Techniques for Wearable Human Activity Recognition** [[Paper](https://arxiv.org/abs/2404.15331)]

*[Sannara Ek](https://scholar.google.com/citations?user=P1F8sQgAAAAJ&hl=en&oi=ao),[Riccardo Presotto](https://sites.google.com/view/riccardopresotto/home), [Gabriele Civitarese](https://civitarese.di.unimi.it/), [François Portet](https://lig-membres.imag.fr/portet/home.php), [Philippe Lalanda](https://lig-membres.imag.fr/lalanda/), [Claudio Bettini](https://sites.google.com/view/claudio-bettini)*


<p align="center">
  <img width="80%" alt="Leave-One-Dataset-Out" src="Figs/LODO_Fig.png">
</p>

If our project is helpful for your research, please consider citing : 
``` 
@misc{ek2024comparing,
      title={Comparing Self-Supervised Learning Techniques for Wearable Human Activity Recognition}, 
      author={Sannara Ek and Riccardo Presotto and Gabriele Civitarese and François Portet and Philippe Lalanda and Claudio Bettini},
      year={2024},
      eprint={2404.15331},
      archivePrefix={arXiv},
      primaryClass={eess.SP}
}
```


## Table of Content
* [1. Updates](#1-Updates)
* [2. Installation](#2-installation)
  * [2.1 Dependencies](#21-dependencies)
  * [2.2 Data](#22-data)
* [3. Quick Start](#3-quick-start)
Loading a pre-trained model to your pipeline

  * [3.1 Loading a pre-trained model to your pipeline](#31-Loading-a-pre-trained-model-to-your-pipeline)
  * [3.2 Using the LODO partition](#32-Using-the-LODO-partition)
  * [3.3 Running Our Self-Supervised Learning Pretraining Pipeline](#33-Running-Our-Self-Supervised-Learning-Pretraining-Pipeline)
* [4. Acknowledgement](#4-acknowledgement)


## 1. Updates


***11/07/2023***
Initial commit: Code of LODO is released.

## 2. Installation
### 2.1 Dependencies

This code was implemented with Python 3.7, Tensorflow 2.11.1 and CUDA 11.2. Please refer to [the official installation](https://www.tensorflow.org/install). If CUDA 11.2 has been properly installed : 
```
pip install tensorflow==2.11.1
```

Another core library of our work is Hickle for the data storage management. Please launch the following command to be able to run our data partitioning scripts: 
```
pip install hickle
```

To run our training and evaluatioin pipeline, additional dependecies are needed. Please launch the following command:

```
pip install -r requirements.txt
```

Our baseline experiments were conducted on a Debian GNU/Linux 10 (buster) machine with the following specs:

CPU : Intel(R) Xeon(R) CPU E5-2623 v4 @ 2.60GHz

GPU : Nvidia GeForce Titan Xp 12GB VRAM

Memory: 80GB 


### 2.2 Data

We provide scripts to automate downloading (With the exception of the MobiAct dataset which requires manual request from  the authors) and proprocessing the datasets used for this study.
See scripts in dataset folders. e.g, for the UCI dataset, run DATA_UCI.py

For running the 'Combined' dataset training and evaluation pipeline, all datasets must first be downloaded and processed.
Please run all scripts in the 'datasets' folder.

Tip: Manually downloading the datasets and placing them in the 'datasets/dataset' folder may be a good alternative for stabiltiy if the download pipeline keeps failing VIA the provided scripts.

UCI
```
https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
```

MotionSense
```
https://github.com/mmalekzadeh/motion-sense/tree/master/data
```

HHAR
```
http://archive.ics.uci.edu/ml/datasets/Heterogeneity+Activity+Recognition
```

RealWorld
```
https://www.uni-mannheim.de/dws/research/projects/activity-recognition/#dataset_dailylog
```

PAMAP2
```
https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring
```

MobiAct

The Mobiact Dataset is only available upon request from the authors. Please contact and request them at:
```
https://bmi.hmu.gr/the-mobifall-and-mobiact-datasets-2/
```

## 3. Quick Start

We provide both a jupyter notebook (.ipynb) and a python script (.py) versions for all the codes.

Due to constraints with Tensorflow, HART currently can only be trained on GPU and will not work when trained with CPU.



### 3.1 Loading a pre-trained model to your work enviroment

The Pre-trained models are provided at the link below:

```
https://zenodo.org/records/11067076?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6Ijk3Y2YxMjdjLTM2ODEtNDM5Yi05ZTg1LTk2ZThmNWUyZjhkOSIsImRhdGEiOnt9LCJyYW5kb20iOiJlMTRiNzE4MGEwZTdkNTg4ZmZjMGE0MDUyYzhhYmRjOSJ9.O9Gt_Nbp9ws44gXAJyAr1ix2U1Pqcei2jL03s74WbdwbiLgJ5tLMge2Lu_9MdHM2tvalUPVE9BultIg8p6RJmQ
```

There are many variations to the provided pre-trained models.

Eg., The architecture used, the SSL technique used and the dataset that was left-out.

We provide scripts to load the pretrained model
After downloading the desired models, please import and add the following code:

```

import utils 

SSL_Model = utils.loadPretrainedModel(method = "Data2vec" ,architecture ="HART", leftOutDataset = "MotionSense", returnType = "classificationModel", activityCount = 8, modelDirectory ="./" )

```



To load a model that was trained with a specific SSL method, change the value of the 'method' parameter to one of the following:

```
Data2vec, MAE, SimCLR
```

To load a different architectures, change the value of the 'architecture' parameter to one of the following:


```
HART,ISPL
```

To load a model that has trained with a specific left-out dataset, change the value of the 'leftOutDataset' parameter to one of the following:

```
'HHAR','MobiAct','MotionSense','RealWorld_Waist','UCI','PAMAP'
```

To specify the state of the model pre-trained model, change the value of the 'returnType' parameter to one of the following:

```
'pipeline','featureExtractor','classificationModel'
```

Passing the 'pipeline' argument will return the entire SSL compononents eg., for MAE, both the decoder and encoder are presents.

Passing the 'featureExtractor' argument will return only the feature extractor of the encoder.

Passing the 'classificationModel' argument will return the feature extrator connected to a dense layer of size 1024 and the classification heads. Note that the added dense and classification heads are not yet trained.

To specify the amount of classifacation heads when 'returnType' is set to 'classificationModel', change the value of the 'activityCount' parameter to your desired classification head count.

To specify the directory of the pre-trained model you have downloaded, change the value of the 'modelDirectory' parameter to your it's corresponding location.

The returned model is a packaged as a conventional tensorflow/keras model. After loading the model, you may further fine-tune the model on your desired tasks.


### 3.2 Using the LODO partition

After downloading and running all the DATA processing scripts in the dataset folder, launch the LODO_Samples.ipynb jupyter notebook OR LODO_Samples.py script to partition the datasets as used in our study.  




### 3.3 Running Our Self-Supervised Learning Pretraining Pipeline

After running the provided LODO scripts, launch the Pretrain.ipynb jupyter notebook OR Pretrain.py script to launch our pre-training pipeline. 

An example to launch the script is below:

```
python Pretrain.py --method Data2vec --architecture hart --testingDataset MotionSense --SSL_epochs 200 --SSL_batch_size 128 --finetune_epoch 50 --finetune_batch_size 64
```

To select different pre-traning methods, change the value of the 'method' flag to one of the following:

```
Data2vec, MAE, SimCLR
```


To select different architectures for the pre-training, change the value of the 'architecture' flag to one of the following:

```
HART,ISPL
```


To select different left-out dataset for the pre-training, change the value of the 'testingDataset' flag to one of the following:
```
'HHAR','MobiAct','MotionSense','RealWorld_Waist','UCI','PAMAP'
```



## 4. Acknowledgement

This work has been partially funded by Naval Group, by MIAI@Grenoble Alpes (ANR-19-P3IA-0003), and granted access to the HPC resources of IDRIS under the allocation 2023-AD011013233R1 made by GENCI.

Part of this research was also supported by projects SERICS (PE00000014) and by project MUSA – Multilayered Urban Sustainability Action,  funded by the European Union – NextGenerationEU, under the National Recovery and Resilience Plan (NRRP) Mission 4 Component 2 Investment Line 1.5: Strengthening of research structures and creation of R\&D “innovation ecosystems”, set up of “territorial leaders in R\&D”.


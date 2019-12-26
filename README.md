# SGAN_vs_Classifier
This repository code will show the impact of the SGAN on classification accuracy with little
supervised data (labeled data).
We will show this on the MNIST dataset with different number of labeled datapoints each time.

The base code of this repository is from: https://machinelearningmastery.com/semi-supervised-generative-adversarial-network/ 
, by Jason Brownlee. The code written here is a direct extension of the article, function added/changed (other than the generation of graphs):
1. train_stand_alone_c
2. train
3. summarize_performance
4. define_standalone_classifier
5. load_mnist

The results:
1. With data augmentation:

<br />
<p align="center">
  <a>
    <img src="Graphs/MODEL_ACC_WITH_AUG.PNG" width="432" height="288">
  </a>
</p>

<br />
<p align="center">
  <a>
    <img src="Graphs/DIFF_WITH_AUG.PNG" width="432" height="288">
  </a>
</p>

2. Without data augmentation:
<br />
<p align="center">
  <a>
    <img src="Graphs/MODEL_ACC_WITHOUT_AUG.PNG" width="432" height="288">
  </a>
</p>

<br />
<p align="center">
  <a>
    <img src="Graphs/DIFF_WITHOUT_AUG.PNG" width="432" height="288">
  </a>
</p>

<!-- GETTING STARTED -->
# GETTING STARTED

1. Change the configuration params in the main.py file
2. Make sure you downloaded the MNIST dataset and you have it locally in the mnist_database dir, if you want to download it programmatically you should change the load_mnist function in main.py (you can also download different datasets in this function.
3. Run the main function
4. Run the generateFinalClassifierGraph.py
The weights and final results presented in the graphs are also available. 


<!-- CONTACT -->
# CONTACT
Zohar Rimon - zohar.rimon@campus.technion.ac.il

Project Link: [https://github.com/zoharri/SGAN_vs_Classifier](https://github.com/zoharri/SGAN_vs_Classifier)





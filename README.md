mc-modes
=========

This repository provides the implementation of modes architecture used to perform the experiments for our ''Dynamic selection and combination of one-class classifiers for multi-class classification” paper.  
Modes is an architecture for multi-class classification using clustering-based ensembles of one-class classifiers.

A natural solution to tackle multi-class problems is employing multi-class classifiers. However, in specific situations, such as imbalanced data or high number of classes, it is more effective to decompose the multi-class problem into several and easier to solve problems. One-class decomposition is an alternative, where one-class classifiers (OCCs) are trained for each class separately. 
However, fitting the data optimally is a challenge for OCCs, especially when it presents a complex intra-class distribution. The literature shows that multiple classifier systems are inherently robust in such cases. Thus, the adoption of multiple OCCs for each class can lead to an improvement for one-class decomposition. 
With that in mind, in this work we introduce the method called One-class Classifier Dynamic Ensemble Selection for Multi-class problems (MODES, for short), which provides competent classifiers for each region of the feature space by decomposing the original multi-class problem into multiple one-class problems. So, each class is segmented using a set of cluster validity indices, and an OCC is trained for each cluster. The rationale is to reduce the complexity of the classification task by defining a region of the feature space where the classifier is supposed to be an expert. The classification of a  test example is performed by dynamically selecting an ensemble of competent OCCs and the final decision is given by the reconstruction of the original multi-class problem. Experiments carried out with 25 databases, 4 OCC models, and 3 aggregation methods showed that the proposed architecture outperforms the literature. When compared with the state-of-the-art, MODES obtained better results, especially for databases with complex decision regions.


============
Installation
============
The package can be installed using pip:

``pip install mc-modes``

=============
Dependencies
=============
The code is tested to work with Python 3.6. The dependency requirements are: 

* numpy
* scipy
* pandas
* scikit-learn

These dependencies are automatically installed using the pip command above.

=========
Examples
=========

We demonstrate the use of modes in the following example, where OcSVM is used as bas OCC for modes.

.. code-block:: python3

    import numpy as np

    from sklearn.datasets import load_iris
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.svm import OneClassSVM
    
    import modes as md
    
    
    X, y = load_iris(return_X_y=True)
    y = y.astype(str)
    
    # modes using ocsvm as base occ
    pipe = make_pipeline(
        # Scaling features
        MinMaxScaler(),
        md.Modes(OneClassSVM()),
    )
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print(np.mean(cross_val_score(pipe, X, y, cv=skf)))
    


==========
Citation
==========

If you find this work useful in your research, please consider citing the following paper:

.. code-block:: latex

    @article{fragoso2021dynamic,
      title={Dynamic selection and combination of one-class classifiers for multi-class classification},
      author={Fragoso, Rog{\'e}rio CP and Cavalcanti, George DC and Pinheiro, Roberto HW and Oliveira, Luiz S},
      journal={Knowledge-Based Systems},
      volume={228},
      pages={107290},
      year={2021},
      publisher={Elsevier}
    }

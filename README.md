# Classification of Mobile Services and Apps through Physical Channel Fingerprinting

The code is used for the paper "Classification of Mobile Services and Apps through Physical Channel Fingerprinting: a Deep Learning Approach" by H.D. Trinh, A. F. Gambin, L. Giupponi, M. Rossi and P. Dini.

LINK to the paper

## Part 1: Supervised Traffic Classification

- ### traffic_classification.py

It performs the supervised traffic classification using the labeled data contained in the file _sessions_df.pkl_, for different session lengths. It produces the accuracy plots shown in Fig.11.

- ### model_init.py

It contains the model definition used to classify the labeled session traces.

## Part 2: Unupervised Traffic Composition

- ### data_prep_for_composition.py

This prepares the data for the traffic composition. You need to install the sniffer [OWL](https://git.networks.imdea.org/nicola_bui/imdeaowl) (follow carefully the instructions) and configure the sniffer output folder. 

- ### traffic_compositon.py

It performs the unsupervised traffic flow decomposition and create the bar plots showed in Fig. 14. Input data is created using _data_prep_for_composition.py_. It uses a CNN model (pretrained), contained in _cnn_model.h5_.

- ### util.py

It contains util methods for data preparation, training/validation sets splits and plots.

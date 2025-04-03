# comp34812
Natural Language Understanding Coursework

TRANSFORMER MODEL:

The transformer model of the project uses Roberta as the pretrained model, and applies a fully connected DNN block with 9 layers on it.
The model is trained to learn the data from train.csv. 
Upon training the model, it was noticed that it struggled to predict 0s with an accuracy as high as that of 1s. To balance it, some synthetic data was created using the techniques used by the paper: Arxiv.org. (2015). Improving the Natural Language Inference robustness to hard dataset by data augmentation and preprocessing. [online] Available at: https://arxiv.org/html/2412.07108v1#S3.

The synthetic data is created by extracting the nouns and verbs from the training set as N and V sets.
1000 iterations are done, where contradictory sentences are created each time with the pattern:
The {n1} {v} the {n2} & The {n1} does not {v} the {n2}; where n1 and n2 are 2 random nouns from N and v is a random verb from V
and with a 30% chance, sentences with entailment are created in the format:
The {n1} {v} the {n2} & The {n1} {v} the {n2}

This is then added back to the training corpus before we start training. This increases the number of examples and hence the accuracy of 0 is raised.

1000 condratictory sentences were added as the training set has fewer examples of contradictions and the model initially didn't learn the contradictions as well. Entailments are added with a 30% probability so that the model does not only predict contradictions for premises with the pattern 'The {n1} {v} the {n2}'

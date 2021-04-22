# Logo classifier
Almost unsupervised logo classifier

The models discriminate logo into two categories: real logo and no logo (dataset consists of logo proposals from the previous pipelines). Classification was done via supervised and semi-supervised approaches on the weakly generated labels from data. For more details of the preprocessing steps, see **Preprocessing.ipynb** or **Preprocessing.html**. 

Weak Precision and weak Recall - metrics on the weak labels - are monitored during training of the model, the results reported on the test dataset. For validation - validation Precision and validation Recall - were used manually labeled ~2000 pictures.

To run training with different, see arguments in the **train.py** file. The unzipped dataset should be located in this folder.
For example, run the supervised training and validate in on the labeled part.
```
python train.py --validate --csv datasets/mixed.csv --target_column mixed_label --validation_csv labeled_part.csv
```
Semi-supervised training pipeline:
```
python train.py --ssl --validate --csv datasets/mixed.csv --target_column mixed_label --validation_csv labeled_part.csv
```

See **Report.ipynb** for more details of the training pipelines.

Total experiments table:
| Method | weak Precision | weak Recall | val. Precision | val. Recall |
| ------------- | ------------- |------------- |------------- |------------- |
| Always predict logo | ~0.7 | 1.0 | 0.3793 | 1.0 |
| EfficientNet-B1 + Weak Labels | 0.8420 | 0.8803 | 0.4378 | 0.7827 |
| Semi-Supervised MixMatch + EfficientNet-B1 + Weak Labels   | 0.8509 | 0.7806 | 0.4500 | 0.7016 |
| Supervised EfficientNet-B1 + Weak Labels + 1000 Strong Labels  | 0.7682 | 0.8093 | 0.4783 | 0.7799 |
| Semi-Supervised MixMatch + EfficientNet-B1 + Weak Labels + 1000 Strong Labels  | 0.7906 | 0.7023 | 0.4990 | 0.7011 |
| Anomaly detection | **TODO** | **TODO** | **TODO** | **TODO** |
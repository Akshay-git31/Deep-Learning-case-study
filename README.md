# AI-Powered E-Commerce Intelligence Platform

A deep-learning case study that explores how product metadata and
product photographs can be used to classify fashion products. The
project uses the **Fashion Product Images (Small)** dataset and compares
a perceptron, metadata-based neural networks, a convolutional neural
network (CNN), and transfer-learning models.

## Project Overview

E-commerce catalogues contain many products with images and descriptive
attributes. Organizing these products manually can be time-consuming.
This project investigates machine-learning approaches that can support
automated catalogue analysis.

The notebook includes these experiments:

1.  **Perceptron and binary image classification** --- demonstrates
    AND/OR logic gates and classifies product images into topwear versus
    bottomwear.
2.  **Metadata classification** --- predicts one of the ten most
    frequent article types from categorical attributes.
3.  **Product-year estimation** --- uses metadata to estimate the
    product year as a regression task.
4.  **Optimizer comparison** --- compares Stochastic Gradient Descent
    (SGD), Momentum, RMSProp, and Adam.
5.  **Regularization** --- explores L2 regularization, dropout, and
    early stopping to help control overfitting.
6.  **CNN trained from scratch** --- classifies product images using a
    custom convolutional neural network.
7.  **Transfer learning** --- compares frozen ImageNet-pretrained VGG16
    and ResNet50 feature extractors.

## Dataset

The project uses [Fashion Product Images
(Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)
by Param Aggarwal.

The notebook downloads the dataset using the Kaggle CLI and expects this
structure:

``` text
data/
├── styles.csv
└── images/
    ├── <product_id>.jpg
    └── ...
```

The notebook cleans records with missing required metadata and missing
images. For the main multiclass experiments, it selects the ten most
frequent `articleType` values. Image-heavy experiments cap the number of
images per class to help keep training time manageable.

> **Dataset access:** You need a Kaggle account and API credentials
> configured for the Kaggle CLI. Do not commit `kaggle.json` or any API
> token to GitHub.

## Technologies Used

-   Python
-   NumPy and Pandas
-   TensorFlow / Keras
-   Scikit-learn
-   Matplotlib and Seaborn
-   Kaggle API / Kaggle CLI
-   Jupyter Notebook or Google Colab
-   LaTeX for the case-study report

## Repository Structure

``` text
.
├── DEEP_LEARNING_CASESTUDY.ipynb   # Main experiment notebook
├── main.tex                        # LaTeX report source (if included)
├── header.png                      # Report title-page header (if included)
├── figures/                        # Figures used by the report (if included)
└── README.md
```

Your repository may have a slightly different layout depending on which
report files you upload. Keep the report's image files in the locations
referenced by `main.tex`.

## Requirements

Recommended Python packages:

``` bash
pip install numpy pandas tensorflow scikit-learn matplotlib seaborn kaggle jupyter
```

A GPU is recommended for the CNN and transfer-learning sections, but it
is not mandatory. CPU training can take significantly longer. The
notebook uses TensorFlow/Keras and downloads pretrained ImageNet weights
for VGG16 and ResNet50 when those models are created.

## How to Run

### Option 1: Google Colab

1.  Upload `DEEP_LEARNING_CASESTUDY.ipynb` to Google Colab.
2.  Enable a GPU through **Runtime → Change runtime type → T4 GPU** (if
    available).
3.  Configure Kaggle API credentials securely in the Colab session.
    Upload `kaggle.json` when prompted by the notebook's comments, then
    place it at `~/.kaggle/kaggle.json` with restricted permissions.
4.  Run the notebook cells from top to bottom.
5.  Review the printed metrics, plots, confusion matrix, and model
    comparison.

### Option 2: Local Jupyter

1.  Install Python and the packages listed above.

2.  Configure the Kaggle CLI with your API credentials. See the [Kaggle
    API documentation](https://github.com/Kaggle/kaggle-api).

3.  Start Jupyter:

    ``` bash
    jupyter notebook
    ```

4.  Open `DEEP_LEARNING_CASESTUDY.ipynb` and run the cells in order.

5.  Ensure the Kaggle download completes and `data/styles.csv` and
    `data/images/` are available before training begins.

The notebook downloads the dataset with:

``` bash
kaggle datasets download -d paramaggarwal/fashion-product-images-small -p ./data --unzip
```

## Model Configuration

The notebook sets a random seed of `42` and defines the following
principal settings:

  Setting                                                                    Value
  ----------------------------------------------- --------------------------------
  Main multiclass labels                            10 most frequent article types
  Maximum images per class in image experiments                                600
  Perceptron image size                                             28 × 28 pixels
  CNN image size                                                    96 × 96 pixels
  Transfer-learning image size                                    128 × 128 pixels
  Shared image batch-size setting                                               32

Individual training sections may use their own batch sizes and epoch
limits.

## Results

The saved report describes these results from the recorded notebook run:

  Model / experiment                                      Reported result
  ----------------------------------- -----------------------------------
  Perceptron: topwear vs bottomwear                        97.0% accuracy
  CNN trained from scratch              Approximately 71.9% test accuracy
  VGG16 transfer learning               Approximately 90.8% test accuracy
  ResNet50 transfer learning            Approximately 90.8% test accuracy

These are results from a particular saved run, not guaranteed outcomes.
Your results may vary with dataset contents, dependency versions,
hardware, and execution conditions. Use the notebook's actual output
when citing results in a report.

## Visualizations

The notebook generates plots for:

-   Perceptron training error
-   Metadata classifier and product-year regression
-   Optimizer loss and accuracy comparison
-   Baseline and regularized training curves
-   Regularization hyperparameter tuning
-   CNN training and validation curves
-   CNN confusion matrix
-   VGG16 and ResNet50 transfer-learning curves
-   Image-model accuracy comparison

If you include the LaTeX report, place the figure files in the paths
expected by `main.tex` (for example, `figures/` if that is how the
source references them).

## Reproducibility and Limitations

-   A fixed random seed is set, but exact reproducibility is not
    guaranteed across hardware and library versions.
-   The main multiclass experiment uses only the ten most frequent
    article types.
-   Image experiments use a capped subset to manage computational cost.
-   The transfer-learning bases are frozen; the notebook does not
    fine-tune all pretrained layers.
-   Results are specific to this dataset and setup and should not be
    assumed to generalize to every e-commerce catalogue.
-   Kaggle credentials and downloaded data should not be committed to a
    public repository unless licensing and security requirements are
    satisfied.

## Authors

-   **Akshay** --- `160124748113`
-   **Raja Sanjay** --- `160124748110`

**Project guide:** Panigrahi Srikanth, Assistant Professor\
**Department:** Computer Science and Engineering --- Artificial
Intelligence and Machine Learning\
**Institution:** Chaitanya Bharathi Institute of Technology (CBIT),
Hyderabad

## References

-   Aggarwal, P. [Fashion Product Images (Small) ---
    Kaggle](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)
-   [TensorFlow documentation](https://www.tensorflow.org/)
-   [Keras documentation](https://keras.io/)
-   [Scikit-learn documentation](https://scikit-learn.org/stable/)
-   Simonyan, K. and Zisserman, A. [Very Deep Convolutional Networks for
    Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
-   He, K. et al. [Deep Residual Learning for Image
    Recognition](https://arxiv.org/abs/1512.03385)

## License

Add a license file before distributing this repository. Check the
dataset's terms on Kaggle separately from the license you choose for
your own code and report.

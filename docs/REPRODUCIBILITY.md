# Reproducibility Guide

## Environment

Recommended:

- Python 3.10 or newer
- Jupyter Notebook/JupyterLab
- A machine with sufficient RAM/storage for the dataset
- GPU recommended for the neural-network and BERT experiments

## Steps

1. Create and activate a virtual environment.
2. Install `requirements.txt`.
3. Open the notebook.
4. Run cells in order.
5. Allow the notebook to download/cache the dataset when required.
6. Review the generated CSV metrics in `artifacts/results/`.
7. Review figures in `figures/`.
8. Review the deployment artifacts generated under `deployment/`.

## Reproducibility caveats

Exact results can vary because of:

- hardware and CUDA/cuDNN versions
- TensorFlow/PyTorch versions
- transformer-library versions
- nondeterministic GPU operations
- package-version differences
- changes to the upstream dataset

The notebook sets a main random seed of `42`, but this does not guarantee bit-for-bit reproducibility across all hardware/software configurations.

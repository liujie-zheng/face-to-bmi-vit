# Face to Body Mass Index
Body mass index (BMI) is a measure of body fat based on height and weight that applies to adult men and women. 
<br>
This project predicts the BMI value with one image of a human face.

## Performance
With original unaugmented dataset, after training 10 epoches, the model has a MAE loss of ``3.45`` on the test dataset. <br>
With augmented dataset, still training...

## Installation
1. Clone this repository by running: (It should take some time since the weight files are large)
```
git clone git@github.com:liujie-zheng/face-to-bmi-vit.git
cd face-to-bmi-vit
```
2. Install conda [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).
3. Depending on your operating system, install dependencies by running: 
```
conda env create -f environment_linux.yml
conda activate face2bmi
```
or
```
conda env create -f environment_mac.yml
conda activate face2bmi
```

## Run a demo in terminal
1. (Optional) replace ./data/test_pic.jpg with your own image. Note: for your own image, a face should occupy a substantial part of the image for optimal results.
2. In root directory, run:
```
cd scripts
conda run -n face2bmi --no-capture-output python demo.py
```
if you encounter a ``PermissionError: [Errno 13] Permission denied`` error, instead run:
```
sudo conda run -n face2bmi --no-capture-output python demo.py
```

## Run a demo with webcam in Jupyter Notebook
1. Install notebook, opencv and prepare ipykernel by running:
```
conda install notebook
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=face2bmi
```
2. Install opencv capturing image with webcam by running:
```
conda install -c conda-forge opencv
```
3. In the root directory, open notebook by running:
```
jupyter notebook demo.ipynb
```
4. In the notebook, use ``Cell`` -> ``Run All`` to capture an image and predict BMI.


## Train it by yourself
In root directory, train the original unaugmented dataset by running:
```
cd scripts
python run.py
```
or train the augmented dataset by running:
```
cd scripts
python run.py -a
```

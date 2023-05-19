# Face to BMI
This project predicts the BMI value with one image of the human face.

## Performance
After training 10 epoches on the original unaugmented dataset, the model has an L1 loss of ``3.45`` on the test dataset.

## Installation (Linux)
1. Clone this repository by running: (It should take some time since the weight files are large)
```
git clone git@github.com:liujie-zheng/face-to-bmi.git
cd face-to-bmi
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

## Run a demo
1. (Optional) replace ./data/test_pic.jpg with your own image. Note: for your own image, a face should occupy a substantial part of the image for optimal results.
2. In root directory, run:
```
cd scripts
conda run -n face2bmi --no-capture-output python demo.py
```

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

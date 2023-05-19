# Face to BMI
This project predicts the BMI value with one image of the human face.

## Installation
1. Clone this repository by running:
```
git clone git@github.com:liujie-zheng/face-to-bmi.git
cd face-to-bmi
```
It should take sometime since the weight files are large.
<br>
2. Install conda [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).
3. Install dependencies by running:
```
conda env create -f environment.yml
conda activate face2bmi
```

## Run a demo
1. (Optional) replace ./data/test_pic.jpg with your own image.
2. In root directory, run:
```
python ./scripts/demo.py
```

## Train it by yourself
In root directory, train the original dataset by running:
```
python ./scripts/run.py
```
or train the augmented dataset by running:
```
python ./scripts/run.py -a
```

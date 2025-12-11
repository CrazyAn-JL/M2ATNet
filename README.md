# M2ATNet: Multi-Scale Multi-Attention Denoising and Feature Fusion Transformer for Low-Light Image Enhancement(CMC 2025)

The official implementation of M2ATNet: Multi-Scale Multi-Attention Denoising and Feature Fusion Transformer for Low-Light Image Enhancement, accepted by Computers, Materials & Continua 2025.

![Fig 1](https://github.com/CrazyAn-JL/M2ATNet/blob/main/MATNet.png)

## pre-trained model

You can refer to the following link to download the pre-trained model.

- LOLv1: [Baidu Pan](https://pan.baidu.com/s/1rZFrF-ePbl_vClMNnv40Qw). (code: `5yjv`)
- LOLv2-real: [Baidu Pan](https://pan.baidu.com/s/1TA_arjB5q94_L_pZfrSNlA). (code: `xgrv`)
- LOLv2-syn: [Baidu Pan](https://pan.baidu.com/s/1dJ3XTn3S5TuzLhoSifjFWQ). (code: `qy23`)

## 1. Create Conda Environment

```bash
conda create --name M2ATNet python=3.7.0
conda activate M2ATNet
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 3. Training
You can modify the dataset configuration you want to train in the ./data/options.py file.
```bash
python train.py
```

## 4. Testing
You can select the dataset you want to test and simply enter the corresponding command, for example: --lol, --lol_v2_real, etc.

```bash
python eval.py --lol
```

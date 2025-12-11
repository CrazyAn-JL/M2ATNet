# M2ATNet: Multi-Scale Multi-Attention Denoising and Feature Fusion Transformer for Low-Light Image Enhancement (CMC 2025)

The official implementation of M2ATNet: Multi-Scale Multi-Attention Denoising and Feature Fusion Transformer for Low-Light Image Enhancement, accepted by Computers, Materials & Continua 2025.

Paper link: [https://link.springer.com/article/10.1007/s00371-024-03414-2](https://www.techscience.com/cmc/v86n1/64460)

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

## Citation
~~~
@Article{wei2025M2atnet,
AUTHOR = {Zhongliang Wei, Jianlong An, Chang Su},
TITLE = {M2ATNet: Multi-Scale Multi-Attention Denoising and Feature Fusion Transformer for Low-Light Image Enhancement},
JOURNAL = {Computers, Materials \& Continua},
VOLUME = {86},
YEAR = {2026},
NUMBER = {1},
PAGES = {1--20},
ISSN = {1546-2226},
DOI = {10.32604/cmc.2025.069335}
}
~~~

## Acknowledgements

<!--ts-->
* [CIDNet](https://github.com/Fediory/HVI-CIDNet)
<!--te-->

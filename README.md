# StyleEntity

This repository contains the code for the paper "Named Entity Driven Zero-Shot Image Manipulation" presented at CVPR 2024.

## Prerequisites

To run the code in this repository, ensure you have the following environment setup:

- `torch>=1.6.0`
- `BasicSR`

## Preparing Pre-trained Models

1. Download the StyleGAN2 checkpoint `stylegan2_ffhq_config_f_1024_official-3ab41b38.pth` from the BasicSR repository: [BasicSR Checkpoints](https://github.com/XPixelGroup/BasicSR?tab=readme-ov-file).
2. Save the downloaded checkpoint in an accessible directory for later use.

## Encoding Named Entities

Before training, you need to encode the Named Entity texts. Run the following script to perform the encoding:

```bash
python scripts/encode_nes.py
```

## Training

To train the model, use the following command with the specified training configuration file:

```bash
python train.py -opt options/train.yaml
```

## Pre-trained Models

You can download pre-trained models from the following link:

- [StyleEntity Pre-trained Models](https://huggingface.co/fengzhida/StyleEntity)

## Citation

If you find this repository useful in your research, please consider citing our paper:

```
@inproceedings{StyleEntity2024,
    author    = {Feng, Zhida and Chen, Li and Tian, Jing and Liu, JiaXiang and Feng, Shikun},
    title     = {Named Entity Driven Zero-Shot Image Manipulation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {9110-9119}
}
```

## License

This repository is released under the MIT License. See the LICENSE file for more details.

## Contact

For any questions or issues, please open an issue on this repository or contact feng.zhida@outlook.com.

"Low-light image enhancement using inverted image normalized by atmospheric light," Signal Processing
[Jong Ju Jeon](triplej@pusan.ac.kr), [Il Kyu Eom*](ikeom@pusan.ac.kr)(https://sites.google.com/view/ispl-pnu)

[Paper](https://doi.org/10.1016/j.sigpro.2022.108523)


### Requirements ###
1. Linux
2. Python (3.10.4)
3. scikit-image (0.19.2)
4. opencv (4.5.5)


### Usage ###
you can just run through
```shell
python Run_LowLight.py 
    --input_dir=/path/to/your/dataset/dir/ \
    --output_dir=/path/to/save/results/ \
    --x_max=0.8                               # defaults x_max=0.8
    --x_min=0.6                               # defaults x_min=0.6

#python Run_LowLight.py --input_dir=/path/to/your/dataset/dir/ --output_dir=/path/to/save/results/ --x_max=0.8 --x_min=0.6

```

### Citation ###
Jong Ju Jeon, Il Kyu Eom,
Low-light image enhancement using inverted image normalized by atmospheric light,
Signal Processing,
2022,
https://doi.org/10.1016/j.sigpro.2022.108523.
(https://www.sciencedirect.com/science/article/pii/S0165168422000706)
Keywords: Low-light image enhancement; Inverted image; Atmospheric light; Normalization; Transmission map; Retinex model; Saturation stretch

 

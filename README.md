# Source code of LACN
## The main requirements. You may need to download some other requirements depending on your environment
```
torch == 1.8.0
python == 3.7.0
transformers == 4.28.1
scikit-learn == 1.0.2
urllib3 == 1.26.6
```

## How to run this code?
Before running the code, you need to download the BERT pre-training model.

you can download it at https://huggingface.co/ or https://pan.quark.cn/s/c529d057d9d6  ExtractionCode:mrfS and place it under the src folder.

Additionally, raw_label.npy and testSub_y.npy are needed, we compress them into a label_data.zip file cause they are too large. Unzip label_data.zip and place the two files at data/EUR/.

### run
* cd src

* run Cla.py

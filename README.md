# Graviti AI Community

[![Official website](https://img.shields.io/badge/website-awesome-blue)](https://www.graviti.com/)
[![Public datasets](https://img.shields.io/badge/Datasets-1%2C233-green)](https://gas.graviti.com/open-datasets)
[![Downloads](https://pepy.tech/badge/tensorbay/month)](https://pepy.tech/project/tensorbay)
![Discord](https://img.shields.io/discord/877084293883625473?color=purple&label=Community%20Discord)

---
## Push the bounddaries of AI
⭐ Welcome to Graviti AI community! We build the community to provide a decentralized way to advocate community-generated datasets, models and applications towards better science understanding.

## Table of Contents

- ▶️ [**Quick start on how to use open datasets**](#quick-start)
- 📖 [**Step-by-step Tutorial**](https://github.com/Graviti-AI/datasets/blob/main/QuickStart.md)
- 📑 [**Open datasets catalog**](#open-datasets-catalog)
- ✍️ [**Become a contributor**](#become-a-contributor)
- 🔍 [**Find a dataset on Graviti**](https://gas.graviti.com/open-datasets)
- ❓ [**Q&A**](#qa)
- 💡 [**Documentation**](https://tensorbay-python-sdk.graviti.com/en/latest/)
- 🧑‍🤝‍🧑 [**Join the community**](#join-the-community)
</br>
<p align="center">
<img src="https://user-images.githubusercontent.com/92721051/161734655-7b76dd90-9dc8-4d40-8d14-a15b8ea4a72f.png" width="509" height="297" border="10"/>
</p>

## Quick start

⭐ You have a complex problem or project involving a large amount of data and lots of variables. You know that finding a public dataset to train your machine learning model would be the best approach. How do you deal with data that’s in a variety of formats? How do you choose the dataset for your model?</br>
We'll walk you through step by step from the basics to advanced techniques and help you get started!

1. Sign up for an account

Go to [graviti.com](https://account.graviti.com/sign-up) to sign up. </br>
Get an AccessKey on [Graviti Developer Tools](https://gas.graviti.com/tensorbay/developer).</br>

> *An AccessKey is needed to authenticate identity when using TensorBay via SDK or CLI.* </br>
> *You have full permissions for the account. Please keep the key properly.*

2. Install Tensorbay Python SDK

- To install TensorBay SDK and CLI by **pip**, run the following command:
```
pip3 install tensorbay
```
- To verify the SDK and CLI version, run the following command:
```
gas --version
```
- Authorize a Client Instance
```
from tensorbay import GAS
gas = GAS("<YOUR_ACCESSKEY>")
```

3. Select an open dataset

> *You need to fork an open dataset from the community to your Graviti workspace before processing the data.*

- Search datasets from the [open dataset](https://gas.graviti.com/open-datasets) catalog [📖](https://docs.graviti.com/guide/opendataset/get)
- Preview the data and annotations</br>
Grasp the data details with the Pharos visualization tool in advance to help you quickly understand a dataset and its semantic information.
<img width="800" alt="annotations" src="https://user-images.githubusercontent.com/92721051/163776918-d9555f16-9e0a-48c9-ac93-e7b6ec4f5ada.png">

- On the dataset page,  choose to fork the dataset in the 'Explore Dataset' drop-down menu.
<img width="800" alt="explore" src="https://user-images.githubusercontent.com/92721051/163776911-3c1b942d-0ab7-49c8-8400-6b91349bcd85.png">

- Then you will find the dataset on the 'Your Datasets' list
<img width="800" alt="yourdatasets" src="https://user-images.githubusercontent.com/92721051/163776902-22a1d05b-5b92-4984-86af-4b19b08b0367.png">

4. Prepare data

You could prepare your data for model training quickly by using the following functions. 

- **Filter** [📖](https://docs.graviti.com/guide/tensorbay/data/filter) 
- **Merge**[📖](https://tensorbay-python-sdk.graviti.com/en/stable/quick_start/examples/move_and_copy.html) 
- **Move and Copy** [📖](https://tensorbay-python-sdk.graviti.com/en/stable/quick_start/examples/move_and_copy.html)

5. Integrate with machine learning frameworks

Integrating with PyTorch, TensorFlow and more.

- **PyTorch** [📖](https://tensorbay-python-sdk.graviti.com/en/latest/integrations/pytorch.htm)

The typical method to integrate a dataset with PyTorch is to build a ‘Segment’ class derived from ‘torch.utils.data.Dataset’.

- **TensorFlow** [📖](https://tensorbay-python-sdk.graviti.com/en/latest/integrations/tensorflow.html)

The typical method to integrate a dataset with TensorFlow is to build a callable ‘Segment’ class.

- We recommend enabling [cache](https://tensorbay-python-sdk.graviti.com/en/stable/advanced_features/cache.html#enable-cache) for a better training experience. Sample code is as below (It requires enough local storage to load dataset)

```
from paddle.io import Dataloader,Dataset
from PIL import Image
from tensorbay.dataset import  Dataset as TensorBay Dataset

class DogsVSCatsSegment(Dataset):
##class for wrapping a DosVsCats segment

    def __init__(self, gas, segment_name, transfors):
        super().__inint__()
        self.dataset = TensorBayDataset('DogsVsCats', gas)
        self.dataset.enable_cache() ## launch cache
        self.segment = self.dataset{segment_name}
        self.category_to_index = self.dataset.catalog.clasification.get_category_to_index()
        self.transform = transform
        print(self.datasdt.cache_enabled) ## confirm if cached has been launched
```   
- Check [**the full tutorial**]((https://github.com/Graviti-AI/datasets/blob/main/QuickStart.md)) for advanced tools and techniques.

---

## Open datasets catolog 

These datasets are great for machine learning learners, researchers and engineers to train models for image classification, object detection, visual relationship detection, instance segmentation, and more. </br>
The [full list](https://gas.graviti.com/open-datasets) is available on Graviti Community. </br>
Please DO NOT modify this file directly. You could direct to the dataset page to contribute. 

[Datasets](https://github.com/Graviti-AI/datasets/tree/main/datasets) repo is a lightweight library of [![1,233](https://img.shields.io/badge/Datasets-1%2C233-green)](https://gas.graviti.com/open-datasets)  datasets in high quality. All are open source carrying a diverse range of tasks, annotation types, and sizes.</br>
Search by task types or keywords if you need a specific dataset. You could fork a dataset on dataset page and read data through [SDK](https://github.com/Graviti-AI/tensorbay-python-sdk). </br>
Popular tasks 
- Object Detection
- Classification
- Keypoints Detection
- Segmentation
- Pose Estimation
- ASR
- OCR

---
## Become a contributor

Contributions are welcomed and greatly appreciated. You can become a community contributor in many different ways, we value all forms of contribution including:

- Improve code
- Improve docs
- Report bugs
- Write blogs
- Give talks
- Provide ideas
- Answer questions

---

## Q&A

**Can I use these datasets for my project?**</br>
Sure! You're totally free to do so. You may check each license further on the dataset link. Refer to [TensorBay Python SDK](https://github.com/Graviti-AI/tensorbay-python-sdk#tensorbay-python-sdk) to read datasets via SDK.

**Can I add a dataset here?**</br>
Send us a pull request and we'll discuss.

---

## Join the community

To connect with all practitioners like you, join our [community discord](https://discord.gg/uJ3uJSsJ2X) for more communication.


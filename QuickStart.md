# Quick start on using open datasets

This repository accompanies Graviti's [open datasets guide](https://docs.graviti.com/guide/opendataset).

The Graviti community provides multiple developer toolkits including Python SDK, CLI and Open API, for quickly integrating open datasets with your machine learning pipeline.

- [Tensorbay Python SDK](https://tensorbay-python-sdk.graviti.com/en/stable/) is a python library to access our data platform and manage your datasets. </br>
- [CLI](https://tensorbay-python-sdk.graviti.com/en/stable/tensorbay_cli/getting_started_with_cli.html) is a tool to operate on datasets. It supports Windows, Linux, and Mac platforms.</br>

## 1. Sign up for an account

Go to [graviti.com](https://account.graviti.com/sign-up) to sign up. </br>
Get an AccessKey on [Graviti Developer Tools](https://gas.graviti.com/tensorbay/developer).</br>

> *An AccessKey is needed to authenticate identity when using TensorBay via SDK or CLI.* </br>
> *You have full permissions for the account. Please keep the key properly.*


## 2. Install Tensorbay Python SDK

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

## 3. Select an open dataset

> *You need to fork an open dataset from the community to your Graviti workspace before processing the data.*

- Search datasets from the [open dataset](https://gas.graviti.com/open-datasets) catalog [[Doc..](https://docs.graviti.com/guide/opendataset/get)]
- Preview the data and annotations</br>
Grasp the data details with the Pharos visualization tool in advance to help you quickly understand a dataset and its semantic information.
- On the dataset page,  choose to fork the dataset in the 'Explore Dataset' drop-down menu.
- Then you will find the dataset on the 'Your Datasets' list

## 4. Prepare data

You could prepare your data for model training quickly by using the following functions. 
### Filter[[Doc..](https://docs.graviti.com/guide/tensorbay/data/filter)] 
When the sheer volume of data gets out of hand, the advanced filtering options help you drill down to see the data you need. Click 'Manage Data' on the page of a forked dataset. Apply filtering options on the left bar and create a subset with the results.</br>
For example, you can filter out the data with traffic cones from an autonomous driving dataset. 

### Merge[[Doc..](https://tensorbay-python-sdk.graviti.com/en/stable/quick_start/examples/move_and_copy.html)] 
Create a dataset by merging different datasets through SDK.

### Move and Copy[[Doc..](https://tensorbay-python-sdk.graviti.com/en/stable/quick_start/examples/move_and_copy.html)]
Copy is supported within a dataset or between datasets. Moving is only supported within one dataset.

## 5. Integrate with machine learning frameworks

You've set up SDK configuration and data preparation. Congrats!</br>
Now we'll take MNIST Dataset as an example to show you the steps of integrating with PyTorch and TensorFlow.

### PyTorch
The typical method to integrate a dataset with PyTorch is to build a ‘Segment’ class derived from ‘torch.utils.data.Dataset’.
```
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from tensorbay import GAS
from tensorbay.dataset import Dataset as TensorBayDataset


class MNISTSegment(Dataset):
    """class for wrapping a MNIST segment."""

    def __init__(self, gas, segment_name, transform):
        super().__init__()
        self.dataset = TensorBayDataset("MNIST", gas)
        self.segment = self.dataset[segment_name]
        self.category_to_index = self.dataset.catalog.classification.get_category_to_index()
        self.transform = transform

    def __len__(self):
        return len(self.segment)

    def __getitem__(self, idx):
        data = self.segment[idx]
        with data.open() as fp:
            image_tensor = self.transform(Image.open(fp))

        return image_tensor, self.category_to_index[data.label.classification.category]
````
Using the following code to create a PyTorch DataLoader and run it.
```
ACCESS_KEY = "Accesskey-*****"

to_tensor = transforms.ToTensor()
normalization = transforms.Normalize(mean=[0.485], std=[0.229])
my_transforms = transforms.Compose([to_tensor, normalization])

train_segment = MNISTSegment(GAS(ACCESS_KEY), segment_name="train", transform=my_transforms)
train_dataloader = DataLoader(train_segment, batch_size=4, shuffle=True, num_workers=4)

for index, (image, label) in enumerate(train_dataloader):
    print(f"{index}: {label}")
```

### TensorFlow

The typical method to integrate a dataset with TensorFlow is to build a callable ‘Segment’ class.
```
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.data import Dataset

from tensorbay import GAS
from tensorbay.dataset import Dataset as TensorBayDataset


class MNISTSegment:
    """class for wrapping a MNIST segment."""

    def __init__(self, gas, segment_name):
        self.dataset = TensorBayDataset("MNIST", gas)
        self.segment = self.dataset[segment_name]
        self.category_to_index = self.dataset.catalog.classification.get_category_to_index()

    def __call__(self):
        """Yield an image and its corresponding label.

        Yields:
            image_tensor: the tensorflow sensor of the image.
            category_tensor: the tensorflow sensor of the category.

        """
        for data in self.segment:
            with data.open() as fp:
                image_tensor = tf.convert_to_tensor(
                    np.array(Image.open(fp)) / 255, dtype=tf.float32
                )
            category = self.category_to_index[data.label.classification.category]
            category_tensor = tf.convert_to_tensor(category, dtype=tf.int32)
            yield image_tensor, category_tensor
```
Using the following code to create a TensorFlow Dataloader and run it.
```
ACCESS_KEY = "Accesskey-*****"

dataset = Dataset.from_generator(
    MNISTSegment(GAS(ACCESS_KEY), "train"),
    output_signature=(
        tf.TensorSpec(shape=(28, 28), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    ),
).batch(4)

for index, (image, label) in enumerate(dataset):
    print(f"{index}: {label}")
```

We recommend enabling [cache](https://tensorbay-python-sdk.graviti.com/en/stable/advanced_features/cache.html#enable-cache) for a better training experience. Sample code is as below (It requires enough local storage to load dataset)
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

## 6. Evaluation

[Sextant](https://gas.graviti.com/apps/application/Sextant) is an efficient model evaluation tool available at Graviti marketplace, supporting data visualization and filtering evaluation data by metrics.


### Prepare
- [Organize dataset](https://tensorbay-python-sdk.graviti.com/en/stable/features/dataset_management.html#organize-dataset)
Transfer a dataset into a uniform TensorBay dataset structure
- [Upload the datasets](https://tensorbay-python-sdk.graviti.com/en/stable/features/dataset_management.html#upload-dataset) via SDK
- Create a project via [SDK](https://tensorbay-python-sdk.graviti.com/en/stable/applications/sextant.html) or through [Web UI](https://gas.graviti.com/tensorbay/evaluation-list).

### Select a benchmark and metrics
- Benchmark can be created via [Web UI](https://gas.graviti.com/tensorbay/evaluation-list).
- Set up your [custom metrics](https://docs.graviti.com/apps/sextant/metrics).

### Data/Model Evaluation
Load a model from Github to start an evaluation.
❓ [How to prepare a suitable algorithm model for Sextant](https://docs.graviti.com/apps/sextant/start-to-evaluate#how-to-prepare-a-suitable-algorithm-model-for-sextant)

---
### Leave a comment in the [Discussion](https://github.com/Graviti-AI/datasets/discussions) if you have any questions.
### See you in the Community! 🤗

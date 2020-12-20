# Deep-Learning
Deep learning (also known as deep structured learning) is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised. Deep-learning architectures such as deep neural networks, deep belief networks, recurrent neural networks and convolutional neural networks have been applied to fields including computer vision, machine vision, speech recognition, natural language processing, audio recognition, social network filtering, machine translation, bioinformatics, drug design, medical image analysis, material inspection and board game programs, where they have produced results comparable to and in some cases surpassing human expert performance. Artificial neural networks (ANNs) were inspired by information processing and distributed communication nodes in biological systems. ANNs have various differences from biological brains. Specifically, neural networks tend to be static and symbolic, while the biological brain of most living organisms is dynamic (plastic) and analog. The adjective "deep" in deep learning comes from the use of multiple layers in the network. Early work showed that a linear perceptron cannot be a universal classifier, and then that a network with a nonpolynomial activation function with one hidden layer of unbounded width can on the other hand so be. Deep learning is a modern variation which is concerned with an unbounded number of layers of bounded size, which permits practical application and optimized implementation, while retaining theoretical universality under mild conditions. In deep learning the layers are also permitted to be heterogeneous and to deviate widely from biologically informed connectionist models, for the sake of efficiency, trainability and understandability, whence the "structured" part.

# Online Website:

![image](https://user-images.githubusercontent.com/74346775/102711716-93371b80-42dd-11eb-9c90-91accc4fc354.png)



In this post I will providing information about the various service that gives us the computation power to us for training models.
Google Colab
Kaggel Kernel
Jupyter Notebook on GCP
Amazon SageMaker
Azure Notebooks
1)Google Colab
Colaboratory is a google research project created to help disseminate machine learning education and research. Colaboratory (colab) provides free Jupyter notebook environment that requires no setup and runs entirely in the cloud.It comes pre-installed with most of the machine learning libraries, it acts as perfect place where you can plug and play and try out stuff where dependency and compute is not an issue.
The notebooks are connected to your google drive, so you can acess it any time you want,and also upload or download notebook from github.
GPU and TPU enabling
First, you’ll need to enable GPU or TPU for the notebook.
Navigate to Edit→Notebook Settings, and select TPU from the Hardware Accelerator drop-down .
Image for post
code to check whether TPU is enabled
import os
import pprint
import tensorflow as tf
if ‘COLAB_TPU_ADDR’ not in os.environ:
 print(‘ERROR: Not connected to a TPU runtime; please see the first cell in this notebook for instructions!’)
else:
 tpu_address = ‘grpc://’ + os.environ[‘COLAB_TPU_ADDR’]
 print (‘TPU address is’, tpu_address)
with tf.Session(tpu_address) as session:
 devices = session.list_devices()
 
 print(‘TPU devices:’)
 pprint.pprint(devices)
Installing libraries
Colab comes with most of ml libraries installed,but you can also add libraries easily which are not pre-installed.
Colab supports both the pip and apt package managers.
!pip install torch
apt command
!apt-get install graphviz -y
both commands work in colab, dont forget the ! (exclamatory) before the command.
Uploading Datasets
There are many ways to upload datasets to the notebook
One can upload files from the local machine.
Upload files from google drive
One can also directly upload datasets from kaggle
Code to upload from local
from google.colab import files
uploaded = files.upload()
you can browse and select the file.
Upload files from google drive
PyDrive library is used to upload and files from google drive
!pip install -U -q PyDrive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
# 1. Authenticate and create the PyDrive client.
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)
# PyDrive reference:
# https://gsuitedevs.github.io/PyDrive/docs/build/html/index.html
# 2. Create & upload a file text file.
uploaded = drive.CreateFile({'title': 'Sample upload.txt'})
uploaded.SetContentString('Sample upload file content')
uploaded.Upload()
print('Uploaded file with ID {}'.format(uploaded.get('id')))
# 3. Load a file by ID and print its contents.
downloaded = drive.CreateFile({'id': uploaded.get('id')})
print('Downloaded content "{}"'.format(downloaded.GetContentString()))
You can get id of the file you want to upload,and use the above code.
For more resource to upload files from google services.
Uploading dataset from kaggle
We need to install kaggle api and add authentication json file which you can download from kaggle website(API_TOKEN).
Image for post
!pip install kaggle
upload the json file to the notebook by, uploading file from the local machine.
create a /.kaggle directory
!mkdir -p ~/.kaggle
copy the json file to the kaggle directory
change the file permision
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
Now you can use command to download any dataset from kaggle
kaggle datasets download -d lazyjustin/pubgplayerstats
Now you can use the below to download competition dataset from kaggle,but for that you have to participate in the competition.
!kaggle competitions download -c tgs-salt-identification-challenge
Image for post
You can train and run fashion_mnist online without any dependency here.
Colab is a great tool for everyone who are interested in machine learning,all the educational resource and code snippets to use colab is provide in the official website itself with notebook examples.
2)Kaggle Kernels
Kaggle Kernels is a cloud computational environment that enables reproducible and collaborative analysis.
One can run both Python and R code in kaggle kernel
Kaggle Kernel runs in a remote computational environment. They provide the hardware needed.
At time of writing, each kernel editing session is provided with the following resources:
CPU Specifications
4 CPU cores
17 Gigabytes of RAM
6 hours execution time
5 Gigabytes of auto-saved disk space (/kaggle/working)
16 Gigabytes of temporary, scratchpad disk space (outside /kaggle/working)
GPU Specifications
2 CPU cores
14 Gigabytes of RAM
Kernels in action
Once we create an account at kaggle.com, we can choose a dataset to play with and spin up a new kernel,with just a few clicks.
Click on create new kernel
Image for post
You will be having jupyter notebook up and running.At the bottom you will be having the console which you can use,and at the right side you will be having various options like
VERSION
When you Commit & Run a kernel, you execute the kernel from top to bottom in a separate session from your interactive session. Once it finishes, you will have generated a new kernel version. A kernel version is a snapshot of your work including your compiled code, log files, output files, data sources, and more. The latest kernel version of your kernel is what is shown to users in the kernel viewer.
Data Environment
When you create a kernel for a dataset ,the dataset will be preloaded into the notebook in the input directory
../input/
you can also click on add data source ,to add other datasets
Image for post
Settings
Sharing: you can keep your kernel private,or you can also make it public so that others can learn from your kernel.
Adding GPU:You can add a single NVIDIA Tesla K80 to your kernel. One of the major benefits to using Kernels as opposed to a local machine or your own VM is that the Kernels environment is already pre-configured with GPU-ready software and packages which can be time consuming and frustrating to set-up.To add a GPU, navigate to the “Settings” pane from the Kernel editor and click the “Enable GPU” option.
Custom pakage:The kernel has the default pakages,if you need any other pakage you can easily add it by the following ways
Just enter the libarary name ,kaggle will download it for you.
Image for post
Enter the user name/repo name
Image for post
both methods work fine in adding custom pakages.
Kaggle acts as a perfect platform for both providing data,and also the compute to work with the great data provided.It also host various competition one can experiment it out to improve one’s skill set.
For more resource regarding kaggle link here. If you are new to kaggle you should definitely try the titanic dataset it comes with awesome tutorials.
Other resources regarding kaggle ,colab and machine learning follow Siraj Raval, and Yufeng G.

# INSTALLING AND TESTING GPU-ENABLED TENSORFLOW ON A GOOGLE CLOUD PLATFORM VIRTUAL MACHINE
This is an updated and streamlined version of [this excellent tutorial](https://hackernoon.com/launch-a-gpu-backed-google-compute-engine-instance-and-setup-tensorflow-keras-and-jupyter-902369ed5272) by Steve Domin.

It will guide you through the minimal steps to spin up a VM in the Google cloud, install the latest version of tensorflow (1.6.0 as of 3/5/2018), and get it to recognize the GPU and rapidly train a simple network.

# Note 1: Remember to stop your instance you're done with it so you don't continue getting charged
# Note 2: Setting up your GPU quota can take a day or two, so plan accordingly
# Note 3: This tutorial uses Python 3, TensorFlow 1.6.0, CUDA 9.0, and CuDNN 7.0.4 for CUDA 9.0
And all software versions must be compatible with each other for the install to be successful, so only change them if you know exactly what you're doing.
# Note 4: The installation is quite involved, but if you save a disk image once things are working you'll only have to do it once

Okay, onwards:

# Set up a virtual machine on Google Cloud Platform

## Get an account
First make a Google Cloud account [here](https://cloud.google.com) if you don't already have one.

You can start with the free trial, which comes with $300 of free credits.

To use a GPU, however, you'll a paid account, so go to the billing page and select the option to upgrade your account.

Note that you won't get charged until after you've spent your $300, so everything is still free.

## Increase your GPU quota
Next, request a quota increase (which you need to do in order to use GPUs) by going to Compute Engine --> Quotas, and clicking through to your list of quotas.

There are a lot to choose from, so click on the dropdown menu under Metrics click on None, then type NVIDIA in the search bar and select NVIDIA K80 GPUs.

Check the box next to one of the rows with an appropriate region. A good option is us-west1.

With the box checked, click EDIT QUOTAS, enter your name, email, and phone, and then type in 1 for the requested limit, and describe why you are requesting a GPU (e.g. "training deep nets for computer vision research").

Click Submit request, and you'll get an email confirmation explaining that your request will be reviewed over the next couple of days.

You may also be asked to pay a ~$70 confirmation charge to ensure you're not a robot, which will appear as additional credits on your account. I found this a bit strange but paid anyway since I assumed I'd probably use it eventually once my free credits ran out. You might not have to do this, however, as my quota increase request was approved even before my payment went through.

## Configure your VM instance
Once your GPU has been increased to 1, go to Compute Engine --> VM instances and click CREATE INSTANCE to spin up a new VM instance.

Name your instance (e.g. 'tf-demo').

For Zone select the zone where you increased your GPU quota (e.g. us-west1-b). If you select the wrong zone you won't be able to use any GPUs.

Under Machine type select 8 vCPUs with 30 GB memory.

Still under Machine type, click Customize, which will display more options.

Click on GPUs and change the number of GPUs to 1 and the GPU type to NVIDIA Tesla K80.

Leave the box under Container unchecked. No need for that now.

Under Boot disk click Change and select Ubuntu 16.04 LTS and change the Standard persistent disk size to 64 GB.

If you want to use run a Jupyter notebook server on your instance at some point check the boxes next to Allow HTTP traffic and Allow HTTPS traffic under Firewall. 

If you only planning on messing around for a little bit or expect to use your instance for less than 24 hours, click Management, disks, networking, SSH keys to show more options and change Preemptibility from Off (recommended) to On. This is cheaper, but leave preemptibility Off if you're doing anything more serious.

## Create your instance

Once you've configured everything click on Create. Not so bad, right?

## Make your VM's external IP address static

Go to VPC Network --> External IP addresses.

On the row corresponding to your new instance, click on the arrow next to Ephemeral and select Static.

Name the address whatever you want. I chose tf-demo, for instance.

## Connect to your VM

Go back to Compute Engine --> VM Instances.

Under Connect, click on the arrow next to SSH and select "Open in browser window", and you'll be dropped into a browser-based SSH session into your VM instance. Sometimes you have to click twice to get this to work for some reason.

Your username will be your Google username or something similar, and it will be identical to how it'd be if you just bought  brand new linux computer with an NVIDIA Tesla K80 GPU installed in it and logged in the for the first time.

# Install basic dependencies

Run the following commands to install some preliminaries:

```
$ sudo apt-get update
$ sudo apt-get upgrade
$ sudo apt-get install -y build-essential
```

# Install miniconda

Miniconda is a lightweight version of Anaconda, which is a Python installation with lots of standard scientific and numerical libraries installed. 

Download miniconda into your /tmp directory and install it using:

```
$ cd /tmp
$ curl -O https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
$ bash Miniconda3-latest-Linux-x84_64.sh
```

Follow the install instructions and make sure to type `yes` at the final prompt asking `Do you wish the installer to prepend the Miniconda3 install location
to PATH in your /home/<username>/.bashrc ? [yes|no]`

Source your `.bashrc` file so you can don't have to start a new SSH session, by typing:

`$ source ~/.bashrc`

Test your install by typing:

`$ conda list`

which will list installed packages.

# Install CUDA toolkit

The CUDA toolkit is the NVIDA software that allows your computer to connect to the GPUs. First check that your GPU is properly installed:

```
$ lspci | grep -i nvidia
00:04.0 3D controller: NVIDIA Corporation GK210GL [Tesla K80] (rev a1)
```

## Download the CUDA installation software

Go to the NVIDA site [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads), click on Legacy Releases, and click on CUDA Toolkit 9.0 [Sept 2017].

On the next page select: Linux, x86_64, Ubuntu, 16.04, deb (network).

Right click on the Download link (which should be a couple kB), and copy it.

Move to your temp directory and download the CUDA toolkit:

```
cd /tmp
curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
```

Note: the line starting with curl is all one line (`$ curl -O https://link.I.copied.to.the.clipboard.deb`).

## Install the toolkit
Run 

```
$ sudo dpkg -i cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
$ sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
$ sudo apt-get update
$ sudo apt-get install -y cuda-9-0
```

This will take several minutes to complete, since it's installing 3.5 GB of software.

Note, make sure the last argument on the last line is `cuda-9-0`, not simply `cuda`, as shown on the NVIDIA website (only specifying `cuda` will make the latest release get downloaded (9.1 at the time of writing), which tensorflow 1.6.0 won't like.

## Set environment variables
These are basically just shortcuts so programs like tensorflow know where to look for CUDA. Basically, we're specifying where CUDA lives, where its lib64 library is, and then adding CUDA's `bin` directory to our `PATH`.

Copy and paste the following into the command line and hit ENTER:

```
cat <<EOF >> ~/.bashrc
export CUDA_HOME=/usr/local/cuda-9.0
export LD_LIBRARY_PATH=\${CUDA_HOME}/lib64
export PATH=\${CUDA_HOME}/bin:\${PATH}
EOF
```

Run:

`source ~/.bashrc`

so you can access these variables immediately.

## Make sure the install was successful

Run 

`$ nvidia-smi`

which should yield an output that looks something like this:

```
Mon Mar 5 14:58:47 2018
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 375.66                 Driver Version: 375.66                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla K80           Off  | 0000:00:04.0     Off |                    0 |
| N/A   33C    P0    57W / 149W |      0MiB / 11439MiB |    100%      Default |
+-------------------------------+----------------------+----------------------+
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID  Type  Process name                               Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

That is, you should see the Tesla K80 GPU that we attached to our VM instance. If you don't see it, make sure you actually increased your quota, that you didn't mix up the zones, and that you indeed added a Tesla K80 GPU when configuring your instance.

# Install cuDNN v7.0.4 for CUDA 9.0
This is a bit tricky, since we will be forced to download this library onto our local machine and then transfer it to our VM.

## Make an NVIDIA developer account
On your local machine go to the NVIDIA developer program [website](https://developer.nvidia.com/developer-program), click "Join Now", and follow the instructions.

## Download cuDNN to your local computer
This is the CUDA library for deep neural networks (cuDNN). Go to the downloads page [here](https://developer.nvidia.com/rdp/cudnn-download) and make sure to select "cuDNN v7.0.4 (Nov 13, 2017), for CUDA 9.0", and click on "cuDNN v7.0.4 Library for Linux" to download it locally.

## Download Google Cloud Platform SDK (software development kit)
Google's CLOUD SDK command-line tools provide a pretty simple way of transferring files between local machines and your VM instances.

Download it at [https://cloud.google.com/sdk/](https://cloud.google.com/sdk/) and unzip it.

To install it, on your local machine, cd into the unzipped directory and run:

`./install.sh`

which sets up your login/connection info and adds the location of the unzipped directory to your `PATH`.

Open a new terminal *on your local machine* and run 

`gcloud --version`

to ensure it was installed correctly.

## Transfer cuDNN to your VM
*On your local machine* change to the directory containing your cuDNN download, e.g.,

`cd ~/Downloads`

and run the following to transfer cuDNN to your VM:

`gcloud compute scp cudnn-9.0-linux-x64-v7.tgz <username>@<instance-name>:/tmp`

Note: replace `<username>` and `<instance-name>` with the appropriate values for your VM. E.g., if your username is `arthurdent` and your instance name was `arthurs-tf-demo` you would run:

`gcloud compute scp cudnn-9.0-linux-x64-v7.tgz arthurdent@arthurs-tf-demo:/tmp`

This copies the cuDNN download to the `/tmp` directory on your VM instance. It'll take a few seconds and you should see the percentage increase to 100% as the transfer proceeds.

## Install cuDNN on your VM
*On your VM instance* (i.e. after you've connected to it by opening an SSH session from the GCP console) unzip the cuDNN download:

```
$ cd /tmp
$ tar xvzf cudnn-9.0-linux-x64-v7.tgz
```

copy the files into the directories where CUDA lives:

```
$ sudo cp -P cuda/include/cudnn.h $CUDA_HOME/include
$ sudo cp -P cuda/lib64/libcudnn* $CUDA_HOME/lib64
```

and change their permissions:

```
$ sudo chmod u+w $CUDA_HOME/include/cudnn.h
$ sudo chmod a+r $CUDA_HOME/lib64/libcudnn*
```

Note 1: if the environment variable `$CUDA_HOME` isn't recognized you probably forgot to run `$ source ~/.bashrc`, so do that.

Note 2: installing CUDA libraries is actually quite easy—all we actually did was copy the header (cudnn.h) and library files (libcudnn*) into the CUDA directory, no building or futzing around required.

# Take a break

Whew. Still hanging in there? it's okay, we're getting close.

# Set up a virtual environment with GPU-enabled tensorflow

On your VM, go back to your home directory:

`$ cd ~`

and clone this repository:

`$ git clone https://github.com/rkp8000/gcp_tf_gpu_installation`

Move into it and create a new virtual environment from the environment configuration file:

```
$ cd gcp_tf_gpu_installation
$ conda env create -f environment-gpu.yml
```

This will create a new virtual environment named "tf_demo" and install a bunch of libraries into it (including tensorflow). It'll probably take a few minutes.

## Activate the environment

`$ source activate tf_demo`

From now on, this is something you'll need to do every time you start a new SSH session, since it tells the system to use the Python environment where all our goodies are installed.

## Test TensorFlow

Start a Python interpreter, import TensorFlow, and print "Hello, World":

```
(tf_demo) $ python
>>> import tensorflow as tf
>>> hello = tf.constant('Hello, World')
>>> sess = tf.Session()
>>> print(sess.run(hello))
```

You may get a couple warnings, but if you get the output `b'Hello, TensorFlow!'` then it worked.

If you get an error or the import hangs, close your SSH connection, establishing a new one, activating your environment, and trying again. If that doesn't fix it, check out [https://www.tensorflow.org/install/install_linux#common_installation_problems](https://www.tensorflow.org/install/install_linux#common_installation_problems) for common tensorflow installation problems.

## Test that TensorFlow recognizes the GPU

While in the "tf_demo" environment, start the Python interpreter again if it's not still open, and run the following:

```
>>> from tensorflow.python.client import device_lib
>>> local_device_protos = device_lib.list_local_devices()
>>> print([x.name for x in local_device_protos if x.device_type == 'GPU'])
```

If you see your GPU listed, hooray. If you see an empty list "[]", then the opposite of hooray. If the import hangs, close the interpreter/SSH session and try once more.

## Train a simple network with the GPU

Move back into your cloned version of this repository and run the test script "tf_test.py":

```
(tf_demo) $ cd ~/gcp_tf_gpu_installation
(tf_demo) $ python tf_test.py
```

The script will take a minute to download and format the MNIST handwritten digits dataset, then training will begin. It should only take a minute or two to run through all 10 epochs once training begins, and you should see an output like the following:

```
Downloading MNIST data...
Successfully downloaded train-images-idx3-ubyte.gz 9912422 bytes.
Extracting MNIST_data/train-images-idx3-ubyte.gz
Successfully downloaded train-labels-idx1-ubyte.gz 28881 bytes.
Extracting MNIST_data/train-labels-idx1-ubyte.gz
Successfully downloaded t10k-images-idx3-ubyte.gz 1648877 bytes.
Extracting MNIST_data/t10k-images-idx3-ubyte.gz
Successfully downloaded t10k-labels-idx1-ubyte.gz 4542 bytes.
Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
Image Shape: (28, 28, 1)
Training Set:   55000 samples
Validation Set: 5000 samples
Test Set:       10000 samples
Updated Image Shape: (32, 32, 1)
WARNING:tensorflow:From tf_test.py:101: softmax_cross_entropy_with_logits (from tensorflow.python.ops.nn_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Future major versions of TensorFlow will allow gradients to flow
into the labels input on backprop by default.
See tf.nn.softmax_cross_entropy_with_logits_v2.
2018-03-06 01:29:24.038497: I tensorflow/core/platform/cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2018-03-06 01:29:24.124551: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:898] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2018-03-06 01:29:24.124873: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1212] Found device 0 with properties: 
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 0000:00:04.0
totalMemory: 11.17GiB freeMemory: 11.09GiB
2018-03-06 01:29:24.124900: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1312] Adding visible gpu devices: 0
2018-03-06 01:29:24.365232: I tensorflow/core/common_runtime/gpu/gpu_device.cc:993] Creating TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10750 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 0000:00:04
.0, compute capability: 3.7)
Training...
EPOCH 1 ...
Validation Accuracy = 0.971
EPOCH 2 ...
Validation Accuracy = 0.978
EPOCH 3 ...
Validation Accuracy = 0.983
EPOCH 4 ...
Validation Accuracy = 0.987
EPOCH 5 ...
Validation Accuracy = 0.988
EPOCH 6 ...
Validation Accuracy = 0.988
EPOCH 7 ...
Validation Accuracy = 0.990
EPOCH 8 ...
Validation Accuracy = 0.990
EPOCH 9 ...
Validation Accuracy = 0.989
EPOCH 10 ...
Validation Accuracy = 0.992
Model saved
2018-03-06 01:29:48.424147: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1312] Adding visible gpu devices: 0
2018-03-06 01:29:48.424292: I tensorflow/core/common_runtime/gpu/gpu_device.cc:993] Creating TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 260 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 0000:00:04.0
, compute capability: 3.7)
Test Accuracy = 0.990
```

Phew.

# Create a disk image

To not have to go through that again every time you spin up a new VM instance, it will be of great benefit to create an image of the disk you so carefully put together.

To do this, exit your browser-based SSH session, go to your [GCP console](https://console.cloud.google.com) and go to Compute Engine --> VM instances, click on the three vertical dots to the right of your instance, and click Stop.

After a couple minutes the instance will stop.

Next, go to Compute Engine --> Images and click "[+] CREATE IMAGE" at the top.

Name your instance something like 'tf-gpu-installation', give it a description, set Encryption to 'Automatic (recommended)', set Source to "Disk", and under "Source disk" select your instance.

Click Create, and after a few minutes you'll have a brand new custom disk image.

This means that the next time you make a new VM, instead of installing everything by hand again, under "Boot disk" you can click "Change", go to the "Custom images" tab, and select your 'tf-gpu-installation' image. Set the rest of the VM configuration parameters (e.g., Zone, vCPUs, GPUs, etc), and click create, and the instance will be created in the same state as the one you just painstakingly created.

# Using Jupyter

If you're a Jupyter fan (which you should be) it's very easy to launch a Jupyter notebook server from your VM instance in the Google cloud (especially compared to installing CUDA/TensorFlow). Basically you just need to configure a couple things, make a new firewall rule in your GCP console, and make sure your instance has HTTP/HTTPS traffic enabled. For details, see [this great tutorial](https://towardsdatascience.com/running-jupyter-notebook-in-google-cloud-platform-in-15-min-61e16da34d52) by Amulya Aankul.

# Sources

In no way did I figure all this out on my own. I basically just copied all the key components from [Steve Domin's article](https://hackernoon.com/launch-a-gpu-backed-google-compute-engine-instance-and-setup-tensorflow-keras-and-jupyter-902369ed5272) and changed all the software versions to be compatible with the latest tensorflow (v1.6.0).

The environment-gpu.yml file and tf_test.py are from the Udacity Github repos [https://github.com/udacity/CarND-Term1-Starter-Kit](https://github.com/udacity/CarND-Term1-Starter-Kit) and [https://github.com/udacity/CarND-LeNet-Lab](https://github.com/udacity/CarND-LeNet-Lab) with a couple very minor changes.

Happy cloud computing.

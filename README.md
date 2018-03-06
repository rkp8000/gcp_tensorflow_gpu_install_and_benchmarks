# Installing and testing GPU-enabled tensorflow on a Google Cloud Platform virtual machine
This is an updated and streamlined version of [this excellent tutorial](https://hackernoon.com/launch-a-gpu-backed-google-compute-engine-instance-and-setup-tensorflow-keras-and-jupyter-902369ed5272) by Steve Domin.

It will guide you through the minimal steps to spin up a VM in the Google cloud, install the latest version of tensorflow (1.6.0 as of 3/5/2018), and get it to recognize the GPU and rapidly train a simple network.

Note 1:
# REMEMBER TO STOP YOUR INSTANCE WHEN YOU'RE DONE USING IT, OTHERWISE YOU WILL CONTINUE GETTING CHARGED
Note 2:
# SETTING UP YOUR GPU QUOTAS CAN TAKE A DAY OR TWO, SO PLAN ACCORDINGLY
Note 3:
# THE INSTALLATION IS FAIRLY INVOLVED, BUT YOU ONLY HAVE TO DO IT ONCE, IF YOU SAVE A DISK IMAGE ONCE YOU GET THINGS WORKING

Okay, onwards:

## Set up a virtual machine on Google Cloud Platform

### Get an account
First make a Google Cloud account [here](https://cloud.google.com) if you don't already have one.

You can start with the free trial, which comes with $300 of free credits.

To use a GPU, however, you'll a paid account, so go to the billing page and select the option to upgrade your account.

Note that you won't get charged until after you've spent your $300, so everything is still free.

### Increase your GPU quota
Next, request a quota increase (which you need to do in order to use GPUs) by going to Compute Engine --> Quotas, and clicking through to your list of quotas.

There are a lot to choose from, so click on the dropdown menu under Metrics click on None, then type NVIDIA in the search bar and select NVIDIA K80 GPUs.

Check the box next to one of the rows with an appropriate region. A good option is us-west1.

With the box checked, click EDIT QUOTAS, enter your name, email, and phone, and then type in 1 for the requested limit, and describe why you are requesting a GPU (e.g. "training deep nets for computer vision research").

Click Submit request, and you'll get an email confirmation explaining that your request will be reviewed over the next couple of days.

You may also be asked to pay a ~$70 confirmation charge to ensure you're not a robot, which will appear as additional credits on your account. I found this a bit strange but paid anyway since I assumed I'd probably use it eventually once my free credits ran out. You might not have to do this, however, as my quota increase request was approved even before my payment went through.

### Configure your VM instance
Once your GPU has been increased to 1, go to Compute Engine --> VM instances and click CREATE INSTANCE to spin up a new VM instance.

Name your instance (e.g. 'tf-demo').

For Zone select the zone where you increased your GPU quota (e.g. us-west1-b). If you select the wrong zone you won't be able to use any GPUs.

Under Machine type select 8 vCPUs with 30 GB memory.

Still under Machine type, clikc Customize, which will display more options.

Click on GPUs and change the number of GPUs to 1 and the GPU type to NVIDIA Tesla K80.

Leave the box under Container unchecked. No need for that now.

Under Boot disk click Change and select Ubuntu 16.04 LTS and change the Standard persistent disk size to 64 GB.

If you want to use run a Jupyter notebook server on your instance at some point check the boxes next to Allow HTTP traffic and Allow HTTPS traffic under Firewall. 

If you only planning on messing around for a little bit or expect to use your instance for less than 24 hours, click Management, disks, networking, SSH keys to show more options and change Preemptibility from Off (recommended) to On. This is cheaper, but leave preemptibility Off if you're doing anything more serious.

### Create your instance

Once you've configured everything click on Create. Not so bad, right?

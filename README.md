# Neptune-Catalyst


In this repo you will learn how to save all you catalyst experiment metadata to Neptune at every level of your experiment.

I have to examples to show you:
 - A basic example -> basic.py
 - And a more robust and customized example-> complex.py


## Basic Example

In this example we are going to log data at 3 levels:
  - Experiment 
  - Loader 
  - Batch
  
### Step 1

Install catalyst, Neptune and psutil & Import libraries 

`pip install catalyst neptune-client psutils`

### Step 2 
Define your hparams, data loaders, model, loss function, optimizer 


### Step 3 
Define you runner and pass arguments to your train() method to setup up your training loop.
One super exciting thing is that the new version of catalyst log_artifacts was introduced and our logger has support for it.

### Step 4
Let’s explore results in Neptune UI
Thanks to our hierarchical structure feature in our client library we can do a 1-to-1 mapping with your catalyst experiment


## Complex example
Same steps as the basic example but here we do 2 things differently.

We create a callback that is going to help us log metadata at different levels of the runner:

**experiment level**:
 - on end: log best model

**stage level**:
 - on start: if "train_frozen" stage, log sample images
 - on end: if "train_unfrozen" stage, log mp4 file

**epoch level**:
 - on start: log audio file

**loader level**:
 - on end: log gif


We customize our runner to showcase how to log metadata at different stages:
 - Frozen
 - Unfrozen



Now, you can explore results in Neptune UI 
	
There you go, you learned how to:

 - connect Neptune to your catalyst experiment 
 - Customize your catalyst code to log metadata at different stages
 - You also learned how to use Neptune to visualize, organize and analyze your experiments metadata.

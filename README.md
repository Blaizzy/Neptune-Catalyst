# Neptune-Catalyst


In this repo you will learn how to save all you catalyst experiment metadata to Neptune at every 'scope' of your experiment.

I have to examples to show you:
 - A basic example -> basic.py
 - And a more robust and customized example-> complex.py


## [Basic Example](https://github.com/Blaizzy/Neptune-Catalyst/blob/master/basic.py)

In this example we are going to log data at 3 scopes:
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
In order to add log data to Neptune UI you just add neptune logger as one of the arguments and run your code - that's it.
```python
my_runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    loggers={
        "neptune": dl.NeptuneLogger(
            project="common/example-project-catalyst",
            tags=["datafest", "basic"],
            name="data-fest",
        )
    },
    loaders=loaders,
    num_epochs=5,
    callbacks=[
        dl.AccuracyCallback(
            input_key="logits",
            target_key="targets",
            topk_args=[1]
        ),
    ],
    hparams=my_hparams,
    logdir="./logs",
    valid_loader="validation",
    valid_metric="loss",
    minimize_valid_metric=True,
)
```


One super exciting thing is that the new version of catalyst log_artifacts was introduced and our logger has support for it.

```python
my_runner.log_artifact(
    path_to_artifact="./logs/checkpoints/best.pth",
    tag="best_model",
    scope="experiment"
)
```

### Step 4
Now you explore results in Neptune UI.

Just follow the link to basic persisted view: https://app.neptune.ai/o/common/org/example-project-catalyst/experiments?split=tbl&dash=leaderboard&viewId=d8ced6bb-61c0-48a2-97e2-bf46fb6e4fdb

## [Complex example](https://github.com/Blaizzy/Neptune-Catalyst/blob/master/complex.py)
Same steps as the basic example but here we do 2 things differently.

We create a callback that is going to help us log metadata at different scopes of the runner:

**experiment scope**:
 - on end: log best model

**stage scope**:
 - on start: if "train_frozen" stage, log sample images
 - on end: if "train_unfrozen" stage, log mp4 file

**epoch scope**:
 - on start: log audio file

**loader scope**:
 - on end: log gif


We customize our runner to showcase how to log metadata at different stages:
 - Frozen
 - Unfrozen



Now you can explore results in Neptune UI.
Just follow the link to complex persisted view:
https://app.neptune.ai/o/common/org/example-project-catalyst/experiments?split=tbl&dash=leaderboard&viewId=b658dc92-b4c6-4a3f-bece-5191600a5c80
	
# Things to remember
There you go, you learned how to:

 - connect Neptune to your catalyst experiment 
 - Customize your catalyst code to log metadata at different stages
 - You also learned how to use Neptune to visualize, organize and analyze your experiments metadata.

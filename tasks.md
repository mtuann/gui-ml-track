ML/DL task:
Write a simple PyTorch model to solve the MNIST challenge

Build a UI for Experiment Management system: design and build a simple web interface
allowing users to tune the above DL task as a blackbox.
- Define some hype parameters for users to tune. Allow user to run several jobs
with different hyper-parameters with a simple click
- Display the progress of currently running jobs
- Display the results of all finished jobs,
o Experiment should be sort-able by a pre-defined metrics (for example
accuracy, run time..) for ease of comparison

- Allow resuming of UI (user can close and open the web browser and still can see
the current state of experiments)
- Adding jobs should be available (for example first set was to grid-search on
learning_rate parameter, then 2 nd set was to grid_search on drop_out or any
other hyper-parameters)
o Should check if exactly same job has been run

Note: the test focuses on the management system and not on the DL task. Simply refer
to any available tutorial on the Internet for a minimal runnable instance.

Note: it is strongly advised that candidate shall not mention WorldQuant when publishing on Github or other platforms.
# List of tasks:
- [] Write a simple PyTorch model to solve the MNIST ( 1 file 50 lines)
    - function training(dataset, model, optimizer, criterion, epochs, batch_size, device)

UI:
- Submit job: backend receive --> append queue -> pick up job -> run job -> update status
- UI: 5s refresh to get status
- streamlit/ dash
    Prameters:
    {
        "learning_rate": [0.01, 0.1, 0.001],
        "batch_size": [32, 64, 128],
        "epochs": [10, 20, 30]
        "model": ["resnet18", "resnet34", "resnet50"]
        "optimizer": ["adam", "sgd"]
        # "criterion": ["cross_entropy", "nll_loss"]
        # "device": ["cpu", "cuda"]
        # "dataset": ["mnist", "cifar10"]
        "num_samples": 10
        "loss": "cross_entropy"
        "accuracy": 0.9
        "time": 10
        "status": "running"
        "job_name": "job1" # Display the progress of currently running jobs
        "job_id": "123" # Display the results of all finished jobs
        "job_status": "running", "finished", "failed"

    }
Database:
- 1 cái MongoDB để lưu, tinydb
    - job status
    - experiment result


Server:
- Flask: json response
    - state UI: 

- worker: 
    - While True:
        - if queue not empty:
            - pick up job
            - run job
            - update status



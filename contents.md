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

import streamlit as st
import requests
import pandas as pd
import yaml

# Load configuration from config.yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)['backend']
URL_API = f"http://{config['host']}:{config['port']}"


# add a header "Training MNIST dataset with color and font formatting
st.title("Training MNIST dataset")
st.write("List of experiments")
# Display all the running experiments and it status
exps = requests.get(f"{URL_API}/experiments").json()
df_exps = pd.DataFrame(exps)
# display is in the form of a table (id, status)
# st.write(df_exps)
list_is_visible = [False for f in range(len(df_exps))]
df_exps["is_visible"] = list_is_visible
table_exps = st.data_editor(df_exps,
    column_config={
        "is_visible": st.column_config.CheckboxColumn(
            "Show",
            # help="Select your **favorite** widgets",
            default=False,
        )
    },
    disabled=["widgets"],
    hide_index=True,
    # on_change=handle_edit(df_exps)
)


st.subheader("Hyperparameters Selection")
learning_rate = st.slider("Learning rate", 0.01, 0.1, 0.05)
num_epochs = st.slider("Number of epochs", 1, 10, 5)
optimizer = st.selectbox("Optimizer", ["Adam", "SGD"])

# add a button to start training
if st.button("Start Training"):
    # add a progress bar
    with st.spinner("Training in progress..."):
        import time
        my_bar = st.progress(0)
        for percent_complete in range(5):
            time.sleep(0.1)
            my_bar.progress(percent_complete + 1)
        st.success("The request has been successfully sent to the worker")
    # Get all the hyperparameters and print them
    st.write("Learning rate:", learning_rate)
    st.write("Number of epochs:", num_epochs)
    st.write("Optimizer:", optimizer)
    
    # put the hyperparameters as a json send to the backend
    hyperparameters = {
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "optimizer": optimizer
    }
    st.write(hyperparameters)
    # send the hyperparameters to the backend using flask
    
    response = requests.post(f"{URL_API}/train", json=hyperparameters)
    st.write(response.json())
    
    exps = requests.get(f"{URL_API}/experiments").json()
    df_exps = pd.DataFrame(exps)
    # display is in the form of a table (id, status)
    table_exps.data = df_exps
    

# adding horizontal line
st.write("---")

st.subheader("Visualize the training progress")
iid_exps = st.selectbox("Select an experiment to visualize", df_exps["iid"])
# adding auto-complete for list of experiments
# iid_exps = st.multiselect("Select an experiment to visualize", df_exps["iid"])



# handle the selection of the experiment
if iid_exps:
    st.write(f"Details of the experiment: {iid_exps}")
    # st.write(experiment)
    id_experiment = iid_exps
    response_data = requests.get(f"{URL_API}/status/{id_experiment}").json()
    print(f"Response data: {response_data}")
    if "in_update" in response_data:
        training_data = response_data['in_update']
        st.write(f"Hyperparameters: {response_data['hyper_params']}")
        st.write(pd.DataFrame(training_data))
        # st.line_chart(training_data)
        # line_chart with x_axis as epoch and y_axis as train_loss
        df_frame = pd.DataFrame(training_data)
        st.write("Loss")
        st.line_chart(data=df_frame, x="epoch", y=["train_loss", "test_loss"])
        # put title for the line chart
        st.write("Accuracy")
        st.line_chart(data=df_frame, x="epoch", y=["train_accuracy", "test_accuracy"])
    else:
        st.write("The experiment got an error")
            

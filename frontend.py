
import streamlit as st
import requests
import pandas as pd

# add a header "Training MNIST dataset with color and font formatting
st.title("Training MNIST dataset")
# st.markdown("## Training MNIST dataset")
URL_API = "http://localhost:5000"

# col1, col2 = st.columns((10, 0))


# with col1:
    # adding different hyperparameters for training mnist dataset
    

def handle_edit(df_edited):
      # Get the row index and edited column name
#   row_index = df_edited.index[0]
#   edited_column = df_edited.columns[0]
#   print(f"Row index: {row_index}, Edited column: {edited_column}")
#   # Check if edit happened in "is_visible" column
#   if edited_column == "is_visible":
#     # Update "clicked" value based on new "is_visible" state
#     df_edited.loc[row_index, "clicked"] = df_edited.loc[row_index, "is_visible"]
  
  # (Optional) Update your app logic based on the clicked value
  # ...
  print(df_edited)
  return df_edited


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

# check interaction with df_exps

st.subheader("Hyperparameters")
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
    table_exps.table(df_exps)
    # check interaction with table_exps
    if st.checkbox("Show raw data"):
        st.write(df_exps)


# iid_exps = st.selectbox("Select an experiment to visualize", df_exps["iid"])
# adding auto-complete for list of experiments
iid_exps = st.multiselect("Select an experiment to visualize", df_exps["iid"])



# handle the selection of the experiment
if iid_exps:
    st.write(f"Details of the experiment: {iid_exps}")
    # st.write(experiment)
    id_experiment = iid_exps[0]
    response_data = requests.get(f"{URL_API}/status/{id_experiment}").json()
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
        

# with col2:
    

#     st.subheader("Visualizations")
#     # adding dropdown menu for list of experiments
#     st.write("Select an experiment to visualize:")
    
#     exps = requests.get(f"{URL_API}/experiments") 
#     # print(exps)
#     experiment = st.selectbox("Experiments", exps.json())
#     # handle the selection of the experiment
#     with st.expander("Experiment details"):
#         st.write(f"Details of the experiment: {experiment}")
    
#         # st.write(experiment)
#         id_experiment = experiment
#         response_data = requests.get(f"http://localhost:5000/status/{id_experiment}").json()
#         training_data = response_data['in_update']
#         st.write(f"Hyperparameters: {response_data['hyper_params']}")
#         st.write(pd.DataFrame(training_data))
#         # st.line_chart(training_data)
#         # line_chart with x_axis as epoch and y_axis as train_loss
#         df_frame = pd.DataFrame(training_data)
#         st.write("Loss")
#         st.line_chart(data=df_frame, x="epoch", y=["train_loss", "test_loss"])
#         # put title for the line chart
#         st.write("Accuracy")
#         st.line_chart(data=df_frame, x="epoch", y=["train_accuracy", "test_accuracy"])
   
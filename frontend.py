# from PIL import Image
# import numpy as np

# import yaml
# import streamlit as st

# # Load configuration from config.yaml
# with open('config.yaml', 'r') as f:
#     config = yaml.safe_load(f)['frontend']


# import pandas as pd
# import matplotlib.pyplot as plt

# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt

# def load_training_data(data_path):
#     """Loads training data from a dictionary or file.

#     Args:
#         data_path (str or dict): Path to the dictionary or file containing training data.

#     Returns:
#         pandas.DataFrame: DataFrame containing training data.
#     """
#     sample_data = {
#         "epoch": [1, 2, 3, 4, 5],
#         "train_loss": [1.0, 0.75, 0.5, 0.25, 0.1],
#         "train_accuracy": [50, 75, 85, 90, 95],
#         "test_loss": [1.0, 0.75, 0.5, 0.25, 0.1],
#         "test_accuracy": [50, 75, 85, 90, 95]
#     }
#     return pd.DataFrame(sample_data)
        
#     if isinstance(data_path, dict):
#         return pd.DataFrame(data_path)
#     elif isinstance(data_path, str):
#         try:
#             return pd.read_csv(data_path)
#         except FileNotFoundError:
#             st.error("Error: File not found at the specified path.")
#             return None
#     else:
#         st.error("Error: Invalid data type provided. Please use a dictionary or a file path.")
#         return None

# def create_visualizations(data, metrics):
#     """Creates interactive visualizations for training metrics.

#     Args:
#         data (pandas.DataFrame): DataFrame containing training data.
#         metrics (list): List of metrics to visualize.

#     Returns:
#         None
#     """

#     if not metrics:
#         st.info("Please select at least one metric to visualize.")
#         return

#     data = data.to_numpy()  # Convert DataFrame to NumPy array
#     fig, ax = plt.subplots()
#     ax.plot(data[:, 0], data[:, 1:])  # Use NumPy indexing
#     ax.set_xlabel("Epoch")
#     ax.set_ylabel("Metrics")
#     ax.legend(metrics)
#     ax.set_title(f"Training Metrics vs Epoch")
#     st.plotly_chart(fig, use_container_width=True)

# def main():
#     st.set_page_config(page_title="Training Results Visualization", page_icon="")

#     # Input validation for data_path
#     data_path = st.text_input("Enter path to training data (dictionary or file):")
#     if not data_path:
#         st.stop()

#     training_data = load_training_data(data_path)
#     if training_data is None:
#         st.stop()

#     # Split the layout
#     col1, col2 = st.columns(ratio=[0.3, 0.7])

#     # Side 1: Parameters and choices
#     with col1:
#         # Add your parameter configuration and choice elements here
#         # (e.g., dropdown menus, text inputs, radio buttons)
#         st.subheader("Parameters")
#         st.write("Add your parameter configuration and choice elements here.")
#         # Ading dropdown menu for learning rate
#         learning_rate = st.selectbox("Select learning rate", [0.01, 0.001, 0.0001])
#         st.write(f"Selected learning rate: {learning_rate}")
        
#         # Adding slider for number of epochs
#         num_epochs = st.slider("Select number of epochs", 1, 100, 10)
#         st.write(f"Selected number of epochs: {num_epochs}")
        
#         # Adding radio buttons for optimizer
#         optimizer = st.radio("Select optimizer", ["Adam", "SGD"])
#         st.write(f"Selected optimizer: {optimizer}")
        
#     # Side 2: Visualization
#     with col2:
#         st.subheader("Training Metrics by Epoch")
#         st.write("Select metrics to visualize:")
#         metrics = st.multiselect("Metrics", list(training_data.columns)[1:])  # Exclude epoch
#         create_visualizations(training_data.copy(), metrics)  # Pass a copy to avoid modifying original data

# if __name__ == "__main__":
#     main()
import streamlit as st

# add a header "Training MNIST dataset with color and font formatting
st.title("Training MNIST dataset")
# st.markdown("## Training MNIST dataset")


col1, col2 = st.columns((3, 7))
# set the ratio size for each column
# col1 = st.columns(3)
# col2 = st.columns(3)
# col3 = st.columns(3)


with col1:
    # adding different hyperparameters for training mnist dataset
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
            st.success("Training completed successfully!")
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
        import requests
        response = requests.post("http://localhost:5000/train", json=hyperparameters)
        st.write(response.json())
        # return_response = response.json()
        # st.write(return_response)
        
        
            
#    st.header("A cat")
#    st.image("https://static.streamlit.io/examples/cat.jpg")
with col2:
    
    
    # adding figure from google drive
    # st.image("https://drive.google.com/uc?export=view&id=18Ss0vrRbi1zv18WqFpXywyHheQE4HdfR")
    # st.image("https://drive.google.com/file/d/18Ss0vrRbi1zv18WqFpXywyHheQE4HdfR/view?usp=sharing")
    
    st.subheader("Visualizations")
    
    # https://drive.google.com/file/d/18Ss0vrRbi1zv18WqFpXywyHheQE4HdfR/view?usp=sharing
    
    st.write("Add your visualizations here.")
    # random data for visualizations
    import pandas as pd
    import numpy as np
    data = {
        "epoch": np.arange(1, num_epochs+1),
        "train_loss": np.random.rand(num_epochs),
        "train_accuracy": np.random.rand(num_epochs),
        "test_loss": np.random.rand(num_epochs),
        "test_accuracy": np.random.rand(num_epochs)
    }
    df = pd.DataFrame(data)
    st.write(df)
    st.line_chart(df)
    
    # st.subheader("Miu")
    # # adding figure from local file .data/miu.jpg
    # st.image("data/miu.jpg")
    

# with col3:
#    st.header("An owl")
#    st.image("https://static.streamlit.io/examples/owl.jpg")
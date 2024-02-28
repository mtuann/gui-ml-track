This is GUI Web Application for Machine Learning Experiments tracking and management

# Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Architecture](#architecture)
5. [Development](#development)
6. [Contributing](#contributing)
7. [License](#license)

## Introduction

This is a simple web application for managing machine learning experiments. It allows users to define hyperparameters for a machine learning model and run multiple jobs with different hyperparameters. The application displays the progress of currently running jobs and the results of all finished jobs. The experiments can be sorted by a pre-defined metric (e.g. accuracy, run time) for ease of comparison. The application also allows users to resume the UI and add new jobs.

## Installation

## Usage

## Architecture
GUI-ML is a web application built with Streamlit, a Python library for creating web applications. The application uses a MongoDB database to store job status and experiment results. The backend is built with Flask, a Python web framework, and the worker is built with Celery, a distributed task queue. The application is designed to be scalable and fault-tolerant.
### Backend
The backend is built with Flask, a Python web framework. It provides a RESTful API for the frontend to interact with the database and the worker. The backend is responsible for updating the job status and experiment results in the database and for starting and stopping the worker.

### Frontend
The frontend is built with Streamlit, a Python library for creating web applications. It provides a user interface for users to define hyperparameters for a machine learning model and run multiple jobs with different hyperparameters. The frontend displays the progress of currently running jobs and the results of all finished jobs. The experiments can be sorted by a pre-defined metric (e.g. accuracy, run time) for ease of comparison. The frontend also allows users to resume the UI and add new jobs.

### Worker
The worker is built with Celery, a distributed task queue. It is responsible for running the machine learning jobs with the specified hyperparameters. The worker picks up jobs from the queue, runs the jobs, and updates the job status and experiment results in the database.

### Database
The database is built with MongoDB, a NoSQL database. It stores the job status and experiment results. The database is designed to be scalable and fault-tolerant.

## Development
To run the application in development mode, follow these steps:
1. Clone the repository
2. Install the required dependencies
3. Start the backend
4. Start the worker
5. Start the frontend

## Contributing
Contributions are welcome! Please read the [contribution guidelines](CONTRIBUTING.md) before getting started.
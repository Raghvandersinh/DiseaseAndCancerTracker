# Disease and Cancer Tracker

## Project Overview 

I have created an Django based web application that uses Machine Learning models to predict preliminary risk assessments for various diseases and cancers, such as Heart disease, Lung Cancer, Pneumonia Tracker, and Breast Cancer Tracker.

This application features four machine learning models for disease prediction. Three are tabular binary classification models for Heart Disease, Lung Cancer, and Breast Cancer, where users fill out a form to receive a risk prediction. The fourth is a multi-class image classification model for Pneumonia, which analyzes uploaded chest X-ray images to detect the presence of pneumonia.

## Key Features
- **Disease Prediction**: Using PyTorch to train cancer and disease related data to predict preliminary risk 
- **User-Friendly Interface**: Simple form inputting health data or inserting a X-ray image from your local computer. 
- **Image Analysis**: Pneumonia detection from X-ray images.
- **Containerized**: Used Docker to store my application in an Linux Debian based image file with python installed. So Users can run my code without worrying about compatibility issues like having different OS type or versions, hidden files, or missing dependencies. 

## Technology Stack 

### List of key technologies used.

#### Core Backend
- **Python 3.13+**: Primary programming language 
- **Django**: Web Framework 
- **Gunicorn**: Production WSGI HTTP Server(used in Docker Container)
- **Whitenoise**: Static file serving for production(used help communicate static files to Gunicorn)

#### Machine Learning & AI
- **Pytorch**: Deep learning framework(realized this was over kill for some of the models)
- **TorchVision**: Computer vision models and transforms
- **scikit-learn**: Machine learning algorithm
- **pandas**: data manipulation and transforms
- **NumPy 2.5.1** - Numerical computing
- **SciPy 1.18.0** - Scientific computing
- **matplotlib 3.11.0** - Data visualization
- **seaborn 0.13.2** - Statistical data visualization
- **timm 1.0.27** - PyTorch image models library
- **Hugging Face Hub 1.22.0** - Model repository and sharing 

#### Data Processing & Utilities

- **joblib 1.5.3** - Python pipelining and serialization
- **pillow 12.3.0** - Python Imaging Library
- **requests 2.34.2** - HTTP library
- **python-dotenv 1.2.2** - Environment variable management

#### APIs & Integration

- **kaggle 2.2.3** - Kaggle API for dataset downloads
- **httpx 0.28.1** - HTTP client

#### Development & Deployment

- **Docker** - Containerization
- **django-distill 3.2.7** - Static site generation
- **setuptools 81.0.0** - Package installation

### Getting Started:

#### Prerequisites
- Have Docker installed in your system
- You must have an Kaggle API Key. Create an Kaggle account and get an API Key

#### Quick start with Docker
- **Pull the Docker Image**
```bash
    docker pull ghcr.io/raghvandersinh/disease-tracker-app:latest
```
- Run the Container(Here set you Kaggle API key as an env variable)

```bash
docker run -p 8000:8000 -e KAGGLE_API_TOKEN="YOUR_TOKEN_HERE" ghcr.io/raghvandersinh/disease-tracker-app:latest
```

#### Access the application

- Open your browser and then go to this url http://localhost:8000. It will take a little
while since its downloading the dataset from kaggle (Thats a poor optimization choice in my part). 

# Advanced Movie Recommendation System

## Overview

This repository contains an advanced movie recommendation system built using Streamlit and PyTorch. The model predicts movie ratings for users based on a deep learning architecture and allows filtering recommendations by genres and release year.

## Features

- Personalized movie recommendations for individual users.
- Advanced filtering options:
  - Select specific genres.
  - Specify a range of release years.
- Utilizes a neural network model for accurate rating predictions.
- Provides metadata for recommended movies, including title, year, genres, and predicted ratings.
- Built-in user-friendly interface powered by Streamlit.

## Technologies Used

- **Python**: Core programming language.
- **PyTorch**: Deep learning framework used to build the neural network model.
- **Streamlit**: For creating an interactive and user-friendly web app interface.
- **Pandas**: For data manipulation and preprocessing.
- **Pickle**: To serialize and load preprocessed data.

## Dataset

The recommendation system is trained and evaluated on the MovieLens 100K dataset. This dataset includes comprehensive metadata for movies and user ratings, enabling accurate predictions and diverse recommendations.

## How It Works

### Model Architecture
The model uses an embedding-based neural network:

- **Embeddings**:
  - User and movie embeddings are learned for a dense representation of features.
- **Neural Network**:
  - The combined user and movie embeddings are passed through a series of fully connected layers.
  - ReLU activation and dropout are applied to enhance performance and reduce overfitting.
- **Output**:
  - A single scalar value representing the predicted rating.


### Recommendations
The app generates a list of top recommendations based on predicted ratings, providing the following details for each movie:
- Title
- Predicted rating
- Release year
- Genres

## 🔗 Try it out here:  
[Movie Recommendation System on Hugging Face Spaces](https://huggingface.co/spaces/dimension-drifter/Movie-Recommendation-System)  

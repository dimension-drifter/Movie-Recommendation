# -*- coding: utf-8 -*-
import torch
print("GPU Available:", torch.cuda.is_available())

# Import Libraries
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import pickle

#Download MovieLens Dataset
!wget https://files.grouplens.org/datasets/movielens/ml-100k.zip
!unzip ml-100k.zip

# Data Preparation Class
class MovieDataPreprocessor:
    def __init__(self, ratings_path='ml-100k/u.data', batch_size=64):
        # Read ratings data
        self.ratings_df = pd.read_csv(
            ratings_path,
            sep='\t',
            names=['user_id', 'movie_id', 'rating', 'timestamp']
        )

        # Create user and movie mappings
        self.users = self.ratings_df['user_id'].unique()
        self.movies = self.ratings_df['movie_id'].unique()

        # Create mapping dictionaries
        self.user_to_index = {user: idx for idx, user in enumerate(self.users)}
        self.movie_to_index = {movie: idx for idx, movie in enumerate(self.movies)}

        # Prepare data for model with batching
        self.batch_size = batch_size
        self.prepare_batched_tensors()

    def prepare_batched_tensors(self):
        # Convert user and movie IDs to indices
        self.ratings_df['user_index'] = self.ratings_df['user_id'].map(self.user_to_index)
        self.ratings_df['movie_index'] = self.ratings_df['movie_id'].map(self.movie_to_index)

        # Create PyTorch Dataset
        dataset = torch.utils.data.TensorDataset(
            torch.LongTensor(self.ratings_df['user_index'].values),
            torch.LongTensor(self.ratings_df['movie_index'].values),
            torch.FloatTensor(self.ratings_df['rating'].values)
        )

        # Create DataLoader with batching
        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

# Recommendation Model
class ImprovedRecommendationModel(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=50):
        super().__init__()

        # Create embeddings for users and movies
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)

        # Simple neural network with clear, progressive layers
        self.network = nn.Sequential(
            # First layer: Combine user and movie information
            nn.Linear(embedding_dim * 2, 128),  # Doubled input layer size
            nn.ReLU(),  # Add non-linearity
            nn.Dropout(0.4),  # Light regularization

            # Second layer: Further process combined information
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Final layer to predict rating
            nn.Linear(64, 1)  # Single output for rating prediction
        )

    def forward(self, users, movies):
        # Get embeddings for users and movies
        user_emb = self.user_embedding(users)
        movie_emb = self.movie_embedding(movies)

        # Combine user and movie embeddings
        combined = torch.cat([user_emb, movie_emb], dim=1)

        # Pass through neural network
        rating_prediction = self.network(combined)

        return rating_prediction.squeeze()

# Training Function with GPU Support
def train_recommendation_model(
    preprocessor,
    epochs=500,  # Reduced epochs for quicker training
    learning_rate=0.001
):
    # Easy device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Create model with clear parameters
    model = ImprovedRecommendationModel(
        num_users=len(preprocessor.user_to_index),
        num_movies=len(preprocessor.movie_to_index)
    ).to(device)

    # Simple loss and optimizer setup
    criterion = nn.MSELoss()  # Mean Squared Error for rating prediction
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop with progress tracking
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        # Batch processing
        for batch_users, batch_movies, batch_ratings in preprocessor.dataloader:
            # Move data to device
            batch_users = batch_users.to(device)
            batch_movies = batch_movies.to(device)
            batch_ratings = batch_ratings.to(device)

            # Zero gradients before each batch
            optimizer.zero_grad()

            # Make predictions
            predictions = model(batch_users, batch_movies)

            # Calculate loss
            loss = criterion(predictions, batch_ratings)

            # Backpropagate and optimize
            loss.backward()
            optimizer.step()

            # Track total loss
            total_loss += loss.item()

        # Print epoch summary
        print(f"Epoch {epoch+1}: Average Loss = {total_loss/len(preprocessor.dataloader):.4f}")

    return model

# Recommendation Function
def get_recommendations(model, preprocessor, user_id, top_k=5):
    # Ensure model is in evaluation mode
    model.eval()
    device = next(model.parameters()).device

    # Convert user ID to index
    user_index = preprocessor.user_to_index[user_id]

    # Predict ratings for all movies
    predictions = []
    for movie_index in range(len(preprocessor.movie_to_index)):
        user_tensor = torch.LongTensor([user_index]).to(device)
        movie_tensor = torch.LongTensor([movie_index]).to(device)

        with torch.no_grad():
            prediction = model(user_tensor, movie_tensor).cpu().item()

        predictions.append((movie_index, prediction))

    # Get top K recommendations
    top_movies = sorted(predictions, key=lambda x: x[1], reverse=True)[:top_k]
    recommended_movies = [preprocessor.movies[movie_index] for movie_index, _ in top_movies]

    return recommended_movies

#Main Execution and Saving
def main():
    # Preprocess Data
    preprocessor = MovieDataPreprocessor()

    # Train Model with GPU
    model = train_recommendation_model(preprocessor)

    # Get Sample Recommendations
    # sample_user = preprocessor.users[0]
    # recommendations = get_recommendations(model, preprocessor, sample_user)
    #
    # print(f"Recommendations for User {sample_user}:")
    # print(recommendations)

    # Save Model and Preprocessor Data
    # 1. Save Model State Dictionary
    torch.save(model.state_dict(), 'recommendation_model.pth')

    # 2. Save Preprocessor Data
    preprocessor_data = {
        'user_to_index': preprocessor.user_to_index,
        'movie_to_index': preprocessor.movie_to_index,
        'users': preprocessor.users,
        'movies': preprocessor.movies
    }

    with open('preprocessor_data.pkl', 'wb') as f:
        pickle.dump(preprocessor_data, f)

    # Download saved files
    from google.colab import files
    files.download('recommendation_model.pth')
    files.download('preprocessor_data.pkl')

]if __name__ == "__main__":
    main()


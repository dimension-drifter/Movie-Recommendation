import streamlit as st
import torch
import torch.nn as nn
import torch.utils.data
import pandas as pd
import pickle

#start of code
class ImprovedRecommendationModel(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=50):
        super().__init__()

        # Create embeddings for users and movies
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)

        self.network = nn.Sequential(
            # First layer: Combine user and movie information
            nn.Linear(embedding_dim * 2, 128),  # Doubled input layer size
            nn.ReLU(),  # Add non-linearity
            nn.Dropout(0.4),  # Light regularization

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Final layer
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


class MovieRecommender:
    def __init__(self,
                 model_path='recommendation_model.pth',
                 preprocessor_path='preprocessor_data.pkl',
                 movies_metadata_path='ml-100k/u.item'):
        # Load preprocessor data
        with open(preprocessor_path, 'rb') as f:
            preprocessor_data = pickle.load(f)

        # Extract preprocessor information
        self.user_to_index = preprocessor_data['user_to_index']
        self.movie_to_index = preprocessor_data['movie_to_index']
        self.users = preprocessor_data['users']
        self.movies = preprocessor_data['movies']

        # Load comprehensive movie metadata
        self.movie_metadata = self.load_comprehensive_movie_data(movies_metadata_path)

        # Determine device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize and load model
        self.model = ImprovedRecommendationModel(
            num_users=len(self.user_to_index),
            num_movies=len(self.movie_to_index)
        ).to(self.device)

        # Load model state
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def load_comprehensive_movie_data(self, movies_metadata_path):
        """
        Load comprehensive movie metadata with additional attributes
        u.item format:
        movie id | movie title | release date | video release date | IMDb URL |
        unknown | Action | Adventure | Animation | Children's | Comedy | Crime |
        Documentary | Drama | Fantasy | Film-Noir | Horror | Musical | Mystery |
        Romance | Sci-Fi | Thriller | War | Western
        """
        try:
            # Read movie metadata with all columns
            movie_data = pd.read_csv(
                movies_metadata_path,
                sep='|',
                encoding='latin-1',
                header=None,
                names=[
                    'movie_id', 'title', 'release_date', 'video_release_date',
                    'imdb_url', 'unknown', 'Action', 'Adventure', 'Animation',
                    "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama',
                    'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                    'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
                ]
            )

            # Extract year from release date
            movie_data['year'] = pd.to_datetime(
                movie_data['release_date'],
                errors='coerce'
            ).dt.year

            # Prepare genre information
            genre_columns = [
                'Action', 'Adventure', 'Animation', "Children's", 'Comedy',
                'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
                'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                'Thriller', 'War', 'Western'
            ]

            # Create metadata dictionary
            movie_metadata = {}
            for _, row in movie_data.iterrows():
                movie_id = row['movie_id']
                genres = [genre for genre in genre_columns if row[genre] == 1]

                movie_metadata[movie_id] = {
                    'title': row['title'],
                    'year': row['year'],
                    'genres': genres
                }

            return movie_metadata

        except Exception as e:
            st.warning(f"Could not load movie metadata: {e}")
            return {}

    def get_recommendations(self, user_id, top_k=5,
                            selected_genres=None,
                            min_year=None,
                            max_year=None):
        # Convert user ID to index
        user_index = self.user_to_index[user_id]

        # Predict ratings for all movies
        all_predictions = []
        for movie_index in range(len(self.movie_to_index)):
            movie_id = self.movies[movie_index]

            # Metadata filtering
            movie_info = self.movie_metadata.get(movie_id, {})

            # Genre filtering
            if selected_genres:
                movie_genres = movie_info.get('genres', [])
                if not any(genre in movie_genres for genre in selected_genres):
                    continue

            # Year filtering
            movie_year = movie_info.get('year')
            if min_year and movie_year and movie_year < min_year:
                continue
            if max_year and movie_year and movie_year > max_year:
                continue

            # Create tensors and move to device
            user_tensor = torch.LongTensor([user_index]).to(self.device)
            movie_tensor = torch.LongTensor([movie_index]).to(self.device)

            with torch.no_grad():
                prediction = self.model(user_tensor, movie_tensor).cpu().item()

            all_predictions.append((movie_index, prediction))

        # Sort and get top K recommendations
        top_movies = sorted(all_predictions, key=lambda x: x[1], reverse=True)[:top_k]

        # Convert to movie names with ratings
        recommended_movies = [
            {
                'name': self.movie_metadata.get(self.movies[movie_index], {}).get('title',
                                                                                  f"Movie {self.movies[movie_index]}"),
                'rating': round(rating, 2),
                'year': self.movie_metadata.get(self.movies[movie_index], {}).get('year'),
                'genres': self.movie_metadata.get(self.movies[movie_index], {}).get('genres', [])
            }
            for movie_index, rating in top_movies
        ]

        return recommended_movies


def main():
    st.title("Advanced Movie Recommendation System")

    # Initialize recommender
    try:
        recommender = MovieRecommender()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    # Sidebar for Filtering
    st.sidebar.header("🎬 Recommendation Filters")

    # Genre Multi-Select
    all_genres = [
        'Action', 'Adventure', 'Animation', "Children's", 'Comedy',
        'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
        'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
        'Thriller', 'War', 'Western'
    ]
    selected_genres = st.sidebar.multiselect(
        "Select Genres",
        all_genres
    )

    # Year Range Slider
    min_year, max_year = st.sidebar.slider(
        "Release Year Range",
        min_value=1900,
        max_value=2023,
        value=(1950, 2023)
    )

    # User selection
    try:
        sorted_users = sorted(
            [int(user) for user in recommender.users],
            key=int
        )

        if not sorted_users:
            st.error("No users found in the dataset!")
            return

        selected_user = st.selectbox(
            "Select a User ID",
            sorted_users,
            index=0
        )
    except Exception as selection_error:
        st.error(f"User Selection Error: {selection_error}")
        return

    # Number of recommendations slider
    top_k = st.slider("Number of Recommendations", 1, 10, 5)

    # Recommendation button
    if st.button("Get Recommendations"):
        with st.spinner("Generating recommendations..."):
            try:
                recommendations = recommender.get_recommendations(
                    selected_user,
                    top_k,
                selected_genres = selected_genres,
                min_year = min_year,
                max_year = max_year
                )

                st.success(f"Top {top_k} Recommendations for User {selected_user}:")
                for i, movie in enumerate(recommendations, 1):
                    st.write(
                        f"{i}. {movie['name']} (Predicted Rating: {movie['rating']}, Year: {movie['year']}, Genres: {', '.join(movie['genres'])})")

            except Exception as e:
                st.error(f"Error generating recommendations: {e}")

        st.sidebar.header("About the Recommendation System")
        st.sidebar.info("""
            💡 Recommendation Insights:
            - Personalized movie suggestions
            - Filter by genre and release year
            - Based on neural network predictions
            - Uses MovieLens 100K dataset
            """)


if __name__ == "__main__":
    main()
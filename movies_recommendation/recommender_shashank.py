# Downloaded the kaggale_movie_rating.csv from kaggale website, which has 4 features FILM; RATING; STARS; and VOTES, I used only FILM and STARS columns for recommendation of movie to customer.
# At the bottom is the output in comments

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class ItemRecommender:
    def __init__(self, items, ratings):
        self.items = items
        self.ratings = ratings
        self.item_vectors = self._calculate_item_vectors()

    def _calculate_item_vectors(self):
        num_items = len(self.items)
        num_users = len(self.ratings)
        item_vectors = np.zeros((num_items, num_users))

        for user_id, user_ratings in enumerate(self.ratings):
            for item_id, rating in enumerate(user_ratings):
                item_vectors[item_id][user_id] = rating

        return item_vectors

    def recommend_similar_items(self, item_id, top_n=5):
        item_vector = self.item_vectors[item_id]
        similarities = cosine_similarity([item_vector], self.item_vectors)[0]
        similar_items_indices = similarities.argsort()[::-1][1:top_n+1]  # Exclude the item itself
        similar_items = [(self.items[i], similarities[i]) for i in similar_items_indices]
        return similar_items

# Load data from CSV
file_path = r"C:\Users\User\Desktop\shashankbaraicollege\Data analtics 2\kaggale_movie_rating.csv"
data = pd.read_csv(file_path)

# Extract FILM and RATING columns
films = data['FILM'].tolist()
ratings = data['RATING'].tolist()

# Fill NaN values with 0
ratings = [rating if not np.isnan(rating) else 0 for rating in ratings]

# Create ItemRecommender instance
recommender = ItemRecommender(films, [ratings])

# Example usage
item_id = 5  # Index of the item for which we want to recommend similar items
similar_items = recommender.recommend_similar_items(item_id)
print(f"Movies similar to '{films[item_id]}':")
for item, similarity in similar_items:
    print(f"{item} (Similarity: {similarity:.2f})")

#-------------------------output------------------------------------------------------------
# Movies similar to 'The Hobbit: The Battle of the Five Armies (2014)':
# Angrej (2015) (Similarity: 1.00)
# Drishyam (2015) (Similarity: 1.00)
# Bolshoi Ballet: Swan Lake (2015) (Similarity: 1.00)
# Insider Access to Disney Pixarâ€™s Inside Out (2015) (Similarity: 1.00)
# The Overnight (2015) (Similarity: 1.00)
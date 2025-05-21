# 🎬 AniMate: Intelligent Anime Recommendation System

**AniMate** is an intelligent anime recommendation system designed for fans who want to discover new shows based on **themes, narrative content, and audience sentiment**—not just genres. Input an anime title or a free-form description, and AniMate returns a curated list of similar shows with explanations for each match.

🔗 **Try the Final Prototype** *(link placeholder)*

## 🧠 How It Works
AniMate combines techniques from **Information Retrieval, Text Mining, and Social Recommendation Systems** to deliver personalized recommendations:
- **TF-IDF** + **Jaccard Similarity**: Filters and compares anime synopses and metadata.
- **SVD (Singular Value Decomposition)**: Identifies deeper thematic matches beyond surface-level vocabulary.
- **Collaborative Filtering**: Surfaces shows liked by users with similar tastes.

## 🔄 Key Evolutions
### Information Retrieval
- Refined Jaccard threshold to improve diversity by loosening strict genre matches.

### Text Mining
- Integrated **SVD** to capture latent thematic connections (e.g., "revenge" → dark fantasy *and* psychological thrillers).

### Social Recommendations
- Leveraged user ratings for crowd-driven suggestions (e.g., "People who liked *Attack on Titan* also enjoyed...").

## 🧪 Example Queries
| **Input** | **Top Results** |
|-----------|----------------|
| *"High school students in a deadly game"* | *Future Diary, Danganronpa* |
| *"Pet dogs and cats"* | *My Roommate is a Cat, Poco's Udon World* |
| *"Revenge, power, war, magic"* | *Berserk, Code Geass* |

## 💡 Core Techniques
- **Jaccard Similarity**: Genre/episode filtering.
- **TF-IDF**: Synopsis-based similarity.
- **SVD**: Latent topic modeling.
- **Collaborative Filtering**: User-review-driven suggestions.

## 👥 Team
| Name |
|------|
| Oluebube Chukwubuikem | 
| Ruth Taddesse | 
| Yoseph Endawoke |
| Srihan Mediboina |

📚 **Cornell University | INFO 4300: Language & Information**

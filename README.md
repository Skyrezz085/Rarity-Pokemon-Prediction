# **Pokémon Rarity Prediction 🌟**  
A machine learning project to predict Pokémon rarity based on battle stats and key attributes.

## **Introduction 🎮**

In the world of Pokémon GO, finding rare Pokémon is one of the most exciting challenges for players. Each Pokémon has different attributes such as attack, defense, stamina, and type combinations that not only impact their battle performance but also indicate their **rarity**. Manually evaluating these features to identify rarity can be time-consuming and subjective.

To simplify this process, we built a machine learning model that predicts a Pokémon's **rarity** based on its stats and characteristics. This project aims to help players make more strategic decisions in collecting rare Pokémon using data-driven insights.

## **Data Overview 📊**

The dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/shreyasur965/pokemon-go/data) and contains **1,007 entries** with **24 features**, including base stats, Pokémon types, rarity levels, and acquisition methods.

Here’s a summary of key columns:

| **Column Name**               | **Description**                                                              |
|-------------------------------|------------------------------------------------------------------------------|
| pokemon_id                    | Unique Pokémon ID                                                            |
| pokemon_name                  | Name of the Pokémon                                                          |
| base_attack                   | Base attack value                                                            |
| base_defense                 | Base defense rating                                                          |
| base_stamina                 | Health/Stamina of the Pokémon                                                |
| type                          | Pokémon type(s) (e.g., Fire, Grass)                                          |
| rarity                        | Rarity level (e.g., Standard, Legendary)                                     |
| max_cp                        | Maximum Combat Power                                                         |
| found_wild, found_raid, etc.  | Boolean values indicating where the Pokémon can be found                     |
| attack_probability            | Chance to successfully land an attack                                        |
| dodge_probability             | Chance to successfully dodge an attack                                       |

> ⚠️ Some columns contain missing values (e.g., `candy_required`, `attack_probability`) which need to be handled during preprocessing.

## **Background 🌍**

In Pokémon GO, many players aim to collect rare and powerful Pokémon. However, rarity is not always easy to assess. Attributes like base attack, defense, stamina, and typing play a major role in how rare or valuable a Pokémon is.

To address this, the goal of this project is to create a predictive model that estimates a Pokémon’s **rarity** based on its core attributes. By using this model, players can more easily identify and prioritize rare Pokémon during gameplay.

## **Methodology 🔍**

1. **Data Cleaning**: Import the dataset and handle missing or inconsistent data.
2. **Exploratory Data Analysis (EDA)**: Analyze attribute distributions, relationships, and type-based trends.
3. **Feature Engineering**: Encode categorical values, extract features from lists, and normalize stat columns.
4. **Model Inference**: Load the trained model from `model.pkl` and perform predictions on new Pokémon entries.
5. **Deployment**: Deploy the best-performing model for real-time rarity predictions.

## **Machine Learning Models Used 🧠**

- K-Nearest Neighbors (KNN)  
- Support Vector Machine (SVM)  
- Decision Tree  
- Random Forest  
- Gradient Boosting  

## **Model Evaluation 🧮**

### Strengths:
- **Instant Predictions**: Pre-trained model in `.pkl` format allows real-time inference.
- **Rich Feature Set**: Model utilizes various battle-related attributes to improve prediction accuracy.

### Weaknesses:
- **Dataset Bias**: Some Pokémon types or encounter methods are overrepresented, which may bias predictions.

### Future Improvements:
- Apply feature selection to remove low-variance or incomplete features.
- Expand the training set with tagged battle outcomes to improve supervised learning quality.

## **Conclusion 📈**

This project demonstrates how machine learning can enhance the Pokémon GO experience by helping players identify rare Pokémon based on battle-related data. With fast and accurate predictions, the model empowers users to make smarter choices during collection and team building.

Future upgrades could include live team recommendations, type matchup simulators, or personalized rarity filters.

## **Try the Model 🚀**

Test and explore the deployed model here:  
👉 [Rarity Pokémon Prediction on Hugging Face](https://huggingface.co/spaces/Skyrezz/rarity_pokemon_prediction)

## **Libraries Used 🛠️**

- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Seaborn  
- Pickle  
- Jupyter Notebook  

## **Author 👨‍💻**

**Reza Syadewo**  
🔗 [LinkedIn](https://www.linkedin.com/in/reza-syadewo-b5801421b/)
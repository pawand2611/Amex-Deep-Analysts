import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample

# Step 1: Load and examine the data
match_level_df = pd.read_csv('664389efa0868_match_level_scorecard.csv')
batsman_level_df = pd.read_csv('663e2b548c98c_batsman_level_scorecard.csv')
bowler_level_df = pd.read_csv('663e2b2c60743_bowler_level_scorecard.csv')
train_df = pd.read_csv('663e2b6d54457_train_data_with_samplefeatures.csv')
test_df = pd.read_csv('6644a1e287df6_test_data_with_samplefeatures.csv')

# Rename 'match id' to 'match_id' in all dataframes
match_level_df.rename(columns={'match id': 'match_id'}, inplace=True)
batsman_level_df.rename(columns={'match id': 'match_id'}, inplace=True)
bowler_level_df.rename(columns={'match id': 'match_id'}, inplace=True)

# Verify common key for merging
common_key = 'match_id'

if common_key in match_level_df.columns and common_key in batsman_level_df.columns and common_key in bowler_level_df.columns:
    # Step 2: Merge data
    match_batsman_df = pd.merge(match_level_df, batsman_level_df, on=common_key, how='left', suffixes=('', '_batsman'))
    full_data_df = pd.merge(match_batsman_df, bowler_level_df, on=common_key, how='left', suffixes=('', '_bowler'))

    # Step 3: Prepare features and target variable
    # Encoding winner as 1 for 'team1' and 0 for 'team2'
    full_data_df['match_outcome'] = full_data_df.apply(lambda row: 1 if row['winner'] == row['team1'] else 0, axis=1)

    # Drop unnecessary columns
    columns_to_drop = [common_key, 'match_dt', 'winner', 'by', 'city', 'series_name', 'season',
                       'umpire1', 'umpire2', 'team1_roster_ids', 'team2_roster_ids', 'series_type', 'batsman',
                       'batsman_details', 'wicket kind', 'out_by_bowler', 'out_by_fielder']

    # Ensure columns to drop exist in the dataframe
    columns_to_drop = [col for col in columns_to_drop if col in full_data_df.columns]
    full_data_df = full_data_df.drop(columns=columns_to_drop)

    # Identify non-numeric columns that need encoding
    non_numeric_columns = full_data_df.select_dtypes(include=['object']).columns

    # Convert categorical features to numerical using one-hot encoding
    full_data_df = pd.get_dummies(full_data_df, columns=non_numeric_columns)

    # Split the data into features and target
    X = full_data_df.drop('match_outcome', axis=1)
    y = full_data_df['match_outcome']

    # Ensure all data is numeric
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Address data imbalance (if necessary)
    if y.value_counts().min() / y.value_counts().max() < 0.5:
        # Combine features and target into one DataFrame for resampling
        combined_df = pd.concat([X, y], axis=1)
        majority_class = combined_df[combined_df.match_outcome == y.mode()[0]]
        minority_class = combined_df[combined_df.match_outcome != y.mode()[0]]

        # Upsample minority class
        minority_upsampled = resample(minority_class,
                                      replace=True,  # Sample with replacement
                                      n_samples=len(majority_class),  # Match number in majority class
                                      random_state=42)  # Reproducible results

        # Combine majority class with upsampled minority class
        combined_df = pd.concat([majority_class, minority_upsampled])

        # Split features and target
        X = combined_df.drop('match_outcome', axis=1)
        y = combined_df['match_outcome']

    # Sample data to reduce size (if necessary)
    X, _, y, _ = train_test_split(X, y, train_size=0.1, random_state=42)

    # Split into train and test datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 4: Train XGBoost model
    model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Feature importance
    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    print("Feature Importances:\n", feature_importances.sort_values(ascending=False))

    # Step 5: Predict using the test dataset
    # Keep the match_id and team columns for reference
    test_match_ids = test_df[common_key]
    test_team1 = test_df['team1']
    test_team2 = test_df['team2']

    # Preprocess the test dataset similarly to the training dataset
    test_df = pd.merge(test_df, match_level_df, on=common_key, how='left', suffixes=('', '_match'))
    test_df = pd.merge(test_df, batsman_level_df, on=common_key, how='left', suffixes=('', '_batsman'))
    test_df = pd.merge(test_df, bowler_level_df, on=common_key, how='left', suffixes=('', '_bowler'))

    # Drop unnecessary columns in test data
    test_df = test_df.drop(columns=columns_to_drop)

    # Encode non-numeric columns
    non_numeric_columns_test = test_df.select_dtypes(include=['object']).columns
    test_df = pd.get_dummies(test_df, columns=non_numeric_columns_test)

    # Ensure all data is numeric
    test_df = test_df.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Align test data with training data
    X_test_final = test_df.reindex(columns=X.columns, fill_value=0)

    # Predict the outcomes
    test_predictions = model.predict(X_test_final)

    # Map predicted outcomes back to team IDs
    predictions_df = pd.DataFrame({
        common_key: test_match_ids,
        'team1': test_team1,
        'team2': test_team2,
        'predicted_outcome': test_predictions
    })
    predictions_df['predicted_winner'] = predictions_df.apply(
        lambda row: row['team1'] if row['predicted_outcome'] == 1 else row['team2'], axis=1
    )

    # Output the predictions with team IDs
    print("\nTest Data Predictions with Team IDs:")
    print(predictions_df.head())

    # Save predictions to a CSV file
    predictions_df[['match_id', 'predicted_winner']].to_csv('test_predictions_with_team_ids.csv', index=False)

else:
    print(f"Column '{common_key}' does not exist in all dataframes.")

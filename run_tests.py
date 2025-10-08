"""
Lab 2 - Weather Prediction Test Harness
Validates model predictions against known test cases
"""

from weather_predictor import predict,build_features,MODELS
from datetime import datetime

# Test cases - format: (date, actual_values, sample_cf6_data)
TEST_CASES = [
    {
        "date": "13 Feb 2025",
        "actual": [True, True, True],
        "input": {
            "ROC": """PRELIMINARY LOCAL CLIMATOLOGICAL DATA
STATION: ROCHESTER NY
MONTH: FEBRUARY
YEAR: 2025
 1 36 32 34 5 0 0 0.15 1.2 3 8.5 22 180"""
        }
    },
    {
        "date": "13 Dec 2024",
        "actual": [False, False, False],
        "input": {
            "ROC": """PRELIMINARY LOCAL CLIMATOLOGICAL DATA
STATION: ROCHESTER NY
MONTH: DECEMBER
YEAR: 2024
 1 28 22 25 -5 40 0 0.00 0.0 0 5.5 15 270"""
        }
    },
    {
        "date": "13 Sep 2024",
        "actual": [False, True, False],
        "input": {
            "ROC": """PRELIMINARY LOCAL CLIMATOLOGICAL DATA
STATION: ROCHESTER NY
MONTH: SEPTEMBER
YEAR: 2024
 1 72 68 70 3 0 5 0.00 0.0 0 7.2 18 190"""
        }
    },
    {
        "date": "13 Jun 2024",
        "actual": [True, True, False],
        "input": {
            "ROC": """PRELIMINARY LOCAL CLIMATOLOGICAL DATA
STATION: ROCHESTER NY
MONTH: JUNE
YEAR: 2024
 1 82 74 78 5 0 13 0.00 0.0 0 8.1 22 210"""
        }
    },
    {
        "date": "13 Mar 2024",
        "actual": [True, True, False],
        "input": {
            "ROC": """PRELIMINARY LOCAL CLIMATOLOGICAL DATA
STATION: ROCHESTER NY
MONTH: MARCH
YEAR: 2024
 1 48 42 45 4 20 0 0.02 0.0 0 10.3 25 280"""
        }
    },
    {
        "date": "25 April",
        "actual": [True, False, False],
        "input": {
            "ROC": """PRELIMINARY LOCAL CLIMATOLOGICAL DATA
STATION: ROCHESTER NY
MONTH: APRIL
YEAR: 2025
 1 55 49 52 -1 13 0 0.00 0.0 0 6.7 16 150"""
        }
    }
]


def run_tests():
    print("=== Weather Prediction Model Test Harness ===")
    print(f"Running {len(TEST_CASES)} test cases...\n")

    total_tree_correct = 0
    total_forest_correct = 0

    for i, test in enumerate(TEST_CASES, 1):
        print(f"\nTest Case {i}: {test['date']}")
        print(f"Actual Values: {test['actual']}")

        # Prepare input data (fill missing days with empty data)
        day_inputs = [{}, {}, {}, {}, {}]  # day5 to day1
        day_inputs[-1] = test['input']  # day1 is most recent

        # Run predictions
        tree_pred = predict("besttree", *day_inputs)
        forest_pred = predict("bestforest", *day_inputs)

        # Calculate accuracy
        tree_correct = sum(t == a for t, a in zip(tree_pred, test['actual']))
        forest_correct = sum(f == a for f, a in zip(forest_pred, test['actual']))

        # Update totals
        total_tree_correct += tree_correct
        total_forest_correct += forest_correct

        # Display results
        print(f"Tree Prediction: {tree_pred} ({tree_correct}/3 correct)")
        print(f"Forest Prediction: {forest_pred} ({forest_correct}/3 correct)")

        print("\nFeature Debug:")
        features = build_features({1: test['input']})  # Show features for day1
        print(f"Generated features: {features}")

        print("\nModel Decisions:")
        for j, model_type in enumerate(['besttree', 'bestforest']):
            model = MODELS[f"{model_type}_temp_higher"]
            if hasattr(model, 'feature_importances_'):
                print(f"{model_type} feature importances:", model.feature_importances_)

        # Show detailed comparison
        labels = ["Temp Higher", "Above Avg", "Precipitation"]
        for j, (label, actual, t_pred, f_pred) in enumerate(zip(labels, test['actual'], tree_pred, forest_pred)):
            print(f"  {label}:")
            print(f"    Actual: {actual}")
            print(f"    Tree:   {t_pred} {'✓' if t_pred == actual else '✗'}")
            print(f"    Forest: {f_pred} {'✓' if f_pred == actual else '✗'}")

    # Final statistics
    total_possible = len(TEST_CASES) * 3
    print("\n=== Final Statistics ===")
    print(
        f"Decision Tree Total Accuracy: {total_tree_correct}/{total_possible} ({total_tree_correct / total_possible:.1%})")
    print(
        f"Random Forest Total Accuracy: {total_forest_correct}/{total_possible} ({total_forest_correct / total_possible:.1%})")


if __name__ == "__main__":
    run_tests()
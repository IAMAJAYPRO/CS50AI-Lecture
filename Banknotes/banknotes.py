def cmd_prompt():
    import argparse
    models = ['Perceptron', 'SVM', 'KNN', 'Naive']
    parser = argparse.ArgumentParser(
        description="Train and test different models on banknotes dataset.", allow_abbrev=True)
    parser.add_argument("--model", "-m", type=str, default="Perceptron", choices=models,
                        help="Choose a model: 'Perceptron', 'SVM', 'KNN', or 'Naive'(NaiveBayes)")
    parser.add_argument("-n", '--n_neighbors', type=int, default=1,
                        help="Number of neighbors for KNN (required with KNN)")
    parser.add_argument("--tests", type=float, default=0.4,
                        help="test size between 0 to 1 (defaut: 0.4)")
    ARGS = parser.parse_args()
    return ARGS


ARGS = cmd_prompt()


if True:
    import csv
    import random
    from sklearn import svm
    from sklearn.linear_model import Perceptron
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier


def load_from_csv():
    with open("banknotes.csv") as f:
        reader = csv.reader(f)
        next(reader)
        data = []
        for row in reader:
            data.append({
                "evidence": [float(cell) for cell in row[:4]],
                "label": "Authentic" if row[4] == "0" else "Counterfeit"
            })
    return data


def main(ARGS):
    models = {
        'Perceptron': Perceptron,
        'SVM': svm.SVC,
        # 'KNN': KNeighborsClassifier(n_neighbors=1),
        'Naive': GaussianNB
    }
    model = models[ARGS.model](
    ) if ARGS.model in models else KNeighborsClassifier(n_neighbors=1)

    # Read data in from file
    data = load_from_csv()
    # Separate data into training and testing groups
    evidence = [row["evidence"] for row in data]
    labels = [row["label"] for row in data]

    X_training, X_testing, y_training, y_testing = train_test_split(
        evidence, labels, test_size=ARGS.tests
    )
    """
    holdout = int(0.40 * len(data))
    random.shuffle(data)
    testing = data[:holdout]
    training = data[holdout:]

    # Train model on training set
    X_training = [row["evidence"] for row in training]
    y_training = [row["label"] for row in training]
    """
    # Fit model
    model.fit(X_training, y_training)

    # Make predictions on the testing set
    predictions = model.predict(X_testing)

    # Compute how well we performed
    correct = (y_testing == predictions).sum()
    incorrect = (y_testing != predictions).sum()
    total = len(predictions)

    # Print results
    print(f"Results for model {type(model).__name__}")
    print(f"Correct: {correct}")
    print(f"Incorrect: {incorrect}")
    print(f"Accuracy: {100 * correct / total:.2f}%")


main(ARGS)

import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv(r"C:\Users\aswin\OneDrive\Desktop\data2.csv")
print(df)


# Function to calculate entropy of a dataset
def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    entropy = np.sum(
        [(-counts[i] / np.sum(counts)) * np.log2(counts[i] / np.sum(counts)) for i in range(len(elements))])
    return entropy


# Function to calculate information gain for a specific attribute
def info_gain(data, split_attribute_name, target_name="buys_computer"):
    # Calculate the total entropy of the target attribute
    total_entropy = entropy(data[target_name])

    # Get the values and counts for the split attribute
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)

    # Calculate the weighted entropy
    weighted_entropy = np.sum([(counts[i] / np.sum(counts)) *
                               entropy(data.where(data[split_attribute_name] == vals[i]).dropna()[target_name])
                               for i in range(len(vals))])

    # Information gain is the reduction in entropy
    info_gain = total_entropy - weighted_entropy
    return info_gain


# Function to find the best attribute to split on
def best_split(data, target_name="buys_computer"):
    info_gains = {column: info_gain(data, column, target_name) for column in data.columns[:-1]}
    return max(info_gains, key=info_gains.get)


# Recursive function to build the decision tree
def build_tree(data, tree=None, target_name="buys_computer"):
    target_values = np.unique(data[target_name])

    # Base case 1: If all target values are the same, return that value
    if len(target_values) == 1:
        return target_values[0]

    # Base case 2: If dataset is empty or no more attributes, return the most common target value
    elif len(data.columns) == 1:
        return target_values[np.argmax(np.unique(data[target_name], return_counts=True)[1])]

    # Recursive case: find the best feature to split on and build the subtree
    else:
        best_feature = best_split(data, target_name)

        # Initialize the tree structure if not already initialized
        if tree is None:
            tree = {}
            tree[best_feature] = {}

        # Recursively build the tree for each value of the best feature
        for value in np.unique(data[best_feature]):
            sub_data = data.where(data[best_feature] == value).dropna()
            subtree = build_tree(sub_data.drop(columns=[best_feature]), target_name=target_name)
            tree[best_feature][value] = subtree

        return tree


# Build the decision tree
decision_tree = build_tree(df)
print("Decision Tree:", decision_tree)


# Function to classify a new instance using the built decision tree
def classify(instance, tree):
    if not isinstance(tree, dict):
        return tree  # If the tree is a leaf node, return the value

    # Traverse the tree
    attribute = next(iter(tree))
    if instance[attribute] in tree[attribute]:
        return classify(instance, tree[attribute][instance[attribute]])
    else:
        return "Unknown"


# Test the classifier with a new instance
new_instance = {'age': '<=30', 'income': 'medium', 'student': 'yes', 'credit_rating': 'excellent'}
print(f"Classified as: {classify(new_instance, decision_tree)}")

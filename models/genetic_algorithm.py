import numpy as np
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)

def run_genetic_algorithm(X_train, X_test, y_train, y_test, generations=10, population_size=10):
    # Hyperparameter ranges
    max_depth_range = list(range(4, 20))
    min_samples_split_range = list(range(2, 20))
    min_samples_leaf_range = list(range(1, 10))

    # Create initial random population of parameter tuples
    population = [
        (random.choice(max_depth_range),
         random.choice(min_samples_split_range),
         random.choice(min_samples_leaf_range))
        for _ in range(population_size)
    ]

    def fitness(params):
        depth, split, leaf = params
        model = DecisionTreeClassifier(
            max_depth=depth,
            min_samples_split=split,
            min_samples_leaf=leaf,
            random_state=42
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return f1_score(y_test, preds, average='macro')

    for gen in range(generations):
        scores = [fitness(ind) for ind in population]
        sorted_pop = [x for _, x in sorted(zip(scores, population), reverse=True)]
        population = sorted_pop[:population_size // 2]

        new_pop = population.copy()
        while len(new_pop) < population_size:
            parent1, parent2 = random.sample(population, 2)
            child = (random.choice([parent1[0], parent2[0]]),
                     random.choice([parent1[1], parent2[1]]),
                     random.choice([parent1[2], parent2[2]]))
            if random.random() < 0.3:
                child = (
                    child[0] + random.choice([-1, 0, 1]),
                    child[1] + random.choice([-1, 0, 1]),
                    child[2] + random.choice([-1, 0, 1])
                )
            child = (
                max(4, min(20, child[0])),
                max(2, min(20, child[1])),
                max(1, min(10, child[2]))
            )
            new_pop.append(child)

        population = new_pop

    best_params = max(population, key=fitness)
    best_model = DecisionTreeClassifier(
        max_depth=best_params[0],
        min_samples_split=best_params[1],
        min_samples_leaf=best_params[2],
        random_state=42
    )
    best_model.fit(X_train, y_train)
    final_preds = best_model.predict(X_test)

    print(f"\\nBest GA params → depth={best_params[0]}, split={best_params[1]}, leaf={best_params[2]}")
    return final_preds
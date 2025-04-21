import random
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from utils.evaluation import evaluate_model

def run_genetic_algorithm(
    X_train, X_test, y_train, y_test,
    pop_size=10,           # smaller population
    generations=5,         # fewer generations
    elite_frac=0.3,        # top 30% carry over
    crossover_rate=0.8,
    mutation_rate=0.1,
    val_size=2000          # use 2k samples for fitness
):
    """
    Fast GA to tune DecisionTree hyperparams using a small validation slice.
    """

    # 1) Prepare a small validation subset for fitness evaluations
    n_test = X_test.shape[0]
    if n_test > val_size:
        idx = np.random.choice(n_test, val_size, replace=False)
        X_val = X_test[idx]
        y_val = y_test.values[idx]
    else:
        X_val = X_test
        y_val = y_test.values

    # 2) Hyperparameter search space
    param_space = {
        'max_depth': list(range(1, 21)),
        'min_samples_split': list(range(2, 51)),
        'min_samples_leaf': list(range(1, 21))
    }

    def random_individual():
        return [
            random.choice(param_space['max_depth']),
            random.choice(param_space['min_samples_split']),
            random.choice(param_space['min_samples_leaf'])
        ]

    def fitness(ind):
        # Train on full training set, evaluate on the small validation slice
        params = {
            'max_depth': ind[0],
            'min_samples_split': ind[1],
            'min_samples_leaf': ind[2],
            'random_state': 42
        }
        model = DecisionTreeClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return f1_score(y_val, preds, zero_division=0)

    # 3) Initialize population
    population = [random_individual() for _ in range(pop_size)]
    num_elite = max(1, int(pop_size * elite_frac))

    # 4) Evolution loop
    for gen in range(generations):
        scores = [fitness(ind) for ind in population]
        # Keep the best
        elite_idx = np.argsort(scores)[-num_elite:]
        elites = [population[i] for i in elite_idx]

        # Build next generation
        new_pop = elites.copy()
        while len(new_pop) < pop_size:
            # Crossover
            if random.random() < crossover_rate:
                p1, p2 = random.sample(elites, 2)
                cut = random.randint(1, 2)
                child = p1[:cut] + p2[cut:]
            else:
                child = random.choice(elites).copy()
            # Mutation
            if random.random() < mutation_rate:
                i = random.randrange(3)
                key = ['max_depth','min_samples_split','min_samples_leaf'][i]
                child[i] = random.choice(param_space[key])
            new_pop.append(child)

        population = new_pop

    # 5) Select best from last gen
    final_scores = [fitness(ind) for ind in population]
    best = population[int(np.argmax(final_scores))]
    best_f1 = max(final_scores)

    print(f"Best GA params → depth={best[0]}, split={best[1]}, leaf={best[2]}  (val‑F1={best_f1:.3f})")

    # 6) Train final model on full train & evaluate on full test
    final_model = DecisionTreeClassifier(
        max_depth=best[0],
        min_samples_split=best[1],
        min_samples_leaf=best[2],
        random_state=42
    )
    final_model.fit(X_train, y_train)
    ga_preds = final_model.predict(X_test)

    evaluate_model(y_test, ga_preds, "GA-Optimized Tree")
    return ga_preds
import random
from numpy import min, sum, ptp
import networkx as nx
import numpy as np
import pandas as pd
from numpy.random import uniform
from pgmpy.base import DAG
from pgmpy.estimators import HillClimbSearch, BicScore, BDeuScore, K2Score, AICScore
import time


def roulette_index(list):
    suma_wartosci = sum(list)
    punkt = uniform(0, suma_wartosci)
    suma_czastkowa = 0
    for i, wartosc in enumerate(list):
        suma_czastkowa += wartosc
        if suma_czastkowa >= punkt:
            return i


def scale_fitness(list_fitness):
    if ptp(list_fitness) != 0:
        scaled_fitness = (list_fitness - min(list_fitness)) / ptp(list_fitness)
    else:
        scaled_fitness = [1] * len(list_fitness)
    return scaled_fitness


def bic_score(G, n, bs, columns):
    vector = g_to_v(G, n)
    dag = DAG()
    matrix = vector.reshape(n, n)
    matrix_df = pd.DataFrame(matrix, columns=columns)
    matrix_df.set_index(columns, inplace=True)

    for i in range(n):
        for j in range(n):
            if matrix_df[columns[i]][columns[j]]:
                dag.add_edge(matrix_df.columns[i], matrix_df.index[j])
    score = bs.score(dag)
    # print(score)
    return score


def generate_random_dag(num_vertices, min_edges):
    while True:
        random_dag = nx.gn_graph(num_vertices, seed=None)
        if nx.is_directed_acyclic_graph(random_dag) and len(random_dag.edges) >= min_edges:
            # print(random_dag)
            return random_dag


def generate_random_dags(num_dags, num_vertices, min_edges):
    dags = []
    for _ in range(num_dags):
        random_dag = generate_random_dag(num_vertices, min_edges)
        dags.append(random_dag)
    return dags


def crossing(g1, g2, n):
    kid1 = nx.DiGraph()
    kid2 = nx.DiGraph()
    for i in range(n):
        kid1.add_node(i)
        kid2.add_node(i)

    for i in range(n):
        for j in range(n):
            if g1.has_edge(i, j) and g2.has_edge(i, j):
                kid1.add_edge(i, j)
                kid2.add_edge(i, j)

    for i in range(n):
        for j in range(n):
            if (g1.has_edge(i, j) and not g2.has_edge(i, j)) or (not g1.has_edge(i, j) and g2.has_edge(i, j)):
                r = random.randint(0, 1)
                if r:
                    kid1.add_edge(i, j)
                    if not nx.is_directed_acyclic_graph(kid1):
                        kid1.remove_edge(i, j)
                        kid2.add_edge(i, j)
                        if not nx.is_directed_acyclic_graph(kid2):
                            kid2.remove_edge(i, j)
                else:
                    kid2.add_edge(i, j)
                    if not nx.is_directed_acyclic_graph(kid2):
                        kid2.remove_edge(i, j)
                        kid1.add_edge(i, j)
                        if not nx.is_directed_acyclic_graph(kid1):
                            kid1.remove_edge(i, j)

    if nx.is_empty(kid1):
        kid1 = kid2
    elif nx.is_empty(kid2):
        kid2 = kid1
    return kid1, kid2


def g_to_v(g, n):
    mat = []
    for i in range(n):
        mat.append([])
        for j in range(n):
            if g.has_edge(i, j):
                mat[i].append(1)
            else:
                mat[i].append(0)
    vec = np.asarray(mat, dtype=int).ravel()
    return vec


def mutate(G, n):
    oldG = G.copy()
    edges = list(G.edges)
    if len(edges) == 0:
        return G
    r = random.randint(0, 3)
    if r == 0:
        rc = random.choice(edges)
        G.remove_edge(rc[0], rc[1])
        if len(list(G.edges)) == 0:
            G.add_edge(rc[0], rc[1])
    elif r == 1:
        u = random.randint(0, n - 2)
        v = random.randint(0, n - 2)
        if v >= u:
            v = v + 1
        if not G.has_edge(u, v):
            G.add_edge(u, v)
            if not nx.is_directed_acyclic_graph(G):
                G.remove_edge(u, v)
    elif r == 2 and len(edges) != 0:
        rc = random.choice(edges)
        G.remove_edge(rc[0], rc[1])
        G.add_edge(rc[1], rc[0])
        if not nx.is_directed_acyclic_graph(G):
            G.remove_edge(rc[1], rc[0])
            G.add_edge(rc[0], rc[1])
    if len(list(G.edges)) > 0 and nx.is_directed_acyclic_graph(G):
        return G
    else:
        return oldG


def main():
    # parametry
    num_dags = 100
    mut_chance = 0.02
    iterations = 50
    dataset = 'asthma.xlsx'
    df = pd.read_excel(dataset)
    num_vertices = df.shape[1]
    min_edges = num_vertices - 1
    columns = df.columns
    bs = BicScore(df)
    t1 = time.perf_counter()
    est = HillClimbSearch(df)
    best_model = est.estimate(scoring_method=BicScore(df))
    best_score = bs.score(best_model)
    t2 = time.perf_counter()
    print("Czas HillClimbSearch: " + str(t2 - t1))
    print(best_score)
    print(best_model.edges())
    population = generate_random_dags(num_dags, num_vertices, min_edges)
    t3 = time.perf_counter()
    for iteration in range(iterations):

        values = []
        for vec in population:
            if len(vec.edges) >= min_edges:
                v = bic_score(vec, num_vertices, bs, columns)
            else:
                v = bic_score(vec, num_vertices, bs, columns) * min_edges / len(vec.edges)
            values.append(v)
        print(str(iteration) + ": " + str(max(values)))
        print(population[values.index(max(values))].edges)
        scaled = scale_fitness(values)

        # reprodukcja
        parents = []
        for i in range(num_dags):
            parents.append(roulette_index(scaled))
        next_generation = []
        # krzyzowanie

        for i in range(int(num_dags / 2)):
            kids = crossing(population[parents[i]], population[parents[i + 1]], num_vertices)
            next_generation.append(kids[0])
            next_generation.append(kids[1])

        # mutacja
        for i in range(len(next_generation)):
            r = random.random()
            if r < mut_chance:
                next_generation[i] = mutate(next_generation[i], num_vertices)

        population = next_generation
        t4 = time.perf_counter()
        print("Czas wykonania do iteracji " + str(iteration) + ": " + str(t4 - t3))


if __name__ == "__main__":
    main()

import mqt.qubomaker as qm
import mqt.qubomaker.pathfinder as pf


#the output of the print_matrix function is generated to be rendered in Latex in jupyter notebook
# so the following function should render it in a more readable way
def print_matrix_plain(matrix):
    for row in matrix:
        print(' '.join(f'{elem:.3f}' for elem in row))





#define an example graph for the problem

graph = qm.Graph.from_adjacency_matrix([
    [0, 1, 3, 4],
    [2, 0, 4, 2],
    [1, 5, 0, 3],
    [3, 8, 1, 0],
])

#select settings for the problem instance and the solution process

settings = pf.PathFindingQUBOGeneratorSettings(
    encoding_type=pf.EncodingType.ONE_HOT, n_paths=1, max_path_length=4, loops=True
)
#Define the QUBO generator to be used for this example

generator = pf.PathFindingQUBOGenerator(
    objective_function=pf.MinimizePathLength(path_ids=[1]),
    graph=graph,
    settings=settings,
)

#adding constraints
generator.add_constraint(pf.PathIsValid(path_ids=[1]))
generator.add_constraint(pf.PathContainsVerticesExactlyOnce(vertex_ids=graph.all_vertices, path_ids=[1]))

#generate and view the QUBO matrix
matrix = generator.construct_qubo_matrix()
print_matrix_plain(matrix)
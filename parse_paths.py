# -*- coding: utf-8 -*-
'''
svakulenko
29 Mar 2018

Parse shortest paths into subgraphs and collect stats:

print len(set(nodes)), 'nodes in the dialogue graph'
print len(set(edges)), 'edges in the dialogue graph'
# length of the shortest paths stats: min max mean
print min(paths_lengths), 'min length of the shortest paths'
print np.mean(paths_lengths), 'mean length of the shortest paths'
print max(paths_lengths), 'max length of the shortest paths'


Results on 6,480 annotated dialogue graphs (lines):
(1 Error parsing)

Size of the graph:
#nodes in the dialogue graph
    min: 4 mean: 27 max: 215
#edges in the dialogue graph
    min: 3 mean: 40 max: 394
#paths in the dialogue graph
    min: 5 mean: 20 max: 255

#nodes per path
min: 0 mean: 1.08736 max: 3
#edges per path
min: 0 mean: 1.64763 max: 4

#nodes per new entity (every 5 paths)
TODO
#edges per new entity (every 5 paths)
TODO

#anchor entities in the dialogue graph
    min: 1 mean: 4 max: 51

min length of the shortest paths
    min: 1 mean: 2.788 max: 6
mean length of the shortest paths
    min: 2.4 mean: 4.45686 max: 6
max length of the shortest paths
    min: 3 mean: 4 max: 7


'''
import numpy as np
from annotate_shortest_paths import PATH_SHORTEST_PATHS


def parse_paths_folder(folder='top5_widipedia', nfiles=1):
    '''
    folder top5_%s_%s top5_random_dbpedia
    nfiles <int> limit the number of files to parse
    '''
    # collect a number of stats for the dialogue graphs
    n_dialogues = 0  # count the number of dialogues managed to parse
    n_nodes = []
    n_edges = []
    n_paths = []
    nodes_per_path = []
    edges_per_path = []
    min_paths_lengths = []
    mean_paths_lengths = []
    max_paths_lengths = []

    metrics = {'nodes in the dialogue graph': n_nodes,
               'edges in the dialogue graph': n_edges,
               'paths in the dialogue graph': n_paths,
               '#nodes per path': nodes_per_path,
               '#edges per path': edges_per_path,
                # length of the shortest paths stats: min max mean
               'min length of the shortest paths': min_paths_lengths,
               'mean length of the shortest paths': mean_paths_lengths,
               'max length of the shortest paths': max_paths_lengths}

    files = os.listdir(folder)
    print len(files), 'files'

    for file in files[:nfiles]:
        print file
        with open(path, 'r') as paths_file:
        for line in paths_file.readlines():
            # skip dialogues with parsing errors
            try:
                # process 1 dialogue per line
                nodes = []
                edges = []
                paths_lengths = []  # collect distribution of the length of the shortest paths

                parse = line.split('\t')
                file = parse[0]
                # print file, 'dialogue'
                paths = parse[1:]
                for path in paths:
                    hops = path[1:-1].split('-<')
                    nhops = len(hops)  # path length
                    paths_lengths.append(nhops)
                    start_node = hops[0]
                    nodes.append(start_node)
                    for hop in hops[1:]:
                        edge_label, next_node = hop.split('>-')
                        nodes.append(next_node)
                        edges.append((start_node, next_node))
                        start_node = next_node
                
                n_nodes.append(len(set(nodes)))
                n_edges.append(len(set(edges)))
                n_paths.append(len(paths))
                min_paths_lengths.append(min(paths_lengths))
                mean_paths_lengths.append(np.mean(paths_lengths))
                max_paths_lengths.append(max(paths_lengths))
            except:
                print "Error parsing"
                continue
    # derivative metrics final
    nodes_per_path.extend(np.divide(n_nodes, n_paths))
    edges_per_path.extend(np.divide(n_edges, n_paths))

    # analyze the distribution for each of the metric (stats)
    report = "min: %f mean: %f max: %f"
    for metric_name, metric in metrics.items():
        print metric_name
        print report % (min(metric), np.mean(metric), max(metric))


def parse_paths(path=PATH_SHORTEST_PATHS, nlines=20000000):
    '''
    nlines <int> limit the number of dialogues (lines) to parse
    '''
    # collect a number of stats for the dialogue graphs
    n_dialogues = 0  # count the number of dialogues managed to parse
    n_nodes = []
    n_edges = []
    n_paths = []
    nodes_per_path = []
    edges_per_path = []
    min_paths_lengths = []
    mean_paths_lengths = []
    max_paths_lengths = []

    metrics = {'nodes in the dialogue graph': n_nodes,
               'edges in the dialogue graph': n_edges,
               'paths in the dialogue graph': n_paths,
               '#nodes per path': nodes_per_path,
               '#edges per path': edges_per_path,
                # length of the shortest paths stats: min max mean
               'min length of the shortest paths': min_paths_lengths,
               'mean length of the shortest paths': mean_paths_lengths,
               'max length of the shortest paths': max_paths_lengths}

    with open(path, 'r') as paths_file:
        for line in paths_file.readlines()[:nlines]:
            # skip dialogues with parsing errors
            try:
                # process 1 dialogue per line
                nodes = []
                edges = []
                paths_lengths = []  # collect distribution of the length of the shortest paths

                parse = line.split('\t')
                file = parse[0]
                # print file, 'dialogue'
                paths = parse[1:]
                for path in paths:
                    hops = path[1:-1].split('-<')
                    nhops = len(hops)  # path length
                    paths_lengths.append(nhops)
                    start_node = hops[0]
                    nodes.append(start_node)
                    for hop in hops[1:]:
                        edge_label, next_node = hop.split('>-')
                        nodes.append(next_node)
                        edges.append((start_node, next_node))
                        start_node = next_node
                
                n_nodes.append(len(set(nodes)))
                n_edges.append(len(set(edges)))
                n_paths.append(len(paths))
                min_paths_lengths.append(min(paths_lengths))
                mean_paths_lengths.append(np.mean(paths_lengths))
                max_paths_lengths.append(max(paths_lengths))
            except:
                print "Error parsing"
                continue

    # derivative metrics final
    nodes_per_path.extend(np.divide(n_nodes, n_paths))
    edges_per_path.extend(np.divide(n_edges, n_paths))

    # analyze the distribution for each of the metric (stats)
    report = "min: %f mean: %f max: %f"
    for metric_name, metric in metrics.items():
        print metric_name
        print report % (min(metric), np.mean(metric), max(metric))


if __name__ == '__main__':
     parse_paths_folder()

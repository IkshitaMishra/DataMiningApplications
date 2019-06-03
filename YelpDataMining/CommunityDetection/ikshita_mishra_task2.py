import sys
from itertools import combinations
from collections import deque
from pyspark import SparkConf, SparkContext
import time

start =  time.time()

sc = SparkContext()
sc.setLogLevel("WARN")
csvrdd = sc.textFile(sys.argv[2])
header = csvrdd.first()
csvrdd = csvrdd.filter(lambda line: line != header).map(lambda x: x.split(","))
rdd = csvrdd.map(lambda a: (str(a[1]), str(a[0]))).groupByKey().mapValues(set)
def filFunc(c):
    c = sorted(c)
    combs = combinations(c, 2)
    res = []
    for user_pair in combs:
        res.append((user_pair, 1))
    return res

threshold = int(sys.argv[1])

filterPhaseA = rdd.flatMap(lambda c: filFunc(c[1])).reduceByKey(lambda a, b: a + b).filter(lambda x: x[1] >= threshold).map(
    lambda x: x[0])
filterPhaseB = filterPhaseA.map(lambda x: (x[1], x[0]))
createEdges = (filterPhaseA.union(filterPhaseB)).groupByKey().mapValues(set)

nodeDict = {}
createEdgesList = createEdges.collect()
for i in createEdgesList:
    nodeDict[i[0]] = i[1]
nodes = createEdges.keys()


def calBetweenness(src, nodeDict):
    # =======================ASSIGN CREDITS======================
    # Select Source for BFS
    root = src
    nodes_seen = []
    parent_dict = {}
    nodes_queue = deque()
    nodes_queue.append(root)
    while nodes_queue:
        # Current node from list
        cur_node = nodes_queue.popleft()
        if cur_node != root:
            nodes_seen.append(cur_node)
        elif cur_node == root:
            # Check current node as root node
            initial_val = 0
            parent_dict[root] = [initial_val, []]
            nodes_seen.append(root)

        # Shortest dist from root
        shortest_dist = parent_dict[cur_node][0]
        adjacent_nodes = nodeDict[cur_node] - set(nodes_seen)
        x = adjacent_nodes - set(nodes_queue)
        for i in x:
            nodes_queue.append(i)

        for j in adjacent_nodes:
            if (j in parent_dict) and (parent_dict[j][0] == shortest_dist + 1):
                parent_dict[j][1].append(cur_node)
            elif j not in parent_dict:
                parent_dict[j] = [parent_dict[cur_node][0] + 1, [cur_node]]

    # Start with last node Hence reverse
    nodes_seen = nodes_seen[::-1]

    # =======================COMPUTATION OF BETWEENESS IN EDGES======================
    # Start from last node
    all_nodes, all_edges = dict(), dict()
    for node_to_calculate in nodes_seen:
        if node_to_calculate in all_nodes:
            all_nodes[node_to_calculate] = all_nodes[node_to_calculate] + 1
        else:
            all_nodes[node_to_calculate] = 1
        # Parents of nodes from dict
        parent_of_node = parent_dict[node_to_calculate][1]
        # Calculate number of parents
        num_of_parents = len(parent_of_node)
        # Get credit of current node
        credit_value = all_nodes[node_to_calculate]
        if num_of_parents != 0:
            add_val = float(float(credit_value) / num_of_parents)

        for parent in parent_of_node:
            edgeList = []
            edgeList.append(node_to_calculate)
            edgeList.append(parent)
            z = sorted(edgeList)
            all_edges[(z[0], z[1])] = add_val
            if parent in all_nodes:
                all_nodes[parent] = all_nodes[parent] + add_val
            else:
                all_nodes[parent] = add_val

    return all_edges.items()


edgeBet = nodes.flatMap(lambda src: calBetweenness(src, nodeDict)).groupByKey().mapValues(list).map(lambda x: (x[0], sum(x[1]) / 2)).collect()
res = sorted(edgeBet, key=lambda x: (-x[1],x[0]))

f = open(sys.argv[3], "w")
for i in res:
    f.write("('" + str(i[0][0]) + "', '" + str(i[0][1])+"')" + ", " + str(i[1]))
    f.write("\n")
curr_modularity = 0
m = len(res)

allNodesList = set(nodes.collect())



def connectedComp(nodeDict,allNodesList):
    allNodesList = allNodesList
    community = []
    while allNodesList:
        vert = allNodesList.pop()
        que = deque()
        visitVertex =  set()
        comp = set()
        que.append(vert)
        while que:
            v = que.popleft()
            visitVertex.add(v)
            comp.add(v)
            for child in nodeDict[v]:
                if child not in visitVertex:
                    que.append(child)
        community.append(comp)
        allNodesList = allNodesList - comp
    return community

maximumMod = -1.0

def getModularity(communities, nodeDict, m):
    edgeNum =m
    new_Mod = 0
    for c in communities:
        #print(c)
        ans = 0
        for m in c:
            for n in c:
                adj = nodeDict[m]
                A=0.0
                #print(adj)
                if n in adj:
                    A=1
                ans += A - (float(len(nodeDict[m]) * len(nodeDict[n])) / (2 * edgeNum))
        new_Mod += ans
    return  new_Mod / (2 * edgeNum)

final = []
n = len(res)

while n!=0:
    #print(n)
    node = res[0]
    i = node[0][0]
    j = node[0][1]
    nodeDict[i].remove(j)
    nodeDict[j].remove(i)
    communities = connectedComp(nodeDict,allNodesList)
    modularity = getModularity(communities, nodeDict, m)
    if maximumMod < modularity:
        maximumMod = modularity
        final = communities[:]
    edgeBet = nodes.flatMap(lambda src: calBetweenness(src, nodeDict)).groupByKey().mapValues(list).map(
        lambda x: (x[0], sum(x[1]) / 2)).collect()
    res = sorted(edgeBet, key=lambda x: (-x[1], x[0]))
    res = res[:]
    n=n-1

#print(maximumMod)
li =[]
for i in final:
    li.append(sorted(i))
#print(li)
reslen = sorted(list(set(len(x) for x in li)))
f1 = open(sys.argv[4], "w")
for i in reslen:
    can = []
    for j in li:
        if len(j) == i:
            can.append(j)
    can = sorted(can)
    for m in can:

        f1.write(str(m).replace("[", "").replace("]", ""))
        f1.write("\n")
end = time.time()
print("Duration", end-start)


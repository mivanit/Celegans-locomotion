from typing import *

import json
import matplotlib.pyplot as plt
import networkx as nx

from util import Path,joinPath

def plot_net(
		params : str = "../input/params/default.json",
		nrvsys : Union[str,List[str]] = ["Head", "VentralCord"],
		strict_fname : bool = True,
		show_weights : bool = True,
	):
	if params.endswith('/') or (strict_fname and not params.endswith('params.json')):
		params = joinPath(params, 'params.json')

	# figure out which nervous systems to plot
	if isinstance(nrvsys, str):
		nrvsys = nrvsys.split(',')

	# load network
	with open(params, 'r') as fin:
		data = json.load(fin)
	
	# create network
	G = nx.DiGraph()
	edges_byType : Dict[str,list] = {
		'ele' : list(), 'chem' : list(),
	}
	weights_byType : Dict[str,dict] = {
		'ele' : dict(), 'chem' : dict(),
	}

	for ns in nrvsys:
		for nrn in data[ns]["neurons"]:
			G.add_node(nrn)

		if ("n_units" in data[ns]) and (int(data[ns]["n_units"]) > 1):
			raise NotImplementedError()
			n_units = int(data[ns]["n_units"])
			for u in range(n_units):
				for conn in data[ns]["connections"]:
					G.add_edge(conn["from"],conn["to"])
				
		else:
			for conn in data[ns]["connections"]:
				G.add_edge(conn["from"], conn["to"])
				edges_byType[conn["type"]].append((conn["from"], conn["to"]))
				weights_byType[conn["type"]][(conn["from"], conn["to"])] = conn["weight"]

	print(G.nodes())
	print(G.edges())

	pos = nx.spring_layout(G)

	nx.draw_networkx_nodes(G, pos, node_size = 1500, node_color = '#E3FFB2')
	nx.draw_networkx_labels(G, pos)
	# draw chem (directed)
	nx.draw_networkx_edges(
		G, pos, 
		edgelist = edges_byType['chem'], edge_color='r', 
		arrows = True, arrowsize = 30, 
		connectionstyle = 'arc3,rad=0.2',
		min_target_margin  = 20, 
	)

	# draw ele (undirected)
	nx.draw_networkx_edges(G, pos, edgelist = edges_byType['ele'], edge_color='b', arrows = False)

	# draw weights
	if show_weights:
		nx.draw_networkx_edge_labels(
			G, pos,
			edge_labels = weights_byType['chem'],
		)
		nx.draw_networkx_edge_labels(
			G, pos,
			edge_labels = weights_byType['ele'],
		)

	plt.title(params)
	plt.show()


if __name__ == '__main__':
	import fire
	fire.Fire(plot_net)
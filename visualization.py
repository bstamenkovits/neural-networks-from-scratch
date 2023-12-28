from graphviz import Digraph


class Visualizer:
    """
    This class can only be used in combination with graphviz: `brew install 
    graphviz`
    """

    @staticmethod
    def trace_tree(root_node):
        '''get all of the nodes and edges of the data tree'''
        
        nodes, edges = set(), set()
        
        def trace(node):
            if node not in nodes:
                nodes.add(node)
                for child_node in node.children:
                    edges.add((child_node, node))
                    trace(child_node)
            
        trace(root_node)
        return nodes, edges

    def draw_diagram(self, tree, direction='TB'):  
        """
        Args:
        -----
            direction: direction in which the tree should be drawn 
            TB = Top-to-Bottom, LR = Left-to-Right
        """
        graph = Digraph(format='svg', graph_attr={'rankdir': direction})  
        
        nodes, edges = self.trace_tree(tree)
        for node in nodes:
            node_id = str(id(node))  # unique node id
            
            # all result_nodes
            graph.node(name=node_id, label=f"{node.get_label()} | {node.data:.3f} | ∇ = {node.gradient:.3f}", shape='record')  # record = rectangular
            
            # arrow from operation_node to result_node
            if node.operation:
                operation_node_name = node_id + node.operation
                graph.node(name=operation_node_name, label=node.operation)
                graph.edge(tail_name=operation_node_name, head_name=node_id)
        
        # arrow from child_node to result_node
        for node_1, node_2 in edges:                
            child_node_name = str(id((node_1)))
            operation_node_name = str(id((node_2))) + node_2.operation
            
            graph.edge(tail_name=child_node_name, head_name=operation_node_name)
        
        return graph

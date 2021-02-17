from src.layers.base_node import G_Node
from tensorflow.keras import Model
import graphviz

class G_Model(G_Node):
    def __init__(self,inputs,outputs,name,color="white"):
        super().__init__({"style":"filled","fillcolor":color,"peripheries":str(0)},**{"inputs":inputs,"outputs":outputs})
        self.inputs_used = 0
        inputs = list(map(lambda x:x. tenseur,inputs))
        outputs = list(map(lambda x:x. tenseur,outputs))
        self.keras_layer = Model(inputs=inputs,outputs=outputs)
        self.graph = None
    def __call__(self, inputs):
        for input in inputs:
            input.enfants.append(self)
        self.tenseur = self.keras_layer(list(map(lambda x:x.tenseur,inputs)))
        return self
    def link_to_inputs(self,input,parent_graph):
        if self.inputs_used == len(self.layer_params["inputs"]):
            raise Exception("All inputs used")
        else:
            parent_graph.edge(str(input.id),str(self.layer_params["inputs"][self.inputs_used].id),
                              label=str(input.tenseur.get_shape().as_list()))
            self.inputs_used += 1

    def render(self,out_path):
        graph = graphviz.Digraph(name="Main",format="png")
        for x in self.layer_params["outputs"]:
            x.output_node = True
        self.build(graph)

        graph.render(out_path)
    def build(self,parent_graph):
        with parent_graph.subgraph(name='cluster_'+str(self.id)) as c:
            for k,v in self.graphviz_params.items():
                c.graph_attr[k] = v
            for input in self.layer_params["inputs"]:
                input.build(c)
        for child,prev_out in zip(self.enfants,self.layer_params["outputs"]):
            parent_graph.edge(str(prev_out.id), str(child.id),
                              label=str(prev_out.tenseur.get_shape().as_list()))
            child.build(parent_graph)
        self.graph_done = True
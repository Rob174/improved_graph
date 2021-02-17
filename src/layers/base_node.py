
class G_Node:
    id = 0
    def __init__(self,graphviz_params,**layer_params):
        self.keras_layer = None
        G_Node.id += 1
        self.graphviz_params = {**graphviz_params,"style":"filled","fillcolor":"white"} \
            if "G_Model" not in self.__class__.__name__ else graphviz_params
        self.id = G_Node.id
        self.layer_params = layer_params
        self.tenseur = None
        self.parents = []# 1 layer n'a qu'un seul layer parent
        self.enfants = [] # un layer peut avoir plusieurs enfants
        self.graph_done = False
        self.output_node = False
    def __call__(self, input_node):
        if input_node.tenseur is None:
            raise Exception("No tensor input")
        else:
            input_node.enfants.append(self)
            self.parent = [input_node]
            if "G_Input" not in self.__class__.__name__:
                self.tenseur = self.keras_layer(input_node.tenseur)

            return self
    def link(self,parent_graph):
        for child in self.enfants:
            if "G_Model" in child.__class__.__name__:
                child.link_to_inputs(self,parent_graph)
            else:
                parent_graph.edge(str(self.id),str(child.id),label=str(self.tenseur.get_shape().as_list()))

        for child in self.enfants:
            child.build(parent_graph)
    def parse_str_args(self,arg_value):
        parsed_value = str(arg_value)
        if parsed_value.strip()[0] != "<":
            return parsed_value
        elif callable(arg_value):
            try:
                return arg_value.__name__
            except:
                return arg_value.__class__.__name__.split(".")[-1]
        else:
            try:
                return arg_value.name
            except Exception as e:
                print("EXCEPTION parsage en str d'un attribut")
                return " "
    def build(self,parent_graph):
        if len(list(filter(lambda p:p.graph_done is False,self.parents))) > 0:
            return
        label_attributs = " | ".join([("{%s|%s}" % (k.capitalize(), self.parse_str_args(v))) for k, v in self.layer_params.items()])
        label = "{{%s%s %d|{%s%s}}}" % (self.output()[0],self.__class__.__name__, self.id, self.output()[1],label_attributs)
        parent_graph.node(str(self.id), label=label, shape="record", **self.graphviz_params)
        self.graph_done = True
        self.link(parent_graph)

    def output(self):
        if self.output_node == True:
            return r"Output\n","{Output_shape|%s}|"%(str(self.tenseur.get_shape().as_list()))
        else:
            return "",""
from .base_node import G_Node
from tensorflow.keras.layers import Input,Conv2D,Dense,\
    MaxPooling2D,AveragePooling2D,\
    Concatenate,Dropout,Flatten


class G_Conv2D(G_Node):
    def __init__(self, **kargs):
        super().__init__({"color":"blue"},**kargs)
        self.keras_layer = Conv2D(**kargs)
class G_Dense(G_Node):
    def __init__(self, **kargs):
        super().__init__({"color":"blue"},**kargs)
        self.keras_layer = Dense(**kargs)
class G_Input(G_Node):
    def __init__(self, **kargs):
        super().__init__({"color":"yellow"},**kargs)
        self.keras_layer = Input(**kargs)
        self.tenseur = self.keras_layer
class G_MaxPooling2D(G_Node):
    def __init__(self, **kargs):
        super().__init__({"color":"red"},**kargs)
        self.keras_layer = MaxPooling2D(**kargs)
class G_AveragePooling2D(G_Node):
    def __init__(self, **kargs):
        super().__init__({"color":"orange"},**kargs)
        self.keras_layer = AveragePooling2D(**kargs)
class G_Dropout(G_Node):
    def __init__(self, **kargs):
        super().__init__({"color":"orange"},**kargs)
        self.keras_layer = Dropout(**kargs)
class G_Flatten(G_Node):
    def __init__(self, **kargs):
        super().__init__({"color":"orange"},**kargs)
        self.keras_layer = Flatten(**kargs)

class G_Concatenate(G_Node):
    def __init__(self, **kargs):
        super().__init__({"color":"black"},**kargs)
        self.keras_layer = Concatenate(**kargs)
    def __call__(self, input_nodes):
        if type(input_nodes) != list:
            raise Exception("A Concatenate layer should be called on a list of inputs")
        elif len(list(filter(lambda x:x.tenseur is None,input_nodes))) > 0:
            raise Exception("No tensor input")
        else:
            for node in input_nodes:
                node.enfants.append(self)
            self.parents = input_nodes
            self.tenseur = self.keras_layer(list(map(lambda x:x.tenseur,input_nodes)))
            return self
    def build(self,parent_graph):
        if len(list(filter(lambda p:p.graph_done is False,self.parents))) > 0:
            return
        label = "{{%s%s %d|%s}}" % (self.output()[0],self.__class__.__name__, self.id,self.output()[1])
        parent_graph.node(str(self.id), label=label, shape="record", **self.graphviz_params)
        self.link(parent_graph)
        self.graph_done = True
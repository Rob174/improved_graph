from IA.improved_graph.src.layers.base_node import G_Node
import tensorflow.keras.layers as layers


class Conv2D(G_Node):
    def __init__(self, **kargs):
        super().__init__({"color": "blue"}, **kargs)
        self.keras_layer = layers.Conv2D(**kargs)


class Dense(G_Node):
    def __init__(self, **kargs):
        super().__init__({"color": "blue"}, **kargs)
        self.keras_layer = layers.Dense(**kargs)


class Input(G_Node):
    def __init__(self, **kargs):
        super().__init__({"color": "yellow"}, **kargs)
        self.keras_layer = layers.Input(**kargs)
        self.tenseur = self.keras_layer


class MaxPooling2D(G_Node):
    def __init__(self, **kargs):
        super().__init__({"color": "red"}, **kargs)
        self.keras_layer = layers.MaxPooling2D(**kargs)


class AveragePooling2D(G_Node):
    def __init__(self, **kargs):
        super().__init__({"color": "orange"}, **kargs)
        self.keras_layer = layers.AveragePooling2D(**kargs)


class GlobalAveragePooling2D(G_Node):
    def __init__(self, **kargs):
        super().__init__({"color": "yellow"}, **kargs)
        self.keras_layer = layers.GlobalAveragePooling2D(**kargs)


class BatchNormalization(G_Node):
    def __init__(self, **kargs):
        super().__init__({"color": "brown"}, **kargs)
        self.keras_layer = layers.BatchNormalization(**kargs)


class Activation(G_Node):
    def __init__(self, **kargs):
        super().__init__({"color": "green"}, **kargs)
        self.keras_layer = layers.Activation(**kargs)


class SeparableConv2D(G_Node):
    def __init__(self, **kargs):
        super().__init__({"color": "blue"}, **kargs)
        self.keras_layer = layers.SeparableConv2D(**kargs)


class Dropout(G_Node):
    def __init__(self, **kargs):
        super().__init__({"color": "black"}, **kargs)
        self.keras_layer = layers.Dropout(**kargs)


class Flatten(G_Node):
    def __init__(self, **kargs):
        super().__init__({"color": "black"}, **kargs)
        self.keras_layer = layers.Flatten(**kargs)


class Concatenate(G_Node):
    def __init__(self, **kargs):
        super().__init__({"color": "black"}, **kargs)
        self.keras_layer = layers.Concatenate(**kargs)

    def __call__(self, input_nodes):
        if type(input_nodes) != list:
            raise Exception("A Concatenate layer should be called on a list of inputs")
        elif len(list(filter(lambda x: x.tenseur is None, input_nodes))) > 0:
            raise Exception("No tensor input")
        else:
            for node in input_nodes:
                node.enfants.append(self)
            self.parents = input_nodes
            self.tenseur = self.keras_layer(list(map(lambda x: x.tenseur, input_nodes)))
            return self

    def build(self, parent_graph):
        if len(list(filter(lambda p: p.graph_done is False, self.parents))) > 0:
            return
        label = "{{%s%s %d|%s}}" % (self.output()[0], self.__class__.__name__, self.id, self.output()[1])
        parent_graph.node(str(self.id), label=label, shape="record", **self.graphviz_params)
        self.link(parent_graph)
        self.graph_done = True


class Add(Concatenate):
    def __init__(self, **kargs):
        G_Node.__init__(self,{"color": "black"}, **kargs)
        self.keras_layer = layers.Add(**kargs)
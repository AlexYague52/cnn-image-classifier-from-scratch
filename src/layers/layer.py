class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        pass

    def backward(self, output_gradient, learning_rate):
        pass

    def get_params(self):
        return {}

    def set_params(self, params):
        pass
import torch 
import torch.nn as nn 
import numpy as np 
from typing import List
 
# Define utility classes 
 
class ConvBnRelu(nn.Module): 
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0): 
        super(ConvBnRelu, self).__init__() 
        # Convert to Python int to avoid numpy types that break TorchScript
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False) 
        self.bn = nn.BatchNorm2d(out_channels) 
        self.relu = nn.ReLU(inplace=True) 
    def forward(self, x): 
        return self.relu(self.bn(self.conv(x))) 
 

def Projection(in_channels, out_channels): 
    return ConvBnRelu(in_channels, out_channels, 1) 
 
def Truncate(x: torch.Tensor, channels: int) -> torch.Tensor:
    # Truncate to channels if needed 
    if x.size(1) > channels: 
        return x[:, :channels, :, :] 
    else:
        return x

def ComputeVertexChannels(in_channels, out_channels, matrix): 
    # Compute the channels for each vertex based on matrix connections to output 
    num_vertices = matrix.shape[0] 
    vertex_channels = [0] * num_vertices 
    vertex_channels[0] = in_channels 
    vertex_channels[-1] = out_channels 
    if num_vertices == 2: 
        return vertex_channels 
    # in-degree (excluding input node) of each vertex 
    in_degree = matrix[1:].sum(axis=0) 
    # Number of edges that go to the output vertex from intermediate nodes
    out_edges = in_degree[-1] 
    # Check if input connects directly to output
    has_input_to_output = bool(matrix[0, -1])
    
    # If input connects to output, we need to account for it in channel distribution
    if has_input_to_output:
        # Total edges to output including input edge
        total_out_edges = out_edges + 1
        # Distribute channels among all edges to output
        interior_channels = out_channels // total_out_edges 
        remainder = out_channels % total_out_edges
    else:
        # Original logic when input doesn't connect to output
        interior_channels = out_channels // out_edges if out_edges > 0 else 0
        remainder = out_channels % out_edges if out_edges > 0 else 0
    
    # assign channels to vertices connecting to output 
    for v in range(1, num_vertices - 1): 
        if matrix[v, -1]: 
            vertex_channels[v] = interior_channels 
            if remainder > 0: 
                vertex_channels[v] += 1 
                remainder -= 1 
    # Propagate channels backwards to interior vertices (not directly connecting to output) 
    for v in range(num_vertices - 3, 0, -1): 
        if not matrix[v, -1]: 
            # find max channel among its successors 
            max_succ = 0 
            for dst in range(v + 1, num_vertices): 
                if matrix[v, dst]: 
                    max_succ = max(max_succ, vertex_channels[dst]) 
            vertex_channels[v] = max_succ 
    # Convert all to Python int to avoid numpy types
    return [int(c) for c in vertex_channels] 
 
class Cell(nn.Module): 
    def __init__(self, matrix, ops, in_channels, out_channels): 
        super(Cell, self).__init__() 
        # Convert numpy array to torch tensor for TorchScript compatibility
        if isinstance(matrix, np.ndarray):
            matrix = matrix.tolist()
        self.matrix = torch.tensor(matrix, dtype=torch.int32)
        self.ops = ops 
        self.num_vertices: int = int(self.matrix.shape[0]) 
        # ComputeVertexChannels still uses numpy internally
        matrix_np = np.array(matrix)
        vertex_channels_list = ComputeVertexChannels(in_channels, out_channels, matrix_np)
        # Store as a list of ints for TorchScript compatibility
        self.vertex_channels: List[int] = vertex_channels_list
        
        # Compute channel count for input-to-output projection if it exists
        self.has_input_to_output = bool(self.matrix[0, -1].item())
        if self.has_input_to_output:
            # Count intermediate vertices connecting to output
            num_intermediate_to_output = sum(1 for v in range(1, self.num_vertices - 1) if self.matrix[v, -1].item())
            total_out_edges = num_intermediate_to_output + 1  # +1 for input edge
            self.input_to_output_channels = out_channels // total_out_edges
            # Give remainder to input projection
            if out_channels % total_out_edges > num_intermediate_to_output:
                self.input_to_output_channels += 1
        else:
            self.input_to_output_channels = 0
            
        # create ops per vertex (except input and output) 
        self.vertex_ops = nn.ModuleList() 
        self.vertex_ops.append(None)  # placeholder for input vertex 
        for i in range(1, self.num_vertices - 1): 
            op_name = ops[i] 
            if op_name == 'conv1x1': 
                self.vertex_ops.append(ConvBnRelu(self.vertex_channels[i], self.vertex_channels[i], 1)) 
            elif op_name == 'conv3x3': 
                self.vertex_ops.append(ConvBnRelu(self.vertex_channels[i], self.vertex_channels[i], 3, padding=1)) 
            elif op_name == 'max_pool_3x3': 
                self.vertex_ops.append(nn.MaxPool2d(kernel_size=3, stride=1, padding=1)) 
            elif op_name == 'conv1x1-bn-relu': 
                self.vertex_ops.append(ConvBnRelu(self.vertex_channels[i], self.vertex_channels[i], 1)) 
            elif op_name == 'conv3x3-bn-relu': 
                self.vertex_ops.append(ConvBnRelu(self.vertex_channels[i], self.vertex_channels[i], 3, padding=1))
            elif op_name == 'maxpool3x3':
                self.vertex_ops.append(nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
            else: 
                raise ValueError(f'Unknown operation: {op_name}') 
        # projection from stem (input node) to each node that has an incoming edge from input 
        self.input_proj = nn.ModuleList() 
        self.input_proj.append(None)  # placeholder for input 
        for i in range(1, self.num_vertices): 
            if self.matrix[0, i]: 
                # Special handling for output node when input connects to it
                if i == self.num_vertices - 1:  # output node
                    target_channels = self.input_to_output_channels
                else:
                    target_channels = self.vertex_channels[i]
                self.input_proj.append(Projection(in_channels, target_channels)) 
            else: 
                self.input_proj.append(None) 
     
    def forward(self, x): 
        tensors = [x]  # tensors[i] holds output of vertex i 
        # process intermediate vertices (excluding input and output) 
        for v in range(1, self.num_vertices - 1): 
            # gather inputs from previous vertices (edges src->v) 
            inputs = [] 
            # from other intermediate nodes 
            for src in range(1, v): 
                if self.matrix[src, v].item() != 0: 
                    inputs.append(Truncate(tensors[src], self.vertex_channels[v])) 
            # from the stem (input node) 
            if self.matrix[0, v].item() != 0: 
                inputs.append(self.input_proj[v](x)) 
            # sum all inputs 
            if len(inputs) == 0: 
                # no incoming edges -> zero tensor 
                vertex_input = torch.zeros_like(x)  # placeholder, maybe not correct, but for safety 
            elif len(inputs) == 1: 
                vertex_input = inputs[0] 
            else: 
                vertex_input = sum(inputs) 
            # apply vertex operation 
            vertex_output = self.vertex_ops[v](vertex_input) 
            tensors.append(vertex_output) 
        # gather outputs from vertices that connect to output node 
        outputs = [] 
        for src in range(1, self.num_vertices - 1): 
            if self.matrix[src, -1].item() != 0: 
                outputs.append(tensors[src]) 
        # also add output from stem if edge from input to output 
        if self.matrix[0, -1].item() != 0: 
            outputs.append(self.input_proj[-1](x)) 
        # combine outputs 
        if len(outputs) == 0: 
            raise RuntimeError('Cell has no connections to output node') 
        elif len(outputs) == 1: 
            return outputs[0] 
        else: 
            # concatenate outputs along channel dimension 
            return torch.cat(outputs, dim=1) 
 
class Network(nn.Module): 
    def __init__(self, matrix, ops, stem_channels=128, num_stacks=3, num_modules_per_stack=3, num_classes=10): 
        super(Network, self).__init__() 
        self.stem = ConvBnRelu(3, stem_channels, 3, padding=1) 
        self.layers = nn.ModuleList() 
        in_channels = stem_channels 
        out_channels = stem_channels 
        for stack in range(num_stacks): 
            if stack > 0: 
                # downsample at start of each stack after the first 
                self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2)) 
                out_channels *= 2 
            for _ in range(num_modules_per_stack): 
                self.layers.append(Cell(matrix, ops, in_channels, out_channels)) 
                in_channels = out_channels 
        # final classifier 
        self.classifier = nn.Linear(out_channels, num_classes) 
     
    def forward(self, x): 
        x = self.stem(x) 
        for layer in self.layers: 
            x = layer(x) 
        # global average pooling 
        x = x.mean([2, 3]) 
        x = self.classifier(x) 
        return x 
 
# Example usage with a random genotype 
if __name__ == '__main__': 
    # Example genotype (5 intermediate nodes) for test 
    matrix = [[0, 1, 1, 0, 0, 0, 0], 
             [0, 0, 0, 1, 1, 0, 0], 
             [0, 0, 0, 0, 1, 0, 0], 
             [0, 0, 0, 0, 0, 1, 0], 
             [0, 0, 0, 0, 0, 1, 0], 
             [0, 0, 0, 0, 0, 0, 1], 
             [0, 0, 0, 0, 0, 0, 0]] 
    ops = ['input', 'conv3x3', 'conv1x1', 'conv3x3', 'max_pool_3x3', 'max_pool_3x3', 'output'] 
    model = Network(matrix, ops) 
    print(model) 
    x = torch.randn(1, 3, 32, 32) 
    y = model(x) 
    print('Output shape:', y.shape) 
     
    # Count parameters 
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad) 
    print('Total trainable parameters:', total_params) 
 
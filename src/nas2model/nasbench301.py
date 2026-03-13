import torch
import torch.nn as nn
from collections import namedtuple

# ============================================================
#  Primitive operations (DARTS-style)
# ============================================================

class Zero(nn.Module):
    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class FactorizedReduce(nn.Module):
    """Downsample by 2 with two parallel 1x1 convs, then concat."""
    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu   = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn     = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x),
                         self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


class SepConv(nn.Module):
    """Depthwise separable conv with BN (used by sep_conv_3x3 / sep_conv_5x5)."""
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride,
                      padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1,
                      padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):
    """Dilated depthwise separable conv with BN (for dil_conv_3x3 / 5x5)."""
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation,
                      groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)


class ReLUConvBN(nn.Module):
    """Conv + BN with a preceding ReLU (used in the cell preprocess)."""
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)


# OPS: map the string in the genotype to an actual nn.Module constructor.
OPS = {
    'none'          : lambda C_in, C_out, stride, affine: Zero(stride),
    'max_pool_3x3'  : lambda C_in, C_out, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
    'avg_pool_3x3'  : lambda C_in, C_out, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1),
    'skip_connect'  : lambda C_in, C_out, stride, affine: Identity() if stride == 1
                                                          else FactorizedReduce(C_in, C_out, affine=affine),
    'sep_conv_3x3'  : lambda C_in, C_out, stride, affine: SepConv(C_in, C_out, 3, stride, 1, affine=affine),
    'sep_conv_5x5'  : lambda C_in, C_out, stride, affine: SepConv(C_in, C_out, 5, stride, 2, affine=affine),
    'dil_conv_3x3'  : lambda C_in, C_out, stride, affine: DilConv(C_in, C_out, 3, stride, 2, dilation=2, affine=affine),
    'dil_conv_5x5'  : lambda C_in, C_out, stride, affine: DilConv(C_in, C_out, 5, stride, 4, dilation=2, affine=affine),
}

# ============================================================
#  Genotype & Cell
# ============================================================

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')


def json_to_genotype(geno_dict):
    """
    geno_dict is the 'genotype' field from your JSON:
      {
        "normal": [...],
        "normal_concat": [...],
        "reduce": [...],
        "reduce_concat": [...]
      }
    """
    return Genotype(
        normal        = geno_dict['normal'],
        normal_concat = geno_dict['normal_concat'],
        reduce        = geno_dict['reduce'],
        reduce_concat = geno_dict['reduce_concat']
    )


class Cell(nn.Module):
    """
    DARTS-style cell that uses the NB301 genotype.
    """
    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        self.reduction = reduction

        # preprocess the two inputs
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)

        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)

        # choose which part of the genotype to use
        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat

        # In DARTS, there are 4 "steps" (4 intermediate nodes),
        # and concat is usually [2,3,4,5].
        self._steps = len(concat)             # 4
        self._concat = concat
        self.multiplier = len(concat)
        self._indices = indices               # list of source node indices

        self._ops = nn.ModuleList()
        for op_name, idx in zip(op_names, indices):
            # For reduction cells, edges from the 2 input nodes (0 or 1)
            # may need stride=2.
            stride = 2 if reduction and idx < 2 else 1
            op = OPS[op_name](C, C, stride, affine=False)
            self._ops.append(op)

    def forward(self, s0, s1):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]

        # Each intermediate node i uses two edges: edges 2*i and 2*i+1
        for i in range(self._steps):
            h1 = states[self._indices[2*i]]
            h2 = states[self._indices[2*i + 1]]
            op1 = self._ops[2*i]
            op2 = self._ops[2*i + 1]
            s = op1(h1) + op2(h2)
            states.append(s)

        # Concatenate selected nodes (normal_concat / reduce_concat)
        return torch.cat([states[i] for i in self._concat], dim=1)

# ============================================================
#  Full network (CIFAR-style DARTS macro-architecture)
# ============================================================

class Network(nn.Module):
    """
    DARTS-style CIFAR network that uses NAS-Bench-301 genotypes.

    Typical settings:
      - C = 36
      - layers = 8, 16, or 20
      - num_classes = 10 (CIFAR-10) or 100 (CIFAR-100)
    """
    def __init__(self, C, num_classes, layers, genotype):
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers

        stem_multiplier = 3
        C_curr = stem_multiplier * C

        # initial stem convolution
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev = C_curr, C_curr
        self.cells = nn.ModuleList()
        reduction_prev = False

        for i in range(layers):
            # place reduction cells at 1/3 and 2/3 of total depth (DARTS convention)
            if i in [layers // 3, 2 * layers // 3]:
                reduction = True
            else:
                reduction = False

            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            self.cells.append(cell)
            reduction_prev = reduction

            # After a cell: update channel bookkeeping
            C_prev_prev, C_prev = C_prev, C_curr * cell.multiplier

            # if just did reduction, increase base C_curr for the next cells
            if reduction:
                C_curr *= 2

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, x):
        s0 = s1 = self.stem(x)
        for cell in self.cells:
            s0, s1 = s1, cell(s0, s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

# ============================================================
#  Example: build a PyTorch model from your JSON
# ============================================================

if __name__ == '__main__':
    import json

    json_str = """
{
  "rank": 2,
  "predicted_accuracy": 94.57183074951172,
  "sample_index": 2475,
  "genotype": {
    "normal": [
      ["sep_conv_5x5", 0],
      ["sep_conv_5x5", 1],
      ["skip_connect", 0],
      ["dil_conv_5x5", 2],
      ["skip_connect", 0],
      ["sep_conv_5x5", 1],
      ["sep_conv_3x3", 1],
      ["dil_conv_5x5", 2]
    ],
    "normal_concat": [2, 3, 4, 5],
    "reduce": [
      ["max_pool_3x3", 0],
      ["sep_conv_3x3", 1],
      ["sep_conv_3x3", 0],
      ["avg_pool_3x3", 1],
      ["sep_conv_5x5", 0],
      ["dil_conv_3x3", 3],
      ["sep_conv_5x5", 0],
      ["dil_conv_3x3", 3]
    ],
    "reduce_concat": [2, 3, 4, 5]
  }
}
    """

    data = json.loads(json_str)
    geno_dict = data['genotype']
    genotype = json_to_genotype(geno_dict)

    # Build a DARTS-style network for CIFAR-10
    model = Network(C=36, num_classes=10, layers=8, genotype=genotype)
    print(model)

    # Quick forward test
    x = torch.randn(1, 3, 32, 32)  # CIFAR-sized fake image
    logits = model(x)
    print('Output shape:', logits.shape)  # should be [1, 10]
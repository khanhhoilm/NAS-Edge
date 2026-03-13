import copy
import torch
import torch.nn as nn


# -----------------------------
# Basic ops used in NAS-Bench-201
# -----------------------------

class Zero(nn.Module):
    def __init__(self, C_in, C_out, stride):
        super(Zero, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.stride = stride

    def forward(self, x):
        if self.C_in == self.C_out:
            if self.stride == 1:
                return x.mul(0.0)
            else:
                return x[:, :, :: self.stride, :: self.stride].mul(0.0)
        else:
            shape = list(x.shape)
            shape[1] = self.C_out
            zeros = x.new_zeros(shape, dtype=x.dtype, device=x.device)
            return zeros

    def extra_repr(self):
        return f"C_in={self.C_in}, C_out={self.C_out}, stride={self.stride}"


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class ReLUConvBN(nn.Module):
    def __init__(
        self,
        C_in,
        C_out,
        kernel_size,
        stride,
        padding,
        dilation,
        affine,
        track_running_stats=True,
    ):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                C_in,
                C_out,
                kernel_size,
                stride,
                padding,
                dilation,
                bias=not affine,
            ),
            nn.BatchNorm2d(
                C_out, affine=affine, track_running_stats=track_running_stats
            ),
        )

    def forward(self, x):
        return self.op(x)


class FactorizedReduce(nn.Module):
    def __init__(
        self, C_in, C_out, stride, affine=True, track_running_stats=True
    ):
        super(FactorizedReduce, self).__init__()
        self.stride = stride
        self.C_in = C_in
        self.C_out = C_out
        self.relu = nn.ReLU(inplace=False)
        if stride == 2:
            C_outs = [C_out // 2, C_out - C_out // 2]
            self.convs = nn.ModuleList()
            self.convs.append(
                nn.Conv2d(C_in, C_outs[0], 1, stride, 0, bias=not affine)
            )
            self.convs.append(
                nn.Conv2d(C_in, C_outs[1], 1, stride, 0, bias=not affine)
            )
            self.pad = nn.ConstantPad2d((0, 1, 0, 1), 0)
        elif stride == 1:
            self.conv = nn.Conv2d(
                C_in, C_out, 1, stride, 0, bias=not affine
            )
        else:
            raise ValueError(f"Invalid stride {stride}")
        self.bn = nn.BatchNorm2d(
            C_out, affine=affine, track_running_stats=track_running_stats
        )

    def forward(self, x):
        if self.stride == 2:
            x = self.relu(x)
            out = self.convs[0](x) + self.convs[1](
                self.pad(x)[:, :, 1:, 1:]
            )
        else:
            out = self.conv(x)
        return self.bn(out)

    def extra_repr(self):
        return f"C_in={self.C_in}, C_out={self.C_out}, stride={self.stride}"


class Pool(nn.Module):
    def __init__(
        self, C_in, C_out, stride, mode, affine=True, track_running_stats=True
    ):
        super(Pool, self).__init__()
        if C_in == C_out:
            self.preprocess = None
        else:
            self.preprocess = ReLUConvBN(
                C_in,
                C_out,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                affine=affine,
                track_running_stats=track_running_stats,
            )
        if mode == "avg":
            self.op = nn.AvgPool2d(
                3, stride=stride, padding=1, count_include_pad=False
            )
        elif mode == "max":
            self.op = nn.MaxPool2d(3, stride=stride, padding=1)
        else:
            raise ValueError(f"Invalid mode {mode} in Pool")

    def forward(self, x):
        if self.preprocess is not None:
            x = self.preprocess(x)
        return self.op(x)


class ResNetBasicblock(nn.Module):
    def __init__(
        self, inplanes, planes, stride, affine=True, track_running_stats=True
    ):
        super(ResNetBasicblock, self).__init__()
        assert stride in (1, 2), f"invalid stride {stride}"
        self.conv_a = ReLUConvBN(
            inplanes, planes, 3, stride, 1, 1, affine, track_running_stats
        )
        self.conv_b = ReLUConvBN(
            planes, planes, 3, 1, 1, 1, affine, track_running_stats
        )
        if stride == 2:
            self.downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                nn.Conv2d(
                    inplanes,
                    planes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
            )
        elif inplanes != planes:
            self.downsample = ReLUConvBN(
                inplanes, planes, 1, 1, 0, 1, affine, track_running_stats
            )
        else:
            self.downsample = None
        self.in_dim = inplanes
        self.out_dim = planes
        self.stride = stride
        self.num_conv = 2

    def extra_repr(self):
        return (
            f"{self.__class__.__name__}(inC={self.in_dim}, "
            f"outC={self.out_dim}, stride={self.stride})"
        )

    def forward(self, inputs):
        basicblock = self.conv_a(inputs)
        basicblock = self.conv_b(basicblock)
        residual = (
            self.downsample(inputs)
            if self.downsample is not None
            else inputs
        )
        return residual + basicblock


# -----------------------------
# NAS-Bench-201 cell Structure
# -----------------------------

class Structure:
    """
    Direct translation of NAS-Bench-201 / xautodl 'Structure' for cells.
    Each node i (1..3) has a tuple of (op_name, input_node_index).
    """

    def __init__(self, genotype):
        assert isinstance(genotype, (list, tuple)), "invalid genotype type"
        self.node_num = len(genotype) + 1
        self.nodes = []
        self.node_N = []
        for idx, node_info in enumerate(genotype):
            assert isinstance(node_info, (list, tuple)), "invalid node_info type"
            assert len(node_info) >= 1, "node must have at least one input"
            for node_in in node_info:
                assert isinstance(node_in, (list, tuple)), "invalid in-node type"
                assert (
                    len(node_in) == 2 and node_in[1] <= idx
                ), f"invalid in-node: {node_in}"
            self.node_N.append(len(node_info))
            self.nodes.append(tuple(copy.deepcopy(node_info)))

    def tostr(self):
        strings = []
        for node_info in self.nodes:
            string = "|".join([f"{op}~{idx}" for op, idx in node_info])
            string = f"|{string}|"
            strings.append(string)
        return "+".join(strings)

    def __len__(self):
        return len(self.nodes) + 1

    def __getitem__(self, index):
        return self.nodes[index]

    @staticmethod
    def str2structure(xstr):
        """
        Parse a NAS-Bench-201 arch string like:
        |nor_conv_3x3~0|+|nor_conv_1x1~0|nor_conv_3x3~1|+|skip_connect~0|...
        into a Structure object.
        """
        if isinstance(xstr, Structure):
            return xstr
        assert isinstance(xstr, str), f"must take string, got {type(xstr)}"
        nodestrs = xstr.split("+")
        genotypes = []
        for node_str in nodestrs:
            inputs = list(filter(lambda x: x != "", node_str.split("|")))
            for xinput in inputs:
                assert (
                    len(xinput.split("~")) == 2
                ), f"invalid input format: {xinput}"
            inputs = (xi.split("~") for xi in inputs)
            input_infos = tuple((op, int(idx)) for (op, idx) in inputs)
            genotypes.append(input_infos)
        return Structure(genotypes)


# -----------------------------
# OPS dict for NAS-Bench-201
# -----------------------------

OPS = {
    "none": lambda C_in, C_out, stride, affine=True, track_running_stats=True: Zero(
        C_in, C_out, stride
    ),
    "skip_connect": lambda C_in, C_out, stride, affine=True, track_running_stats=True: (
        Identity()
        if stride == 1 and C_in == C_out
        else FactorizedReduce(
            C_in, C_out, stride, affine, track_running_stats
        )
    ),
    "nor_conv_1x1": lambda C_in, C_out, stride, affine=True, track_running_stats=True: ReLUConvBN(
        C_in, C_out, 1, stride, 0, 1, affine, track_running_stats
    ),
    "nor_conv_3x3": lambda C_in, C_out, stride, affine=True, track_running_stats=True: ReLUConvBN(
        C_in, C_out, 3, stride, 1, 1, affine, track_running_stats
    ),
    "avg_pool_3x3": lambda C_in, C_out, stride, affine=True, track_running_stats=True: Pool(
        C_in, C_out, stride, "avg", affine, track_running_stats
    ),
    "max_pool_3x3": lambda C_in, C_out, stride, affine=True, track_running_stats=True: Pool(
        C_in, C_out, stride, "max", affine, track_running_stats
    ),
}


# -----------------------------
# NAS-Bench-201 InferCell
# -----------------------------

class InferCell(nn.Module):
    """
    A fixed NAS-Bench-201 cell (4 nodes: 0..3).
    genotype: Structure (list of node infos for nodes 1,2,3).
    """

    def __init__(
        self, genotype, C_in, C_out, stride, affine=True, track_running_stats=True
    ):
        super(InferCell, self).__init__()
        self.layers = nn.ModuleList()
        self.node_IN = []
        self.node_IX = []
        self.genotype = copy.deepcopy(genotype)

        # genotype.nodes: for node-1, node-2, node-3
        for i in range(1, len(genotype.nodes) + 1):
            node_info = genotype[i - 1]  # inputs to node i
            cur_index = []
            cur_innod = []
            for (op_name, op_in) in node_info:
                if op_in == 0:
                    # from cell input
                    layer = OPS[op_name](
                        C_in, C_out, stride, affine, track_running_stats
                    )
                else:
                    # from intermediate node
                    layer = OPS[op_name](
                        C_out, C_out, 1, affine, track_running_stats
                    )
                cur_index.append(len(self.layers))
                cur_innod.append(op_in)
                self.layers.append(layer)
            self.node_IX.append(cur_index)
            self.node_IN.append(cur_innod)

        self.nodes = len(genotype.nodes)
        self.in_dim = C_in
        self.out_dim = C_out

    def forward(self, x):
        nodes = [x]  # node-0 is input
        for node_layers, node_innods in zip(self.node_IX, self.node_IN):
            node_feature = sum(
                self.layers[l_idx](nodes[in_idx])
                for l_idx, in_idx in zip(node_layers, node_innods)
            )
            nodes.append(node_feature)
        return nodes[-1]


# -----------------------------
# TinyNetwork macro-architecture
# -----------------------------

class TinyNetwork(nn.Module):
    """
    NAS-Bench-201 CIFAR network:
    - Stem conv
    - 3 stages of cells with channel sizes [C, 2C, 4C]
    - N cells per stage, with 2 reduction blocks between stages
    """

    def __init__(self, C, N, genotype, num_classes):
        super(TinyNetwork, self).__init__()
        self._C = C
        self._layerN = N

        self.stem = nn.Sequential(
            nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )

        layer_channels = (
            [C] * N
            + [C * 2]
            + [C * 2] * N
            + [C * 4]
            + [C * 4] * N
        )
        layer_reductions = (
            [False] * N
            + [True]
            + [False] * N
            + [True]
            + [False] * N
        )

        C_prev = C
        self.cells = nn.ModuleList()
        for C_curr, reduction in zip(layer_channels, layer_reductions):
            if reduction:
                cell = ResNetBasicblock(C_prev, C_curr, 2, True)
            else:
                cell = InferCell(genotype, C_prev, C_curr, 1)
            self.cells.append(cell)
            C_prev = cell.out_dim

        self._Layer = len(self.cells)
        self.lastact = nn.Sequential(
            nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True)
        )
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward_features(self, x):
        """Return pooled penultimate features used before the classifier."""
        feature = self.stem(x)
        for cell in self.cells:
            feature = cell(feature)
        out = self.lastact(feature)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        return out

    def forward(self, x):
        out = self.forward_features(x)
        logits = self.classifier(out)
        return logits


# -----------------------------
# Example: build your arch and test
# -----------------------------

def build_nasbench201_model_from_string(
    arch_str,
    num_classes=10,
    C=16,
    N=5,
):
    """
    Convenience function:
    - parses NAS-Bench-201 arch_str
    - returns TinyNetwork ready for CIFAR-10.
    """
    genotype = Structure.str2structure(arch_str)
    model = TinyNetwork(C=C, N=N, genotype=genotype, num_classes=num_classes)
    return model


if __name__ == "__main__":
    # Your architecture string from JSON
    arch_str = (
        "|nor_conv_3x3~0|"
        "+|nor_conv_1x1~0|nor_conv_3x3~1|"
        "+|skip_connect~0|nor_conv_1x1~1|nor_conv_3x3~2|"
    )

    model = build_nasbench201_model_from_string(
        arch_str, num_classes=10, C=16, N=5
    )
    print(model)

    # Test with dummy CIFAR-10 input
    x = torch.randn(1, 3, 32, 32)
    features = model.forward_features(x)
    logits = model(x)
    print("Features shape:", features.shape)
    print("Logits shape:", logits.shape)
# Experiments

This section aims to apply variable open-source for pruning. More deeply, it summarizes each open-sources feature and profile why the performance is high or low.


## PyTorch Native

`torch.nn.utils.prune` is the pytorch native package for pruning. It supports magnitude (Ln-norm) based criteria for pruning. More deeply, It works like the following:

1. It computes Ln-norm for each weight matrix
2. Select low-k weight using `sort()` mechanism
3. Zeroize low-k weights

In other words, it requires computation cost for comparing Ln-norm and selecting low-k weight. This is the primitive approach for pruning, and there are many approaches like gradient-based (e.g.., taylor expansion) pruning.


## DepGraph : Towards any Structural Pruning

DepGraph aims to build schema for global structural pruning. In other words, it aims to
consider all connections in neural network architectures. ([[11](https://github.com/VainF/Torch-Pruning)], [Paper Review](https://www.canva.com/design/DAGnN0T2VIU/Re-ovkxrAu09dorbA2Q_qg/edit?utm_content=DAGnN0T2VIU&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)).

For more explaination about schema, suppose CNN architecture; `ConV > ReLU > Pooling`. Since `ReLU` is the injective (one-to-one) function, input of `ReLU` is connected to output of `ReLU`. In other example, if there is `BatchNorm`, input and output is grouped. Therefore, each group in `BatchNorm` shares its schema. As you can see, schema means **allocated shape by the layers structure**.

> Note: DepgGraph only suggests approaches for **group coupling**. Others sections for pruning are adopted by primitive approaches. For examples, DepGraph used Ln-norm for pruning criteria

![depgraph_process](images/depgraph_pruning_step.png)

And even more, `torch_pruning` supports Isomorphic pruning ([[6](https://arxiv.org/abs/2407.04616)]) in "importance evaluation" section, enhancing its performance in vision task.

## SparseML

[SparseML](https://github.com/neuralmagic/sparseml) open-source library by Neural Magic, acquired by Red Hat foundation.
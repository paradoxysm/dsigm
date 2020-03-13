# dsigm.Core

A CoreCluster is `dsigm`'s cluster, grouping a collection of Cores.
CoreCluster is capable of being structured in a hierarchical manner.

**Attributes**
```
cores : array-like, shape (some_cores,), default=[]
	A list of Cores associated with this CoreCluster.

parent : CoreCluster, default=None
	The parent CoreCluster in a hierarchical manner.

children : array-like, shape (some_cores,), default=[]
	A list of CoreCluster children in a hierarchical manner.
```

**Methods**
[`__init__`](https://github.com/paradoxysm/dsigm/tree/master/doc/pydoc/CoreCluster.md#CoreCluster) : Instantiates a CoreCluster.
[`_validate_init`](https://github.com/paradoxysm/dsigm/tree/master/doc/pydoc/CoreCluster.md#_validate_init) : Ensures all attributes have valid values.

## __init__
```python
CoreCluster(self, cores=[], parents=[], children=[])
```

A CoreCluster defines a collection of Cores that belong to a cluster.

**Parameters**
```
cores : array-like, shape (some_cores,), default=[]
	A list of Cores associated with this CoreCluster.

parent : CoreCluster, default=None
	The parent CoreCluster in a hierarchical manner.

children : array-like, shape (some_cores,), default=[]
	A list of CoreCluster children in a hierarchical manner.
```

## _validate_init
```python
CoreCluster._validate_init()
```
Validate the attributes of the CoreCluster.

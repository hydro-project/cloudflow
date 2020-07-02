# Cloudflow: Serverless Dataflow for Prediction Pipelines

Cloudflow is a dataflow DSL built on top of [Cloudburst](https://github.com/hydro-project/cloudburst) that is designed to easily support prediction serving pipelines. Cloudflow enables users to construct simple prediction pipelines using familiar dataflow operators like `map`, `filter`, and `join`. These pipelines can optionally be optimized by Cloudflow before being compiled into Cloudburst [DAGs](https://github.com/hydro-project/cloudburst/blob/master/docs/function-execution.md) -- optimizations include techniques like operator fusion (merging multiple logical operators into a single physical operator) and competitive execution (executing multiple models of a replica in parallel to reduce tail latencies). For more details on the motivation, implementation, and evaluation of Cloudflow, please check out [our paper](TODO INSERT LINK).

## API

The table below provides an overview of the Cloudflow API. The core abstraction in Cloudflow is a `Table`, which (like a relational table) has a pre-defined schema with 0 or more rows. Every operator in Cloudflow takes in a `Table` and returns a `Table`. For example, `map` takes an input, applies a provided function to each row in the `Table` and returns another `Table`. Operators can be chained together in the standard dataflow dot-notation style.

A new dataflow is created by creating a `Flow` object. Once that object is created, operators can be chained together, and `deploy` is called to compile and deploy the dataflow to Cloudburst. Users can optionally call `optimize` to optimize the dataflow (see the Optimization section below), and users can execute deployed dataflows with `run`. A `map` function can be set to run on a GPU by setting the `gpu` flag when calling `map`.

| API name | Arguments | Descrption |
|----------|-----------|------------|
| `map`    | `fn`, `col`, `names`| Apply a function to each row in a `Table`. `names` allows you to specify the name of the output columns|
| `filter` | `fn`, `group` | Apply a boolean function to each row in a `Table` and keep only `True` results. If `group` is true and the input is a grouped table, whole groups are filtered rather than individual rows. |
| `groupby`| `key` | Group a `Table` by the value in `key`. |
| `join`   | `other` | Joins the current `Table` with `other`.
| `lookup` | `key` | Retrieve `key` from Anna and add it to the input `Table`. |
| `agg`    | `agg_fn`, `col` | Apply the function `agg_fn` (one of `count`, `sum`, `min`, `max`, `avg`) to `col`.|

Here is an example that uses the above API. You can find more complex examples in the `flow/benchmarks` directory and some real worl pipelines in the `flow/pipelines` directory.

```python
from flow.operators.flow import Flow
from flow.types.basic import IntType
from flow.types.table import Table

def map_fn(row: Row) -> int:
	return row['a'] + row['b']
	
def filter_fn(row: Row) -> bool:
	return row['sum] % 2 == 0

dataflow = Flow('example-flow', CLOUDBURST_ADDR)
dataflow.map(map_fn, names=['sum']).filter(filter_fn)

table = Table([('a', IntType), ('b', IntType)])

table.insert([1, 2])
table.insert([1, 3])
table.insert([1, 4])

dataflow.deploy()
dataflow.run(table)
> [ 4 ] # Only one result which comes from the middle row in the input table.
```

## Optimization

Cloudflow supports 5 optimizations: operator fusion, competitive execution, data locality for dynamic lookups, batching, and operator autoscaling. All optimziations except for operator autoscaling are configurable through the optimization API. We describe each optimization briefly here -- if you are interesetd in learning more, please see the [full paper](TODO INSERT LINK).

```python
dataflow = Flow('optimized-flow', CLOUDBURST_ADDR)
... # Construct a dataflow.

rules = {'fusion': True, 'compete': False, 'breakpoint'; True}
from flow.optimize import optimize
optimized_flow = optimize(dataflow, rules)

# optimized_flow is another dataflow which can be deployed and run.
optimized_flow.deploy()
optimized_flow.run(input)
```

**Operator Fusion**: When multiple operators are in a linear chain, they can be grouped into a single physical operator in order to reduce the cost of data movement and context switching between multiple threads responsible for executing them. This optimization can be enabled or disabled with the `fusion` flag when calling `optimize`.

**Competitive Execution**: Machine learning models can have extremely highly variable execution times, even on fixed inputs. Competitive execution can be enabled to reduce this variance by running multiple replicas of an operator in parallel and picking the first that returns. A `map` operator can be marked for competitive execution by passing the `high_variance=True` flag when calling `map` -- when calling optimize, the `compete` flag should be enabled. By default, one extra replica is added per `high_variance` operator -- you can control this by setting the `compete_replica` key in the `rules` argument to `optimize`.

**Data Locality**: `lookup` operators can either have fixed input keys, or they can have dynamic lookups based on the results of previous stages in the dataflow. If the lookup is dynamic and data locality ensures that the operator takes advantage of Cloudburst's locality aware scheduling. Set the `breakpoint` flag to true when calling `optimize`.

**Batching**: Batching is a straightforward optimization: Multiple inputs will be passed into a function in a batch instead of a single input. Batching can be enabled by setting `batching=True` when calling `map`.

**Operator Autosclaing**: By default, Cloudburst automatically scales the resources required for each function up and down. We extended the Cloudburst autoscaler to support GPUs, so users do not have to worry about manually managing scaling.
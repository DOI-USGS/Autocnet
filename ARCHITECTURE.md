# Global Map

## Major Data Structures

### Network vs. Non-Network
The major data structures: graph, node, and edge have network versions (e.g., `NetworkNode` and `NetworkCandidateGraph`). These inherit all functionality from the non-network versions and add I/O operations that use network resources. Network resources can be processing queues, databases, and remote job submission through job orchestrators like slurm.

It is preferable to implement analytic capabilities in the non-network versions of the data structures. The goal is that the network versions simply retrieve data and push results to non-memory locations.

### Graphs: CandidateGraph & NetworkCandidateGraph

### Nodes: Node & NetworkNode

### Edges: Edge & NetworkEdge

### ControlNetworks
ISIS compatible control networks are managed as pandas DataFrames within the library. When the control network is persisted to a database (usually via a `NetworkCandidateGraph`), pandas `read_sql` calls are used with the `db_to_df_sql_string` to select a dataframe representation. This data structure can be analyzed, manipulated, and ultimately either written back to the database or to a file (via the `to_isis` method in `plio`).

## Testing
The library is tested via unit and integration tests using pytest. Endeavour to test as much of the software as is feasible, understanding that the code base has been developed to create a product and not unto itself.

Tests are parameritized using the `pytest.ini` file in the repository root. This way, developers are all invoking the tests in a standard way and best practices can be propagated under version control.

### Unit Tests
The majority of the library is tested via unit tests.

### Integration Tests
The integration tests are longer running. Strive to keep integration tests less than 120s each. We assume that tests can run in parallel with at least 4 CPUs (GitHub free runners, as of March 2024 have 4 CPUs on ubuntu). Integration tests do not make use of any external resources. They should make liberal use of mocked REDIS and SQLAlchemy session objects.

## Sensor Models
Sensor models are abstracted through a common interface. That interface is stored in 

# Gotchas
Here are a number of different gotchas in the library that deal with stuff.

## Latitude: Ocentric vs. Ographic
There are two different ways to define latitude. See [the proj docs](https://proj.org/en/9.4/operations/conversions/geoc.html) for a in-depth description of ocentric vs. ographic latitudes. This library is impacted when working with other libraries that default to a particular latitude definition. As of the 1.2.0 release, all latitudes inside this library are ographic, conforming to the PROJ standard. The CSM (usgscsm) working in XYZ/BCBF coordinates, so anytime one converts those to lat/lons, they use PROJ and have ographic latitudes. The ISIS library defaults to ocentric latitudes. So, passing a lat/lon pair into a program like `campt` assumes that that latitude is ocentric. So, the autocnet.camera.sensor_models `ISISSensor` object takes can of transformations to ocentric when a lat/lon pair is passed. That same object defaults to returning ographic latitudes when one queries for ground coordinates. When writing a control network, one writes the XYZ/BCBF coordinates. Therefore, no conversion has to take place. 

When adding any functionality to the library dealing with sensor models and ISIS either (1) use the `ISISSensor` interface or (2) be very careful to convert to/from ographic/ocentric latitudes as appropriate.
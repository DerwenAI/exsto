## ETL in PySpark with Spark SQL

Let's use PySpark and Spark SQL to prepare the data for ML and graph
analysis.
We can perform *data discovery* while reshaping the data for later
work.
These early results can help guide our deeper analysis.

NB: if this ETL needs to run outside of the `bin/pyspark` shell, first
set up a `SparkContext` variable:

```python
from pyspark import SparkContext
sc = SparkContext(appName="Exsto", master="local[*]")
```

Import the JSON data produced by the scraper and register its schema
for ad-hoc SQL queries later.
Each message has the fields: 
`date`, `sender`, `id`, `next_thread`, `prev_thread`, `next_url`, `subject`, `text`

```python
from pyspark.sql import SQLContext, Row
sqlCtx = SQLContext(sc)

msg = sqlCtx.jsonFile("data").cache()
msg.registerTempTable("msg")
```

NB: note the persistence used for the JSON message data.
We may need to unpersist at a later stage of this ETL work.

### Question: Who are the senders?

Who are the people in the developer community sending email to the list?
We will use this as a dimension in our analysis and reporting.
Let's create a map, with a unique ID for each email address --
this will be required for the graph analysis.
It may come in handy later for some
[named-entity recognition](https://en.wikipedia.org/wiki/Named-entity_recognition).

```python
who = msg.map(lambda x: x.sender).distinct().zipWithUniqueId()
who.take(10)

whoMap = who.collectAsMap()

print "\nsenders"
print len(whoMap)
```

### Question: Who are the top K senders?

[Apache Spark](http://spark.apache.org/) is one of the most
active open source developer communities on Apache, so it
will tend to have several thousand people engaged.
Let's identify the most active ones.
Then we can show a leaderboard and track changes in it over time.

```python
from operator import add

top_sender = msg.map(lambda x: (x.sender, 1,)).reduceByKey(add) \
 .map(lambda (a, b): (b, a)) \
 .sortByKey(0, 1) \
 .map(lambda (a, b): (b, a))

print "\ntop senders"
print top_sender.take(11)
```

You many notice that code... it comes from *word count*.


### Question: Which are the top K conversations?

Clearly, some people discuss over the email list more than others.
Let's identify *who* those people are.
Later we can leverage our graph analysis to determine *what* they discuss.

NB: note the use case for `groupByKey` transformations; 
sometimes its usage is indicated.

```python
import itertools

def nitems (replier, senders):
  for sender, g in itertools.groupby(senders):
    yield len(list(g)), (replier, sender,)

senders = msg.map(lambda x: (x.id, x.sender,))
replies = msg.map(lambda x: (x.prev_thread, x.sender,))

convo = replies.join(senders).values() \
 .filter(lambda (a, b): a != b)

top_convo = convo.groupByKey() \
 .flatMap(lambda (a, b): list(nitems(a, b))) \
 .sortByKey(0)

print "\ntop convo"
print top_convo.take(10)
```

### Prepare for Sender/Reply Graph Analysis

Given the RDDs that we have created to help answer some of the
questions so far, let's persist those data sets using
[Parquet](http://parquet.io) --
starting with the graph of sender/message/reply:

```python
edge = top_convo.map(lambda (a, b): (whoMap.get(b[0]), whoMap.get(b[1]), a,))
edgeSchema = edge.map(lambda p: Row(replier=p[0], sender=p[1], count=int(p[2])))
edgeTable = sqlCtx.inferSchema(edgeSchema)
edgeTable.saveAsParquetFile("reply_edge.parquet")

node = who.map(lambda (a, b): (b, a))
nodeSchema = node.map(lambda p: Row(id=int(p[0]), sender=p[1]))
nodeTable = sqlCtx.inferSchema(nodeSchema)
nodeTable.saveAsParquetFile("reply_node.parquet")
```


### Prepare for TextRank Analysis per paragraph

```python
def map_graf_edges (x):
  j = json.loads(x)

  for pair in j["tile"]:
    n0 = int(pair[0])
    n1 = int(pair[1])

    if n0 > 0 and n1 > 0:
      yield (j["id"], n0, n1,)
      yield (j["id"], n1, n0,)

graf = sc.textFile("parsed")
n = graf.flatMap(map_graf_edges).count()
print "\ngraf edges", n

edgeSchema = graf.map(lambda p: Row(id=p[0], node0=p[1], node1=p[2]))

edgeTable = sqlCtx.inferSchema(edgeSchema)
edgeTable.saveAsParquetFile("graf_edge.parquet")
```

```python
def map_graf_nodes (x):
  j = json.loads(x)

  for word in j["graf"]:
    yield [j["id"]] + word

graf = sc.textFile("parsed")
n = graf.flatMap(map_graf_nodes).count()
print "\ngraf nodes", n

nodeSchema = graf.map(lambda p: Row(id=p[0], node_id=p[1], raw=p[2], root=p[3], pos=p[4], keep=p[5], num=p[6]))

nodeTable = sqlCtx.inferSchema(nodeSchema)
nodeTable.saveAsParquetFile("graf_node.parquet")
```

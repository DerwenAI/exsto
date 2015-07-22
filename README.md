# Exsto: Developer Community Insights

A frequently asked question on the [Apache Spark](http://spark.apache.org/) 
user email list concerns where to find data sets for evaluating the code.
Oddly enough, the collection of archived messages for this email list
provides an excellent data set for evaluating Spark capabilities, e.g.,
machine learning, graph algorithms, text analytics, time-series analysis, etc.

Herein, an open source developer community considers itself algorithmically.
This project shows work-in-progress for how to surface data insights from 
the developer email forums for an Apache open source project. 
It leverages advanced technologies for natural language processing, machine 
learning, graph algorithms, time series analysis, etc.
As an example, we use data from the `<user@spark.apache.org>` 
[email list archives](http://mail-archives.apache.org) to help understand 
its community better.

See these talks about `Exsto`:
<ul>
<li>
DataDayTexas 2015 session talk,
[Microservices, Containers, and Machine Learning]
(http://www.slideshare.net/pacoid/microservices-containers-and-machine-learning)
</li>
<li>
Scala Days EU 2015 session,
[GraphX: Graph analytics for insights about developer communities]
(http://www.slideshare.net/pacoid/graphx-graph-analytics-for-insights-about-developer-communities)
</li>
</ul>

In particular, we will shows production use of NLP tooling in Python, 
integrated with
[MLlib](http://spark.apache.org/docs/latest/mllib-guide.html)
(machine learning) and 
[GraphX](http://spark.apache.org/docs/latest/graphx-programming-guide.html)
(graph algorithms) in Apache Spark. 
Machine learning approaches used include: 
[Word2Vec](https://code.google.com/p/word2vec/), 
[TextRank](http://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf),
Connected Components, Streaming K-Means, etc.

Keep in mind that "One Size Fits All" is an anti-pattern, especially for 
Big Data tools. 
This project illustrates how to leverage microservices and containers to 
scale-out the code+data components that do not fit well in Spark, Hadoop, etc.

In addition to Spark, other technologies used include: 
[Mesos](http://mesos.apache.org/),
[Docker](https://www.docker.com/),
[Anaconda](http://continuum.io/downloads),
[Flask](http://flask.pocoo.org/),
[NLTK](http://www.nltk.org/),
[TextBlob](https://textblob.readthedocs.org/en/dev/).


## Dependencies

  * https://github.com/opentable/docker-anaconda

```bash
conda config --add channels https://conda.binstar.org/sloria
conda install textblob
python -m textblob.download_corpora
python -m nltk.downloader -d ~/nltk_data all
pip install -U textblob textblob-aptagger
pip install lxml
pip install python-dateutil
pip install Flask
```

NLTK and TextBlob require some
[data downloads](https://s3.amazonaws.com/textblob/nltk_data.tar.gz)
which may also require updating the NLTK data path:

```python
import nltk
nltk.data.path.append("~/nltk_data/")
```


## Running

To change the project configuration simply edit the `defaults.cfg`
file.


### scrape the email list via scripts

```bash
./scrape.py data/foo.json
```

### parse the email text via scripts

```bash
./parse.py data/foo.json parsed/foo.json
```

## Public Data

Example data from the Apache Spark email list is available as JSON:

  * original: [https://s3-us-west-1.amazonaws.com/paco.dbfs.public/exsto/original/2015_01.json](https://s3-us-west-1.amazonaws.com/paco.dbfs.public/exsto/original/2015_01.json)
  * parsed: [https://s3-us-west-1.amazonaws.com/paco.dbfs.public/exsto/parsed/2015_01.json](https://s3-us-west-1.amazonaws.com/paco.dbfs.public/exsto/parsed/2015_01.json)


# What's in a name?

The word [exsto](http://en.wiktionary.org/wiki/exsto) is the Latin
verb meaning "to stand out", in its present active form.

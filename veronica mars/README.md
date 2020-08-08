# Veronica Mars Codebase

This is a collection of all the Jupyter Notebooks we worked on during the UCLA 2020 REU Knowledge Graphs Group. Within our project many of us were exploring different techniques in parallel. We aimed for proof of concept techniques and results, consequently some of the code was created ad-hoc to facilitate our exploration. If our code is difficult to understand and you really want to know what we have done, please feel free to contact the group. We have all of our data sets and notebooks here.

We do not have an interface to interact with our code. Instead this codebase is
supplemental to the REU report. We recommend reading the report and if you are interested
in how we implemented a particular functionality then we would recommend reading through
the appropriate notebooks. Our code is not easily extendible as we had a focus on proof of
concept. If you would like to build upon what we have then it might be best to find the
appropriate code block and then refactor it to refit your purposes.

Note that OpenKE is a copy of the OpenKE github repo with our changes and includeddatasets. [https://github.com/thunlp/OpenKE](https://github.com/thunlp/OpenKE)
Similarly SEAL/SEAL-repo is a copy of the SEAL github repo with our training files
included.
[https://github.com/muhanzhang/SEAL/tree/master/Python](https://github.com/muhanzhang/SEAL/tree/master/Python)

## File Descriptions
Veronica\_Mars\_to\_TransE\_Conversion.ipynb :: This notebook generates trainng data for
OpenKE using our knowledge graph. 

VM KG Topic Modelling.ipynb :: This notebook contains the code which performs topic
modelling on random walks of our knowledge graph.

VM RDFTriples and KG + SPARQL.ipynb :: This notebook contains some example SPARQL queries
for our knowledge graph.

VM Subgraph Visualisations :: Here we perform visualisations of subgraphs such as
subgraphs which describe all the clues of a case, or all the triples which have a given
character as a subject.

VM pred\_emb.txt :: This text file contains the predicate embedding of our
knowledge graph. Please refer to the report understand what this means. 

VM\_TransE\_emb.txt :: This text file ontains the resultant TransE embedding of our
knowledge graph. This embedding was generated using OpenKE.

vmars ontology.owl :: This file contains the ontology we used for our knowledge graph. Note
that this file may not be up to date.

VMars Triples.xlsx :: This file contains all the triples for our Veronica Mars Knowledge
graph. 

VMars\_Topics.xlsx :: This file contains examples of the topics which were output
by our topic modelling script.

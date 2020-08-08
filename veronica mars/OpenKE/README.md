This file is a copy of the OpenKE package with our datasets and extensions. Our dataset
can be found in VM/ 

We recommend inspecting train\_transe\_VM.py to inspect our training and testing code.
Note that our experimental code has not been well refactored as we focused on rapid prototyping and proof of concept experiements. Once you have inspected train\_transe\_VM.py you can run that file and run your own experiemnts.
Note that this comes with a pretrained model and can be found under checkpoint/
There are various models which each contain different sets of testing and training triples.

04Aug: Has 50% of has_financial_status and 50% of perpatrator triples removed.
31Jul: Has 50% of perpatrator triples removed.
VM_transE: trained on entire knowledge graph, although the knowledge graph has been updated since this has been trained.

# XAI Mini Project - EDGE Framework

## Dataset
**Videogame**  
A custom RDF-based dataset containing metadata about various video games, genres, developers, and platforms. This dataset is used for benchmarking explainability techniques on graph-based models.

---

## Installation Guide for the EDGE Framework

Follow these steps to set up the EDGE environment on your system:

### Step 1: Clone the Repository
```bash
git clone https://github.com/21Siddy/xai-miniproject.git
cd xai-miniproject
```

### Step 2: Install Conda
If you don't have Conda installed, download and install it from Anaconda's official website.

### Step 3: Create and Activate the Conda Environment
```bash
conda create --name edge python=3.10
conda activate edge
```

### Step 4: Install Dependencies 
```bash
pip install -r requirements.txt
```

### Step 5: Install DGL
```bash
conda install -c dglteam/label/th23_cu121 dgl
```

### Step 6: cd into the EDGE directory and test with a command similar to this
```bash
(edge) xaimonster@LAPTOP-UGEQJG77:~/xai-miniproject/EDGE$ python main.py --datasets videogames --explainers EvoLearner --model RGCN --train --num_runs 5 --print_results
```

### Sample Results
```bash
(edge) xaimonster@LAPTOP-UGEQJG77:~/xai-miniproject/EDGE$ python main.py --datasets videogames --explainers EvoLearner --model RGCN --train --num_runs 5 --print_results
Datasets: ['videogames']
Explainers ['EvoLearner']
Model name: RGCN
Running explainers for 5 runs for dataset videogames
Starting the  Run 1
Starting VideoGameDataset.process() (Full Custom Implementation).
Parsing data/videogames/videogame_f.rdf
Parsing data/KGs/videogame_f.rdf
Prepared 4825 raw rdflib triples after pre-scan.
DGL graph built. Details:
Graph(num_nodes={'Country': 20, 'Developer': 154, 'Game': 299, 'Genre': 107, 'Literal': 781, 'OntologyClass': 6, 'OntologyConstruct': 3, 'Platform': 127, 'Publisher': 153, 'Unknown': 7},
      num_edges={('Country', 'rdf_type', 'OntologyClass'): 20, ('Country', 'rev-hasCountry', 'Game'): 294, ('Developer', 'rdf_type', 'OntologyClass'): 154, ('Developer', 'rev-developedBy', 'Game'): 213, ('Game', 'availableOn', 'Platform'): 1236, ('Game', 'developedBy', 'Developer'): 213, ('Game', 'developedBy', 'Publisher'): 125, ('Game', 'hasCountry', 'Country'): 294, ('Game', 'hasGenre', 'Genre'): 472, ('Game', 'hasReleaseDate', 'Literal'): 580, ('Game', 'publishedBy', 'Publisher'): 381, ('Game', 'rdf_type', 'OntologyClass'): 299, ('Game', 'rdf_type', 'Unknown'): 299, ('Game', 'rdfs_label', 'Literal'): 299, ('Genre', 'rdf_type', 'OntologyClass'): 107, ('Genre', 'rev-hasGenre', 'Game'): 472, ('Literal', 'rev-hasReleaseDate', 'Game'): 580, ('Literal', 'rev-rdfs_label', 'Game'): 299, ('OntologyClass', 'rdf_type', 'OntologyConstruct'): 6, ('OntologyClass', 'rev-rdf_type', 'Country'): 20, ('OntologyClass', 'rev-rdf_type', 'Developer'): 154, ('OntologyClass', 'rev-rdf_type', 'Game'): 299, ('OntologyClass', 'rev-rdf_type', 'Genre'): 107, ('OntologyClass', 'rev-rdf_type', 'Platform'): 127, ('OntologyClass', 'rev-rdf_type', 'Publisher'): 207, ('OntologyConstruct', 'rev-rdf_type', 'OntologyClass'): 6, ('OntologyConstruct', 'rev-rdf_type', 'Unknown'): 6, ('Platform', 'rdf_type', 'OntologyClass'): 127, ('Platform', 'rev-availableOn', 'Game'): 1236, ('Publisher', 'rdf_type', 'OntologyClass'): 207, ('Publisher', 'rev-developedBy', 'Game'): 125, ('Publisher', 'rev-publishedBy', 'Game'): 381, ('Unknown', 'rdf_type', 'OntologyConstruct'): 6, ('Unknown', 'rev-rdf_type', 'Game'): 299},
      metagraph=[('Country', 'OntologyClass', 'rdf_type'), ('Country', 'Game', 'rev-hasCountry'), ('OntologyClass', 'OntologyConstruct', 'rdf_type'), ('OntologyClass', 'Country', 'rev-rdf_type'), ('OntologyClass', 'Developer', 'rev-rdf_type'), ('OntologyClass', 'Game', 'rev-rdf_type'), ('OntologyClass', 'Genre', 'rev-rdf_type'), ('OntologyClass', 'Platform', 'rev-rdf_type'), ('OntologyClass', 'Publisher', 'rev-rdf_type'), ('Game', 'Platform', 'availableOn'), ('Game', 'Developer', 'developedBy'), ('Game', 'Publisher', 'developedBy'), ('Game', 'Publisher', 'publishedBy'), ('Game', 'Country', 'hasCountry'), ('Game', 'Genre', 'hasGenre'), ('Game', 'Literal', 'hasReleaseDate'), ('Game', 'Literal', 'rdfs_label'), ('Game', 'OntologyClass', 'rdf_type'), ('Game', 'Unknown', 'rdf_type'), ('Developer', 'OntologyClass', 'rdf_type'), ('Developer', 'Game', 'rev-developedBy'), ('Platform', 'OntologyClass', 'rdf_type'), ('Platform', 'Game', 'rev-availableOn'), ('Publisher', 'OntologyClass', 'rdf_type'), ('Publisher', 'Game', 'rev-developedBy'), ('Publisher', 'Game', 'rev-publishedBy'), ('Genre', 'OntologyClass', 'rdf_type'), ('Genre', 'Game', 'rev-hasGenre'), ('Literal', 'Game', 'rev-hasReleaseDate'), ('Literal', 'Game', 'rev-rdfs_label'), ('Unknown', 'OntologyConstruct', 'rdf_type'), ('Unknown', 'Game', 'rev-rdf_type'), ('OntologyConstruct', 'OntologyClass', 'rev-rdf_type'), ('OntologyConstruct', 'Unknown', 'rev-rdf_type')])
Generated 117 MultiGenre labels and 182 SingleGenre labels.
VideoGameDataset.process() completed. Final graph has 1657 nodes.
Train/Valid/Test split sizes: Train=149, Valid=74, Test=76
Done saving data into cached files.
VideoGameDataset initialized.
Initializing RGCN  model
Start training...
Epoch 00000 | Train Acc: 0.6107 | Train Loss: 0.6759 | Valid Acc: 0.6081 | Valid loss: 0.6763 | Time: 0.5802
Epoch 00001 | Train Acc: 0.7450 | Train Loss: 0.5928 | Valid Acc: 0.4595 | Valid loss: 0.7075 | Time: 0.3483
Epoch 00002 | Train Acc: 0.6242 | Train Loss: 0.6486 | Valid Acc: 0.6081 | Valid loss: 0.8201 | Time: 0.2776
Epoch 00003 | Train Acc: 0.7785 | Train Loss: 0.4750 | Valid Acc: 0.5541 | Valid loss: 0.7071 | Time: 0.2368
Epoch 00004 | Train Acc: 0.8389 | Train Loss: 0.4589 | Valid Acc: 0.4730 | Valid loss: 0.7558 | Time: 0.2112
Epoch 00005 | Train Acc: 0.8725 | Train Loss: 0.4009 | Valid Acc: 0.5000 | Valid loss: 0.7817 | Time: 0.1991
Epoch 00006 | Train Acc: 0.9060 | Train Loss: 0.3117 | Valid Acc: 0.5405 | Valid loss: 0.7779 | Time: 0.1912
Epoch 00007 | Train Acc: 0.9128 | Train Loss: 0.2768 | Valid Acc: 0.5946 | Valid loss: 0.8432 | Time: 0.1820
Epoch 00008 | Train Acc: 0.9262 | Train Loss: 0.2268 | Valid Acc: 0.5811 | Valid loss: 0.9028 | Time: 0.1768
Epoch 00009 | Train Acc: 0.9463 | Train Loss: 0.1671 | Valid Acc: 0.5676 | Valid loss: 0.9670 | Time: 0.1697
Epoch 00010 | Train Acc: 0.9732 | Train Loss: 0.1451 | Valid Acc: 0.5541 | Valid loss: 1.0814 | Time: 0.1641
Epoch 00011 | Train Acc: 0.9866 | Train Loss: 0.1063 | Valid Acc: 0.5676 | Valid loss: 1.1437 | Time: 0.1610
Epoch 00012 | Train Acc: 0.9933 | Train Loss: 0.0752 | Valid Acc: 0.5946 | Valid loss: 1.2071 | Time: 0.1565
Epoch 00013 | Train Acc: 0.9933 | Train Loss: 0.0606 | Valid Acc: 0.5811 | Valid loss: 1.3091 | Time: 0.1531
Epoch 00014 | Train Acc: 1.0000 | Train Loss: 0.0397 | Valid Acc: 0.5946 | Valid loss: 1.4115 | Time: 0.1508
Epoch 00015 | Train Acc: 0.9933 | Train Loss: 0.0281 | Valid Acc: 0.5811 | Valid loss: 1.5223 | Time: 0.1555
Epoch 00016 | Train Acc: 1.0000 | Train Loss: 0.0212 | Valid Acc: 0.5811 | Valid loss: 1.6455 | Time: 0.1524
Epoch 00017 | Train Acc: 1.0000 | Train Loss: 0.0124 | Valid Acc: 0.5541 | Valid loss: 1.7702 | Time: 0.1523
Epoch 00018 | Train Acc: 1.0000 | Train Loss: 0.0077 | Valid Acc: 0.5676 | Valid loss: 1.9071 | Time: 0.1497
Epoch 00019 | Train Acc: 1.0000 | Train Loss: 0.0059 | Valid Acc: 0.5676 | Valid loss: 2.0534 | Time: 0.1489
Epoch 00020 | Train Acc: 1.0000 | Train Loss: 0.0045 | Valid Acc: 0.5676 | Valid loss: 2.1991 | Time: 0.1470
Epoch 00021 | Train Acc: 1.0000 | Train Loss: 0.0030 | Valid Acc: 0.5676 | Valid loss: 2.3374 | Time: 0.1484
Epoch 00022 | Train Acc: 1.0000 | Train Loss: 0.0019 | Valid Acc: 0.5541 | Valid loss: 2.4681 | Time: 0.1469
Early stopping
End Training
Final validation accuracy of the model RGCN on unseen dataset: 0.631578947368421
Prepared learning problem for concept 'http://example.org/videogame#MultiGenreConcept':
  Train Pos: 87, Neg: 136
  Test Pos: 30, Neg: 46
Prepared learning problem for concept 'http://example.org/videogame#MultiGenreConcept':
  Train Pos: 85, Neg: 138
  Test Pos: 36, Neg: 40
Training EvoLearner (explanation) on videogames
<class 'ontolearn.search.EvoLearnerNode'> at 0xa4b84b3  ≥ 2 hasGenre.Genre      Quality:0.84753 Length:4        Tree Length:3   Tree Depth:1    |Indv.|:117
Total time taken for EvoLearner (explanation)  on videogames: 10.45
Starting the  Run 2
Starting VideoGameDataset.process() (Full Custom Implementation).
Parsing data/videogames/videogame_f.rdf
Parsing data/KGs/videogame_f.rdf
Prepared 4825 raw rdflib triples after pre-scan.
DGL graph built. Details:
Graph(num_nodes={'Country': 20, 'Developer': 154, 'Game': 299, 'Genre': 107, 'Literal': 781, 'OntologyClass': 6, 'OntologyConstruct': 3, 'Platform': 127, 'Publisher': 153, 'Unknown': 7},
      num_edges={('Country', 'rdf_type', 'OntologyClass'): 20, ('Country', 'rev-hasCountry', 'Game'): 294, ('Developer', 'rdf_type', 'OntologyClass'): 154, ('Developer', 'rev-developedBy', 'Game'): 213, ('Game', 'availableOn', 'Platform'): 1236, ('Game', 'developedBy', 'Developer'): 213, ('Game', 'developedBy', 'Publisher'): 125, ('Game', 'hasCountry', 'Country'): 294, ('Game', 'hasGenre', 'Genre'): 472, ('Game', 'hasReleaseDate', 'Literal'): 580, ('Game', 'publishedBy', 'Publisher'): 381, ('Game', 'rdf_type', 'OntologyClass'): 299, ('Game', 'rdf_type', 'Unknown'): 299, ('Game', 'rdfs_label', 'Literal'): 299, ('Genre', 'rdf_type', 'OntologyClass'): 107, ('Genre', 'rev-hasGenre', 'Game'): 472, ('Literal', 'rev-hasReleaseDate', 'Game'): 580, ('Literal', 'rev-rdfs_label', 'Game'): 299, ('OntologyClass', 'rdf_type', 'OntologyConstruct'): 6, ('OntologyClass', 'rev-rdf_type', 'Country'): 20, ('OntologyClass', 'rev-rdf_type', 'Developer'): 154, ('OntologyClass', 'rev-rdf_type', 'Game'): 299, ('OntologyClass', 'rev-rdf_type', 'Genre'): 107, ('OntologyClass', 'rev-rdf_type', 'Platform'): 127, ('OntologyClass', 'rev-rdf_type', 'Publisher'): 207, ('OntologyConstruct', 'rev-rdf_type', 'OntologyClass'): 6, ('OntologyConstruct', 'rev-rdf_type', 'Unknown'): 6, ('Platform', 'rdf_type', 'OntologyClass'): 127, ('Platform', 'rev-availableOn', 'Game'): 1236, ('Publisher', 'rdf_type', 'OntologyClass'): 207, ('Publisher', 'rev-developedBy', 'Game'): 125, ('Publisher', 'rev-publishedBy', 'Game'): 381, ('Unknown', 'rdf_type', 'OntologyConstruct'): 6, ('Unknown', 'rev-rdf_type', 'Game'): 299},
      metagraph=[('Country', 'OntologyClass', 'rdf_type'), ('Country', 'Game', 'rev-hasCountry'), ('OntologyClass', 'OntologyConstruct', 'rdf_type'), ('OntologyClass', 'Country', 'rev-rdf_type'), ('OntologyClass', 'Developer', 'rev-rdf_type'), ('OntologyClass', 'Game', 'rev-rdf_type'), ('OntologyClass', 'Genre', 'rev-rdf_type'), ('OntologyClass', 'Platform', 'rev-rdf_type'), ('OntologyClass', 'Publisher', 'rev-rdf_type'), ('Game', 'Platform', 'availableOn'), ('Game', 'Developer', 'developedBy'), ('Game', 'Publisher', 'developedBy'), ('Game', 'Publisher', 'publishedBy'), ('Game', 'Country', 'hasCountry'), ('Game', 'Genre', 'hasGenre'), ('Game', 'Literal', 'hasReleaseDate'), ('Game', 'Literal', 'rdfs_label'), ('Game', 'OntologyClass', 'rdf_type'), ('Game', 'Unknown', 'rdf_type'), ('Developer', 'OntologyClass', 'rdf_type'), ('Developer', 'Game', 'rev-developedBy'), ('Platform', 'OntologyClass', 'rdf_type'), ('Platform', 'Game', 'rev-availableOn'), ('Publisher', 'OntologyClass', 'rdf_type'), ('Publisher', 'Game', 'rev-developedBy'), ('Publisher', 'Game', 'rev-publishedBy'), ('Genre', 'OntologyClass', 'rdf_type'), ('Genre', 'Game', 'rev-hasGenre'), ('Literal', 'Game', 'rev-hasReleaseDate'), ('Literal', 'Game', 'rev-rdfs_label'), ('Unknown', 'OntologyConstruct', 'rdf_type'), ('Unknown', 'Game', 'rev-rdf_type'), ('OntologyConstruct', 'OntologyClass', 'rev-rdf_type'), ('OntologyConstruct', 'Unknown', 'rev-rdf_type')])
Generated 117 MultiGenre labels and 182 SingleGenre labels.
VideoGameDataset.process() completed. Final graph has 1657 nodes.
Train/Valid/Test split sizes: Train=149, Valid=74, Test=76
Done saving data into cached files.
VideoGameDataset initialized.
Initializing RGCN  model
Start training...
Epoch 00000 | Train Acc: 0.4094 | Train Loss: 0.7744 | Valid Acc: 0.4054 | Valid loss: 0.7720 | Time: 0.1094
Epoch 00001 | Train Acc: 0.6107 | Train Loss: 0.9034 | Valid Acc: 0.6081 | Valid loss: 0.9252 | Time: 0.1167
Epoch 00002 | Train Acc: 0.6107 | Train Loss: 0.6513 | Valid Acc: 0.6081 | Valid loss: 0.7125 | Time: 0.1236
Epoch 00003 | Train Acc: 0.7047 | Train Loss: 0.6063 | Valid Acc: 0.5811 | Valid loss: 0.6838 | Time: 0.1299
Epoch 00004 | Train Acc: 0.8591 | Train Loss: 0.6073 | Valid Acc: 0.5000 | Valid loss: 0.7061 | Time: 0.1255
Epoch 00005 | Train Acc: 0.8188 | Train Loss: 0.5866 | Valid Acc: 0.4730 | Valid loss: 0.7140 | Time: 0.1213
Epoch 00006 | Train Acc: 0.8523 | Train Loss: 0.5442 | Valid Acc: 0.5811 | Valid loss: 0.7049 | Time: 0.1212
Epoch 00007 | Train Acc: 0.8255 | Train Loss: 0.4949 | Valid Acc: 0.5405 | Valid loss: 0.6940 | Time: 0.1188
Epoch 00008 | Train Acc: 0.8255 | Train Loss: 0.4564 | Valid Acc: 0.5811 | Valid loss: 0.7063 | Time: 0.1192
Epoch 00009 | Train Acc: 0.8322 | Train Loss: 0.4171 | Valid Acc: 0.5811 | Valid loss: 0.7403 | Time: 0.1245
Epoch 00010 | Train Acc: 0.8523 | Train Loss: 0.3598 | Valid Acc: 0.5541 | Valid loss: 0.7777 | Time: 0.0248
Epoch 00011 | Train Acc: 0.8658 | Train Loss: 0.3129 | Valid Acc: 0.5676 | Valid loss: 0.8385 | Time: 0.0317
Epoch 00012 | Train Acc: 0.9128 | Train Loss: 0.2793 | Valid Acc: 0.5270 | Valid loss: 0.9124 | Time: 0.0421
Epoch 00013 | Train Acc: 0.9195 | Train Loss: 0.2323 | Valid Acc: 0.5405 | Valid loss: 0.9571 | Time: 0.0476
Epoch 00014 | Train Acc: 0.9262 | Train Loss: 0.1900 | Valid Acc: 0.5541 | Valid loss: 0.9990 | Time: 0.0511
Epoch 00015 | Train Acc: 0.9262 | Train Loss: 0.1589 | Valid Acc: 0.5541 | Valid loss: 1.0567 | Time: 0.0560
Epoch 00016 | Train Acc: 0.9530 | Train Loss: 0.1238 | Valid Acc: 0.5676 | Valid loss: 1.1095 | Time: 0.0595
Epoch 00017 | Train Acc: 0.9732 | Train Loss: 0.0912 | Valid Acc: 0.5541 | Valid loss: 1.1606 | Time: 0.0644
Epoch 00018 | Train Acc: 0.9866 | Train Loss: 0.0696 | Valid Acc: 0.5405 | Valid loss: 1.2240 | Time: 0.0680
Epoch 00019 | Train Acc: 1.0000 | Train Loss: 0.0495 | Valid Acc: 0.5541 | Valid loss: 1.2734 | Time: 0.0704
Epoch 00020 | Train Acc: 1.0000 | Train Loss: 0.0334 | Valid Acc: 0.5541 | Valid loss: 1.3151 | Time: 0.0742
Epoch 00021 | Train Acc: 1.0000 | Train Loss: 0.0242 | Valid Acc: 0.5676 | Valid loss: 1.3762 | Time: 0.0761
Epoch 00022 | Train Acc: 1.0000 | Train Loss: 0.0172 | Valid Acc: 0.5676 | Valid loss: 1.4530 | Time: 0.0783
Early stopping
End Training
Final validation accuracy of the model RGCN on unseen dataset: 0.7236842105263158
Prepared learning problem for concept 'http://example.org/videogame#MultiGenreConcept':
  Train Pos: 87, Neg: 136
  Test Pos: 30, Neg: 46
Prepared learning problem for concept 'http://example.org/videogame#MultiGenreConcept':
  Train Pos: 83, Neg: 140
  Test Pos: 31, Neg: 45
Training EvoLearner (explanation) on videogames
<class 'ontolearn.search.EvoLearnerNode'> at 0xd1fa6a7  ≥ 2 hasGenre.Genre      Quality:0.8565  Length:4        Tree Length:3   Tree Depth:1    |Indv.|:117
Total time taken for EvoLearner (explanation)  on videogames: 10.24
Starting the  Run 3
Starting VideoGameDataset.process() (Full Custom Implementation).
Parsing data/videogames/videogame_f.rdf
Parsing data/KGs/videogame_f.rdf
Prepared 4825 raw rdflib triples after pre-scan.
DGL graph built. Details:
Graph(num_nodes={'Country': 20, 'Developer': 154, 'Game': 299, 'Genre': 107, 'Literal': 781, 'OntologyClass': 6, 'OntologyConstruct': 3, 'Platform': 127, 'Publisher': 153, 'Unknown': 7},
      num_edges={('Country', 'rdf_type', 'OntologyClass'): 20, ('Country', 'rev-hasCountry', 'Game'): 294, ('Developer', 'rdf_type', 'OntologyClass'): 154, ('Developer', 'rev-developedBy', 'Game'): 213, ('Game', 'availableOn', 'Platform'): 1236, ('Game', 'developedBy', 'Developer'): 213, ('Game', 'developedBy', 'Publisher'): 125, ('Game', 'hasCountry', 'Country'): 294, ('Game', 'hasGenre', 'Genre'): 472, ('Game', 'hasReleaseDate', 'Literal'): 580, ('Game', 'publishedBy', 'Publisher'): 381, ('Game', 'rdf_type', 'OntologyClass'): 299, ('Game', 'rdf_type', 'Unknown'): 299, ('Game', 'rdfs_label', 'Literal'): 299, ('Genre', 'rdf_type', 'OntologyClass'): 107, ('Genre', 'rev-hasGenre', 'Game'): 472, ('Literal', 'rev-hasReleaseDate', 'Game'): 580, ('Literal', 'rev-rdfs_label', 'Game'): 299, ('OntologyClass', 'rdf_type', 'OntologyConstruct'): 6, ('OntologyClass', 'rev-rdf_type', 'Country'): 20, ('OntologyClass', 'rev-rdf_type', 'Developer'): 154, ('OntologyClass', 'rev-rdf_type', 'Game'): 299, ('OntologyClass', 'rev-rdf_type', 'Genre'): 107, ('OntologyClass', 'rev-rdf_type', 'Platform'): 127, ('OntologyClass', 'rev-rdf_type', 'Publisher'): 207, ('OntologyConstruct', 'rev-rdf_type', 'OntologyClass'): 6, ('OntologyConstruct', 'rev-rdf_type', 'Unknown'): 6, ('Platform', 'rdf_type', 'OntologyClass'): 127, ('Platform', 'rev-availableOn', 'Game'): 1236, ('Publisher', 'rdf_type', 'OntologyClass'): 207, ('Publisher', 'rev-developedBy', 'Game'): 125, ('Publisher', 'rev-publishedBy', 'Game'): 381, ('Unknown', 'rdf_type', 'OntologyConstruct'): 6, ('Unknown', 'rev-rdf_type', 'Game'): 299},
      metagraph=[('Country', 'OntologyClass', 'rdf_type'), ('Country', 'Game', 'rev-hasCountry'), ('OntologyClass', 'OntologyConstruct', 'rdf_type'), ('OntologyClass', 'Country', 'rev-rdf_type'), ('OntologyClass', 'Developer', 'rev-rdf_type'), ('OntologyClass', 'Game', 'rev-rdf_type'), ('OntologyClass', 'Genre', 'rev-rdf_type'), ('OntologyClass', 'Platform', 'rev-rdf_type'), ('OntologyClass', 'Publisher', 'rev-rdf_type'), ('Game', 'Platform', 'availableOn'), ('Game', 'Developer', 'developedBy'), ('Game', 'Publisher', 'developedBy'), ('Game', 'Publisher', 'publishedBy'), ('Game', 'Country', 'hasCountry'), ('Game', 'Genre', 'hasGenre'), ('Game', 'Literal', 'hasReleaseDate'), ('Game', 'Literal', 'rdfs_label'), ('Game', 'OntologyClass', 'rdf_type'), ('Game', 'Unknown', 'rdf_type'), ('Developer', 'OntologyClass', 'rdf_type'), ('Developer', 'Game', 'rev-developedBy'), ('Platform', 'OntologyClass', 'rdf_type'), ('Platform', 'Game', 'rev-availableOn'), ('Publisher', 'OntologyClass', 'rdf_type'), ('Publisher', 'Game', 'rev-developedBy'), ('Publisher', 'Game', 'rev-publishedBy'), ('Genre', 'OntologyClass', 'rdf_type'), ('Genre', 'Game', 'rev-hasGenre'), ('Literal', 'Game', 'rev-hasReleaseDate'), ('Literal', 'Game', 'rev-rdfs_label'), ('Unknown', 'OntologyConstruct', 'rdf_type'), ('Unknown', 'Game', 'rev-rdf_type'), ('OntologyConstruct', 'OntologyClass', 'rev-rdf_type'), ('OntologyConstruct', 'Unknown', 'rev-rdf_type')])
Generated 117 MultiGenre labels and 182 SingleGenre labels.
VideoGameDataset.process() completed. Final graph has 1657 nodes.
Train/Valid/Test split sizes: Train=149, Valid=74, Test=76
Done saving data into cached files.
VideoGameDataset initialized.
Initializing RGCN  model
Start training...
Epoch 00000 | Train Acc: 0.6107 | Train Loss: 0.8003 | Valid Acc: 0.6081 | Valid loss: 0.7559 | Time: 0.1713
Epoch 00001 | Train Acc: 0.3893 | Train Loss: 0.9050 | Valid Acc: 0.3919 | Valid loss: 0.9174 | Time: 0.1590
Epoch 00002 | Train Acc: 0.4564 | Train Loss: 0.6957 | Valid Acc: 0.4189 | Valid loss: 0.7374 | Time: 0.1534
Epoch 00003 | Train Acc: 0.6980 | Train Loss: 0.6110 | Valid Acc: 0.5135 | Valid loss: 0.6758 | Time: 0.1515
Epoch 00004 | Train Acc: 0.6242 | Train Loss: 0.5904 | Valid Acc: 0.6081 | Valid loss: 0.6770 | Time: 0.1432
Epoch 00005 | Train Acc: 0.6174 | Train Loss: 0.5782 | Valid Acc: 0.6081 | Valid loss: 0.6987 | Time: 0.1383
Epoch 00006 | Train Acc: 0.6242 | Train Loss: 0.5546 | Valid Acc: 0.6081 | Valid loss: 0.7161 | Time: 0.1335
Epoch 00007 | Train Acc: 0.6846 | Train Loss: 0.5153 | Valid Acc: 0.6081 | Valid loss: 0.7217 | Time: 0.1347
Epoch 00008 | Train Acc: 0.7450 | Train Loss: 0.4671 | Valid Acc: 0.6081 | Valid loss: 0.7197 | Time: 0.1318
Epoch 00009 | Train Acc: 0.8456 | Train Loss: 0.4194 | Valid Acc: 0.6486 | Valid loss: 0.7230 | Time: 0.1315
Epoch 00010 | Train Acc: 0.8456 | Train Loss: 0.3750 | Valid Acc: 0.5811 | Valid loss: 0.7442 | Time: 0.1336
Epoch 00011 | Train Acc: 0.8792 | Train Loss: 0.3309 | Valid Acc: 0.5405 | Valid loss: 0.7869 | Time: 0.1314
Epoch 00012 | Train Acc: 0.8926 | Train Loss: 0.2824 | Valid Acc: 0.5676 | Valid loss: 0.8440 | Time: 0.1329
Epoch 00013 | Train Acc: 0.9128 | Train Loss: 0.2385 | Valid Acc: 0.5811 | Valid loss: 0.9102 | Time: 0.1307
Epoch 00014 | Train Acc: 0.9463 | Train Loss: 0.1970 | Valid Acc: 0.5946 | Valid loss: 0.9776 | Time: 0.1300
Epoch 00015 | Train Acc: 0.9530 | Train Loss: 0.1527 | Valid Acc: 0.5811 | Valid loss: 1.0399 | Time: 0.1282
Epoch 00016 | Train Acc: 0.9597 | Train Loss: 0.1169 | Valid Acc: 0.5541 | Valid loss: 1.1125 | Time: 0.1282
Epoch 00017 | Train Acc: 0.9732 | Train Loss: 0.0857 | Valid Acc: 0.5676 | Valid loss: 1.1913 | Time: 0.1270
Epoch 00018 | Train Acc: 0.9933 | Train Loss: 0.0595 | Valid Acc: 0.5676 | Valid loss: 1.2827 | Time: 0.1265
Epoch 00019 | Train Acc: 0.9933 | Train Loss: 0.0413 | Valid Acc: 0.5676 | Valid loss: 1.3874 | Time: 0.1260
Epoch 00020 | Train Acc: 1.0000 | Train Loss: 0.0262 | Valid Acc: 0.5676 | Valid loss: 1.5018 | Time: 0.1280
Epoch 00021 | Train Acc: 1.0000 | Train Loss: 0.0169 | Valid Acc: 0.5946 | Valid loss: 1.6237 | Time: 0.1274
Epoch 00022 | Train Acc: 1.0000 | Train Loss: 0.0104 | Valid Acc: 0.5541 | Valid loss: 1.7399 | Time: 0.1283
Epoch 00023 | Train Acc: 1.0000 | Train Loss: 0.0063 | Valid Acc: 0.5811 | Valid loss: 1.8529 | Time: 0.1273
Epoch 00024 | Train Acc: 1.0000 | Train Loss: 0.0041 | Valid Acc: 0.5811 | Valid loss: 1.9720 | Time: 0.1268
Epoch 00025 | Train Acc: 1.0000 | Train Loss: 0.0028 | Valid Acc: 0.5811 | Valid loss: 2.0944 | Time: 0.1263
Epoch 00026 | Train Acc: 1.0000 | Train Loss: 0.0018 | Valid Acc: 0.5811 | Valid loss: 2.2152 | Time: 0.1265
Epoch 00027 | Train Acc: 1.0000 | Train Loss: 0.0012 | Valid Acc: 0.5811 | Valid loss: 2.3329 | Time: 0.1273
Epoch 00028 | Train Acc: 1.0000 | Train Loss: 0.0009 | Valid Acc: 0.5676 | Valid loss: 2.4490 | Time: 0.1270
Epoch 00029 | Train Acc: 1.0000 | Train Loss: 0.0007 | Valid Acc: 0.5676 | Valid loss: 2.5606 | Time: 0.1260
Early stopping
End Training
Final validation accuracy of the model RGCN on unseen dataset: 0.631578947368421
Prepared learning problem for concept 'http://example.org/videogame#MultiGenreConcept':
  Train Pos: 87, Neg: 136
  Test Pos: 30, Neg: 46
Prepared learning problem for concept 'http://example.org/videogame#MultiGenreConcept':
  Train Pos: 87, Neg: 136
  Test Pos: 36, Neg: 40
Training EvoLearner (explanation) on videogames
<class 'ontolearn.search.EvoLearnerNode'> at 0xa4582c0  ≥ 2 hasGenre.Genre      Quality:0.8565  Length:4        Tree Length:3   Tree Depth:1    |Indv.|:117
Total time taken for EvoLearner (explanation)  on videogames: 10.37
Starting the  Run 4
Starting VideoGameDataset.process() (Full Custom Implementation).
Parsing data/videogames/videogame_f.rdf
Parsing data/KGs/videogame_f.rdf
Prepared 4825 raw rdflib triples after pre-scan.
DGL graph built. Details:
Graph(num_nodes={'Country': 20, 'Developer': 154, 'Game': 299, 'Genre': 107, 'Literal': 781, 'OntologyClass': 6, 'OntologyConstruct': 3, 'Platform': 127, 'Publisher': 153, 'Unknown': 7},
      num_edges={('Country', 'rdf_type', 'OntologyClass'): 20, ('Country', 'rev-hasCountry', 'Game'): 294, ('Developer', 'rdf_type', 'OntologyClass'): 154, ('Developer', 'rev-developedBy', 'Game'): 213, ('Game', 'availableOn', 'Platform'): 1236, ('Game', 'developedBy', 'Developer'): 213, ('Game', 'developedBy', 'Publisher'): 125, ('Game', 'hasCountry', 'Country'): 294, ('Game', 'hasGenre', 'Genre'): 472, ('Game', 'hasReleaseDate', 'Literal'): 580, ('Game', 'publishedBy', 'Publisher'): 381, ('Game', 'rdf_type', 'OntologyClass'): 299, ('Game', 'rdf_type', 'Unknown'): 299, ('Game', 'rdfs_label', 'Literal'): 299, ('Genre', 'rdf_type', 'OntologyClass'): 107, ('Genre', 'rev-hasGenre', 'Game'): 472, ('Literal', 'rev-hasReleaseDate', 'Game'): 580, ('Literal', 'rev-rdfs_label', 'Game'): 299, ('OntologyClass', 'rdf_type', 'OntologyConstruct'): 6, ('OntologyClass', 'rev-rdf_type', 'Country'): 20, ('OntologyClass', 'rev-rdf_type', 'Developer'): 154, ('OntologyClass', 'rev-rdf_type', 'Game'): 299, ('OntologyClass', 'rev-rdf_type', 'Genre'): 107, ('OntologyClass', 'rev-rdf_type', 'Platform'): 127, ('OntologyClass', 'rev-rdf_type', 'Publisher'): 207, ('OntologyConstruct', 'rev-rdf_type', 'OntologyClass'): 6, ('OntologyConstruct', 'rev-rdf_type', 'Unknown'): 6, ('Platform', 'rdf_type', 'OntologyClass'): 127, ('Platform', 'rev-availableOn', 'Game'): 1236, ('Publisher', 'rdf_type', 'OntologyClass'): 207, ('Publisher', 'rev-developedBy', 'Game'): 125, ('Publisher', 'rev-publishedBy', 'Game'): 381, ('Unknown', 'rdf_type', 'OntologyConstruct'): 6, ('Unknown', 'rev-rdf_type', 'Game'): 299},
      metagraph=[('Country', 'OntologyClass', 'rdf_type'), ('Country', 'Game', 'rev-hasCountry'), ('OntologyClass', 'OntologyConstruct', 'rdf_type'), ('OntologyClass', 'Country', 'rev-rdf_type'), ('OntologyClass', 'Developer', 'rev-rdf_type'), ('OntologyClass', 'Game', 'rev-rdf_type'), ('OntologyClass', 'Genre', 'rev-rdf_type'), ('OntologyClass', 'Platform', 'rev-rdf_type'), ('OntologyClass', 'Publisher', 'rev-rdf_type'), ('Game', 'Platform', 'availableOn'), ('Game', 'Developer', 'developedBy'), ('Game', 'Publisher', 'developedBy'), ('Game', 'Publisher', 'publishedBy'), ('Game', 'Country', 'hasCountry'), ('Game', 'Genre', 'hasGenre'), ('Game', 'Literal', 'hasReleaseDate'), ('Game', 'Literal', 'rdfs_label'), ('Game', 'OntologyClass', 'rdf_type'), ('Game', 'Unknown', 'rdf_type'), ('Developer', 'OntologyClass', 'rdf_type'), ('Developer', 'Game', 'rev-developedBy'), ('Platform', 'OntologyClass', 'rdf_type'), ('Platform', 'Game', 'rev-availableOn'), ('Publisher', 'OntologyClass', 'rdf_type'), ('Publisher', 'Game', 'rev-developedBy'), ('Publisher', 'Game', 'rev-publishedBy'), ('Genre', 'OntologyClass', 'rdf_type'), ('Genre', 'Game', 'rev-hasGenre'), ('Literal', 'Game', 'rev-hasReleaseDate'), ('Literal', 'Game', 'rev-rdfs_label'), ('Unknown', 'OntologyConstruct', 'rdf_type'), ('Unknown', 'Game', 'rev-rdf_type'), ('OntologyConstruct', 'OntologyClass', 'rev-rdf_type'), ('OntologyConstruct', 'Unknown', 'rev-rdf_type')])
Generated 117 MultiGenre labels and 182 SingleGenre labels.
VideoGameDataset.process() completed. Final graph has 1657 nodes.
Train/Valid/Test split sizes: Train=149, Valid=74, Test=76
Done saving data into cached files.
VideoGameDataset initialized.
Initializing RGCN  model
Start training...
Epoch 00000 | Train Acc: 0.5973 | Train Loss: 0.7040 | Valid Acc: 0.6081 | Valid loss: 0.6997 | Time: 0.1084
Epoch 00001 | Train Acc: 0.3893 | Train Loss: 1.0277 | Valid Acc: 0.3919 | Valid loss: 1.0685 | Time: 0.1059
Epoch 00002 | Train Acc: 0.6980 | Train Loss: 0.6238 | Valid Acc: 0.4865 | Valid loss: 0.7122 | Time: 0.1095
Epoch 00003 | Train Acc: 0.6174 | Train Loss: 0.5887 | Valid Acc: 0.5946 | Valid loss: 0.7118 | Time: 0.1068
Epoch 00004 | Train Acc: 0.6107 | Train Loss: 0.6104 | Valid Acc: 0.6081 | Valid loss: 0.7654 | Time: 0.1065
Epoch 00005 | Train Acc: 0.6242 | Train Loss: 0.5623 | Valid Acc: 0.6081 | Valid loss: 0.7344 | Time: 0.1048
Epoch 00006 | Train Acc: 0.6980 | Train Loss: 0.5179 | Valid Acc: 0.6081 | Valid loss: 0.6968 | Time: 0.1042
Epoch 00007 | Train Acc: 0.7919 | Train Loss: 0.4875 | Valid Acc: 0.6081 | Valid loss: 0.6847 | Time: 0.1058
Epoch 00008 | Train Acc: 0.8658 | Train Loss: 0.4545 | Valid Acc: 0.5946 | Valid loss: 0.6903 | Time: 0.1047
Epoch 00009 | Train Acc: 0.8859 | Train Loss: 0.4146 | Valid Acc: 0.5000 | Valid loss: 0.7145 | Time: 0.1049
Epoch 00010 | Train Acc: 0.8993 | Train Loss: 0.3728 | Valid Acc: 0.4865 | Valid loss: 0.7622 | Time: 0.1089
Epoch 00011 | Train Acc: 0.9060 | Train Loss: 0.3333 | Valid Acc: 0.4865 | Valid loss: 0.8390 | Time: 0.1082
Epoch 00012 | Train Acc: 0.9128 | Train Loss: 0.2905 | Valid Acc: 0.4865 | Valid loss: 0.9264 | Time: 0.1075
Epoch 00013 | Train Acc: 0.9195 | Train Loss: 0.2426 | Valid Acc: 0.4730 | Valid loss: 1.0145 | Time: 0.1072
Epoch 00014 | Train Acc: 0.9329 | Train Loss: 0.1998 | Valid Acc: 0.4595 | Valid loss: 1.1031 | Time: 0.1066
Epoch 00015 | Train Acc: 0.9732 | Train Loss: 0.1595 | Valid Acc: 0.4459 | Valid loss: 1.1820 | Time: 0.1100
Epoch 00016 | Train Acc: 0.9732 | Train Loss: 0.1248 | Valid Acc: 0.4730 | Valid loss: 1.2442 | Time: 0.1095
Epoch 00017 | Train Acc: 0.9732 | Train Loss: 0.0976 | Valid Acc: 0.4865 | Valid loss: 1.3091 | Time: 0.1090
Epoch 00018 | Train Acc: 0.9933 | Train Loss: 0.0713 | Valid Acc: 0.4865 | Valid loss: 1.3881 | Time: 0.1093
Epoch 00019 | Train Acc: 0.9933 | Train Loss: 0.0511 | Valid Acc: 0.5135 | Valid loss: 1.4891 | Time: 0.1089
Epoch 00020 | Train Acc: 1.0000 | Train Loss: 0.0349 | Valid Acc: 0.5270 | Valid loss: 1.6029 | Time: 0.1118
Epoch 00021 | Train Acc: 1.0000 | Train Loss: 0.0230 | Valid Acc: 0.5270 | Valid loss: 1.7322 | Time: 0.1128
Epoch 00022 | Train Acc: 1.0000 | Train Loss: 0.0151 | Valid Acc: 0.5135 | Valid loss: 1.8807 | Time: 0.1122
Epoch 00023 | Train Acc: 1.0000 | Train Loss: 0.0092 | Valid Acc: 0.5270 | Valid loss: 2.0441 | Time: 0.1118
Epoch 00024 | Train Acc: 1.0000 | Train Loss: 0.0057 | Valid Acc: 0.5135 | Valid loss: 2.2194 | Time: 0.1115
Epoch 00025 | Train Acc: 1.0000 | Train Loss: 0.0037 | Valid Acc: 0.5135 | Valid loss: 2.3950 | Time: 0.1118
Epoch 00026 | Train Acc: 1.0000 | Train Loss: 0.0024 | Valid Acc: 0.5135 | Valid loss: 2.5636 | Time: 0.1116
Epoch 00027 | Train Acc: 1.0000 | Train Loss: 0.0016 | Valid Acc: 0.5270 | Valid loss: 2.7198 | Time: 0.1114
Early stopping
End Training
Final validation accuracy of the model RGCN on unseen dataset: 0.6842105263157895
Prepared learning problem for concept 'http://example.org/videogame#MultiGenreConcept':
  Train Pos: 87, Neg: 136
  Test Pos: 30, Neg: 46
Prepared learning problem for concept 'http://example.org/videogame#MultiGenreConcept':
  Train Pos: 98, Neg: 125
  Test Pos: 36, Neg: 40
Training EvoLearner (explanation) on videogames
<class 'ontolearn.search.EvoLearnerNode'> at 0xe887809  ≥ 2 hasGenre.Genre      Quality:0.84305 Length:4        Tree Length:3   Tree Depth:1    |Indv.|:117
Total time taken for EvoLearner (explanation)  on videogames: 9.53
Starting the  Run 5
Starting VideoGameDataset.process() (Full Custom Implementation).
Parsing data/videogames/videogame_f.rdf
Parsing data/KGs/videogame_f.rdf
Prepared 4825 raw rdflib triples after pre-scan.
DGL graph built. Details:
Graph(num_nodes={'Country': 20, 'Developer': 154, 'Game': 299, 'Genre': 107, 'Literal': 781, 'OntologyClass': 6, 'OntologyConstruct': 3, 'Platform': 127, 'Publisher': 153, 'Unknown': 7},
      num_edges={('Country', 'rdf_type', 'OntologyClass'): 20, ('Country', 'rev-hasCountry', 'Game'): 294, ('Developer', 'rdf_type', 'OntologyClass'): 154, ('Developer', 'rev-developedBy', 'Game'): 213, ('Game', 'availableOn', 'Platform'): 1236, ('Game', 'developedBy', 'Developer'): 213, ('Game', 'developedBy', 'Publisher'): 125, ('Game', 'hasCountry', 'Country'): 294, ('Game', 'hasGenre', 'Genre'): 472, ('Game', 'hasReleaseDate', 'Literal'): 580, ('Game', 'publishedBy', 'Publisher'): 381, ('Game', 'rdf_type', 'OntologyClass'): 299, ('Game', 'rdf_type', 'Unknown'): 299, ('Game', 'rdfs_label', 'Literal'): 299, ('Genre', 'rdf_type', 'OntologyClass'): 107, ('Genre', 'rev-hasGenre', 'Game'): 472, ('Literal', 'rev-hasReleaseDate', 'Game'): 580, ('Literal', 'rev-rdfs_label', 'Game'): 299, ('OntologyClass', 'rdf_type', 'OntologyConstruct'): 6, ('OntologyClass', 'rev-rdf_type', 'Country'): 20, ('OntologyClass', 'rev-rdf_type', 'Developer'): 154, ('OntologyClass', 'rev-rdf_type', 'Game'): 299, ('OntologyClass', 'rev-rdf_type', 'Genre'): 107, ('OntologyClass', 'rev-rdf_type', 'Platform'): 127, ('OntologyClass', 'rev-rdf_type', 'Publisher'): 207, ('OntologyConstruct', 'rev-rdf_type', 'OntologyClass'): 6, ('OntologyConstruct', 'rev-rdf_type', 'Unknown'): 6, ('Platform', 'rdf_type', 'OntologyClass'): 127, ('Platform', 'rev-availableOn', 'Game'): 1236, ('Publisher', 'rdf_type', 'OntologyClass'): 207, ('Publisher', 'rev-developedBy', 'Game'): 125, ('Publisher', 'rev-publishedBy', 'Game'): 381, ('Unknown', 'rdf_type', 'OntologyConstruct'): 6, ('Unknown', 'rev-rdf_type', 'Game'): 299},
      metagraph=[('Country', 'OntologyClass', 'rdf_type'), ('Country', 'Game', 'rev-hasCountry'), ('OntologyClass', 'OntologyConstruct', 'rdf_type'), ('OntologyClass', 'Country', 'rev-rdf_type'), ('OntologyClass', 'Developer', 'rev-rdf_type'), ('OntologyClass', 'Game', 'rev-rdf_type'), ('OntologyClass', 'Genre', 'rev-rdf_type'), ('OntologyClass', 'Platform', 'rev-rdf_type'), ('OntologyClass', 'Publisher', 'rev-rdf_type'), ('Game', 'Platform', 'availableOn'), ('Game', 'Developer', 'developedBy'), ('Game', 'Publisher', 'developedBy'), ('Game', 'Publisher', 'publishedBy'), ('Game', 'Country', 'hasCountry'), ('Game', 'Genre', 'hasGenre'), ('Game', 'Literal', 'hasReleaseDate'), ('Game', 'Literal', 'rdfs_label'), ('Game', 'OntologyClass', 'rdf_type'), ('Game', 'Unknown', 'rdf_type'), ('Developer', 'OntologyClass', 'rdf_type'), ('Developer', 'Game', 'rev-developedBy'), ('Platform', 'OntologyClass', 'rdf_type'), ('Platform', 'Game', 'rev-availableOn'), ('Publisher', 'OntologyClass', 'rdf_type'), ('Publisher', 'Game', 'rev-developedBy'), ('Publisher', 'Game', 'rev-publishedBy'), ('Genre', 'OntologyClass', 'rdf_type'), ('Genre', 'Game', 'rev-hasGenre'), ('Literal', 'Game', 'rev-hasReleaseDate'), ('Literal', 'Game', 'rev-rdfs_label'), ('Unknown', 'OntologyConstruct', 'rdf_type'), ('Unknown', 'Game', 'rev-rdf_type'), ('OntologyConstruct', 'OntologyClass', 'rev-rdf_type'), ('OntologyConstruct', 'Unknown', 'rev-rdf_type')])
Generated 117 MultiGenre labels and 182 SingleGenre labels.
VideoGameDataset.process() completed. Final graph has 1657 nodes.
Train/Valid/Test split sizes: Train=149, Valid=74, Test=76
Done saving data into cached files.
VideoGameDataset initialized.
Initializing RGCN  model
Start training...
Epoch 00000 | Train Acc: 0.4027 | Train Loss: 0.7927 | Valid Acc: 0.4324 | Valid loss: 0.7647 | Time: 0.2554
Epoch 00001 | Train Acc: 0.6107 | Train Loss: 0.9652 | Valid Acc: 0.6081 | Valid loss: 0.9924 | Time: 0.1761
Epoch 00002 | Train Acc: 0.6107 | Train Loss: 0.6714 | Valid Acc: 0.6081 | Valid loss: 0.7431 | Time: 0.1499
Epoch 00003 | Train Acc: 0.6711 | Train Loss: 0.5892 | Valid Acc: 0.5946 | Valid loss: 0.6720 | Time: 0.1395
Epoch 00004 | Train Acc: 0.8725 | Train Loss: 0.6005 | Valid Acc: 0.5135 | Valid loss: 0.6945 | Time: 0.1383
Epoch 00005 | Train Acc: 0.8255 | Train Loss: 0.5916 | Valid Acc: 0.4189 | Valid loss: 0.7090 | Time: 0.1317
Epoch 00006 | Train Acc: 0.8859 | Train Loss: 0.5556 | Valid Acc: 0.5135 | Valid loss: 0.7075 | Time: 0.1273
Epoch 00007 | Train Acc: 0.9128 | Train Loss: 0.5069 | Valid Acc: 0.5676 | Valid loss: 0.7035 | Time: 0.1237
Epoch 00008 | Train Acc: 0.8993 | Train Loss: 0.4529 | Valid Acc: 0.5541 | Valid loss: 0.7097 | Time: 0.1212
Epoch 00009 | Train Acc: 0.8993 | Train Loss: 0.3984 | Valid Acc: 0.5541 | Valid loss: 0.7315 | Time: 0.1228
Epoch 00010 | Train Acc: 0.8993 | Train Loss: 0.3481 | Valid Acc: 0.5811 | Valid loss: 0.7703 | Time: 0.1206
Epoch 00011 | Train Acc: 0.9262 | Train Loss: 0.3008 | Valid Acc: 0.5946 | Valid loss: 0.8269 | Time: 0.1193
Epoch 00012 | Train Acc: 0.9396 | Train Loss: 0.2548 | Valid Acc: 0.6081 | Valid loss: 0.8878 | Time: 0.1176
Epoch 00013 | Train Acc: 0.9396 | Train Loss: 0.2109 | Valid Acc: 0.5946 | Valid loss: 0.9398 | Time: 0.1177
Epoch 00014 | Train Acc: 0.9396 | Train Loss: 0.1705 | Valid Acc: 0.5946 | Valid loss: 0.9818 | Time: 0.1186
Epoch 00015 | Train Acc: 0.9597 | Train Loss: 0.1347 | Valid Acc: 0.6081 | Valid loss: 1.0234 | Time: 0.1174
Epoch 00016 | Train Acc: 0.9799 | Train Loss: 0.1028 | Valid Acc: 0.6216 | Valid loss: 1.0739 | Time: 0.1162
Epoch 00017 | Train Acc: 0.9933 | Train Loss: 0.0746 | Valid Acc: 0.6216 | Valid loss: 1.1206 | Time: 0.1152
Epoch 00018 | Train Acc: 0.9933 | Train Loss: 0.0515 | Valid Acc: 0.6351 | Valid loss: 1.1725 | Time: 0.1151
Epoch 00019 | Train Acc: 1.0000 | Train Loss: 0.0343 | Valid Acc: 0.6351 | Valid loss: 1.2370 | Time: 0.1160
Epoch 00020 | Train Acc: 1.0000 | Train Loss: 0.0234 | Valid Acc: 0.6216 | Valid loss: 1.3075 | Time: 0.1169
Epoch 00021 | Train Acc: 1.0000 | Train Loss: 0.0161 | Valid Acc: 0.6216 | Valid loss: 1.3864 | Time: 0.1161
Epoch 00022 | Train Acc: 1.0000 | Train Loss: 0.0114 | Valid Acc: 0.6486 | Valid loss: 1.4792 | Time: 0.1158
Epoch 00023 | Train Acc: 1.0000 | Train Loss: 0.0080 | Valid Acc: 0.6622 | Valid loss: 1.5710 | Time: 0.1151
Epoch 00024 | Train Acc: 1.0000 | Train Loss: 0.0053 | Valid Acc: 0.6486 | Valid loss: 1.6546 | Time: 0.1147
Epoch 00025 | Train Acc: 1.0000 | Train Loss: 0.0034 | Valid Acc: 0.6486 | Valid loss: 1.7427 | Time: 0.1141
Epoch 00026 | Train Acc: 1.0000 | Train Loss: 0.0021 | Valid Acc: 0.6486 | Valid loss: 1.8355 | Time: 0.1143
Epoch 00027 | Train Acc: 1.0000 | Train Loss: 0.0013 | Valid Acc: 0.6351 | Valid loss: 1.9292 | Time: 0.1142
Epoch 00028 | Train Acc: 1.0000 | Train Loss: 0.0008 | Valid Acc: 0.6351 | Valid loss: 2.0197 | Time: 0.1138
Epoch 00029 | Train Acc: 1.0000 | Train Loss: 0.0006 | Valid Acc: 0.6351 | Valid loss: 2.1066 | Time: 0.1134
Epoch 00030 | Train Acc: 1.0000 | Train Loss: 0.0004 | Valid Acc: 0.6486 | Valid loss: 2.1880 | Time: 0.1134
Epoch 00031 | Train Acc: 1.0000 | Train Loss: 0.0003 | Valid Acc: 0.6486 | Valid loss: 2.2640 | Time: 0.1132
Epoch 00032 | Train Acc: 1.0000 | Train Loss: 0.0002 | Valid Acc: 0.6622 | Valid loss: 2.3333 | Time: 0.1132
Epoch 00033 | Train Acc: 1.0000 | Train Loss: 0.0002 | Valid Acc: 0.6757 | Valid loss: 2.3973 | Time: 0.1131
Epoch 00034 | Train Acc: 1.0000 | Train Loss: 0.0002 | Valid Acc: 0.6757 | Valid loss: 2.4557 | Time: 0.1130
Epoch 00035 | Train Acc: 1.0000 | Train Loss: 0.0001 | Valid Acc: 0.6622 | Valid loss: 2.5090 | Time: 0.1129
Epoch 00036 | Train Acc: 1.0000 | Train Loss: 0.0001 | Valid Acc: 0.6757 | Valid loss: 2.5570 | Time: 0.1126
Epoch 00037 | Train Acc: 1.0000 | Train Loss: 0.0001 | Valid Acc: 0.6757 | Valid loss: 2.5997 | Time: 0.1128
Epoch 00038 | Train Acc: 1.0000 | Train Loss: 0.0001 | Valid Acc: 0.6757 | Valid loss: 2.6375 | Time: 0.1127
Epoch 00039 | Train Acc: 1.0000 | Train Loss: 0.0001 | Valid Acc: 0.6757 | Valid loss: 2.6703 | Time: 0.1136
Epoch 00040 | Train Acc: 1.0000 | Train Loss: 0.0001 | Valid Acc: 0.6757 | Valid loss: 2.6985 | Time: 0.1133
Epoch 00041 | Train Acc: 1.0000 | Train Loss: 0.0001 | Valid Acc: 0.6757 | Valid loss: 2.7222 | Time: 0.1132
Epoch 00042 | Train Acc: 1.0000 | Train Loss: 0.0001 | Valid Acc: 0.6757 | Valid loss: 2.7420 | Time: 0.1131
Epoch 00043 | Train Acc: 1.0000 | Train Loss: 0.0001 | Valid Acc: 0.6757 | Valid loss: 2.7578 | Time: 0.1136
Epoch 00044 | Train Acc: 1.0000 | Train Loss: 0.0001 | Valid Acc: 0.6757 | Valid loss: 2.7701 | Time: 0.1133
Epoch 00045 | Train Acc: 1.0000 | Train Loss: 0.0001 | Valid Acc: 0.6757 | Valid loss: 2.7794 | Time: 0.1132
Epoch 00046 | Train Acc: 1.0000 | Train Loss: 0.0001 | Valid Acc: 0.6757 | Valid loss: 2.7858 | Time: 0.1131
Epoch 00047 | Train Acc: 1.0000 | Train Loss: 0.0001 | Valid Acc: 0.6757 | Valid loss: 2.7896 | Time: 0.1137
Epoch 00048 | Train Acc: 1.0000 | Train Loss: 0.0001 | Valid Acc: 0.6757 | Valid loss: 2.7911 | Time: 0.1139
Epoch 00049 | Train Acc: 1.0000 | Train Loss: 0.0001 | Valid Acc: 0.6757 | Valid loss: 2.7905 | Time: 0.1140
End Training
Final validation accuracy of the model RGCN on unseen dataset: 0.618421052631579
Prepared learning problem for concept 'http://example.org/videogame#MultiGenreConcept':
  Train Pos: 87, Neg: 136
  Test Pos: 30, Neg: 46
Prepared learning problem for concept 'http://example.org/videogame#MultiGenreConcept':
  Train Pos: 85, Neg: 138
  Test Pos: 33, Neg: 43
Training EvoLearner (explanation) on videogames
<class 'ontolearn.search.EvoLearnerNode'> at 0xa409383  (≥ 2 hasGenre.Genre) ⊔ Genre    Quality:0.89238 Length:6        Tree Length:5   Tree Depth:2    |Indv.|:224
<class 'ontolearn.search.EvoLearnerNode'> at 0xa40939c  Genre ⊔ (≥ 2 hasGenre.Genre)    Quality:0.89238 Length:6        Tree Length:5   Tree Depth:2    |Indv.|:224
Total time taken for EvoLearner (explanation)  on videogames: 10.90
+-------------+------------+---------------+----------------+-------------+---------------+--------------+---------------+------------+--------------+
|    Model    |  Dataset   | Pred Accuracy | Pred Precision | Pred Recall | Pred F1 Score | Exp Accuracy | Exp Precision | Exp Recall | Exp F1 Score |
+-------------+------------+---------------+----------------+-------------+---------------+--------------+---------------+------------+--------------+
|    CELOE    |    aifb    |     0.722     |     0.647      |    0.733    |     0.688     |     0.75     |     0.706     |    0.75    |    0.727     |
|  EvoLearner |    aifb    |     0.639     |     0.536      |     1.0     |     0.698     |    0.611     |      0.5      |    1.0     |    0.667     |
| PGExplainer |    aifb    |     0.861     |      0.75      |     1.0     |     0.857     |    0.889     |      0.8      |    1.0     |    0.889     |
|  SubGraphX  |    aifb    |     0.778     |     0.667      |    0.933    |     0.778     |    0.806     |     0.714     |   0.938    |    0.811     |
|    CELOE    |    bgs     |     0.483     |      0.4       |     1.0     |     0.571     |    0.414     |      0.32     |    1.0     |    0.485     |
|  EvoLearner |    bgs     |     0.483     |      0.4       |     1.0     |     0.571     |    0.414     |      0.32     |    1.0     |    0.485     |
| PGExplainer |    bgs     |     0.759     |      0.8       |     0.4     |     0.533     |    0.759     |      0.6      |   0.375    |    0.462     |
|  SubGraphX  |    bgs     |     0.655     |      0.5       |     0.4     |     0.444     |    0.655     |     0.375     |   0.375    |    0.375     |
|    CELOE    |   mutag    |     0.706     |     0.698      |    0.978    |     0.815     |    0.706     |      0.73     |   0.939    |    0.821     |
|  EvoLearner |   mutag    |      0.75     |     0.726      |     1.0     |     0.841     |    0.721     |     0.726     |   0.957    |    0.826     |
| PGExplainer |   mutag    |     0.588     |     0.698      |    0.667    |     0.682     |    0.735     |      0.86     |   0.755    |    0.804     |
|  SubGraphX  |   mutag    |     0.588     |     0.689      |    0.689    |     0.689     |    0.735     |     0.844     |   0.776    |    0.809     |
|  EvoLearner | videogames |      1.0      |      1.0       |     1.0     |      1.0      |    0.632     |     0.633     |   0.528    |    0.576     |
+-------------+------------+---------------+----------------+-------------+---------------+--------------+---------------+------------+--------------+
```

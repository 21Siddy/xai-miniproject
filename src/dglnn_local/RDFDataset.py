import abc
import itertools
import os
import re
from collections import OrderedDict,defaultdict
import rdflib as rdf
import dgl
import dgl.backend as F
import networkx as nx
import numpy as np
import torch as th
from dgl.data.dgl_dataset import DGLBuiltinDataset
from dgl.data.utils import (_get_dgl_url, generate_mask_tensor, idx2mask,
                            load_graphs, load_info, save_graphs, save_info)

__all__ = ["AIFBDataset", "MUTAGDataset", "BGSDataset", "AMDataset", "VideoGameDataset"] # Added VideoGameDataset

# Dictionary for renaming reserved node/edge type names to the ones
# that are allowed by nn.Module.
RENAME_DICT = {
    "type": "rdftype",
    "rev-type": "rev-rdftype",
}


class Entity:
    """Class for entities
    Parameters
    ----------
    id : str
        ID of this entity
    cls : str
        Type of this entity
    """

    def __init__(self, e_id, cls):
        self.id = e_id
        self.cls = cls

    def __str__(self):
        return "{}/{}".format(self.cls, self.id)


class Relation:
    """Class for relations
    Parameters
    ----------
    cls : str
        Type of this relation
    """

    def __init__(self, cls):
        self.cls = cls

    def __str__(self):
        return str(self.cls)


class RDFGraphDataset(DGLBuiltinDataset):
    """Base graph dataset class from RDF tuples.

    To derive from this, implement the following abstract methods:
    * ``parse_entity``
    * ``parse_relation``
    * ``process_tuple``
    * ``process_idx_file_line``
    * ``predict_category``
    Preprocessed graph and other data will be cached in the download folder
    to speedup data loading.
    The dataset should contain a "trainingSet.tsv" and a "testSet.tsv" file
    for training and testing samples.

    Attributes
    ----------
    num_classes : int
        Number of classes to predict
    predict_category : str
        The entity category (node type) that has labels for prediction

    Parameters
    ----------
    name : str
        Name of the dataset
    url : str or path
        URL to download the raw dataset.
    predict_category : str
        Predict category.
    print_every : int, optional
        Preprocessing log for every X tuples.
    insert_reverse : bool, optional
        If true, add reverse edge and reverse relations to the final graph.
        This parameter is primarily for the `RDFGraphDataset`'s default `process()` behavior.
        Custom `process()` implementations should handle reverse edges as needed.
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool, optional
        If true, force load and process from raw data. Ignore cached pre-processed data.
    verbose : bool
        Whether to print out progress information. Default: True.
    transform : callable, optional
        A transform that takes in a :class:`~dgl.DGLGraph` object and returns
        a transformed version. The :class:`~dgl.DGLGraph` object will be
        transformed before every access.
    """

    def __init__(
        self,
        name,
        url,
        predict_category,
        print_every=10000,
        insert_reverse=True,
        raw_dir=None,
        force_reload=False,
        verbose=True,
        transform=None,
    ):
        self._insert_reverse = insert_reverse
        self._print_every = print_every
        self._predict_category = predict_category

        super(RDFGraphDataset, self).__init__(
            name,
            url,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            transform=transform,
        )

    def process(self):
        """
        Default processing method. Subclasses are encouraged to override this
        if their RDF data structure or processing needs differ significantly.
        """
        raw_tuples = self.load_raw_tuples(self.raw_path)
        self.process_raw_tuples(raw_tuples, self.raw_path)

    def load_raw_tuples(self, root_path):
        """Loading raw RDF dataset

        Parameters
        ----------
        root_path : str
            Root path containing the data

        Returns
        -------
            Loaded rdf data (iterator of rdflib triples)
        """
        raw_rdf_graphs = []
        for _, filename in enumerate(os.listdir(root_path)):
            fmt = None
            if filename.endswith("nt"):
                fmt = "nt"
            elif filename.endswith("n3"):
                fmt = "n3"
            elif filename.endswith("ttl"):
                fmt = "turtle"
            elif filename.endswith("rdf") or filename.endswith("xml"):
                fmt = "xml"

            if fmt is None:
                continue
            g = rdf.Graph()
            print("Parsing file %s ..." % filename)
            try:
                g.parse(os.path.join(root_path, filename), format=fmt)
                raw_rdf_graphs.append(g)
            except Exception as e:
                print(f"Error parsing {filename} as {fmt}: {e}")

        return itertools.chain(*raw_rdf_graphs)

    def process_raw_tuples(self, raw_tuples, root_path):
        """Processing raw RDF dataset into a DGL graph.
        This method is generally intended for simpler RDF structures.
        More complex datasets often need a fully custom `process()` override.

        Parameters
        ----------
        raw_tuples: iterator
            Raw rdf triples (rdflib objects)
        root_path: str
            Root path containing the data
        """
        mg = nx.MultiDiGraph()
        ent_classes = OrderedDict() # Stores node type names and their IDs
        rel_classes = OrderedDict() # Stores relation type names and their IDs
        entities = OrderedDict()    # Stores original entity URIs/literals and their global DGL IDs
        src = []
        dst = []
        ntid = [] # Global node type IDs for each node
        etid = [] # Global edge type IDs for each edge
        
        # Sorting triples to ensure consistent entity/relation ID assignment across runs
        # Convert raw_tuples (an iterator) to a list for sorting
        sorted_tuples = sorted(list(raw_tuples)) 

        for i, (sbj_term, pred_term, obj_term) in enumerate(sorted_tuples):
            if self.verbose and i % self._print_every == 0:
                print("Processed %d tuples, found %d valid tuples." % (i, len(src)))
            
            sbjent = self.parse_entity(sbj_term)
            rel = self.parse_relation(pred_term)
            objent = self.parse_entity(obj_term)
            
            processed = self.process_tuple((sbj_term, pred_term, obj_term), sbjent, rel, objent)
            if processed is None:
                # ignored
                continue
            
            # Use the processed (sbj, rel, obj) if the method modifies them
            sbjent, rel, objent = processed

            # Populate metadata for graph construction
            sbj_cls = sbjent.cls
            obj_cls = objent.cls
            rel_cls = rel.cls

            sbj_cls_id = _get_id(ent_classes, sbj_cls)
            obj_cls_id = _get_id(ent_classes, obj_cls)
            rel_cls_id = _get_id(rel_classes, rel_cls)
            
            # Add to metagraph (schema graph)
            mg.add_edge(sbj_cls, obj_cls, key=rel_cls)
            if self._insert_reverse:
                mg.add_edge(obj_cls, sbj_cls, key="rev-%s" % rel_cls)
            
            # Map original entity string (URI/Literal) to a global DGL ID
            src_id = _get_id(entities, str(sbjent.id)) # Use entity ID, not its string representation
            if len(entities) > len(ntid): # If a new entity was added
                ntid.append(sbj_cls_id)
            
            dst_id = _get_id(entities, str(objent.id))
            if len(entities) > len(ntid): # If a new entity was added
                ntid.append(obj_cls_id)
            
            src.append(src_id)
            dst.append(dst_id)
            etid.append(rel_cls_id)

        src = np.asarray(src)
        dst = np.asarray(dst)
        ntid = np.asarray(ntid)
        etid = np.asarray(etid)
        ntypes = list(ent_classes.keys())
        etypes = list(rel_classes.keys())

        # add reverse edge with reverse relation
        if self._insert_reverse:
            if self.verbose:
                print("Adding reverse edges ...")
            newsrc = np.hstack([src, dst])
            newdst = np.hstack([dst, src])
            src = newsrc
            dst = newdst
            etid = np.hstack([etid, etid + len(etypes)])
            etypes.extend(["rev-%s" % t for t in etypes])

        hg = self.build_graph(mg, src, dst, ntid, etid, ntypes, etypes)

        # Map global DGL NIDs to original string IDs for prediction category
        self._entity_id_to_original_str_map = {v: k for k, v in entities.items()}
        # This mapping is crucial for `load_data_split` if it needs original strings.
        # It's better to make `_entity_id_to_original_str_map` a direct attribute of the dataset.

        if self.verbose:
            print("Load training/validation/testing split ...")
        
        # Ensure the predict_category exists in the graph
        if self.predict_category not in hg.ntypes:
            raise RuntimeError(f"Critical: Predict category '{self.predict_category}' not found in the graph. "
                               "Ensure `parse_entity` and input data correctly define nodes of this type.")

        # DGL NID for the predict_category nodes (local IDs within that node type)
        # These are the IDs that `train_idx`/`test_idx` will refer to.
        predict_cat_local_nids = F.asnumpy(hg.nodes[self.predict_category].data[dgl.NID])
        
        # Create a mapping from local DGL NID for predict_category to its global DGL NID
        local_predict_cat_id_to_global_id = {}
        for local_id, global_nid_val in enumerate(predict_cat_local_nids):
             local_predict_cat_id_to_global_id[local_id] = global_nid_val
        
        # findidfn maps original entity string to local DGL NID within predict_category
        def findidfn(original_entity_str):
            global_id = entities.get(original_entity_str)
            if global_id is None:
                return None
            
            # Find the local ID for this global ID within the predict_category nodes
            # This requires knowing the list of global NIDs for the predict_category.
            # This logic might be fragile if global_nids_for_predict_category is not ordered correctly.
            # A direct lookup of original_entity_str -> local_id is safer if entity map stores local IDs.
            # Given the current setup (global_ids in 'entities' map), we need to reverse map.
            
            # Get the list of global NIDs that belong to predict_category, sorted by their local NID
            global_nids_in_predict_category = sorted(list(F.asnumpy(hg.nodes[self.predict_category].data[dgl.NID])))
            
            try:
                # Find the local index (0-based) of the global_id within this list
                local_idx = global_nids_in_predict_category.index(global_id)
                return local_idx
            except ValueError:
                return None # Global ID not found in predict_category nodes

        self._hg = hg
        (
            train_idx,
            test_idx,
            labels,
            num_classes,
            idx_map, # This idx_map maps local predict_category NID to its original IRI
        ) = self.load_data_split(findidfn, root_path)

        train_mask = idx2mask(
            train_idx, self._hg.number_of_nodes(self.predict_category)
        )
        test_mask = idx2mask(test_idx, self._hg.number_of_nodes(self.predict_category))
        labels = F.tensor(labels, F.data_type_dict["int64"])

        train_mask = generate_mask_tensor(train_mask)
        test_mask = generate_mask_tensor(test_mask)
        
        # Assign masks and labels to the predict_category node data
        self._hg.nodes[self.predict_category].data["train_mask"] = train_mask
        self._hg.nodes[self.predict_category].data["test_mask"] = test_mask
        self._hg.nodes[self.predict_category].data["labels"] = labels
        self._hg.nodes[self.predict_category].data["label"] = labels # DGL's convention
        
        self._num_classes = num_classes
        self.idx_map = idx_map # Store the map from local NID to original IRI/info

    def build_graph(self, mg, src, dst, ntid, etid, ntypes, etypes):
        """Build the graphs

        Parameters
        ----------
        mg: MultiDiGraph
            Input metagraph
        src: Numpy array
            Source nodes global IDs
        dst: Numpy array
            Destination nodes global IDs
        ntid: Numpy array
            Node types (global type ID) for each node (global DGL ID)
        etid: Numpy array
            Edge types (global type ID) for each edge
        ntypes: list
            Node type names (ordered)
        etypes: list
            Edge type names (ordered)

        Returns
        -------
        g: DGLGraph (heterogeneous)
        """
        # create homo graph
        if self.verbose:
            print("Creating one whole graph ...")
        g = dgl.graph((src, dst))
        g.ndata[dgl.NTYPE] = F.tensor(ntid)
        g.edata[dgl.ETYPE] = F.tensor(etid)
        if self.verbose:
            print("Total #nodes:", g.number_of_nodes())
            print("Total #edges:", g.number_of_edges())

        # rename names such as 'type' so that they an be used as keys
        # to nn.ModuleDict. This applies to metagraph edge keys.
        etypes_renamed = [RENAME_DICT.get(ty, ty) for ty in etypes]
        
        # Reconstruct metagraph with renamed edge types if necessary
        mg_renamed = nx.MultiDiGraph()
        for u, v, k in mg.edges(keys=True):
            mg_renamed.add_edge(u, v, key=RENAME_DICT.get(k, k))

        # convert to heterograph
        if self.verbose:
            print("Convert to heterograph ...")
        
        # dgl.to_heterogeneous will use the global NTYPE and ETYPE data
        # along with the provided ntypes and etypes lists (which map type IDs to names)
        # to construct the heterogeneous graph.
        hg = dgl.to_heterogeneous(g, ntypes, etypes_renamed, metagraph=mg_renamed)
        
        if self.verbose:
            print("#Node types:", len(hg.ntypes))
            print("#Canonical edge types:", len(hg.etypes))
            print("#Unique edge type names:", len(set(hg.etypes)))
        return hg

    def load_data_split(self, ent2id_func, root_path):
        """Load data split from trainingSet.tsv and testSet.tsv.
        
        This method relies on external TSV files for splits.
        Custom datasets (like VideoGameDataset below) might override this
        or generate splits internally within their `process` method.

        Parameters
        ----------
        ent2id_func: func
            A function mapping original entity string to local DGL NID of predict_category.
        root_path: str
            Root path containing the data

        Return
        ------
        train_idx: Numpy array
            Local NIDs (within predict_category) for the training set
        test_idx: Numpy array
            Local NIDs (within predict_category) for the testing set
        labels: Numpy array
            Labels corresponding to the nodes in `predict_category`
        num_classes: int
            Number of classes
        idx_map: dict
            Mapping from local NID (within predict_category) to original entity info (e.g., IRI).
        """
        label_dict = {}
        # Initialize labels array for all nodes of predict_category
        labels = np.zeros((self._hg.number_of_nodes(self.predict_category),)) - 1
        
        train_idx, train_idx_map = self.parse_idx_file(
            os.path.join(root_path, "trainingSet.tsv"), ent2id_func, label_dict, labels
        )
        test_idx, test_idx_map = self.parse_idx_file(
            os.path.join(root_path, "testSet.tsv"), ent2id_func, label_dict, labels
        )
        
        train_idx = np.array(train_idx, dtype=np.int64)
        test_idx = np.array(test_idx, dtype=np.int64)
        labels = np.array(labels, dtype=np.int64)
        num_classes = len(label_dict)
        
        idx_map = train_idx_map
        idx_map.update(test_idx_map)
        
        return train_idx, test_idx, labels, num_classes, idx_map

    def parse_idx_file(self, filename, ent2id_func, label_dict, labels_array):
        """Parse idx files (trainingSet.tsv or testSet.tsv)

        Parameters
        ----------
        filename: str
            Path to the TSV file to parse
        ent2id_func: func
            A function mapping original entity string to local DGL NID of predict_category.
        label_dict: dict
            Map label string to label id (populated by this function)
        labels_array: np.array
            Array to populate with label IDs for entities (modified in place)

        Return
        ------
        idx: list
            List of local NIDs (within predict_category) found in this file
        idx_to_ent: dict
            Mapping from local NID to dictionary with original entity info (id, cls, IRI)
        """
        idx = []
        idx_to_ent = {} # Maps local predict_category NID to info about original entity
        if not os.path.exists(filename):
            print(f"Warning: Index file not found at: {filename}. Skipping.")
            return [], {}

        with open(filename, "r") as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue  # Skip header line
                
                # Process the line to get the original entity string and its label string
                # This abstract method needs concrete implementation in subclasses
                sample_entity_str, label_str = self.process_idx_file_line(line)
                
                # Map the original entity string to its local DGL NID within predict_category
                ent_local_id = ent2id_func(sample_entity_str)
                
                if ent_local_id is None:
                    print(
                        'Warning: entity "%s" from index file does not have a valid mapping in graph. Ignored.'
                        % sample_entity_str
                    )
                else:
                    idx.append(ent_local_id)
                    
                    # Get integer ID for the label string
                    # _get_id_aifb is specific to AIFB, use _get_id for general case
                    lbl_id = _get_id(label_dict, label_str) 
                    
                    labels_array[ent_local_id] = lbl_id
                    
                    # Store original info for the local ID
                    # This relies on parse_entity returning an Entity object that has id and cls
                    parsed_ent = self.parse_entity(rdf.URIRef(sample_entity_str)) # Re-parse to get Entity object
                    if parsed_ent:
                         idx_to_ent[ent_local_id] = {"id": parsed_ent.id, "cls": parsed_ent.cls, "IRI": sample_entity_str}
        return idx, idx_to_ent

    def has_cache(self):
        """check if there is a processed data"""
        graph_path = os.path.join(self.save_path, self.save_name + ".bin")
        info_path = os.path.join(self.save_path, self.save_name + ".pkl")
        if os.path.exists(graph_path) and os.path.exists(info_path):
            return True

        return False

    def save(self):
        """save the graph list and the labels"""
        graph_path = os.path.join(self.save_path, self.save_name + ".bin")
        info_path = os.path.join(self.save_path, self.save_name + ".pkl")
        save_graphs(str(graph_path), self._hg)
        save_info(
            str(info_path),
            {
                "num_classes": self.num_classes,
                "predict_category": self.predict_category,
                "idx_map": self.idx_map,
            },
        )

    def load(self):
        """load the graph list and the labels from disk"""
        graph_path = os.path.join(self.save_path, self.save_name + ".bin")
        info_path = os.path.join(self.save_path, self.save_name + ".pkl")
        graphs, _ = load_graphs(str(graph_path))

        info = load_info(str(info_path))
        self._num_classes = info["num_classes"]
        self._predict_category = info["predict_category"]
        self.idx_map = info["idx_map"]
        self._hg = graphs[0]
        # For backward compatibility if "label" not in self._hg.nodes[self.predict_category].data:
        if "labels" in self._hg.nodes[self.predict_category].data:
            self._hg.nodes[self.predict_category].data["label"] = self._hg.nodes[
                self.predict_category
            ].data["labels"]

    def __getitem__(self, idx):
        r"""Gets the graph object"""
        g = self._hg
        if self._transform is not None:
            g = self._transform(g)
        return g

    def __len__(self):
        r"""The number of graphs in the dataset."""
        return 1

    @property
    def save_name(self):
        return self.name + "_dgl_graph"

    @property
    def predict_category(self):
        return self._predict_category

    @property
    def num_classes(self):
        return self._num_classes

    @abc.abstractmethod
    def parse_entity(self, term):
        """Parse one entity from an RDF term (rdflib.term.Identifier).
        Return an Entity object or None if the term does not represent a valid entity
        and the whole triple should be ignored.

        Parameters
        ----------
        term : rdflib.term.Identifier
            RDF term (e.g., rdflib.URIRef, rdflib.Literal, rdflib.BNode)

        Returns
        -------
        Entity or None
            An entity object containing its ID and class (node type).
        """
        pass

    @abc.abstractmethod
    def parse_relation(self, term):
        """Parse one relation from an RDF term (rdflib.term.Identifier).
        Return a Relation object or None if the term does not represent a valid relation
        and the whole triple should be ignored.

        Parameters
        ----------
        term : rdflib.term.Identifier
            RDF term (e.g., rdflib.URIRef)

        Returns
        -------
        Relation or None
            A relation object containing its class (edge type).
        """
        pass

    @abc.abstractmethod
    def process_tuple(self, raw_tuple, sbj_parsed, rel_parsed, obj_parsed):
        """Process the parsed tuple. Return (Entity, Relation, Entity) tuple for as
        the final tuple to be added to the graph. Return None if the tuple should be ignored.
        This method can be used to filter or transform triples.

        Parameters
        ----------
        raw_tuple : tuple of rdflib.term.Identifier
            (subject, predicate, object) original RDF triple
        sbj_parsed : Entity
            Subject entity parsed by `parse_entity`
        rel_parsed : Relation
            Relation parsed by `parse_relation`
        obj_parsed : Entity
            Object entity parsed by `parse_entity`

        Returns
        -------
        (Entity, Relation, Entity) or None
            The final tuple to be used for graph construction or None if should be ignored.
        """
        pass

    @abc.abstractmethod
    def process_idx_file_line(self, line):
        """Process one line of a training/test index file (e.g., trainingSet.tsv or testSet.tsv).
        This method is used by `parse_idx_file`.

        Parameters
        ----------
        line : str
            One line from the index file

        Returns
        -------
        (str, str)
            A tuple containing the original entity string (e.g., its IRI) and its label string.
        """
        pass


def _get_id(id_map, key):
    """Helper function to get a unique integer ID for a given key,
    populating the map if the key is new."""
    id_val = id_map.get(key)
    if id_val is None:
        id_val = len(id_map)
        id_map[key] = id_val
    return id_val


# New code
# COnverting AIFB to Binray with fixed id1instance to class 1
def _get_id_aifb(id_map, key):
    """Specific ID mapping for AIFB dataset, converting to binary labels."""
    id_val = id_map.get(key)
    if id_val is None:
        if (
            key
            == "http://www.aifb.uni-karlsruhe.de/Forschungsgruppen/viewForschungsgruppeOWL/id1instance"
        ):
            id_val = int(1)
            id_map[key] = id_val  # Set the value to 1 for the given key
        else:
            id_val = int(0)
            id_map[key] = id_val
    return id_val


class AIFBDataset(RDFGraphDataset):
    r"""AIFB dataset for node classification task

    AIFB DataSet is a Semantic Web (RDF) dataset used as a benchmark
    in data mining. It records the organizational structure of AIFB at
    the University of Karlsruhe.

    AIFB dataset statistics:
    - Nodes: 7262
    - Edges: 48810 (including reverse edges)
    - Target Category: Person
    - Number of Classes: 4 (originally, but processed to binary for this implementation)

    Parameters
    ----------
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool, optional
        If true, force load and process from raw data. Default: False
    verbose : bool, optional
        Whether to print out progress information. Default: True
    transform : callable, optional
        A transform that takes in a :class:`~dgl.DGLGraph` object and returns
        a transformed version. The :class:`~dgl.DGLGraph` object will be
        transformed before every access.
    """

    def __init__(
        self, raw_dir=None, force_reload=False, verbose=True, transform=None
    ):
        name = "aifb"
        url = _get_dgl_url("dataset/aifb.zip")
        predict_category = "Person"
        self.entity_prefix = "http://www.aifb.uni-karlsruhe.de/Dir/"
        self.relation_prefix = "http://www.aifb.uni-karlsruhe.de/vocab#"
        super(AIFBDataset, self).__init__(
            name,
            url,
            predict_category,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            transform=transform,
        )

    def parse_entity(self, term):
        """Parses an AIFB RDF term into an Entity object."""
        # Special handling for schema URIs that might appear as subject/object
        if term == rdf.URIRef(self.entity_prefix + "Person"):
            return Entity(e_id="Person", cls="Person")
        if term == rdf.URIRef(self.entity_prefix + "Publication"):
            return Entity(e_id="Publication", cls="Publication")
        if term == rdf.URIRef(self.entity_prefix + "ResearchGroup"):
            return Entity(e_id="ResearchGroup", cls="ResearchGroup")
        if term == rdf.URIRef(self.entity_prefix + "Project"):
            return Entity(e_id="Project", cls="Project")
        
        entstr = str(term)
        if entstr.startswith(self.entity_prefix):
            # Assumes direct instances have format entity_prefix + ID
            return Entity(e_id=entstr[len(self.entity_prefix) :], cls="Person")
        elif isinstance(term, rdf.Literal):
            # Treat literals as a generic 'Literal' type if they appear as entities
            return Entity(e_id=entstr, cls="Literal")
        else:
            # For other URIs not matching known prefixes, or BNodes
            # Fallback to a generic entity type or ignore based on context
            return Entity(e_id=entstr, cls="Unknown") # or None to ignore

    def parse_relation(self, term):
        """Parses an AIFB RDF predicate into a Relation object."""
        relstr = str(term)
        if relstr.startswith(self.relation_prefix):
            return Relation(cls=relstr[len(self.relation_prefix) :])
        elif relstr == str(rdf.RDF.type):
            return Relation(cls='rdf_type')
        elif relstr == str(rdf.RDFS.subClassOf):
            return Relation(cls='rdfs_subClassOf')
        else:
            # Handle other common RDF/RDFS/OWL predicates explicitly if needed
            # Or sanitize and use as is. Replacing non-alphanumeric chars for valid DGL etype names.
            sanitized_rel_cls = re.sub(r'[^a-zA-Z0-9_]', '_', relstr)
            return Relation(cls=sanitized_rel_cls)


    def process_tuple(self, raw_tuple, sbj, rel, obj):
        # For AIFB, we only preserve Person->*->* relations
        if sbj is None or rel is None or obj is None:
            return None
        if sbj.cls == "Person":
            return (sbj, rel, obj)
        else:
            return None

    def process_idx_file_line(self, line):
        """Processes one line from AIFB's trainingSet.tsv/testSet.tsv."""
        person_uri, _, label_uri = line.strip().split("\t")
        return person_uri, label_uri


class MUTAGDataset(RDFGraphDataset):
    r"""MUTAG dataset for node classification task

    The MUTAG dataset is an RDF dataset, describing chemical compounds
    and their mutagenic effect.

    MUTAG dataset statistics:
    - Nodes: 2364
    - Edges: 7120 (including reverse edges)
    - Target Category: Compound
    - Number of Classes: 2

    Parameters
    ----------
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool, optional
        If true, force load and process from raw data. Default: False
    verbose : bool, optional
        Whether to print out progress information. Default: True
    transform : callable, optional
        A transform that takes in a :class:`~dgl.DGLGraph` object and returns
        a transformed version. The :class:`~dgl.DGLGraph` object will be
        transformed before every access.
    """

    def __init__(
        self, raw_dir=None, force_reload=False, verbose=True, transform=None
    ):
        name = "mutag"
        url = _get_dgl_url("dataset/mutag.zip")
        predict_category = "Compound"
        self.compound_prefix = "http://reach.in.tum.de/data/mutag#compound"  # not explicitly used in parsing logic below
        self.entity_prefix = "http://dl-learner.org/mutag#"
        self.relation_prefix = "http://dl-learner.org/mutag#"
        super(MUTAGDataset, self).__init__(
            name,
            url,
            predict_category,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            transform=transform,
        )

    def parse_entity(self, term):
        """Parses a MUTAG RDF term into an Entity object."""
        entstr = str(term)
        if entstr.startswith(self.entity_prefix):
            # Split by '_' to distinguish between compound (single part) and atom (type_instance)
            sp = entstr[len(self.entity_prefix) :].split("_")
            if len(sp) == 1:
                return Entity(e_id=sp[0], cls="Compound") # e.g., '1' for compound_1
            elif len(sp) == 2:
                # atom: e.g., 'C_1' -> type 'C', instance '1'
                cls, inst = sp
                return Entity(e_id=inst, cls=cls.capitalize()) # Capitalize type for node name consistency
            else:
                return Entity(e_id=entstr, cls="UnknownMutagEntity") # Fallback for unexpected formats
        elif isinstance(term, rdf.Literal):
            return Entity(e_id=entstr, cls="Literal")
        else:
            return None # Ignore terms not from this ontology or literals for parsing

    def parse_relation(self, term):
        """Parses a MUTAG RDF predicate into a Relation object."""
        relstr = str(term)
        if relstr.startswith(self.relation_prefix):
            return Relation(cls=relstr[len(self.relation_prefix) :])
        else:
            # Sanitize relation names that might not be simple local names
            relstr = relstr.replace(".", "_").replace("#", "_").replace("/", "_").replace(":", "_")
            return Relation(cls=relstr)

    def process_tuple(self, raw_tuple, sbj, rel, obj):
        # For MUTAG, all valid parsed triples are accepted
        if sbj is None or rel is None or obj is None:
            return None
        return (sbj, rel, obj)

    def process_idx_file_line(self, line):
        """Processes one line from MUTAG's trainingSet.tsv/testSet.tsv."""
        compound_uri, _, label_str = line.strip().split("\t")
        return compound_uri, label_str


class BGSDataset(RDFGraphDataset):
    r"""BGS dataset for node classification task

    BGS dataset is an RDF dataset, consisting of properties of different
    rocks, rock units and geological timescales in the UK.

    BGS dataset statistics:
    - Nodes: 12015
    - Edges: 104538 (including reverse edges)
    - Target Category: Sample
    - Number of Classes: 2

    Parameters
    ----------
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool, optional
        If true, force load and process from raw data. Default: False
    verbose : bool, optional
        Whether to print out progress information. Default: True
    transform : callable, optional
        A transform that takes in a :class:`~dgl.DGLGraph` object and returns
        a transformed version. The :class:`~dgl.DGLGraph` object will be
        transformed before every access.
    """

    def __init__(
        self, raw_dir=None, force_reload=False, verbose=True, transform=None
    ):
        name = "bgs"
        url = _get_dgl_url("dataset/bgs.zip")
        predict_category = "Sample"
        self.entity_prefix = "http://data.bgs.ac.uk/id/sample/"
        self.relation_prefix = "http://data.bgs.ac.uk/ref/lexicon/"
        super(BGSDataset, self).__init__(
            name,
            url,
            predict_category,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            transform=transform,
        )

    def parse_entity(self, term):
        """Parses a BGS RDF term into an Entity object."""
        if str(term) == "http://www.w3.org/2002/07/owl#Thing":
            return Entity(e_id="Thing", cls="Thing")
        entstr = str(term)
        if entstr.startswith(self.entity_prefix):
            sp = entstr[len(self.entity_prefix) :].split("#")
            if len(sp) == 1:
                return Entity(e_id=sp[0], cls="Sample") # Sample instances
            else:
                return None # For unexpected hash splits within entity prefix
        elif isinstance(term, rdf.Literal):
            return Entity(e_id=entstr, cls="Literal")
        else:
            return None # Ignore terms not matching BGS samples or literals

    def parse_relation(self, term):
        """Parses a BGS RDF predicate into a Relation object."""
        if str(term) == "http://www.opengis.net/ont/geosparql#hasGeometry":
            return Relation(cls="hasGeometry")
        relstr = str(term)
        if relstr.startswith(self.relation_prefix):
            return Relation(cls=relstr[len(self.relation_prefix) :])
        else:
            # Sanitize relation names
            relstr = relstr.replace(".", "_").replace("#", "_").replace("/", "_").replace(":", "_")
            return Relation(cls=relstr)

    def process_tuple(self, raw_tuple, sbj, rel, obj):
        # For BGS, all valid parsed triples are accepted
        if sbj is None or rel is None or obj is None:
            return None
        return (sbj, rel, obj)

    def process_idx_file_line(self, line):
        """Processes one line from BGS's trainingSet.tsv/testSet.tsv."""
        sample_uri, _, label_str = line.strip().split("\t")
        return sample_uri, label_str


class AMDataset(RDFGraphDataset):
    r"""AM dataset for node classification task

    AM dataset is an RDF dataset, containing information about more than
    1,000 components of the "Aircraft Maintenance" domain.

    AM dataset statistics:
    - Nodes: 1693
    - Edges: 104538 (including reverse edges)
    - Target Category: Component
    - Number of Classes: 2

    Parameters
    ----------
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool, optional
        If true, force load and process from raw data. Default: False
    verbose : bool, optional
        Whether to print out progress information. Default: True
    transform : callable, optional
        A transform that takes in a :class:`~dgl.DGLGraph` object and returns
        a transformed version. The :class:`~dgl.DGLGraph` object will be
        transformed before every access.
    """

    def __init__(
        self, raw_dir=None, force_reload=False, verbose=True, transform=None
    ):
        name = "am"
        url = _get_dgl_url("dataset/am.zip")
        predict_category = "Component"
        self.entity_prefix = "http://am.isi.edu/ontologies/domain-ontology#"
        self.relation_prefix = "http://am.isi.edu/ontologies/domain-ontology#"
        self.objectCategory = (
            "http://www.w3.org/2002/07/owl#ObjectProperty"
        )
        self.material = "http://am.isi.edu/ontologies/domain-ontology#material"
        super(AMDataset, self).__init__(
            name,
            url,
            predict_category,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            transform=transform,
        )

    def __len__(self):
        """
        Overwrite the super class's __len__ method.

        Return
        -------
        int
        """
        return super(AMDataset, self).__len__()

    def parse_entity(self, term):
        """Parses an AM RDF term into an Entity object."""
        if isinstance(term, rdf.Literal):
            return None # Do not create entities for literals in this dataset
        elif isinstance(term, rdf.BNode):
            return Entity(e_id=str(term), cls="_BNode")
        entstr = str(term)
        if entstr.startswith(self.entity_prefix):
            # Expected format: http://am.isi.edu/ontologies/domain-ontology#TYPE-ID
            # Or http://am.isi.edu/ontologies/domain-ontology#ConceptName (for classes)
            parts = entstr.split("#")
            if len(parts) == 2:
                local_name = parts[1]
                if "-" in local_name:
                    # Likely an instance like 'Component-123'
                    cls_name, inst_id = local_name.split("-", 1) # Split only on first '-'
                    return Entity(e_id=inst_id, cls=cls_name)
                else:
                    # Likely a class name or un-instantiated concept
                    return Entity(e_id=local_name, cls="Concept") # Generic concept type
            else:
                return Entity(e_id=entstr, cls="UnknownAMEntity") # Fallback for unexpected format
        else:
            return None # Ignore terms not from AM ontology

    def parse_relation(self, term):
        """Parses an AM RDF predicate into a Relation object."""
        # Specific relations to ignore
        if term == rdf.URIRef(self.objectCategory) or term == rdf.URIRef(self.material):
            return None
        
        relstr = str(term)
        if relstr.startswith(self.relation_prefix):
            # Expected format: http://am.isi.edu/ontologies/domain-ontology#relationName
            local_name = relstr.split("#")[-1]
            return Relation(cls=local_name)
        else:
            # Sanitize other relation URIs
            sanitized_rel_cls = re.sub(r'[^a-zA-Z0-9_]', '_', relstr)
            return Relation(cls=sanitized_rel_cls)

    def process_tuple(self, raw_tuple, sbj, rel, obj):
        # For AM, only valid parsed triples are accepted, no specific filtering
        if sbj is None or rel is None or obj is None:
            return None
        return (sbj, rel, obj)

    def process_idx_file_line(self, line):
        """Processes one line from AM's trainingSet.tsv/testSet.tsv."""
        component_uri, label_str = line.strip().split("\t")
        return component_uri, label_str

# New class for VideoGameDataset, inheriting from RDFGraphDataset but fully overriding process
class VideoGameDataset(RDFGraphDataset):
    def __init__(self, raw_dir=None, force_reload=True, verbose=True, transform=None):
        dataset_name = 'videogames'
        dataset_url = None # No direct download URL, data assumed to be local

        self.data_file = 'videogame_f.rdf'
        self.kb_file = 'videogame_f.rdf'
        self.label_property = 'http://example.org/videogame#hasGenre'
        self.entity_prefix = 'http://example.org/videogame#'
        self.relation_prefix = 'http://example.org/videogame#'
        _predict_category = 'Game' 

        self.original_id_feat_name = dgl.NID 
        
        # Initialize helper attributes that `process()` will use.
        self._entity_id_to_original_str_map = {}
        self._original_str_to_entity_id_map = {}
        self._next_entity_id = 0
        self.instance_types = {} # To store inferred types for instances

        # Initialize dataset properties
        self.has_feature = False
        self.multi_label = False
        self.meta_paths_dict = None
        
        self.g = None
        self.labels = None
        self.train_idx = None
        self.valid_idx = None
        self.test_idx = None
        self.idx_map = None

        super().__init__(
            name=dataset_name,
            url=dataset_url,
            predict_category=_predict_category,
            force_reload=force_reload,
            raw_dir=raw_dir,
            verbose=verbose,
            transform=transform
        )

        print("VideoGameDataset initialized.")

    def _get_id(self, original_str):
        """Helper to get a unique global integer ID for an entity string/literal."""
        if original_str not in self._original_str_to_entity_id_map:
            new_id = self._next_entity_id
            self._original_str_to_entity_id_map[original_str] = new_id
            self._entity_id_to_original_str_map[new_id] = original_str
            self._next_entity_id += 1
        return self._original_str_to_entity_id_map[original_str]

    # In class VideoGameDataset in RDFDataset.py
    
    def load_raw_tuples(self, root_path):
        """
        Loads raw RDF triples and performs a multi-stage pre-scan to
        infer entity types from schema and explicit rdf:type statements.
        """
        
        rdf_file_path = os.path.join(root_path, self.name, self.data_file)
        kb_file_path = os.path.join(self.raw_dir, 'KGs', self.kb_file) 

        g = rdf.Graph()
        for path in (rdf_file_path, kb_file_path):
            if not os.path.exists(path):
                print(f"Warning: RDF data file not found at: {path}, skipping.")
                continue
            
            print(f"Parsing {path}")
            # --- Reverting to robust parsing loop ---
            parsed = False
            for fmt in ['xml', 'turtle', 'nt', 'n3']: # Try multiple formats
                try:
                    g.parse(path, format=fmt)
                    parsed = True
                    break 
                except Exception: # Suppress errors for incorrect format trials
                    continue
            
            if not parsed:
                print(f"ERROR: Could not parse {path} with any of 'xml', 'turtle', 'nt', 'n3' formats.")

        # --- Pre-scanning Logic (Keep this) ---
        self.instance_types.clear()
        domain_map = {}
        range_map = {}

        # 1. Scan for schema definitions (domain and range) to infer types
        for p, o in g.subject_objects(predicate=rdf.RDFS.domain):
            if isinstance(p, rdf.URIRef) and isinstance(o, rdf.URIRef):
                domain_map[str(p)] = str(o).split('#')[-1]
        for p, o in g.subject_objects(predicate=rdf.RDFS.range):
            if isinstance(p, rdf.URIRef) and isinstance(o, rdf.URIRef):
                range_map[str(p)] = str(o).split('#')[-1]

        # 2. Scan for explicit rdf:type triples, which are the most reliable source
        for s, o in g.subject_objects(predicate=rdf.RDF.type):
            if isinstance(s, rdf.URIRef) and isinstance(o, rdf.URIRef):
                type_name = str(o).split('#')[-1]
                # Filter out schema definitions from being instance types
                if 'Class' not in type_name and 'Property' not in type_name:
                    self.instance_types[str(s)] = type_name

        # 3. Use schema to infer types for any remaining untyped entities
        for s, p, o in g:
            s_str, p_str, o_str = str(s), str(p), str(o)
            # Infer subject type from domain
            if isinstance(s, rdf.URIRef) and s_str not in self.instance_types:
                domain_type = domain_map.get(p_str)
                if domain_type:
                    self.instance_types[s_str] = domain_type
            # Infer object type from range
            if isinstance(o, rdf.URIRef) and o_str not in self.instance_types:
                range_type = range_map.get(p_str)
                if range_type:
                    self.instance_types[o_str] = range_type
        
        raw_rdflib_triples = list(g)
        print(f"Prepared {len(raw_rdflib_triples)} raw rdflib triples after pre-scan.")
        return raw_rdflib_triples
    
    def parse_entity(self, term):
        """
        Parses an RDF term into an Entity object, using pre-scanned type info.
        This version prioritizes explicit instance types over other checks.
        """
        if isinstance(term, rdf.Literal):
            return Entity(e_id=str(term), cls="Literal")
        elif isinstance(term, rdf.BNode):
            return Entity(e_id=str(term), cls="_BNode")
        elif isinstance(term, rdf.URIRef):
            entstr = str(term)
            
            # 1. Prioritize explicit instance typing from the pre-scan.
            # This correctly identifies nodes like "...#Tetris" as "Game".
            instance_class = self.instance_types.get(entstr)
            if instance_class:
                return Entity(e_id=entstr, cls=instance_class)

            # 2. If it's not an explicitly typed instance, check if it's a class definition.
            # This correctly identifies nodes like "...#Game" as "OntologyClass".
            known_classes = ["Game", "Genre", "Platform", "Developer", "Publisher", "Country", "Company", "VideoGameEntity"]
            local_name = entstr.split('#')[-1]
            if entstr.startswith(self.entity_prefix) and local_name in known_classes:
                return Entity(e_id=entstr, cls="OntologyClass")

            # 3. Handle other ontology constructs (e.g., owl:Class).
            if entstr.endswith(("#Class", "#ObjectProperty", "#DatatypeProperty")):
                return Entity(e_id=entstr, cls="OntologyConstruct")

            # 4. Fallback for any other URI that wasn't typed.
            return Entity(e_id=entstr, cls="Unknown")
        
        return None


    def parse_relation(self, term):
        """
        Parses an RDF predicate into a Relation object.
        """
        if isinstance(term, rdf.URIRef):
            relstr = str(term)
            if relstr.startswith(self.relation_prefix):
                return Relation(cls=relstr.split('#')[-1])
            
            # Handle common RDF/RDFS/OWL predicates
            common_predicates = {
                str(rdf.RDFS.label): 'rdfs_label',
                str(rdf.RDF.type): 'rdf_type',
                str(rdf.RDFS.domain): 'rdfs_domain',
                str(rdf.RDFS.range): 'rdfs_range',
                str(rdf.RDFS.subClassOf): 'rdfs_subClassOf'
            }
            if relstr in common_predicates:
                return Relation(cls=common_predicates[relstr])
            else:
                # Sanitize other URIs
                local_name = relstr.split('/')[-1].split('#')[-1].replace('.', '_').replace('-', '_')
                return Relation(cls=local_name if local_name else 'unknown_relation')
        return None

    def process_tuple(self, raw_tuple, sbj, rel, obj):
        if sbj is None or rel is None or obj is None:
            return None 
        return (sbj, rel, obj) 

    def process(self):
        """
        Main processing method for the VideoGame dataset.
        """
        print("Starting VideoGameDataset.process() (Full Custom Implementation).")

        raw_rdflib_triples = self.load_raw_tuples(self.raw_dir)

        self._entity_id_to_original_str_map.clear()
        self._original_str_to_entity_id_map.clear()
        self._next_entity_id = 0
        
        canonical_edges_data = defaultdict(lambda: defaultdict(list))
        node_type_local_id_map = defaultdict(dict)
        node_type_current_local_id = defaultdict(int)
        all_node_types_encountered = set()

        for s_rdflib, p_rdflib, o_rdflib in raw_rdflib_triples:
            sbj_entity = self.parse_entity(s_rdflib)
            obj_entity = self.parse_entity(o_rdflib)
            rel_entity = self.parse_relation(p_rdflib)

            processed_triple = self.process_tuple((s_rdflib, p_rdflib, o_rdflib), sbj_entity, rel_entity, obj_entity)
            if not processed_triple:
                continue

            sbj_entity, rel_entity, obj_entity = processed_triple
            
            # Assign global and local IDs
            for ent in [sbj_entity, obj_entity]:
                self._get_id(ent.id) # Ensure global ID exists
                all_node_types_encountered.add(ent.cls)
                if ent.id not in node_type_local_id_map[ent.cls]:
                    node_type_local_id_map[ent.cls][ent.id] = node_type_current_local_id[ent.cls]
                    node_type_current_local_id[ent.cls] += 1
            
            s_local_id = node_type_local_id_map[sbj_entity.cls][sbj_entity.id]
            o_local_id = node_type_local_id_map[obj_entity.cls][obj_entity.id]

            canonical_etype = (sbj_entity.cls, rel_entity.cls, obj_entity.cls)
            canonical_edges_data[canonical_etype]['src'].append(s_local_id)
            canonical_edges_data[canonical_etype]['dst'].append(o_local_id)

            if self._insert_reverse:
                rev_canonical_etype = (obj_entity.cls, f"rev-{rel_entity.cls}", sbj_entity.cls)
                canonical_edges_data[rev_canonical_etype]['src'].append(o_local_id)
                canonical_edges_data[rev_canonical_etype]['dst'].append(s_local_id)

        data_dict = {
            etype: (th.tensor(data['src'], dtype=th.int64), th.tensor(data['dst'], dtype=th.int64))
            for etype, data in canonical_edges_data.items() if data['src']
        }
        
        num_nodes_dict = {ntype: count for ntype, count in node_type_current_local_id.items()}
        for ntype in all_node_types_encountered:
            if ntype not in num_nodes_dict:
                num_nodes_dict[ntype] = 0

        self._hg = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)
        self.g = self._hg

        print("DGL graph built. Details:")
        print(self.g)

        if self.predict_category not in self.g.ntypes:
            raise RuntimeError(f"Target node type '{self.predict_category}' not found in graph.")
        
        # --- Label and Split Generation ---
        g_rdflib_for_labels = rdf.Graph()
        for triple in raw_rdflib_triples:
             g_rdflib_for_labels.add(triple)
        
        has_genre_rel_uri = rdf.URIRef(self.label_property)
        
        # Get game nodes, sorted by their local ID
        game_nodes_sorted_by_local_id = sorted(
            node_type_local_id_map[self.predict_category].items(),
            key=lambda item: item[1]
        )
        original_game_uris = [item[0] for item in game_nodes_sorted_by_local_id]

        labels_list = []
        for original_game_uri in original_game_uris:
            game_uri_ref = rdf.URIRef(original_game_uri)
            genre_count = len(list(g_rdflib_for_labels.triples((game_uri_ref, has_genre_rel_uri, None))))
            labels_list.append(1 if genre_count >= 2 else 0)

        self.labels = th.tensor(labels_list, dtype=th.long)
        self.g.nodes[self.predict_category].data['labels'] = self.labels
        self.g.nodes[self.predict_category].data['label'] = self.labels
        self._num_classes = 2

        print(f"Generated {self.labels.sum().item()} MultiGenre labels and "
              f"{len(self.labels) - self.labels.sum().item()} SingleGenre labels.")

        num_game_nodes = len(self.labels)
        if num_game_nodes == 0:
            raise RuntimeError("No 'Game' nodes found to generate labels and masks for.")

        np.random.seed(42)
        indices = np.random.permutation(num_game_nodes)
        train_size = int(0.5 * num_game_nodes)
        val_size = int(0.25 * num_game_nodes)
        
        train_idx_th = th.tensor(indices[:train_size], dtype=th.long)
        val_idx_th = th.tensor(indices[train_size : train_size + val_size], dtype=th.long)
        test_idx_th = th.tensor(indices[train_size + val_size :], dtype=th.long)

        self.g.nodes[self.predict_category].data['train_mask'] = generate_mask_tensor(idx2mask(train_idx_th, num_game_nodes))
        self.g.nodes[self.predict_category].data['val_mask'] = generate_mask_tensor(idx2mask(val_idx_th, num_game_nodes))
        self.g.nodes[self.predict_category].data['test_mask'] = generate_mask_tensor(idx2mask(test_idx_th, num_game_nodes))
        
        self.train_idx = train_idx_th
        self.valid_idx = val_idx_th
        self.test_idx = test_idx_th

        self.idx_map = {
            local_id: {"IRI": uri} 
            for uri, local_id in node_type_local_id_map[self.predict_category].items()
        }
        
        print(f"VideoGameDataset.process() completed. Final graph has {self.g.num_nodes()} nodes.")
        print(f"Train/Valid/Test split sizes: Train={len(self.train_idx)}, Valid={len(self.valid_idx)}, Test={len(self.test_idx)}")

    def load_data_split(self, ent2id_func=None, root_path=None):
        if self.g is None:
             raise RuntimeError("Dataset not fully processed. Call `dataset.process()` first.")
        return (
            self.train_idx.tolist(),
            self.test_idx.tolist(),
            self.labels.tolist(),
            self.num_classes,
            self.idx_map,
        )

    def process_idx_file_line(self, line):
        raise NotImplementedError("VideoGameDataset does not use external index files.")

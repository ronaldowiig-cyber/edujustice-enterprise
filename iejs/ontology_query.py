"""Read and query the project's RDF/Turtle and RDF/XML ontology files."""

from __future__ import annotations
from pathlib import Path
from rdflib import Graph, URIRef
from rdflib.namespace import OWL, RDF, RDFS

def load_graph(path: str | Path, *, format: str | None = None) -> Graph:
    ontology_path = Path(path)
    if not ontology_path.is_file():
        raise FileNotFoundError(f"Ontology file does not exist: {ontology_path}")

    parser_format = format
    if parser_format is None:
        suffix = ontology_path.suffix.lower()
        parser_format = {
            ".ttl": "turtle",
            ".owl": "xml",
            ".rdf": "xml",
            ".xml": "xml",
        }.get(suffix)

    if parser_format is None:
        raise ValueError("Could not determine RDF format from file extension.")

    graph = Graph()
    graph.parse(str(ontology_path), format=parser_format)
    return graph

def load_project_ontology(*, source: str = "ttl", base_dir: str | Path | None = None) -> Graph:
    project_root = (
        Path(base_dir).resolve()
        if base_dir is not None
        else Path(__file__).resolve().parent.parent
    )
    source_key = source.lower()
    if source_key in {"ttl", "turtle"}:
        return load_graph(project_root / "ontology" / "classes.ttl")
    if source_key in {"owl", "xml", "rdf", "rdfxml"}:
        return load_graph(project_root / "ontology" / "educational_justice.owl", format="xml")
    raise ValueError("source must be 'ttl' or 'owl'")

def class_uris(graph: Graph, *, include_custom_class_predicates: bool = True) -> set[URIRef]:
    classes: set[URIRef] = set()

    for subject in graph.subjects(RDF.type, OWL.Class):
        if isinstance(subject, URIRef):
            classes.add(subject)

    for subject in graph.subjects(RDF.type, RDFS.Class):
        if isinstance(subject, URIRef):
            classes.add(subject)

    if include_custom_class_predicates:
        for subject, predicate, object_ in graph:
            if (
                predicate == RDF.type
                and isinstance(object_, URIRef)
                and str(object_).endswith("#Class")
                and isinstance(subject, URIRef)
            ):
                classes.add(subject)

    return classes

def class_names(graph: Graph) -> set[str]:
    return {
        graph.namespace_manager.normalizeUri(uri)
        for uri in class_uris(graph)
    }

def has_class(graph: Graph, class_uri: str) -> bool:
    return URIRef(class_uri) in class_uris(graph)

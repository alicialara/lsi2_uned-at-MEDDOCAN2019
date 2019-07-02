import os
import argparse
# from .classes import i2b2Annotation, BratAnnotation, NER_Evaluation, Span_Evaluation
from collections import defaultdict

# from main.classes import i2b2Annotation, BratAnnotation, NER_Evaluation, Span_Evaluation
from main.classes import i2b2Annotation, BratAnnotation, NER_Evaluation, Span_Evaluation

def get_document_dict_by_system_id(system_dirs, annotation_format):
    """Takes a list of directories and returns annotations. """

    documents = defaultdict(lambda: defaultdict(int))

    for d in system_dirs:
        for fn in os.listdir(d):
            if fn.endswith(".ann") or fn.endswith(".xml"):
                sa = annotation_format(os.path.join(d, fn))
                documents[sa.sys_id][sa.id] = sa

    return documents


def evaluate(gs, system, annotation_format, subtrack, **kwargs):
    """Evaluate the system by calling either NER_evaluation or Span_Evaluation.
    'system' can be a list containing either one file,  or one or more
    directories. 'gs' can be a file or a directory. """

    gold_ann = {}
    evaluations = []

    # Strip verbose keyword if it exists
    try:
        verbose = kwargs['verbose']
        del kwargs['verbose']
    except KeyError:
        verbose = False

    # Handle if two files were passed on the command line
    if os.path.isfile(system[0]) and os.path.isfile(gs):
        if (system[0].endswith(".ann") and gs.endswith(".ann")) or \
                (system[0].endswith(".xml") or gs.endswith(".xml")):
            gs = annotation_format(gs)
            sys = annotation_format(system[0])
            e = subtrack({sys.id: sys}, {gs.id: gs}, **kwargs)
            e.print_docs()
            evaluations.append(e)

    # Handle the case where 'gs' is a directory and 'system' is a list of directories.
    elif all([os.path.isdir(sys) for sys in system]) and os.path.isdir(gs):
        # Get a dict of gold annotations indexed by id

        for filename in os.listdir(gs):
            if filename.endswith(".ann") or filename.endswith(".xml"):
                annotations = annotation_format(os.path.join(gs, filename))
                gold_ann[annotations.id] = annotations

        for system_id, system_ann in sorted(get_document_dict_by_system_id(system, annotation_format).items()):
            e = subtrack(system_ann, gold_ann, **kwargs)
            e.print_report(verbose=verbose)
            evaluations.append(e)

    else:
        Exception("Must pass file file or [directory/]+ directory/"
                  "on command line!")

    return evaluations[0] if len(evaluations) == 1 else evaluations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation script for the MEDDOCAN track.")

    parser.add_argument("format",
                        choices=["i2b2", "brat"],
                        help="Format")
    parser.add_argument("subtrack",
                        choices=["ner", "spans"],
                        help="Subtrack")
    parser.add_argument('-v', '--verbose',
                        help="List also scores for each document",
                        action="store_true")
    parser.add_argument("gs_dir",
                        help="Directory to load GS from")
    parser.add_argument("sys_dir",
                        help="Directories with system outputs (one or more)",
                        nargs="+")

    args = parser.parse_args()

    evaluate(args.gs_dir,
             args.sys_dir,
             i2b2Annotation if args.format == "i2b2" else BratAnnotation,
             NER_Evaluation if args.subtrack == "ner" else Span_Evaluation,
             verbose=args.verbose)

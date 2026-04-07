from __future__ import annotations

import argparse

from phylometa.io import read_tree, read_metadata, get_tip_names
from phylometa.plotting import plot_tree_with_traits
from phylometa.qc import check_tip_metadata_overlap
from phylometa.stats_categorical import (
    batch_categorical_tests,
    categorical_clustering_test,
    clade_enrichment_test,
)
from phylometa.stats_continuous import continuous_trait_association_test
from phylometa.tree import patristic_distance_matrix
from phylometa.utils import write_result_dict


def build_parser():
    parser = argparse.ArgumentParser(
        prog="phylometa",
        description="Analyze associations between phylogenetic trees and sample metadata.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--tree", required=True, help="Input tree file in Newick format")
    common.add_argument("--meta", required=True, help="Metadata table (TSV or CSV)")
    common.add_argument("--id-column", required=True, help="Metadata column matching tree tip names")

    subparsers.add_parser("check", parents=[common], help="Check tree/metadata compatibility")

    p_cat = subparsers.add_parser("test-categorical", parents=[common], help="Test categorical trait clustering")
    p_cat.add_argument("--trait", required=True, help="Categorical trait column")
    p_cat.add_argument("--n-perm", type=int, default=1000, help="Number of permutations")
    p_cat.add_argument("--out", required=True, help="Output TSV/CSV file")

    p_batch = subparsers.add_parser("test-categorical-batch", parents=[common], help="Run clustering tests for multiple categorical traits")
    p_batch.add_argument("--traits", required=True, help="Comma-separated categorical trait columns")
    p_batch.add_argument("--n-perm", type=int, default=1000, help="Number of permutations")
    p_batch.add_argument("--out", required=True, help="Output TSV/CSV file")

    p_enrich = subparsers.add_parser("clade-enrichment", parents=[common], help="Test clade enrichment for a trait value")
    p_enrich.add_argument("--trait", required=True, help="Categorical trait column")
    p_enrich.add_argument("--value", required=True, help="Target categorical value to test")
    p_enrich.add_argument("--min-clade-size", type=int, default=3, help="Minimum clade size")
    p_enrich.add_argument("--out", required=True, help="Output TSV/CSV file")

    p_cont = subparsers.add_parser("test-continuous", parents=[common], help="Test continuous trait association")
    p_cont.add_argument("--trait", required=True, help="Continuous trait column")
    p_cont.add_argument("--n-perm", type=int, default=1000, help="Number of permutations")
    p_cont.add_argument("--out", required=True, help="Output TSV/CSV file")

    p_plot = subparsers.add_parser("plot", parents=[common], help="Plot tree with metadata traits")
    p_plot.add_argument("--traits", required=True, help="Comma-separated list of trait columns")
    p_plot.add_argument("--out", required=True, help="Output figure file (PDF/PNG/SVG)")

    return parser


def cmd_check(args):
    tree = read_tree(args.tree)
    meta = read_metadata(args.meta)
    tips = get_tip_names(tree)
    result = check_tip_metadata_overlap(tips, meta, args.id_column)

    print(f"Tree tips: {result['n_tree_tips']}")
    print(f"Metadata IDs: {result['n_metadata_ids']}")
    print(f"Shared IDs: {result['n_shared']}")

    if result["only_tree"]:
        print("\nIDs present only in tree:")
        for x in result["only_tree"][:20]:
            print(f"  {x}")
        if len(result["only_tree"]) > 20:
            print(f"  ... and {len(result['only_tree']) - 20} more")

    if result["only_metadata"]:
        print("\nIDs present only in metadata:")
        for x in result["only_metadata"][:20]:
            print(f"  {x}")
        if len(result["only_metadata"]) > 20:
            print(f"  ... and {len(result['only_metadata']) - 20} more")


def cmd_test_categorical(args):
    tree = read_tree(args.tree)
    meta = read_metadata(args.meta)
    tips = get_tip_names(tree)
    dist_df = patristic_distance_matrix(tree, tips)
    result = categorical_clustering_test(dist_df=dist_df, meta=meta, id_column=args.id_column, trait=args.trait, n_perm=args.n_perm)
    write_result_dict(result, args.out)
    print(result)


def cmd_test_categorical_batch(args):
    tree = read_tree(args.tree)
    meta = read_metadata(args.meta)
    tips = get_tip_names(tree)
    dist_df = patristic_distance_matrix(tree, tips)
    traits = [x.strip() for x in args.traits.split(",") if x.strip()]
    result = batch_categorical_tests(dist_df=dist_df, meta=meta, id_column=args.id_column, traits=traits, n_perm=args.n_perm)
    if args.out.endswith(".csv"):
        result.to_csv(args.out, index=False)
    else:
        result.to_csv(args.out, sep="\t", index=False)
    print(result)


def cmd_clade_enrichment(args):
    tree = read_tree(args.tree)
    meta = read_metadata(args.meta)
    result = clade_enrichment_test(tree=tree, meta=meta, id_column=args.id_column, trait=args.trait, target_value=args.value, min_clade_size=args.min_clade_size)
    if args.out.endswith(".csv"):
        result.to_csv(args.out, index=False)
    else:
        result.to_csv(args.out, sep="\t", index=False)
    print(result.head())


def cmd_test_continuous(args):
    tree = read_tree(args.tree)
    meta = read_metadata(args.meta)
    tips = get_tip_names(tree)
    dist_df = patristic_distance_matrix(tree, tips)
    result = continuous_trait_association_test(dist_df=dist_df, meta=meta, id_column=args.id_column, trait=args.trait, n_perm=args.n_perm)
    write_result_dict(result, args.out)
    print(result)


def cmd_plot(args):
    tree = read_tree(args.tree)
    meta = read_metadata(args.meta)
    traits = [x.strip() for x in args.traits.split(",") if x.strip()]
    plot_tree_with_traits(tree=tree, meta=meta, id_column=args.id_column, traits=traits, out_file=args.out)
    print(f"Saved plot to {args.out}")


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "check":
        cmd_check(args)
    elif args.command == "test-categorical":
        cmd_test_categorical(args)
    elif args.command == "test-categorical-batch":
        cmd_test_categorical_batch(args)
    elif args.command == "clade-enrichment":
        cmd_clade_enrichment(args)
    elif args.command == "test-continuous":
        cmd_test_continuous(args)
    elif args.command == "plot":
        cmd_plot(args)
    else:
        parser.error("Unknown command")


if __name__ == "__main__":
    main()

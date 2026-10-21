#!/usr/bin/env python3
# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0


"""
Script to extract metrics from text files and format them into aligned tables.
"""

import sys
import re
from collections import defaultdict
import argparse


def parse_metrics_file(file_path):
    """Parse a metrics file and extract all metrics into a dictionary."""
    # Define the exact metric names to search for
    target_metrics = [
        # Planning metrics: core
        "plan_L2_1s",
        "plan_L2_2s",
        "plan_L2_3s",
        "plan_obj_box_col_1s",
        "plan_obj_box_col_2s",
        "plan_obj_box_col_3s",
        # Detection metrics
        "mAP",
        "mATE",
        "mASE",
        "mAOE",
        "mAVE",
        "mAAE",
        "NDS",
        # Segmentation metrics (mIoU)
        "map",
        "divider",
        "ped_crossing",
        "boundary",
        # Motion Prediction metrics
        "EPA_car",
        "EPA_pedestrian",
        "ADE_car",
        "ADE_pedestrian",
        "FDE_car",
        "FDE_pedestrian",
        "MR_car",
        "MR_pedestrian",
        # Planning metrics: additional
        "gt_car",
        "gt_pedestrian",
        "cnt_ade_car",
        "cnt_ade_pedestrian",
        "cnt_fde_car",
        "cnt_fde_pedestrian",
        "hit_car",
        "hit_pedestrian",
        "fp_car",
        "fp_pedestrian",
        "ADE_car",
        "ADE_pedestrian",
        "FDE_car",
        "FDE_pedestrian",
        "MR_car",
        "MR_pedestrian",
        "plan_obj_col_1s",
        "plan_obj_col_2s",
        "plan_obj_col_3s",
        "fut_valid_flag",
    ]

    metrics = {}

    try:
        with open(file_path, "r") as f:
            content = f.read()

        # Search for each target metric specifically
        for metric_name in target_metrics:
            # Create pattern for this specific metric: metric_name: value
            pattern = rf"{re.escape(metric_name)}\s*:\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)"

            match = re.search(pattern, content)
            if match:
                metric_value = float(match.group(1))
                metrics[metric_name] = metric_value

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return {}
    except Exception as e:
        print(f"Error reading file '{file_path}': {e}")
        return {}

    return metrics


def format_value_rounded(value, metric_name=None):
    """Format value for the rounded table."""
    # Define metrics that should be converted to percentages when < 1
    percentage_metrics = [
        "MR_car",
        "MR_pedestrian",  # Miss Rate
        "mAP",
        "NDS",  # Detection metrics (already in 0-1 range)
        "plan_obj_col_1s",
        "plan_obj_col_2s",
        "plan_obj_col_3s",  # Collision rates
        "plan_obj_box_col_1s",
        "plan_obj_box_col_2s",
        "plan_obj_box_col_3s",  # Box collision rates
        "fut_valid_flag",  # Flag (if < 1)
        "divider",
        "ped_crossing",
        "boundary",
        "map",  # mIoU metrics
    ]

    # Convert to percentage if metric is in percentage list and value is between 0 and 1
    if metric_name and 0 < abs(value) <= 1 and metric_name in percentage_metrics:
        return f"{value * 100:.2f}%"
    else:
        return f"{value:.3f}"


def format_value_original(value):
    """Format value for the original table."""
    return f"{value:.8f}".rstrip("0").rstrip(".")


def create_tables(data_dict, output_file):
    """Create and save formatted tables."""
    if not data_dict:
        print("No data to process.")
        return

    # Define the metric order (same as in parse_metrics_file)
    target_metrics_order = [
        # Planning metrics: core
        "plan_L2_1s",
        "plan_L2_2s",
        "plan_L2_3s",
        "plan_obj_box_col_1s",
        "plan_obj_box_col_2s",
        "plan_obj_box_col_3s",
        # Detection metrics
        "mAP",
        "mATE",
        "mASE",
        "mAOE",
        "mAVE",
        "mAAE",
        "NDS",
        # Segmentation metrics (mIoU)
        "map",
        "divider",
        "ped_crossing",
        "boundary",
        # Motion Prediction metrics
        "EPA_car",
        "EPA_pedestrian",
        "ADE_car",
        "ADE_pedestrian",
        "FDE_car",
        "FDE_pedestrian",
        "MR_car",
        "MR_pedestrian",
        # Planning metrics: additional
        "gt_car",
        "gt_pedestrian",
        "cnt_ade_car",
        "cnt_ade_pedestrian",
        "cnt_fde_car",
        "cnt_fde_pedestrian",
        "hit_car",
        "hit_pedestrian",
        "fp_car",
        "fp_pedestrian",
        "ADE_car",
        "ADE_pedestrian",
        "FDE_car",
        "FDE_pedestrian",
        "MR_car",
        "MR_pedestrian",
        "plan_obj_col_1s",
        "plan_obj_col_2s",
        "plan_obj_col_3s",
        "fut_valid_flag",
    ]

    # Get all unique metrics that actually exist in the data
    all_metrics = set()
    for metrics in data_dict.values():
        all_metrics.update(metrics.keys())

    # Keep only metrics that exist in data, in the predefined order
    sorted_metrics = [metric for metric in target_metrics_order if metric in all_metrics]

    # Add any additional metrics found in data that weren't in our predefined list
    additional_metrics = sorted(all_metrics - set(target_metrics_order))
    sorted_metrics.extend(additional_metrics)
    sorted_identifiers = list(data_dict.keys())

    # Calculate column widths
    identifier_width = max(len("Identifier"), max(len(id) for id in sorted_identifiers))
    metric_widths = {}

    for metric in sorted_metrics:
        # Check width needed for metric name
        metric_width = len(metric)

        # Check width needed for rounded values
        for identifier in sorted_identifiers:
            if metric in data_dict[identifier]:
                rounded_val = format_value_rounded(data_dict[identifier][metric], metric)
                metric_width = max(metric_width, len(rounded_val))

        # Check width needed for original values
        for identifier in sorted_identifiers:
            if metric in data_dict[identifier]:
                original_val = format_value_original(data_dict[identifier][metric])
                metric_width = max(metric_width, len(original_val))

        metric_widths[metric] = max(metric_width, 8)  # Minimum width of 8

    with open(output_file, "w") as f:
        # Write rounded table
        f.write("=" * 80 + "\n")
        f.write("ROUNDED VALUES TABLE\n")
        f.write("=" * 80 + "\n\n")

        # Header
        header = f"{'Identifier':<{identifier_width}}"
        for metric in sorted_metrics:
            header += f" | {metric:>{metric_widths[metric]}}"
        f.write(header + "\n")

        # Separator line
        separator = "-" * identifier_width
        for metric in sorted_metrics:
            separator += "-+-" + "-" * metric_widths[metric]
        f.write(separator + "\n")

        # Data rows
        for identifier in sorted_identifiers:
            row = f"{identifier:<{identifier_width}}"
            for metric in sorted_metrics:
                if metric in data_dict[identifier]:
                    value_str = format_value_rounded(data_dict[identifier][metric], metric)
                else:
                    value_str = "N/A"
                row += f" | {value_str:>{metric_widths[metric]}}"
            f.write(row + "\n")

        f.write("\n\n")

        # Write original values table
        f.write("=" * 80 + "\n")
        f.write("ORIGINAL VALUES TABLE\n")
        f.write("=" * 80 + "\n\n")

        # Header
        header = f"{'Identifier':<{identifier_width}}"
        for metric in sorted_metrics:
            header += f" | {metric:>{metric_widths[metric]}}"
        f.write(header + "\n")

        # Separator line
        separator = "-" * identifier_width
        for metric in sorted_metrics:
            separator += "-+-" + "-" * metric_widths[metric]
        f.write(separator + "\n")

        # Data rows
        for identifier in sorted_identifiers:
            row = f"{identifier:<{identifier_width}}"
            for metric in sorted_metrics:
                if metric in data_dict[identifier]:
                    value_str = format_value_original(data_dict[identifier][metric])
                else:
                    value_str = "N/A"
                row += f" | {value_str:>{metric_widths[metric]}}"
            f.write(row + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Extract metrics from text files and create formatted tables",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python metrics_extractor.py --files file1.txt file2.txt file3.txt --ids epoch_10 epoch_20 epoch_30
  python metrics_extractor.py -f results1.txt results2.txt -i exp1 exp2 --output my_table.txt
        """,
    )

    parser.add_argument("--files", "-f", nargs="+", required=True, help="List of file paths")
    parser.add_argument(
        "--ids", "-i", nargs="+", required=True, help="List of identifiers (must match number of files)"
    )
    parser.add_argument(
        "--output", "-o", default="metrics_table.txt", help="Output file name (default: metrics_table.txt)"
    )

    args = parser.parse_args()

    if len(args.files) != len(args.ids):
        print(f"Error: Number of files ({len(args.files)}) must match number of identifiers ({len(args.ids)}).")
        print("Usage: python script.py --files file1.txt file2.txt --ids id1 id2")
        sys.exit(1)

    # Parse file paths and identifiers
    data_dict = {}
    for file_path, identifier in zip(args.files, args.ids):
        print(f"Processing {file_path} with identifier '{identifier}'...")
        metrics = parse_metrics_file(file_path)

        if metrics:
            data_dict[identifier] = metrics
            print(f"  Found {len(metrics)} metrics")
        else:
            print(f"  No metrics found or error reading file")

    if not data_dict:
        print("No valid data found. Exiting.")
        sys.exit(1)

    # Create tables
    print(f"\nCreating tables and saving to '{args.output}'...")
    create_tables(data_dict, args.output)
    print(f"Tables saved successfully to '{args.output}'!")

    # Print summary
    print(f"\nSummary:")
    print(f"  Processed {len(data_dict)} files")
    total_metrics = len(set().union(*(metrics.keys() for metrics in data_dict.values())))
    print(f"  Found {total_metrics} unique metrics")


if __name__ == "__main__":
    main()

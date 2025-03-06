# -*- coding: utf-8 -*-
"""
This module contains functions that extract data from NASTRAN '.dat' simulation files.

Functions:
- nastran_float(s): Converts a NASTRAN short-format number string to standard Python float notation.
- split_into_chunks(line, size): Splits a string into fixed-size chunks.
- read_node_coords(filename, nodes): Reads node coordinates from a NASTRAN .dat file.
- read_damping(filename): Extracts a constant damping definition from a .dat file.
"""
import numpy as np


def nastran_float(s):
    """Convert a string containing a number in NASTRAN short format, that is, with only
    an '+' or '-' indicating exponential notation, and no 'e', into python format,
    by adding said 'e'."""
    s = s.replace('-','e-')
    s = s.replace('+','e+')
    if s[0] == 'e':
        s = s[1:]
    return s


def split_into_chunks(line, size):
    """Helper function to split a string into fixed-size chunks."""
    return [line[idx:idx + size].strip() for idx in range(0, len(line), size)]


def read_node_coords(filename, nodes):
    """Reads node coordinates from a NASTRAN .dat file for specified nodes.

    Parameters:
        filename (str): Path to the .dat file.
        nodes (list of str): Node numbers to search for in the file.

    Returns:
        coords (dict): Node numbers as keys, and their coordinates as values (numpy arrays).

    Raises:
        FileNotFoundError: If the file cannot be found.
        ValueError: If the file is empty.
        ValueError: If no matches are found for the given nodes.
    """
    grid_str = 'GRID'
    long_grid_str = 'GRID*'
    short_chunk_size = 8
    long_chunk_size = 16
    coords = {}

    try:
        with open(filename, 'r') as f:
            lines = f.read().splitlines()

        if not lines:
            raise ValueError(f"File '{filename}' is empty.")

        for idx, line in enumerate(lines):
            line_split = line.split()
            if grid_str in line_split[0] and line_split[1] in nodes:
                node = line_split[1]

                if line.startswith(long_grid_str):  # Long format detected
                    try:
                        next_line = lines[idx + 1]
                    except StopIteration:
                        raise ValueError(f"Incomplete long-format GRID entry for node {node} in '{filename}'.")
                    chunks_1 = split_into_chunks(line[short_chunk_size:], long_chunk_size)
                    chunks_2 = split_into_chunks(next_line[short_chunk_size:], long_chunk_size)
                    x = chunks_1[2]
                    y = chunks_1[3]
                    z = chunks_2[0]
                    coords[node] = np.array([x, y, z], dtype=float)

                else:  # Standard short format
                    chunks = [nastran_float(c) for c in split_into_chunks(line, short_chunk_size)]
                    coords[node] = np.array(chunks[3:6], dtype=float)

        if not coords:
            raise ValueError(f"No matching GRID entries found in '{filename}'.")

        missing_nodes = set(nodes) - coords.keys()
        if missing_nodes:
            print(f"WARNING: The following nodes where not found, {sorted(missing_nodes)}.")

        return coords

    except FileNotFoundError:
        raise FileNotFoundError(f"File '{filename}' not found.")


def read_damping(filename):
    """
    Reads a constant damping definition from a NASTRAN .dat file.

    The function searches for a TABDMP1 entry and extracts the damping value.
    Only constant damping defined by two identical values is supported.

    Parameters:
        filename (str): Path to the .dat file.

    Returns:
        float: The extracted damping value.

    Raises:
        ValueError: If damping is not constant or incorrectly defined.
        FileNotFoundError: If the file cannot be found.
        ValueError: If the file is empty or TABDMP1 is not found.
    """
    damp_table_str = 'TABDMP1'
    expected_table_str_length = 6
    short_chunk_size = 8

    try:
        with open(filename, 'r') as f:
            lines = f.read().splitlines()

        if not lines:
            raise ValueError(f"File '{filename}' is empty.")

        for idx, line in enumerate(lines):
            if damp_table_str in line.split():
                # Ensure the next line exists
                if idx + 1 >= len(lines):
                    raise ValueError(f"File '{filename}' has an incomplete TABDMP1 entry.")

                next_line = lines[idx + 1]
                chunks = split_into_chunks(next_line, short_chunk_size)

                # Validate chunk length
                if len(chunks) <= expected_table_str_length:
                    raise ValueError(f"Malformed TABDMP1 entry in '{filename}'.")

                # Ensure damping is constant
                if chunks[2] != chunks[4]:
                    raise ValueError("Only constant damping defined by two identical values is supported.")

                return float(nastran_float(chunks[2]))

        raise ValueError(f"No TABDMP1 entry found in '{filename}'.")
    except FileNotFoundError:
        raise FileNotFoundError(f"File '{filename}' not found.")
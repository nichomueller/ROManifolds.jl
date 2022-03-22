using Pkg
using DataFrames, Random, DataStructures, FillArrays, Tables, LinearAlgebra, Statistics, Distributions, Parameters, Classes, BlockArrays 
using SuiteSparse, SparseArrays, Serialization, ThreadedSparseCSR, SparseMatricesCSR, LazyAlgebra, LazyAlgebra.SparseMethods
using MAT, CSV, JLD, DelimitedFiles, Arrow, CodecLz4
using Test, InteractiveUtils, TimerOutputs, BenchmarkTools, Interact, Logging, Plots
using Conda, PyCall
using Flux, Flux.Data.MNIST
using Gridap, Gridap.FESpaces, Gridap.ReferenceFEs, Gridap.Arrays, Gridap.Geometry, Gridap.Fields, Gridap.CellData

py"""
import numpy as np
"""

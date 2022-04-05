using Pkg
using DataFrames, Random, DataStructures, FillArrays, Tables, LinearAlgebra, Statistics, Distributions, Parameters, Classes, BlockArrays
using Base: @kwdef
using ReusePatterns
using SuiteSparse, SparseArrays, Serialization, ThreadedSparseCSR, SparseMatricesCSR, LazyAlgebra, LazyAlgebra.SparseMethods, Arpack
using MAT, CSV, JLD, DelimitedFiles, Arrow, CodecLz4
using Test, InteractiveUtils, TimerOutputs, BenchmarkTools, Interact, Logging, Plots
using Conda, PyCall
using Flux, Flux.Data.MNIST
using Gridap, Gridap.FESpaces, Gridap.ReferenceFEs, Gridap.Arrays, Gridap.Geometry, Gridap.Fields, Gridap.CellData
import Gridap: ∇
py"""
import numpy as np
"""

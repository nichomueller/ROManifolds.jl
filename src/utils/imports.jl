using DrWatson
#@quickactivate "Mabla.jl"
using Pkg
using DataFrames, Random, DataStructures, FillArrays, Tables, TypedTables, LinearAlgebra, Statistics, Distributions, Parameters, Classes, BlockArrays
using Base:@kwdef
using SuiteSparse, SparseArrays, Serialization, ThreadedSparseCSR, SparseMatricesCSR, LazyAlgebra, LazyAlgebra.SparseMethods, Arpack
using MAT, CSV, JLD, DelimitedFiles, Arrow, CodecLz4
using Test, InteractiveUtils, TimerOutputs, BenchmarkTools, Interact, Logging, Plots, WriteVTK
using Conda, PyCall
using Gridap, Gridap.FESpaces, Gridap.ReferenceFEs, Gridap.Arrays, Gridap.Geometry, Gridap.Fields, Gridap.CellData, GridapGmsh, Gridap.Io, Gridap.TensorValues
import Gridap:∇
using Revise
py"""
import numpy as np
"""
using LineSearches:BackTracking
using ReusePatterns, TensorOperations
#= using MLDatasets
using MLDatasets:CIFAR10
using Flux, Flux.Data.MNIST, Flux.Optimise
using Flux:gradient
using Flux:params
using Flux:Descent
using Flux:crossentropy
using Flux:Momentum
using Flux.Optimise:update!
using Flux:onehotbatch
using Flux:onecold
using Optim, CUDA
using IterTools
using Zygote, FluxOptTools =#

function assemble_fe_snaps(op::ParamOperator,args...;fun_mdeim=false)
  assemble_fe_snaps(Val{fun_mdeim}(),op,args...)
end

function assemble_fe_snaps(val::Val{false},op::ParamOperator,args...)
  id = get_id(op)
  Nt = get_Nt(op)
  assembler = get_assembler(op)
  snap,findnz_idx = assembler(unpack_for_assembly(val,op)...,args...)
  nsnap = get_nsnap(snap,Nt)
  s = Snapshots(id,snap,nsnap)
  s,findnz_idx
end

function assemble_fe_snaps(val::Val{true},op::ParamOperator,args...)
  id = get_id(op)
  assembler = get_assembler(op)
  snap,findnz_idx = assembler(unpack_for_assembly(val,op)...,args...)
  nsnap = size(snap,2)
  s = Snapshots(id,snap,nsnap)
  s,findnz_idx
end

get_assembler(::ParamLinOperator) = assemble_vectors

get_assembler(::ParamBilinOperator) = assemble_matrices

abstract type AssemblerLoop end

struct VectorAssemblerLoop <: AssemblerLoop
  fdata::Function
  nsnap::Int
end

struct MatrixAssemblerLoop <: AssemblerLoop
  fdata::Function
  nsnap::Int
end

get_fdata(al::AssemblerLoop) = al.fdata

get_nsnap(al::AssemblerLoop) = al.nsnap

function assemble_vectors(f::Function,V::FESpace,args...)
  asmblr,vecdata = assembler_setup(f,V,args...)
  v1,findnz_idx = assembler_invariables(asmblr,vecdata)
  assembler_loop(asmblr,vecdata,v1,findnz_idx)
end

function assembler_setup(
  f::Function,
  V::FESpace,
  args...)

  dv = get_fe_basis(V)
  asmblr = SparseMatrixAssembler(V,V)
  vecdata = get_vecdata(f,V,dv,args...)
  asmblr,vecdata
end

function get_vecdata(
  f::Function,
  V::FESpace,
  dv::Gridap.FESpaces.SingleFieldFEBasis,
  μvec::Vector{Param})

  vecdata(i) = collect_cell_vector(V,f(μvec[i],dv))
  nsnap = length(μvec)
  VectorAssemblerLoop(vecdata,nsnap)
end

function get_vecdata(
  f::Function,
  V::FESpace,
  dv::Gridap.FESpaces.SingleFieldFEBasis,
  dir::Function,
  μvec::Vector{Param})

  vecdata(i) = collect_cell_vector(V,f(μvec[i],dir(μvec[i]),dv))
  nsnap = length(μvec)
  VectorAssemblerLoop(vecdata,nsnap)
end

function get_vecdata(
  f::Function,
  V::FESpace,
  dv::Gridap.FESpaces.SingleFieldFEBasis,
  dir::Function,
  μvec::Vector{Param},
  uvec::Vector{<:FEFunction})

  vecdata(i) = collect_cell_vector(V,f(uvec[i],dir(μvec[i]),dv))
  nsnap = length(μvec)
  VectorAssemblerLoop(vecdata,nsnap)
end

function arg_functions(
  μvec::Vector{Param},
  tvec::Vector{Float})

  Nt = length(tvec)
  μ(i) = μvec[slow_idx(i,Nt)]
  t(i) = tvec[fast_idx(i,Nt)]
  Nt,μ,t
end

function arg_functions(
  μvec::Vector{Param},
  tvec::Vector{Float},
  uvec::Vector{T}) where {T<:Union{FEFunction,Function}}

  Nt,μ,t = arg_functions(μvec,tvec)
  u(i) = uvec[slow_idx(i,Nt)]
  Nt,μ,t,u
end

function get_vecdata(
  f::Function,
  V::FESpace,
  dv::Gridap.FESpaces.SingleFieldFEBasis,
  μvec::Vector{Param},
  tvec::Vector{Float})

  Nt,μ,t = arg_functions(μvec,tvec)
  vecdata(i) = collect_cell_vector(V,f(μ(i),t(i),dv))
  nsnap = length(μvec)*Nt
  VectorAssemblerLoop(vecdata,nsnap)
end

function get_vecdata(
  f::Function,
  V::FESpace,
  dv::Gridap.FESpaces.SingleFieldFEBasis,
  dir::Function,
  μvec::Vector{Param},
  tvec::Vector{Float})

  Nt,μ,t = arg_functions(μvec,tvec)
  vecdata(i) = collect_cell_vector(V,f(μ(i),t(i),dir(μ(i),t(i)),dv))
  nsnap = length(μvec)*Nt
  VectorAssemblerLoop(vecdata,nsnap)
end

function get_vecdata(
  f::Function,
  V::FESpace,
  dv::Gridap.FESpaces.SingleFieldFEBasis,
  dir::Function,
  μvec::Vector{Param},
  tvec::Vector{Float},
  uvec::Vector{<:FEFunction})

  Nt,μ,t,u = arg_functions(μvec,tvec,uvec)
  vecdata(i) = collect_cell_vector(V,f(u(i),dir(μ(i),t(i)),dv))
  nsnap = length(μvec)*Nt
  VectorAssemblerLoop(vecdata,nsnap)
end

function assembler_invariables(
  asmblr::SparseMatrixAssembler,
  vecdata::VectorAssemblerLoop)

  data = get_fdata(vecdata)
  vecdata1 = data(1)
  v1 = Gridap.Algebra.nz_counter(get_vector_builder(asmblr),(get_rows(asmblr),))
  symbolic_loop_vector!(v1,asmblr,vecdata1)
  findnz_idx = collect(eachindex(get_rows(asmblr)))

  v1,findnz_idx
end

function assembler_loop(
  a::SparseMatrixAssembler,
  vecdata::VectorAssemblerLoop,
  v1::Gridap.Algebra.ArrayCounter{Vector{Float},Tuple{Base.OneTo{Int}}},
  findnz_idx::Vector{Int})

  data = get_fdata(vecdata)
  nsnap = get_nsnap(vecdata)
  Nz = length(findnz_idx)

  vecs = Elemental.zeros(EMatrix{Float},Nz,nsnap)
  for i = 1:nsnap
    vecdata_i = data(i)
    v2 = Gridap.Algebra.nz_allocation(v1)
    numeric_loop_vector!(v2,a,vecdata_i)
    copyto!(view(vecs,:,i),v2)
  end

  vecs[findnz_idx,:],findnz_idx
end

function assemble_matrices(f::Function,U,V::FESpace,args...)
  asmblr,matdata = assembler_setup(f,U,V,args...)
  m1,findnz_idx = assembler_invariables(asmblr,matdata)
  assembler_loop(asmblr,matdata,m1,findnz_idx)
end

function assembler_setup(
  f::Function,
  U::T,
  V::FESpace,
  args...) where T<:Union{TrialFESpace,ParamTrialFESpace,ParamTransientTrialFESpace}

  U1 = realization_trial(U,args...)
  dv = get_fe_basis(V)
  du = get_trial_fe_basis(U1)
  asmblr = SparseMatrixAssembler(U1,V)
  matdata = get_matdata(f,U,V,du,dv,args...)
  asmblr,matdata
end

function get_matdata(
  f::Function,
  U::ParamTrialFESpace,
  V::FESpace,
  du::Gridap.FESpaces.SingleFieldFEBasis,
  dv::Gridap.FESpaces.SingleFieldFEBasis,
  μvec::Vector{Param})

  matdata(i) = collect_cell_matrix(U(μvec[i]),V,f(μvec[i],du,dv))
  nsnap = length(μvec)
  MatrixAssemblerLoop(matdata,nsnap)
end

function get_matdata(
  f::Function,
  U::ParamTrialFESpace,
  V::FESpace,
  du::Gridap.FESpaces.SingleFieldFEBasis,
  dv::Gridap.FESpaces.SingleFieldFEBasis,
  μvec::Vector{Param},
  uvec::Vector{<:FEFunction})

  matdata(i) = collect_cell_matrix(U(μvec[i]),V,f(uvec[i],du,dv))
  nsnap = length(μvec)
  MatrixAssemblerLoop(matdata,nsnap)
end

function get_matdata(
  f::Function,
  U::ParamTransientTrialFESpace,
  V::FESpace,
  du::Gridap.FESpaces.SingleFieldFEBasis,
  dv::Gridap.FESpaces.SingleFieldFEBasis,
  μvec::Vector{Param},
  tvec::Vector{Float})

  Nt,μ,t = arg_functions(μvec,tvec)
  matdata(i) = collect_cell_matrix(U(μ(i),t(i)),V,f(μ(i),t(i),du,dv))
  nsnap = length(μvec)*Nt
  MatrixAssemblerLoop(matdata,nsnap)
end

function get_matdata(
  f::Function,
  U::ParamTransientTrialFESpace,
  V::FESpace,
  du::Gridap.FESpaces.SingleFieldFEBasis,
  dv::Gridap.FESpaces.SingleFieldFEBasis,
  μvec::Vector{Param},
  tvec::Vector{Float},
  uvec::Vector{<:FEFunction})

  Nt,μ,t,u = arg_functions(μvec,tvec,uvec)
  matdata(i) = collect_cell_matrix(U(μ(i),t(i)),V,f(u(i),du,dv))
  nsnap = length(μvec)*Nt
  MatrixAssemblerLoop(matdata,nsnap)
end

# FUNCTIONAL MDEIM
function get_matdata(
  f::Function,
  U::TrialFESpace,
  V::FESpace,
  du::Gridap.FESpaces.SingleFieldFEBasis,
  dv::Gridap.FESpaces.SingleFieldFEBasis,
  fvec::Vector{<:FEFunction})

  matdata(i) = collect_cell_matrix(U,V,f(fvec[i],du,dv))
  nsnap = length(fvec)
  MatrixAssemblerLoop(matdata,nsnap)
end

function assembler_invariables(
  asmblr::SparseMatrixAssembler,
  matdata::MatrixAssemblerLoop)

  data = get_fdata(matdata)
  matdata1 = data(1)
  m1 = Gridap.Algebra.nz_counter(get_matrix_builder(asmblr),(get_rows(asmblr),get_cols(asmblr)))
  symbolic_loop_matrix!(m1,asmblr,matdata1)
  m2 = Gridap.Algebra.nz_allocation(m1)
  numeric_loop_matrix!(m2,asmblr,matdata1)
  m3 = Gridap.Algebra.create_from_nz(m2)
  findnz_idx = get_findnz_idx(m3)

  m1,findnz_idx
end

function assembler_loop(
  a::SparseMatrixAssembler,
  matdata::MatrixAssemblerLoop,
  m1::Gridap.Algebra.CounterCSC{Float,Int,Gridap.Algebra.Loop},
  findnz_idx::Vector{Int})

  data = get_fdata(matdata)
  nsnap = get_nsnap(matdata)
  Nz = length(findnz_idx)

  mats = Elemental.zeros(EMatrix{Float},Nz,nsnap)
  for i = 1:nsnap
    matdata_i = data(i)
    m2 = Gridap.Algebra.nz_allocation(m1)
    numeric_loop_matrix!(m2,a,matdata_i)
    copyto!(view(mats,:,i),my_create_from_nz(m2))
  end

  mats,findnz_idx
end

function my_create_from_nz(
  asmblr::Gridap.Algebra.InserterCSC{Tv,Ti}) where {Tv,Ti}

  k = 1
  for j in 1:asmblr.ncols
    pini = Int(asmblr.colptr[j])
    pend = pini + Int(asmblr.colnnz[j]) - 1
    for p in pini:pend
      asmblr.nzval[k] = asmblr.nzval[p]
      k += 1
    end
  end
  @inbounds for j in 1:asmblr.ncols
    asmblr.colptr[j+1] = asmblr.colnnz[j]
  end
  Gridap.FESpaces.length_to_ptrs!(asmblr.colptr)
  nnz = asmblr.colptr[end]-1
  resize!(asmblr.nzval,nnz)

  my_nnz = findall(v -> abs.(v) .>= eps(),asmblr.nzval)
  asmblr.nzval[my_nnz]
end

function get_findnz_idx(mat::EMatrix{Float})
  sum_cols = sum(mat,dims=2)[:]
  findall(x -> abs(x) ≥ eps(),sum_cols)
end

function get_findnz_idx(mat::SparseMatrixCSC{Float,Int})
  findnz_idx, = findnz(mat[:])
  findnz_idx
end

function assemble_functional_snaps(
  op::ParamOperator,
  μvec::Vector{Param})

  quadp = get_phys_quad_points(op)
  Nq = length(quadp)
  matdata = evaluate_param_function(op,quadp,μvec)
  data = get_fdata(matdata)
  nsnap = get_nsnap(matdata)

  snaps = Elemental.zeros(EMatrix{Float},Nq,nsnap)
  for i = 1:nsnap
    copyto!(view(snaps,:,i),data(i))
  end

  Snapshots(get_id(op),snaps,length(μvec))
end

function assemble_functional_snaps(
  ::ParamOperator{Nonlinear,Ttr},
  ::Vector{Param},
  snaps::Snapshots,
  snaps_dir::Snapshots) where Ttr

  snaps,snaps_dir
end

function evaluate_param_function(
  ::ParamSteadyOperator,
  quadp,
  μvec::Vector{Param})

  nsnap = length(μvec)
  param_fun = get_param_function(op)

  paramdata(i) = Broadcasting(x->param_fun(x,μvec[i]))(quadp)
  MatrixAssemblerLoop(paramdata,nsnap)
end

function evaluate_param_function(
  op::ParamUnsteadyOperator,
  quadp,
  μvec::Vector{Param})

  tvec = get_times(op)
  Nt,μ,t = arg_functions(μvec,tvec)
  nsnap = length(μvec)*Nt
  param_fun = get_param_function(op)

  paramdata(i) = Broadcasting(x->param_fun(x,μ(i),t(i)))(quadp)
  MatrixAssemblerLoop(paramdata,nsnap)
end
